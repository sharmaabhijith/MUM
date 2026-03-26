"""Main evaluation runner — orchestrates memory method + answer generation + judging.

Supports multi-model evaluation with resume, per-question incremental save,
and detailed file + console logging.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from datagen.llm.client import LLMClient
from datagen.llm.cost_tracker import CostTracker
from datagen.models.schemas import (
    ConversationSession,
    EvalQuestion,
    SessionSummary,
)
from datagen.utils.io import read_json, write_json

from memory.evaluation.judge import JudgeScore, LLMJudge
from memory.methods.base import BaseMemoryMethod
from memory.prompts.answer_gen import build_answer_prompt

logger = logging.getLogger("mum.memory")
console = Console()

BENCHMARK_DIR = Path("MUMBench")

# ── Short names for display ───────────────────────────────────────────────────

MODEL_SHORT_NAMES: dict[str, str] = {
    "Qwen/Qwen3-14B": "Qwen3-14B",
    "Qwen/Qwen3-32B": "Qwen3-32B",
    "Qwen/Qwen3-Next-80B-A3B-Instruct": "Qwen3-Next-80B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama3.1-8B",
    "meta-llama/Meta-Llama-3.1-70B-Instruct": "Llama3.1-70B",
    "deepseek-ai/DeepSeek-V3.2": "DeepSeek-V3.2",
    "google/gemini-2.5-flash": "Gemini-2.5-Flash",
    "google/gemini-2.5-pro": "Gemini-2.5-Pro",
}

# The 7 models that will be evaluated by default
EVAL_MODELS: list[str] = [
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "deepseek-ai/DeepSeek-V3.2",
    "google/gemini-2.5-flash",
]


def short_model_name(model: str) -> str:
    return MODEL_SHORT_NAMES.get(model, model.rsplit("/", 1)[-1])


# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass
class QuestionResult:
    """Result for a single evaluation question."""

    question_id: str
    category: str
    difficulty: str
    question: str
    gold_answer: str
    predicted_answer: str
    judge_score: JudgeScore
    context_tokens: int
    answer_time_s: float

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "category": self.category,
            "difficulty": self.difficulty,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "predicted_answer": self.predicted_answer,
            "judge_score": self.judge_score.to_dict(),
            "context_tokens": self.context_tokens,
            "answer_time_s": round(self.answer_time_s, 2),
        }


@dataclass
class ScenarioResult:
    """Aggregate results for one scenario with one memory method + one model."""

    scenario_id: str
    method_name: str
    model: str
    question_results: list[QuestionResult] = field(default_factory=list)
    total_time_s: float = 0.0

    @property
    def num_questions(self) -> int:
        return len(self.question_results)

    def overall_score(self) -> float:
        if not self.question_results:
            return 0.0
        return sum(qr.judge_score.overall for qr in self.question_results) / len(
            self.question_results
        )

    def scores_by_category(self) -> dict[str, dict]:
        by_cat: dict[str, list[float]] = defaultdict(list)
        for qr in self.question_results:
            by_cat[qr.category].append(qr.judge_score.overall)
        return {
            cat: {
                "count": len(scores),
                "mean": round(sum(scores) / len(scores), 4),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
            }
            for cat, scores in sorted(by_cat.items())
        }

    def scores_by_difficulty(self) -> dict[str, dict]:
        by_diff: dict[str, list[float]] = defaultdict(list)
        for qr in self.question_results:
            by_diff[qr.difficulty].append(qr.judge_score.overall)
        return {
            diff: {
                "count": len(scores),
                "mean": round(sum(scores) / len(scores), 4),
            }
            for diff, scores in sorted(by_diff.items())
        }

    def dimension_averages(self) -> dict[str, float]:
        if not self.question_results:
            return {}
        n = len(self.question_results)
        return {
            "correctness": sum(qr.judge_score.correctness for qr in self.question_results) / n,
            "completeness": sum(qr.judge_score.completeness for qr in self.question_results) / n,
            "attribution": sum(qr.judge_score.attribution for qr in self.question_results) / n,
            "hallucination": sum(qr.judge_score.hallucination for qr in self.question_results) / n,
        }

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "method_name": self.method_name,
            "model": self.model,
            "model_short": short_model_name(self.model),
            "num_questions": self.num_questions,
            "overall_score": round(self.overall_score(), 4),
            "scores_by_category": self.scores_by_category(),
            "scores_by_difficulty": self.scores_by_difficulty(),
            "dimension_averages": {
                k: round(v, 4) for k, v in self.dimension_averages().items()
            },
            "total_time_s": round(self.total_time_s, 2),
            "question_results": [qr.to_dict() for qr in self.question_results],
        }


# ── File logger setup ─────────────────────────────────────────────────────────


def _setup_eval_file_logger(results_dir: Path) -> logging.FileHandler:
    """Create a rotating file handler that writes detailed eval logs."""
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"eval_{ts}.log"

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)

    # Attach to both loggers
    logging.getLogger("mum").addHandler(handler)
    logging.getLogger("mum.memory").addHandler(handler)
    logger.info(f"Evaluation log file: {log_path}")
    return handler


# ── Evaluation runner ─────────────────────────────────────────────────────────


class EvaluationRunner:
    """Runs evaluation for a given memory method against the benchmark.

    Supports per-question incremental save and resume from partial runs.
    """

    def __init__(
        self,
        method: BaseMemoryMethod,
        answer_model: str = "google/gemini-2.5-pro",
        judge_model: str = "google/gemini-2.5-pro",
        benchmark_dir: Path | str = BENCHMARK_DIR,
    ):
        self.method = method
        self.answer_model = answer_model
        self.benchmark_dir = Path(benchmark_dir)
        self.cost_tracker = CostTracker()
        self.answer_llm = LLMClient(
            model=answer_model,
            temperature=0.3,
            cost_tracker=self.cost_tracker,
        )
        self.judge = LLMJudge(model=judge_model, cost_tracker=self.cost_tracker)

    # ── data loading ──────────────────────────────────────────────────────

    def load_scenario_data(
        self, scenario_id: str
    ) -> tuple[list[ConversationSession], list[SessionSummary], list[EvalQuestion]]:
        scenario_dir = self.benchmark_dir / f"scenario_{scenario_id}"

        conv_dir = scenario_dir / "conversations"
        conversations = []
        if conv_dir.exists():
            for f in sorted(conv_dir.glob("*.json")):
                conversations.append(
                    ConversationSession.model_validate(read_json(f))
                )

        summ_dir = scenario_dir / "summaries"
        summaries = []
        if summ_dir.exists():
            summ_files = sorted(summ_dir.glob("*_summary.json"))
            if summ_files:
                for f in summ_files:
                    summaries.append(SessionSummary.model_validate(read_json(f)))
            else:
                combined = summ_dir / "session_summaries.json"
                if combined.exists():
                    for item in read_json(combined):
                        summaries.append(SessionSummary.model_validate(item))

        eval_path = scenario_dir / "evaluation" / "eval_questions.json"
        questions = []
        if eval_path.exists():
            for item in read_json(eval_path):
                questions.append(EvalQuestion.model_validate(item))

        logger.info(
            f"Loaded scenario {scenario_id}: "
            f"{len(conversations)} convs, {len(summaries)} summaries, "
            f"{len(questions)} eval questions"
        )
        return conversations, summaries, questions

    # ── resume support ────────────────────────────────────────────────────

    def _result_path(self, scenario_id: str, output_dir: Path | None = None) -> Path:
        d = output_dir or (self.benchmark_dir / "results")
        model_slug = short_model_name(self.answer_model).replace(".", "_").lower()
        return d / f"eval_{self.method.name}__{model_slug}__s{scenario_id}.json"

    def _load_completed_ids(self, path: Path) -> set[str]:
        """Load question IDs already completed from a partial result file."""
        if not path.exists():
            return set()
        try:
            data = read_json(path)
            return {qr["question_id"] for qr in data.get("question_results", [])}
        except Exception:
            return set()

    def _load_partial_results(self, path: Path) -> list[QuestionResult]:
        """Reload QuestionResult objects from a partial result file."""
        if not path.exists():
            return []
        try:
            data = read_json(path)
            results = []
            for qr in data.get("question_results", []):
                js = qr["judge_score"]
                results.append(QuestionResult(
                    question_id=qr["question_id"],
                    category=qr["category"],
                    difficulty=qr["difficulty"],
                    question=qr["question"],
                    gold_answer=qr["gold_answer"],
                    predicted_answer=qr["predicted_answer"],
                    judge_score=JudgeScore(
                        question_id=js["question_id"],
                        correctness=js["correctness"],
                        completeness=js["completeness"],
                        attribution=js["attribution"],
                        hallucination=js["hallucination"],
                        reasoning=js["reasoning"],
                        is_binary=js.get("is_binary", False),
                        binary_correct=js.get("binary_correct"),
                    ),
                    context_tokens=qr["context_tokens"],
                    answer_time_s=qr["answer_time_s"],
                ))
            return results
        except Exception as e:
            logger.warning(f"Failed to load partial results from {path}: {e}")
            return []

    # ── main run ──────────────────────────────────────────────────────────

    def run_scenario(
        self,
        scenario_id: str,
        resume: bool = True,
        output_dir: Path | None = None,
    ) -> ScenarioResult:
        """Run full evaluation for one scenario, with resume + incremental save."""
        start_time = time.time()
        m_short = short_model_name(self.answer_model)

        conversations, summaries, questions = self.load_scenario_data(scenario_id)

        if not questions:
            logger.warning(f"No eval questions found for scenario {scenario_id}")
            return ScenarioResult(
                scenario_id=scenario_id,
                method_name=self.method.name,
                model=self.answer_model,
            )

        # Resume support
        result_path = self._result_path(scenario_id, output_dir)
        completed_ids: set[str] = set()
        prior_results: list[QuestionResult] = []
        if resume:
            completed_ids = self._load_completed_ids(result_path)
            if completed_ids:
                prior_results = self._load_partial_results(result_path)
                logger.info(
                    f"Resuming: {len(completed_ids)}/{len(questions)} already done "
                    f"for {m_short} / {self.method.name} / S{scenario_id}"
                )

        pending = [q for q in questions if q.question_id not in completed_ids]
        if not pending:
            logger.info(
                f"All {len(questions)} questions already completed for "
                f"{m_short} / {self.method.name} / S{scenario_id} — skipping"
            )
            result = ScenarioResult(
                scenario_id=scenario_id,
                method_name=self.method.name,
                model=self.answer_model,
                question_results=prior_results,
                total_time_s=0.0,
            )
            return result

        # Ingest
        logger.info(
            f"{'=' * 60}\n"
            f"  Model:    {m_short} ({self.answer_model})\n"
            f"  Method:   {self.method.name}\n"
            f"  Scenario: {scenario_id}\n"
            f"  Questions: {len(pending)} pending / {len(questions)} total\n"
            f"{'=' * 60}"
        )
        self.method.ingest(conversations, summaries)

        result = ScenarioResult(
            scenario_id=scenario_id,
            method_name=self.method.name,
            model=self.answer_model,
            question_results=list(prior_results),
        )

        # Process pending questions with rich progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"S{scenario_id} | {m_short} | {self.method.name}",
                total=len(pending),
            )

            for i, q in enumerate(pending):
                q_start = time.time()

                logger.info(
                    f"[{len(result.question_results) + 1}/{len(questions)}] "
                    f"{q.category} | {q.difficulty} | {q.question_id}"
                )

                qr = self._answer_and_judge(q)
                result.question_results.append(qr)

                q_elapsed = time.time() - q_start
                logger.info(
                    f"  -> score={qr.judge_score.overall:.3f} "
                    f"ctx={qr.context_tokens:,}tok "
                    f"time={q_elapsed:.1f}s "
                    f"[C={qr.judge_score.correctness:.0f} "
                    f"Co={qr.judge_score.completeness:.0f} "
                    f"A={qr.judge_score.attribution:.0f} "
                    f"H={qr.judge_score.hallucination:.0f}]"
                )

                # Incremental save every 5 questions
                if (i + 1) % 5 == 0 or (i + 1) == len(pending):
                    result.total_time_s = time.time() - start_time
                    self._save_result(result, result_path)

                progress.update(task, advance=1)

        result.total_time_s = time.time() - start_time
        self._save_result(result, result_path)
        self.method.reset()

        # Log summary
        logger.info(
            f"Scenario {scenario_id} DONE | {m_short} | {self.method.name}\n"
            f"  overall={result.overall_score():.4f} | "
            f"questions={result.num_questions} | "
            f"time={result.total_time_s:.0f}s | "
            f"cost=${self.cost_tracker.get_total_cost():.4f}"
        )

        cat_scores = result.scores_by_category()
        for cat, info in cat_scores.items():
            logger.info(f"  {cat}: {info['mean']:.3f} (n={info['count']})")

        return result

    def _answer_and_judge(self, q: EvalQuestion) -> QuestionResult:
        t0 = time.time()
        memory_ctx = self.method.retrieve(q.question)

        messages = build_answer_prompt(
            question=q.question,
            context=memory_ctx.context_text,
            method_name=memory_ctx.method_name,
        )
        predicted_answer, _ = self.answer_llm.generate(
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
            phase="eval_answer",
        )
        answer_time = time.time() - t0

        judge_score = self.judge.score(
            question_id=q.question_id,
            question=q.question,
            gold_answer=q.gold_answer,
            predicted_answer=predicted_answer,
            category=q.category,
        )

        return QuestionResult(
            question_id=q.question_id,
            category=q.category,
            difficulty=q.difficulty,
            question=q.question,
            gold_answer=q.gold_answer,
            predicted_answer=predicted_answer,
            judge_score=judge_score,
            context_tokens=memory_ctx.token_count,
            answer_time_s=answer_time,
        )

    def _save_result(self, result: ScenarioResult, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(result.to_dict(), path)

    def save_results(
        self, result: ScenarioResult, output_dir: Path | str | None = None
    ) -> Path:
        path = self._result_path(result.scenario_id, Path(output_dir) if output_dir else None)
        self._save_result(result, path)
        logger.info(f"Results saved to {path}")
        return path


# ── Multi-model orchestrator ──────────────────────────────────────────────────


class MultiModelEvaluator:
    """Orchestrates evaluation across multiple models x methods x scenarios."""

    def __init__(
        self,
        models: list[str] | None = None,
        method_names: list[str] | None = None,
        scenario_ids: list[str] | None = None,
        judge_model: str = "google/gemini-2.5-pro",
        benchmark_dir: Path | str = BENCHMARK_DIR,
        resume: bool = True,
    ):
        from memory.methods import METHODS

        self.models = models or EVAL_MODELS
        self.method_names = method_names or list(METHODS.keys())
        self.scenario_ids = scenario_ids or ["1", "2", "3", "4", "5"]
        self.judge_model = judge_model
        self.benchmark_dir = Path(benchmark_dir)
        self.resume = resume

        # Compute total number of runs
        self.total_runs = (
            len(self.models) * len(self.method_names) * len(self.scenario_ids)
        )

    def run_all(self) -> list[ScenarioResult]:
        """Run the full evaluation grid."""
        from memory.methods import METHODS

        results_dir = self.benchmark_dir / "results"
        file_handler = _setup_eval_file_logger(results_dir)

        logger.info(
            f"\n{'#' * 70}\n"
            f"  MUM BENCHMARK EVALUATION\n"
            f"  Models:    {len(self.models)} — {', '.join(short_model_name(m) for m in self.models)}\n"
            f"  Methods:   {len(self.method_names)} — {', '.join(self.method_names)}\n"
            f"  Scenarios: {len(self.scenario_ids)} — S{', S'.join(self.scenario_ids)}\n"
            f"  Judge:     {short_model_name(self.judge_model)}\n"
            f"  Total runs: {self.total_runs}\n"
            f"  Resume:    {self.resume}\n"
            f"{'#' * 70}"
        )

        all_results: list[ScenarioResult] = []
        run_idx = 0

        for model in self.models:
            m_short = short_model_name(model)
            logger.info(f"\n{'~' * 50}\n  MODEL: {m_short}\n{'~' * 50}")

            for method_name in self.method_names:
                method_cls = METHODS[method_name]
                method_kwargs: dict = {"model": model}
                if method_name == "rag":
                    method_kwargs["top_k"] = 20
                    method_kwargs["chunk_size"] = 512

                for sid in self.scenario_ids:
                    run_idx += 1
                    console.print(
                        f"\n[bold]Run {run_idx}/{self.total_runs}: "
                        f"{m_short} / {method_name} / S{sid}[/bold]"
                    )

                    method_instance = method_cls(**method_kwargs)
                    runner = EvaluationRunner(
                        method=method_instance,
                        answer_model=model,
                        judge_model=self.judge_model,
                        benchmark_dir=self.benchmark_dir,
                    )

                    try:
                        result = runner.run_scenario(
                            sid, resume=self.resume, output_dir=results_dir
                        )
                        all_results.append(result)

                        console.print(
                            f"  [green]score={result.overall_score():.3f}[/green] | "
                            f"n={result.num_questions} | "
                            f"time={result.total_time_s:.0f}s"
                        )
                    except Exception as e:
                        logger.error(
                            f"FAILED: {m_short}/{method_name}/S{sid}: {e}",
                            exc_info=True,
                        )
                        console.print(f"  [red]FAILED: {e}[/red]")

        # Final summary
        logger.info(
            f"\n{'#' * 70}\n"
            f"  EVALUATION COMPLETE — {len(all_results)} successful runs\n"
            f"{'#' * 70}"
        )

        # Remove file handler
        logging.getLogger("mum").removeHandler(file_handler)
        logging.getLogger("mum.memory").removeHandler(file_handler)
        file_handler.close()

        return all_results
