"""MUMM-Core Answer Generation Runner.

Orchestrates: for each baseline × each question in mumm_core.json, generate an
answer and save incrementally. Supports resume.

Usage:
    python -m MUMBench.runner --baselines no_memory,rag,long_context
    python -m MUMBench.runner --baselines rag --scenario 1
    python -m MUMBench.runner --resume
"""

from __future__ import annotations

import json
import logging
import time
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

from datagen.llm.cost_tracker import CostTracker
from datagen.models.schemas import ConversationSession, SessionSummary
from datagen.utils.io import read_json, write_json

from MUMBench.baselines import BASELINES, MUMMBaseline
from MUMBench.config import ALL_SCENARIOS, MUMM_CONFIG
from MUMBench.subset_selector import get_questions_for_scenario, load_mumm_core

logger = logging.getLogger("mum.mummbench")
console = Console()

BENCHMARK_DIR = Path("MUMBench")
OUTPUT_DIR = Path("output") / "eval_results"


# ── Scenario data loading ─────────────────────────────────────────────────────


def load_scenario_data(
    scenario_id: int,
    benchmark_dir: Path = BENCHMARK_DIR,
) -> tuple[list[ConversationSession], list[SessionSummary], str, dict]:
    """Load conversations, summaries, relationship_type, authority_context for a scenario."""
    scenario_dir = benchmark_dir / f"scenario_{scenario_id}"

    conversations: list[ConversationSession] = []
    conv_dir = scenario_dir / "conversations"
    if conv_dir.exists():
        for f in sorted(conv_dir.glob("*.json")):
            try:
                conversations.append(ConversationSession.model_validate(read_json(f)))
            except Exception as e:
                logger.warning(f"Failed to load conversation {f}: {e}")

    summaries: list[SessionSummary] = []
    summ_dir = scenario_dir / "summaries"
    if summ_dir.exists():
        summ_files = sorted(summ_dir.glob("*_summary.json"))
        if summ_files:
            for f in summ_files:
                try:
                    summaries.append(SessionSummary.model_validate(read_json(f)))
                except Exception as e:
                    logger.warning(f"Failed to load summary {f}: {e}")
        else:
            combined = summ_dir / "session_summaries.json"
            if combined.exists():
                for item in read_json(combined):
                    try:
                        summaries.append(SessionSummary.model_validate(item))
                    except Exception as e:
                        logger.warning(f"Failed to parse summary: {e}")

    # Load scenario config for relationship_type and authority_context
    relationship_type = ""
    authority_context: dict = {}
    try:
        from datagen.utils.io import read_yaml

        config_path = Path("config") / "scenarios" / f"scenario_{scenario_id}.yaml"
        if config_path.exists():
            cfg = read_yaml(config_path)
            relationship_type = cfg.get("relationship_type", "")
            # Build authority context from users
            users = cfg.get("users", [])
            authority_context = {
                u["user_id"]: {
                    "authority_level": u.get("authority_level", "equal"),
                    "authority_weight": u.get("authority_weight", 1.0),
                }
                for u in users
                if "user_id" in u
            }
    except Exception as e:
        logger.warning(f"Could not load scenario config for {scenario_id}: {e}")

    logger.info(
        f"Loaded scenario {scenario_id}: "
        f"{len(conversations)} conversations, {len(summaries)} summaries, "
        f"relationship={relationship_type}"
    )
    return conversations, summaries, relationship_type, authority_context


# ── Answer file management ─────────────────────────────────────────────────────


def _answer_path(baseline_name: str, scenario_id: int, output_dir: Path) -> Path:
    return output_dir / "answers" / baseline_name / f"scenario_{scenario_id}_answers.json"


def _load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = read_json(path)
        return {item["question_id"] for item in data if "question_id" in item}
    except Exception:
        return set()


def _load_partial_answers(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return read_json(path)
    except Exception:
        return []


def _save_answers(answers: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(answers, path)


# ── Core runner ────────────────────────────────────────────────────────────────


def run_baseline_scenario(
    baseline: MUMMBaseline,
    scenario_id: int,
    questions: list[dict],
    benchmark_dir: Path = BENCHMARK_DIR,
    output_dir: Path = OUTPUT_DIR,
    resume: bool = True,
) -> list[dict]:
    """Run one baseline on one scenario's questions. Returns list of answer dicts."""
    answer_path = _answer_path(baseline.name, scenario_id, output_dir)

    # Resume: skip already-answered questions
    completed_ids: set[str] = set()
    answers: list[dict] = []
    if resume:
        completed_ids = _load_completed_ids(answer_path)
        if completed_ids:
            answers = _load_partial_answers(answer_path)
            logger.info(
                f"[{baseline.name}] Scenario {scenario_id}: "
                f"resuming — {len(completed_ids)}/{len(questions)} already done"
            )

    pending = [q for q in questions if q["question_id"] not in completed_ids]

    if not pending:
        logger.info(
            f"[{baseline.name}] Scenario {scenario_id}: all {len(questions)} already complete"
        )
        return answers

    # Ingest scenario data
    conversations, summaries, relationship_type, authority_context = load_scenario_data(
        scenario_id, benchmark_dir
    )
    baseline.ingest(
        scenario_id=scenario_id,
        conversations=conversations,
        summaries=summaries,
        relationship_type=relationship_type,
        authority_context=authority_context,
    )

    # Answer questions with progress bar
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
            f"{baseline.name} | S{scenario_id}",
            total=len(pending),
        )

        for i, q in enumerate(pending):
            qid = q["question_id"]
            t0 = time.time()

            try:
                predicted_answer, metadata = baseline.answer(q["question"])
            except Exception as e:
                logger.error(f"Answer failed for {qid}: {e}")
                predicted_answer = f"[ERROR: {e}]"
                metadata = {"error": str(e)}

            latency_ms = int((time.time() - t0) * 1000)

            answer_entry = {
                "question_id": qid,
                "scenario": scenario_id,
                "category": q["category"],
                "difficulty": q.get("difficulty", "medium"),
                "question": q["question"],
                "gold_answer": q["gold_answer"],
                "predicted_answer": predicted_answer,
                "evidence_links": q.get("evidence_links", []),
                "latency_ms": latency_ms,
                "metadata": metadata,
            }
            answers.append(answer_entry)

            # Save every 5 questions
            if (i + 1) % 5 == 0 or (i + 1) == len(pending):
                _save_answers(answers, answer_path)

            progress.update(task, advance=1)

    baseline.reset()
    _save_answers(answers, answer_path)
    logger.info(
        f"[{baseline.name}] Scenario {scenario_id}: "
        f"answered {len(pending)} questions → {answer_path}"
    )
    return answers


# ── Multi-baseline orchestrator ───────────────────────────────────────────────


def run_evaluation(
    core_manifest_path: str | Path = "output/mumm_core.json",
    baseline_names: list[str] | None = None,
    scenario_ids: list[int] | None = None,
    benchmark_dir: Path = BENCHMARK_DIR,
    output_dir: Path = OUTPUT_DIR,
    resume: bool = True,
    model: str | None = None,
    dry_run: bool = False,
) -> None:
    """Run answer generation for all baselines × scenarios.

    Args:
        core_manifest_path: Path to mumm_core.json.
        baseline_names: Which baselines to run. Defaults to all 3.
        scenario_ids: Which scenarios. Defaults to all in the manifest.
        benchmark_dir: Root of benchmark data.
        output_dir: Root for answer output.
        resume: Skip already-answered questions.
        model: Override answer model for all baselines.
        dry_run: Print plan without making LLM calls.
    """
    manifest = load_mumm_core(core_manifest_path)
    baseline_names = baseline_names or MUMM_CONFIG["default_baselines"]
    answer_model = model or MUMM_CONFIG["answer_model"]

    if scenario_ids is None:
        scenario_ids = [
            int(k.replace("scenario_", ""))
            for k in manifest["per_scenario"].keys()
        ]

    total_questions = sum(
        len(get_questions_for_scenario(manifest, sid)) for sid in scenario_ids
    )
    total_calls = total_questions * len(baseline_names)

    console.print(f"\n[bold]MUMM-Core Answer Generation[/bold]")
    console.print(f"  Manifest:  {core_manifest_path}")
    console.print(f"  Baselines: {', '.join(baseline_names)}")
    console.print(f"  Scenarios: {', '.join(f'S{s}' for s in scenario_ids)}")
    console.print(f"  Model:     {answer_model}")
    console.print(f"  Questions: {total_questions} per baseline, {total_calls} total LLM calls")
    console.print(f"  Resume:    {resume}\n")

    if dry_run:
        console.print("[yellow][dry-run] Would make the calls above. Exiting.[/yellow]")
        return

    # Initialize cost tracker shared across all baselines
    global_cost_tracker = CostTracker()

    run_metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "core_manifest": str(core_manifest_path),
        "baselines": baseline_names,
        "scenarios": scenario_ids,
        "answer_model": answer_model,
        "resume": resume,
    }
    write_json(run_metadata, output_dir / "run_metadata.json")

    for baseline_name in baseline_names:
        if baseline_name not in BASELINES:
            logger.error(f"Unknown baseline: {baseline_name}. Available: {list(BASELINES.keys())}")
            continue

        baseline_cls = BASELINES[baseline_name]
        baseline = baseline_cls(model=answer_model, cost_tracker=global_cost_tracker)

        console.print(f"\n[bold cyan]Baseline: {baseline_name}[/bold cyan]")

        for sid in scenario_ids:
            questions = get_questions_for_scenario(manifest, sid)
            if not questions:
                logger.warning(f"No questions for scenario {sid} in manifest")
                continue

            console.print(
                f"  Scenario {sid}: {len(questions)} questions "
                f"({len([q for q in questions if q['question_id'] not in _load_completed_ids(_answer_path(baseline_name, sid, output_dir))])} pending)"
            )

            run_baseline_scenario(
                baseline=baseline,
                scenario_id=sid,
                questions=questions,
                benchmark_dir=benchmark_dir,
                output_dir=output_dir,
                resume=resume,
            )

    # Save final run metadata with cost
    run_metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    run_metadata["total_cost_usd"] = round(global_cost_tracker.get_total_cost(), 4)
    run_metadata["total_tokens"] = global_cost_tracker.get_total_tokens()
    write_json(run_metadata, output_dir / "run_metadata.json")

    console.print(
        f"\n[bold green]Answer generation complete.[/bold green]\n"
        f"  Total cost: ${global_cost_tracker.get_total_cost():.4f}\n"
        f"  Output: {output_dir}/answers/\n"
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="MUMM-Core answer generation runner")
    parser.add_argument("--core", default="output/mumm_core.json")
    parser.add_argument("--baselines", default="no_memory,rag,long_context",
                        help="Comma-separated baseline names")
    parser.add_argument("--scenario", type=int, default=None, help="Single scenario ID")
    parser.add_argument("--benchmark-dir", default="MUMBench")
    parser.add_argument("--output-dir", default="output/eval_results")
    parser.add_argument("--model", default=None, help="Override answer model")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_evaluation(
        core_manifest_path=args.core,
        baseline_names=args.baselines.split(","),
        scenario_ids=[args.scenario] if args.scenario else None,
        benchmark_dir=Path(args.benchmark_dir),
        output_dir=Path(args.output_dir),
        resume=not args.no_resume,
        model=args.model,
        dry_run=args.dry_run,
    )
