"""Export evaluation results to CSV and structured JSON for analysis."""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console

from datagen.utils.io import read_json, write_json

from memory.evaluation.runner import short_model_name

logger = logging.getLogger("mum.memory")
console = Console()

BENCHMARK_DIR = Path("MUMBench")


class ResultsExporter:
    """Reads raw eval result JSONs and exports structured CSV + JSON files."""

    def __init__(self, results_dir: Path | None = None):
        self.results_dir = results_dir or (BENCHMARK_DIR / "results")

    def _load_all(self) -> list[dict]:
        if not self.results_dir.exists():
            return []
        results = []
        for f in sorted(self.results_dir.glob("eval_*.json")):
            if f.name in ("eval_summary.json",):
                continue
            try:
                results.append(read_json(f))
            except Exception as e:
                logger.warning(f"Skipping {f.name}: {e}")
        return results

    # ── public API ────────────────────────────────────────────────────

    def export_all(self, output_dir: Path | None = None) -> Path:
        """Export all results to CSV and JSON.

        Returns the output directory path.
        """
        out = output_dir or (self.results_dir / "exports")
        out.mkdir(parents=True, exist_ok=True)

        all_results = self._load_all()
        if not all_results:
            console.print("[red]No evaluation results found to export.[/red]")
            return out

        self._export_question_level_csv(all_results, out)
        self._export_experiment_summary_csv(all_results, out)
        self._export_category_breakdown_csv(all_results, out)
        self._export_full_json(all_results, out)

        console.print(f"\n[bold green]Exported {len(all_results)} experiments to {out}/[/bold green]")
        return out

    # ── question-level CSV ────────────────────────────────────────────

    def _export_question_level_csv(self, results: list[dict], out: Path) -> None:
        path = out / "question_level_results.csv"
        fieldnames = [
            "model", "model_full", "method", "scenario",
            "question_id", "category", "difficulty",
            "question", "gold_answer", "predicted_answer",
            "correctness", "completeness", "attribution", "hallucination",
            "overall_score", "is_binary", "binary_correct",
            "context_tokens", "answer_time_s", "judge_reasoning",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in results:
                model_short = r.get(
                    "model_short", short_model_name(r.get("model", "unknown"))
                )
                model_full = r.get("model", "")
                method = r["method_name"]
                scenario = r["scenario_id"]

                for qr in r.get("question_results", []):
                    js = qr.get("judge_score", {})
                    writer.writerow({
                        "model": model_short,
                        "model_full": model_full,
                        "method": method,
                        "scenario": scenario,
                        "question_id": qr["question_id"],
                        "category": qr["category"],
                        "difficulty": qr["difficulty"],
                        "question": qr["question"],
                        "gold_answer": qr["gold_answer"],
                        "predicted_answer": qr["predicted_answer"],
                        "correctness": js.get("correctness", ""),
                        "completeness": js.get("completeness", ""),
                        "attribution": js.get("attribution", ""),
                        "hallucination": js.get("hallucination", ""),
                        "overall_score": js.get("overall", ""),
                        "is_binary": js.get("is_binary", False),
                        "binary_correct": js.get("binary_correct", ""),
                        "context_tokens": qr.get("context_tokens", ""),
                        "answer_time_s": qr.get("answer_time_s", ""),
                        "judge_reasoning": js.get("reasoning", ""),
                    })

        n_rows = sum(len(r.get("question_results", [])) for r in results)
        console.print(f"  question_level_results.csv  ({n_rows:,} rows)")

    # ── experiment summary CSV ────────────────────────────────────────

    def _export_experiment_summary_csv(self, results: list[dict], out: Path) -> None:
        path = out / "experiment_summary.csv"
        fieldnames = [
            "model", "model_full", "method", "scenario",
            "overall_score",
            "avg_correctness", "avg_completeness",
            "avg_attribution", "avg_hallucination",
            "num_questions", "total_time_s",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in results:
                dims = r.get("dimension_averages", {})
                writer.writerow({
                    "model": r.get(
                        "model_short",
                        short_model_name(r.get("model", "unknown")),
                    ),
                    "model_full": r.get("model", ""),
                    "method": r["method_name"],
                    "scenario": r["scenario_id"],
                    "overall_score": round(r.get("overall_score", 0), 4),
                    "avg_correctness": round(dims.get("correctness", 0), 4),
                    "avg_completeness": round(dims.get("completeness", 0), 4),
                    "avg_attribution": round(dims.get("attribution", 0), 4),
                    "avg_hallucination": round(dims.get("hallucination", 0), 4),
                    "num_questions": r.get("num_questions", 0),
                    "total_time_s": round(r.get("total_time_s", 0), 2),
                })

        console.print(f"  experiment_summary.csv      ({len(results)} rows)")

    # ── category breakdown CSV ────────────────────────────────────────

    def _export_category_breakdown_csv(self, results: list[dict], out: Path) -> None:
        path = out / "category_breakdown.csv"
        fieldnames = [
            "model", "method", "scenario",
            "category", "mean_score", "count", "min_score", "max_score",
        ]

        rows = 0
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in results:
                model_short = r.get(
                    "model_short",
                    short_model_name(r.get("model", "unknown")),
                )
                for cat, info in sorted(
                    r.get("scores_by_category", {}).items()
                ):
                    writer.writerow({
                        "model": model_short,
                        "method": r["method_name"],
                        "scenario": r["scenario_id"],
                        "category": cat,
                        "mean_score": round(info.get("mean", 0), 4),
                        "count": info.get("count", 0),
                        "min_score": round(info.get("min", 0), 4),
                        "max_score": round(info.get("max", 0), 4),
                    })
                    rows += 1

        console.print(f"  category_breakdown.csv      ({rows} rows)")

    # ── full JSON export ──────────────────────────────────────────────

    def _export_full_json(self, results: list[dict], out: Path) -> None:
        path = out / "full_results.json"

        total_questions = sum(
            len(r.get("question_results", [])) for r in results
        )

        export = {
            "exported_at": datetime.now().isoformat(),
            "num_experiments": len(results),
            "num_questions_total": total_questions,
            "experiments": [],
        }

        for r in results:
            model_short = r.get(
                "model_short",
                short_model_name(r.get("model", "unknown")),
            )
            entry = {
                "model": model_short,
                "model_full": r.get("model", ""),
                "method": r["method_name"],
                "scenario": r["scenario_id"],
                "overall_score": round(r.get("overall_score", 0), 4),
                "dimension_averages": {
                    k: round(v, 4)
                    for k, v in r.get("dimension_averages", {}).items()
                },
                "scores_by_category": r.get("scores_by_category", {}),
                "scores_by_difficulty": r.get("scores_by_difficulty", {}),
                "num_questions": r.get("num_questions", 0),
                "total_time_s": round(r.get("total_time_s", 0), 2),
                "questions": [],
            }

            for qr in r.get("question_results", []):
                js = qr.get("judge_score", {})
                entry["questions"].append({
                    "question_id": qr["question_id"],
                    "category": qr["category"],
                    "difficulty": qr["difficulty"],
                    "question": qr["question"],
                    "gold_answer": qr["gold_answer"],
                    "predicted_answer": qr["predicted_answer"],
                    "correctness": js.get("correctness"),
                    "completeness": js.get("completeness"),
                    "attribution": js.get("attribution"),
                    "hallucination": js.get("hallucination"),
                    "overall_score": js.get("overall"),
                    "is_binary": js.get("is_binary", False),
                    "binary_correct": js.get("binary_correct"),
                    "context_tokens": qr.get("context_tokens"),
                    "answer_time_s": qr.get("answer_time_s"),
                    "judge_reasoning": js.get("reasoning", ""),
                })

            export["experiments"].append(entry)

        write_json(export, path)
        console.print(f"  full_results.json           ({len(results)} experiments, {total_questions:,} questions)")
