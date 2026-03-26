"""Generate comparison reports across models, memory methods, and scenarios."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from datagen.utils.io import read_json, write_json

from memory.evaluation.runner import short_model_name

logger = logging.getLogger("mum.memory")
console = Console()

BENCHMARK_DIR = Path("MUMBench")


def load_all_results(results_dir: Path | None = None) -> list[dict]:
    if results_dir is None:
        results_dir = BENCHMARK_DIR / "results"
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    results = []
    for f in sorted(results_dir.glob("eval_*.json")):
        if f.name == "eval_summary.json":
            continue
        try:
            results.append(read_json(f))
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")
    return results


# ── Per-scenario report (all models x methods) ──────────────────────────────


def print_scenario_report(scenario_id: str, results_dir: Path | None = None) -> None:
    """Detailed report for one scenario: model x method grid."""
    all_results = load_all_results(results_dir)
    scenario_results = [r for r in all_results if r["scenario_id"] == scenario_id]
    if not scenario_results:
        console.print(f"[red]No results found for scenario {scenario_id}[/red]")
        return

    # ── Overall scores: model x method ───────────────────────────────────
    table = Table(
        title=f"Scenario {scenario_id} — Overall Scores (model x method)",
        show_lines=True,
    )
    table.add_column("Model", style="cyan", min_width=16)
    table.add_column("Method", style="yellow", min_width=12)
    table.add_column("Overall", justify="right", style="bold green")
    table.add_column("Correct", justify="right")
    table.add_column("Complete", justify="right")
    table.add_column("Attrib", justify="right")
    table.add_column("Halluc", justify="right")
    table.add_column("N", justify="right", style="dim")
    table.add_column("Time", justify="right", style="dim")

    for r in sorted(
        scenario_results,
        key=lambda x: (x.get("model_short", ""), x["method_name"]),
    ):
        dims = r.get("dimension_averages", {})
        table.add_row(
            r.get("model_short", short_model_name(r.get("model", "?"))),
            r["method_name"],
            f"{r['overall_score']:.3f}",
            f"{dims.get('correctness', 0):.2f}",
            f"{dims.get('completeness', 0):.2f}",
            f"{dims.get('attribution', 0):.2f}",
            f"{dims.get('hallucination', 0):.2f}",
            str(r["num_questions"]),
            f"{r.get('total_time_s', 0):.0f}s",
        )

    console.print()
    console.print(table)

    # ── Category breakdown ───────────────────────────────────────────────
    models_in = sorted(set(
        r.get("model_short", short_model_name(r.get("model", "?")))
        for r in scenario_results
    ))
    methods_in = sorted(set(r["method_name"] for r in scenario_results))

    cat_table = Table(
        title=f"Scenario {scenario_id} — Scores by Category",
        show_lines=True,
    )
    cat_table.add_column("Category", style="cyan", min_width=24)
    for model in models_in:
        for method in methods_in:
            cat_table.add_column(f"{model}\n{method}", justify="right", max_width=10)

    all_cats = set()
    for r in scenario_results:
        all_cats.update(r.get("scores_by_category", {}).keys())

    for cat in sorted(all_cats):
        row = [cat]
        for model in models_in:
            for method in methods_in:
                match = next(
                    (
                        r
                        for r in scenario_results
                        if r.get("model_short", short_model_name(r.get("model"))) == model
                        and r["method_name"] == method
                    ),
                    None,
                )
                if match and cat in match.get("scores_by_category", {}):
                    row.append(f"{match['scores_by_category'][cat]['mean']:.3f}")
                else:
                    row.append("-")
        cat_table.add_row(*row)

    console.print()
    console.print(cat_table)


# ── Cross-scenario comparison ────────────────────────────────────────────────


def print_comparison_report(results_dir: Path | None = None) -> None:
    """Print model x method scores averaged across scenarios."""
    all_results = load_all_results(results_dir)
    if not all_results:
        console.print("[red]No evaluation results found.[/red]")
        return

    key_fn = lambda r: (
        r.get("model_short", short_model_name(r.get("model", "?"))),
        r["method_name"],
    )
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in all_results:
        grouped[key_fn(r)].append(r)

    scenario_ids = sorted(set(r["scenario_id"] for r in all_results))

    # ── Main grid ────────────────────────────────────────────────────────
    table = Table(title="MUM Benchmark — Model x Method x Scenario", show_lines=True)
    table.add_column("Model", style="cyan", min_width=16)
    table.add_column("Method", style="yellow", min_width=12)
    for sid in scenario_ids:
        table.add_column(f"S{sid}", justify="right")
    table.add_column("Avg", justify="right", style="bold green")

    for (model, method), results in sorted(grouped.items()):
        row = [model, method]
        scores = []
        for sid in scenario_ids:
            match = next((r for r in results if r["scenario_id"] == sid), None)
            if match:
                scores.append(match["overall_score"])
                row.append(f"{match['overall_score']:.3f}")
            else:
                row.append("-")
        avg = sum(scores) / len(scores) if scores else 0
        row.append(f"{avg:.3f}")
        table.add_row(*row)

    console.print()
    console.print(table)

    # ── Model-level summary (avg across methods) ─────────────────────────
    by_model: dict[str, list[float]] = defaultdict(list)
    for (model, _), results in grouped.items():
        for r in results:
            by_model[model].append(r["overall_score"])

    model_table = Table(title="MUM Benchmark — Model Summary (avg across methods & scenarios)")
    model_table.add_column("Model", style="cyan")
    model_table.add_column("Avg Score", justify="right", style="bold green")
    model_table.add_column("Runs", justify="right", style="dim")

    for model in sorted(by_model, key=lambda m: -_avg(by_model[m])):
        scores = by_model[model]
        model_table.add_row(model, f"{_avg(scores):.3f}", str(len(scores)))

    console.print()
    console.print(model_table)

    # ── Method-level summary (avg across models) ─────────────────────────
    by_method: dict[str, list[float]] = defaultdict(list)
    for (_, method), results in grouped.items():
        for r in results:
            by_method[method].append(r["overall_score"])

    method_table = Table(title="MUM Benchmark — Method Summary (avg across models & scenarios)")
    method_table.add_column("Method", style="yellow")
    method_table.add_column("Avg Score", justify="right", style="bold green")
    method_table.add_column("Runs", justify="right", style="dim")

    for method in sorted(by_method, key=lambda m: -_avg(by_method[m])):
        scores = by_method[method]
        method_table.add_row(method, f"{_avg(scores):.3f}", str(len(scores)))

    console.print()
    console.print(method_table)

    # ── Dimension averages per model ─────────────────────────────────────
    dim_table = Table(title="MUM Benchmark — Dimension Averages by Model", show_lines=True)
    dim_table.add_column("Model", style="cyan")
    dim_table.add_column("Correct", justify="right")
    dim_table.add_column("Complete", justify="right")
    dim_table.add_column("Attrib", justify="right")
    dim_table.add_column("Halluc", justify="right")

    model_dims: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        m = r.get("model_short", short_model_name(r.get("model", "?")))
        for dim, val in r.get("dimension_averages", {}).items():
            model_dims[m][dim].append(val)

    for model in sorted(model_dims, key=lambda m: -_avg(model_dims[m].get("correctness", []))):
        d = model_dims[model]
        dim_table.add_row(
            model,
            f"{_avg(d.get('correctness', [])):.2f}",
            f"{_avg(d.get('completeness', [])):.2f}",
            f"{_avg(d.get('attribution', [])):.2f}",
            f"{_avg(d.get('hallucination', [])):.2f}",
        )

    console.print()
    console.print(dim_table)


def save_comparison_report(
    results_dir: Path | None = None, output_path: Path | None = None
) -> None:
    """Save a structured JSON comparison report."""
    all_results = load_all_results(results_dir)
    if not all_results:
        return

    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        model = r.get("model_short", short_model_name(r.get("model", "?")))
        method = r["method_name"]
        grouped[model][method].append(r)

    report: dict = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "models": {},
        "methods": sorted(set(r["method_name"] for r in all_results)),
        "scenarios": sorted(set(r["scenario_id"] for r in all_results)),
    }

    for model, methods_dict in sorted(grouped.items()):
        model_entry: dict = {"methods": {}}
        all_model_scores: list[float] = []

        for method, results in sorted(methods_dict.items()):
            scores = [r["overall_score"] for r in results]
            all_model_scores.extend(scores)
            model_entry["methods"][method] = {
                "average_score": round(_avg(scores), 4),
                "per_scenario": {
                    r["scenario_id"]: {
                        "overall_score": r["overall_score"],
                        "scores_by_category": r.get("scores_by_category", {}),
                        "dimension_averages": r.get("dimension_averages", {}),
                        "num_questions": r["num_questions"],
                    }
                    for r in results
                },
            }

        model_entry["average_score"] = round(_avg(all_model_scores), 4)
        report["models"][model] = model_entry

    if output_path is None:
        output_path = (results_dir or BENCHMARK_DIR / "results") / "comparison_report.json"
    write_json(report, output_path)
    console.print(f"\n[green]Comparison report saved to {output_path}[/green]")


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
