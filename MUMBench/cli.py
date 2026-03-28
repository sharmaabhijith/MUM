"""MUMM-Core CLI — evaluation pipeline commands.

Commands:
    mum-bench select-core   Build the MUMM-Core subset manifest
    mum-bench evaluate      Run answer generation across baselines
    mum-bench score         Score answers and write score files
    mum-bench report        Generate aggregate report + radar chart
    mum-bench run-eval      End-to-end pipeline: select → evaluate → score → report
"""

from __future__ import annotations

from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console

from datagen.utils.logging import setup_logging

console = Console()
load_dotenv()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose (DEBUG) logging")
def main(verbose: bool):
    """MUMM-Core Benchmark Evaluation Pipeline."""
    setup_logging("DEBUG" if verbose else "INFO")


# ── select-core ────────────────────────────────────────────────────────────────


@main.command("select-core")
@click.option("--benchmark-dir", default="MUMBench", show_default=True,
              help="Root dir with scenario_N/ data")
@click.option("--output", default="output/mumm_core.json", show_default=True,
              help="Output path for mumm_core.json")
@click.option("--seed", default=42, type=int, show_default=True,
              help="Random seed for stratified sampling")
@click.option("--dry-run", is_flag=True, help="Print stats without writing file")
def select_core(benchmark_dir: str, output: str, seed: int, dry_run: bool):
    """Build the MUMM-Core subset (~448 questions) from existing eval_questions.json files.

    Applies stratified-by-difficulty sampling according to the CORE_BUDGET
    defined in MUMBench/config.py. Does NOT modify any source data.
    """
    from MUMBench.subset_selector import build_mumm_core

    manifest = build_mumm_core(
        benchmark_dir=Path(benchmark_dir),
        output_path=Path(output),
        seed=seed,
        dry_run=dry_run,
    )
    if not dry_run:
        total = manifest.get("total_questions", 0)
        console.print(
            f"\n[bold green]MUMM-Core manifest created:[/bold green] {output}\n"
            f"  Total questions: {total}\n"
            f"  Seed: {seed}"
        )


# ── evaluate ───────────────────────────────────────────────────────────────────


@main.command("evaluate")
@click.option("--core", default="output/mumm_core.json", show_default=True,
              help="Path to mumm_core.json manifest")
@click.option("--baselines", default="rag,long_context", show_default=True,
              help="Comma-separated baseline names (rag, long_context, no_memory)")
@click.option("--scenario", "-s", type=int, default=None,
              help="Single scenario ID (omit to run all scenarios in manifest)")
@click.option("--benchmark-dir", default="MUMBench", show_default=True)
@click.option("--output-dir", default="output/eval_results", show_default=True)
@click.option("--model", default=None,
              help="Override answer model (default: from MUMBench/config.py)")
@click.option("--no-resume", is_flag=True, help="Do NOT resume from partial results")
@click.option("--dry-run", is_flag=True, help="Show plan without making LLM calls")
def evaluate(
    core: str, baselines: str, scenario, benchmark_dir: str,
    output_dir: str, model, no_resume: bool, dry_run: bool,
):
    """Generate answers for all baselines × questions in the MUMM-Core manifest.

    Supports resume: already-answered questions are skipped automatically.
    """
    from MUMBench.runner import run_evaluation

    run_evaluation(
        core_manifest_path=core,
        baseline_names=baselines.split(","),
        scenario_ids=[scenario] if scenario else None,
        benchmark_dir=Path(benchmark_dir),
        output_dir=Path(output_dir),
        resume=not no_resume,
        model=model,
        dry_run=dry_run,
    )


# ── score ─────────────────────────────────────────────────────────────────────


@main.command("score")
@click.option("--results-dir", default="output/eval_results", show_default=True,
              help="Root dir containing answers/ subdirectory")
@click.option("--judge-model", default=None,
              help="Model for LLM-as-judge (default: from MUMBench/config.py)")
@click.option("--factscore-model", default=None,
              help="Model for FactScore LLM calls (optional, uses heuristic if not set)")
@click.option("--no-resume", is_flag=True, help="Re-score all (do not skip existing)")
def score(results_dir: str, judge_model, factscore_model, no_resume: bool):
    """Score all answer files and write score files to results-dir/scores/.

    Routes each question to the appropriate metric:
      - exact_match (T1), binary_accuracy (T6)
      - set_prf1 (T3, T7)
      - factscore (T2)
      - llm_judge with category-specific rubrics (T4, T5, T8–T12)
    """
    from MUMBench.aggregator import build_score_files_from_answers

    results_path = Path(results_dir)
    build_score_files_from_answers(
        answer_dir=results_path / "answers",
        scores_dir=results_path / "scores",
        judge_model=judge_model,
        factscore_model=factscore_model,
        resume=not no_resume,
    )
    console.print(
        f"\n[bold green]Scoring complete.[/bold green]\n"
        f"  Score files: {results_path}/scores/"
    )


# ── report ────────────────────────────────────────────────────────────────────


@main.command("report")
@click.option("--results-dir", default="output/eval_results", show_default=True)
@click.option("--detailed", is_flag=True, help="Show per-scenario breakdown")
@click.option("--no-chart", is_flag=True, help="Skip radar chart generation")
@click.option("--bootstrap", default=1000, type=int, show_default=True,
              help="Number of bootstrap resamples for CI")
def report(results_dir: str, detailed: bool, no_chart: bool, bootstrap: int):
    """Generate aggregate MUMM report: console table + JSON + radar chart.

    Reads score files from results-dir/scores/ and writes:
      - Console: 12-category table per baseline + MUMM score
      - results-dir/mumm_report.json
      - results-dir/radar_chart.png + radar_chart.svg
    """
    from MUMBench.report import generate_report

    results_path = Path(results_dir)
    generate_report(
        scores_dir=results_path / "scores",
        output_dir=results_path,
        n_bootstrap=bootstrap,
        show_detailed=detailed,
        skip_chart=no_chart,
    )


# ── run-eval (end-to-end) ─────────────────────────────────────────────────────


@main.command("run-eval")
@click.option("--benchmark-dir", default="MUMBench", show_default=True)
@click.option("--core", default="output/mumm_core.json", show_default=True)
@click.option("--baselines", default="rag,long_context", show_default=True)
@click.option("--scenario", "-s", type=int, default=None)
@click.option("--output-dir", default="output/eval_results", show_default=True)
@click.option("--seed", default=42, type=int, show_default=True)
@click.option("--model", default=None, help="Override answer model")
@click.option("--judge-model", default=None)
@click.option("--skip-select", is_flag=True,
              help="Skip subset selection (use existing mumm_core.json)")
@click.option("--skip-evaluate", is_flag=True,
              help="Skip answer generation (use existing answers)")
@click.option("--skip-score", is_flag=True,
              help="Skip scoring (use existing score files)")
@click.option("--no-resume", is_flag=True)
@click.option("--dry-run", is_flag=True)
def run_eval(
    benchmark_dir: str, core: str, baselines: str, scenario,
    output_dir: str, seed: int, model, judge_model,
    skip_select: bool, skip_evaluate: bool, skip_score: bool,
    no_resume: bool, dry_run: bool,
):
    """End-to-end MUMM evaluation pipeline.

    Runs: select-core → evaluate → score → report

    Each step can be skipped independently with --skip-select, --skip-evaluate,
    --skip-score flags.
    """
    from MUMBench.aggregator import build_score_files_from_answers
    from MUMBench.report import generate_report
    from MUMBench.runner import run_evaluation
    from MUMBench.subset_selector import build_mumm_core

    core_path = Path(core)
    results_path = Path(output_dir)
    bench_dir = Path(benchmark_dir)

    # Step 1: Select core subset
    if not skip_select:
        console.print("\n[bold]Step 1/4: Building MUMM-Core subset...[/bold]")
        build_mumm_core(
            benchmark_dir=bench_dir,
            output_path=core_path,
            seed=seed,
            dry_run=dry_run,
        )
    else:
        console.print("[dim]Step 1/4: Skipping subset selection[/dim]")

    # Step 2: Generate answers
    if not skip_evaluate:
        console.print("\n[bold]Step 2/4: Generating answers...[/bold]")
        run_evaluation(
            core_manifest_path=core_path,
            baseline_names=baselines.split(","),
            scenario_ids=[scenario] if scenario else None,
            benchmark_dir=bench_dir,
            output_dir=results_path,
            resume=not no_resume,
            model=model,
            dry_run=dry_run,
        )
    else:
        console.print("[dim]Step 2/4: Skipping answer generation[/dim]")

    # Step 3: Score answers
    if not skip_score:
        console.print("\n[bold]Step 3/4: Scoring answers...[/bold]")
        if not dry_run:
            build_score_files_from_answers(
                answer_dir=results_path / "answers",
                scores_dir=results_path / "scores",
                judge_model=judge_model,
                resume=not no_resume,
            )
    else:
        console.print("[dim]Step 3/4: Skipping scoring[/dim]")

    # Step 4: Generate report
    console.print("\n[bold]Step 4/4: Generating report...[/bold]")
    if not dry_run:
        generate_report(
            scores_dir=results_path / "scores",
            output_dir=results_path,
        )
    else:
        console.print("[yellow][dry-run] Would generate report.[/yellow]")

    console.print("\n[bold green]MUMM evaluation pipeline complete.[/bold green]")


# ── list-baselines ────────────────────────────────────────────────────────────


@main.command("list-baselines")
def list_baselines():
    """List available memory baselines with descriptions."""
    from MUMBench.baselines import BASELINES

    console.print("\n[bold]Available MUMM Baselines[/bold]\n")
    for name, cls in BASELINES.items():
        doc = cls.__doc__ or "No description"
        first_line = doc.strip().split("\n")[0].strip(".")
        console.print(f"  [cyan]{name:<15}[/cyan]  {first_line}")
    console.print()


if __name__ == "__main__":
    main()
