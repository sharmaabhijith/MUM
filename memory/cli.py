"""CLI for MUM memory evaluation pipeline."""

from __future__ import annotations

from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console

from datagen.utils.logging import setup_logging

from memory.evaluation.config import load_eval_config
from memory.evaluation.runner import EVAL_MODELS, short_model_name

console = Console()
load_dotenv()


def _resolve_models(answer_model: str, config_models: list[str] | None = None) -> list[str]:
    """Resolve model argument to a list of full model IDs.

    When answer_model is "config" (the default), uses models from
    config/evaluation.yaml.  Falls back to hardcoded EVAL_MODELS when
    the YAML list is empty or when "all" is passed explicitly.
    """
    if answer_model == "config":
        if config_models:
            return config_models
        return list(EVAL_MODELS)
    if answer_model == "all":
        return list(EVAL_MODELS)
    # Allow comma-separated short or full names
    requested = [m.strip() for m in answer_model.split(",")]
    # Build reverse lookup: short_name -> full_id
    short_to_full = {short_model_name(m): m for m in EVAL_MODELS}
    short_to_full.update({m: m for m in EVAL_MODELS})  # full name also works
    resolved = []
    for r in requested:
        if r in short_to_full:
            resolved.append(short_to_full[r])
        else:
            # Fuzzy match: check if r is a substring of any known model
            matches = [m for m in EVAL_MODELS if r.lower() in m.lower()]
            if matches:
                resolved.extend(matches)
            else:
                # Treat as literal model ID (user may know a model not in EVAL_MODELS)
                resolved.append(r)
    return resolved


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool):
    """MUM Memory Evaluation Pipeline."""
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level)


@main.command()
@click.option("--scenario", "-s", type=str, help="Scenario ID (1-5)")
@click.option("--all-scenarios", "run_all", is_flag=True, help="Run all scenarios")
@click.option(
    "--method",
    "-m",
    type=click.Choice(["no_memory", "full_context", "rag", "summary", "structured", "all"]),
    default="all",
    help="Memory method (or 'all')",
)
@click.option(
    "--answer-model",
    "-a",
    default="config",
    help=(
        "Answer model(s): 'config' (default) reads from config/evaluation.yaml, "
        "'all' for all 7 hardcoded eval models, or comma-separated "
        "names/IDs (e.g. 'Qwen3-14B,Llama3.1-8B' or full DeepInfra IDs)"
    ),
)
@click.option("--judge-model", default=None, help="Model for LLM judge (default: from config)")
@click.option("--benchmark-dir", default="MUMBench", help="Benchmark data directory")
@click.option("--no-resume", is_flag=True, help="Do NOT resume from partial results")
@click.option("--top-k", default=None, type=int, help="Top-K for RAG retrieval (default: from config)")
@click.option("--chunk-size", default=None, type=int, help="Chunk size for RAG (default: from config)")
@click.option("--config", "config_path", default=None, help="Path to evaluation YAML config")
def evaluate(
    scenario, run_all, method, answer_model, judge_model,
    benchmark_dir, no_resume, top_k, chunk_size, config_path,
):
    """Run evaluation: model(s) x method(s) x scenario(s).

    By default reads models, methods, and parameters from config/evaluation.yaml.
    CLI flags override config values when provided.
    """
    from memory.evaluation.runner import MultiModelEvaluator
    from memory.methods import METHODS

    cfg = load_eval_config(config_path)
    console.print(f"[dim]Config: {config_path or 'config/evaluation.yaml'}[/dim]")

    scenario_ids = []
    if run_all:
        scenario_ids = ["1", "2", "3", "4", "5"]
    elif scenario:
        scenario_ids = [scenario]
    else:
        console.print("[red]Specify --scenario <id> or --all-scenarios[/red]")
        return

    models = _resolve_models(answer_model, cfg.eval_models)

    # Resolve methods: "all" uses config methods if available, else all registered
    if method == "all":
        if cfg.method_names:
            method_names = [m for m in cfg.method_names if m in METHODS]
        else:
            method_names = list(METHODS.keys())
    else:
        method_names = [method]

    # CLI flags override config values
    resolved_judge = judge_model or cfg.judge_model
    rag_cfg = cfg.rag_kwargs()
    resolved_top_k = top_k if top_k is not None else rag_cfg["top_k"]
    resolved_chunk_size = chunk_size if chunk_size is not None else rag_cfg["chunk_size"]

    console.print(f"\n[bold]MUM Evaluation Grid[/bold]")
    console.print(f"  Models:    {', '.join(short_model_name(m) for m in models)}")
    console.print(f"  Methods:   {', '.join(method_names)}")
    console.print(f"  Scenarios: {', '.join('S' + s for s in scenario_ids)}")
    console.print(f"  Judge:     {short_model_name(resolved_judge)}")
    console.print(
        f"  Total runs: {len(models) * len(method_names) * len(scenario_ids)}"
    )
    console.print(f"  Resume:    {not no_resume}\n")

    evaluator = MultiModelEvaluator(
        models=models,
        method_names=method_names,
        scenario_ids=scenario_ids,
        judge_model=resolved_judge,
        benchmark_dir=Path(benchmark_dir),
        resume=not no_resume,
        eval_config=cfg,
    )
    evaluator.run_all()

    console.print("\n[bold green]Evaluation complete. Run 'mum-eval compare' to see results.[/bold green]")


@main.command()
@click.option("--scenario", "-s", type=str, help="Scenario ID for detailed report")
@click.option("--results-dir", default=None, help="Results directory")
def report(scenario, results_dir):
    """Print evaluation report."""
    from memory.evaluation.report import print_comparison_report, print_scenario_report

    results_path = Path(results_dir) if results_dir else None
    if scenario:
        print_scenario_report(scenario, results_path)
    else:
        print_comparison_report(results_path)


@main.command()
@click.option("--results-dir", default=None, help="Results directory")
def compare(results_dir):
    """Print cross-model, cross-method, cross-scenario comparison + save JSON."""
    from memory.evaluation.report import print_comparison_report, save_comparison_report

    results_path = Path(results_dir) if results_dir else None
    print_comparison_report(results_path)
    save_comparison_report(results_path)


@main.command("export-results")
@click.option("--results-dir", default=None, help="Results directory")
@click.option("--output-dir", default=None, help="Output directory for exported files")
def export_results(results_dir, output_dir):
    """Export evaluation results to CSV and structured JSON."""
    from memory.evaluation.export import ResultsExporter

    exporter = ResultsExporter(
        results_dir=Path(results_dir) if results_dir else None,
    )
    out = exporter.export_all(
        output_dir=Path(output_dir) if output_dir else None,
    )
    console.print(f"\n[green]Results exported to {out}/[/green]")


@main.command("list-models")
def list_models():
    """Show all registered evaluation models."""
    from datagen.llm.cost_tracker import MODEL_PRICING

    from memory.evaluation.runner import EVAL_MODELS

    console.print("\n[bold]Registered Evaluation Models[/bold]\n")
    for m in EVAL_MODELS:
        p = MODEL_PRICING.get(m, {})
        console.print(
            f"  {short_model_name(m):<20s}  {m:<50s}  "
            f"${p.get('input', 0):.2f}/${p.get('output', 0):.2f} per 1M tok"
        )
    console.print()


if __name__ == "__main__":
    main()
