"""CLI for MUM memory evaluation pipeline."""

from __future__ import annotations

from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console

from datagen.utils.logging import setup_logging

from memory.evaluation.runner import EVAL_MODELS, short_model_name

console = Console()
load_dotenv()


def _resolve_models(answer_model: str) -> list[str]:
    """Resolve model argument to a list of full model IDs."""
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
    default="all",
    help=(
        "Answer model(s): 'all' for all 7 eval models, or comma-separated "
        "names/IDs (e.g. 'Qwen3-14B,Llama3.1-8B' or full DeepInfra IDs)"
    ),
)
@click.option("--judge-model", default="google/gemini-2.5-pro", help="Model for LLM judge")
@click.option("--benchmark-dir", default="MUMBench", help="Benchmark data directory")
@click.option("--no-resume", is_flag=True, help="Do NOT resume from partial results")
@click.option("--top-k", default=20, help="Top-K for RAG retrieval")
@click.option("--chunk-size", default=512, help="Chunk size for RAG")
def evaluate(
    scenario, run_all, method, answer_model, judge_model,
    benchmark_dir, no_resume, top_k, chunk_size,
):
    """Run evaluation: model(s) x method(s) x scenario(s)."""
    from memory.evaluation.runner import MultiModelEvaluator
    from memory.methods import METHODS

    scenario_ids = []
    if run_all:
        scenario_ids = ["1", "2", "3", "4", "5"]
    elif scenario:
        scenario_ids = [scenario]
    else:
        console.print("[red]Specify --scenario <id> or --all-scenarios[/red]")
        return

    models = _resolve_models(answer_model)
    method_names = list(METHODS.keys()) if method == "all" else [method]

    console.print(f"\n[bold]MUM Evaluation Grid[/bold]")
    console.print(f"  Models:    {', '.join(short_model_name(m) for m in models)}")
    console.print(f"  Methods:   {', '.join(method_names)}")
    console.print(f"  Scenarios: {', '.join('S' + s for s in scenario_ids)}")
    console.print(f"  Judge:     {short_model_name(judge_model)}")
    console.print(
        f"  Total runs: {len(models) * len(method_names) * len(scenario_ids)}"
    )
    console.print(f"  Resume:    {not no_resume}\n")

    evaluator = MultiModelEvaluator(
        models=models,
        method_names=method_names,
        scenario_ids=scenario_ids,
        judge_model=judge_model,
        benchmark_dir=Path(benchmark_dir),
        resume=not no_resume,
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
