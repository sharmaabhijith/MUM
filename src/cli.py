from __future__ import annotations

import click
from dotenv import load_dotenv
from rich.console import Console

from src.utils.logging import setup_logging

console = Console()
load_dotenv()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool):
    """MUM Benchmark Generation Pipeline."""
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level)


@main.command()
@click.option("--scenario", "-s", type=str, help="Scenario ID (1-5)")
@click.option("--all", "run_all", is_flag=True, help="Run all scenarios")
@click.option("--model", "-m", default="deepseek-ai/DeepSeek-V3.2", help="Model for conversation generation")
@click.option("--question-model", default="google/gemini-2.5-pro", help="Model for question generation")
@click.option("--dry-run", is_flag=True, help="Use mock LLM client")
@click.option("--output-dir", default="output", help="Output directory")
def generate(scenario, run_all, model, question_model, dry_run, output_dir):
    """Generate benchmark data (full pipeline)."""
    from src.pipeline.orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator(
        model=model,
        question_model=question_model,
        dry_run=dry_run,
        output_dir=output_dir,
    )

    if run_all:
        dataset = orchestrator.run_all()
        console.print(f"\n[green]Generated benchmark with {len(dataset.scenarios)} scenarios[/green]")
        console.print(f"Total cost: ${dataset.generation_report.total_cost:.4f}")
    elif scenario:
        output = orchestrator.run_scenario(scenario)
        console.print(f"\n[green]Generated scenario {scenario}[/green]")
        console.print(f"  Conversations: {len(output.conversations)}")
        console.print(f"  Eval questions: {len(output.eval_questions)}")
    else:
        console.print("[red]Specify --scenario <id> or --all[/red]")


@main.command("generate-conversations")
@click.option("--scenario", "-s", required=True, type=str, help="Scenario ID")
@click.option("--model", "-m", default="deepseek-ai/DeepSeek-V3.2", help="Model")
def generate_conversations(scenario, model):
    """Generate conversations only (Phase 2)."""
    from src.pipeline.orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator(model=model)
    orchestrator.run_conversations_only(scenario)
    console.print(f"[green]Conversations generated for scenario {scenario}[/green]")


@main.command("validate-docs")
@click.option("--scenario", "-s", type=str, help="Scenario ID (or all if omitted)")
def validate_docs(scenario):
    """Validate source documents exist and meet token targets."""
    from scripts.validate_documents import validate_scenario_docs
    from src.llm.token_counter import TokenCounter

    token_counter = TokenCounter()
    scenario_ids = [scenario] if scenario else ["1", "2", "3", "4", "5"]

    all_ok = True
    for sid in scenario_ids:
        try:
            ok = validate_scenario_docs(sid, token_counter)
            if not ok:
                all_ok = False
        except Exception as e:
            console.print(f"[red]Error validating scenario {sid}: {e}[/red]")
            all_ok = False

    if all_ok:
        console.print("\n[green]All documents validated.[/green]")
    else:
        console.print("\n[red]Some documents missing or out of range.[/red]")


@main.command("estimate-cost")
@click.option("--scenario", "-s", type=str, help="Scenario ID")
@click.option("--all", "run_all", is_flag=True, help="Estimate for all scenarios")
@click.option(
    "--mode",
    type=click.Choice(["iteration", "final"]),
    default="iteration",
    help="Model configuration (iteration=DeepSeek+Gemini, final=same)",
)
def estimate_cost(scenario, run_all, mode):
    """Estimate API costs for conversation generation and question generation."""
    from rich.table import Table

    from src.llm.cost_tracker import MODEL_PRICING
    from src.llm.token_counter import TokenCounter
    from src.pipeline.phase1_document_prep import DocumentPreparer
    from src.scenarios import create_scenario, load_scenario

    token_counter = TokenCounter()
    doc_preparer = DocumentPreparer(token_counter=token_counter)

    model_configs = {
        "iteration": {
            "conversation": "deepseek-ai/DeepSeek-V3.2",
            "summary": "deepseek-ai/DeepSeek-V3.2",
            "question_gen": "google/gemini-2.5-pro",
        },
        "final": {
            "conversation": "deepseek-ai/DeepSeek-V3.2",
            "summary": "deepseek-ai/DeepSeek-V3.2",
            "question_gen": "google/gemini-2.5-pro",
        },
    }
    models = model_configs[mode]

    scenario_ids = (
        ["1", "2", "3", "4", "5"] if run_all else [scenario] if scenario else []
    )
    if not scenario_ids:
        console.print("[red]Specify --scenario <id> or --all[/red]")
        return

    phase_totals: dict[str, dict[str, dict[str, int]]] = {}

    def _accum(phase: str, model: str, inp: int, out: int):
        phase_totals.setdefault(phase, {}).setdefault(model, {"input": 0, "output": 0})
        phase_totals[phase][model]["input"] += inp
        phase_totals[phase][model]["output"] += out

    scenario_table = Table(title=f"MUM Cost Estimate — {mode} mode")
    scenario_table.add_column("Scenario", style="cyan")
    scenario_table.add_column("Users", justify="right")
    scenario_table.add_column("Sessions", justify="right")
    scenario_table.add_column("Doc Tokens\n(actual)", justify="right")
    scenario_table.add_column("Conv Gen", justify="right", style="yellow")
    scenario_table.add_column("Summary", justify="right", style="yellow")
    scenario_table.add_column("Questions", justify="right", style="yellow")
    scenario_table.add_column("Scenario $", justify="right", style="bold green")

    grand_total_cost = 0.0

    for sid in scenario_ids:
        config = load_scenario(sid)
        scenario_obj = create_scenario(config)
        n_users = len(config.users)
        n_sessions = config.sessions_per_user
        avg_turns = sum(
            config.get_turns_for_session(s) for s in range(1, n_sessions + 1)
        ) / n_sessions
        turns = avg_turns
        total_sessions = n_users * n_sessions

        # Actual document tokens from PDFs
        doc_context = doc_preparer.prepare_scenario(config)
        actual_doc_tokens = doc_context.total_tokens
        if actual_doc_tokens == 0:
            actual_doc_tokens = sum(d.target_tokens for d in config.documents)

        # Phase 2a: Conversation Generation
        avg_prior_summary_tokens = int(
            sum(range(n_sessions)) * 270 / n_sessions
        )
        conv_overhead = 600 + 100 + 500 + avg_prior_summary_tokens + 50
        conv_input_per_session = actual_doc_tokens + conv_overhead
        conv_output_per_session = turns * 2 * 90

        conv_total_input = total_sessions * conv_input_per_session
        conv_total_output = total_sessions * conv_output_per_session

        conv_model = models["conversation"]
        _accum("conversation", conv_model, conv_total_input, conv_total_output)

        # Phase 2b: Session Summary
        turns_json_tokens = turns * 2 * 120
        summary_input_per_session = 350 + turns_json_tokens + avg_prior_summary_tokens
        summary_output_per_session = 500

        summary_total_input = total_sessions * summary_input_per_session
        summary_total_output = total_sessions * summary_output_per_session

        summary_model = models["summary"]
        _accum("summary", summary_model, summary_total_input, summary_total_output)

        # Phase 3: Question Generation
        # 1 call per active eval category.
        # Input: conversations (truncated ~12k tokens) + summaries (~2k) + prompt overhead (~500)
        # Output: target_count questions * ~300 tokens each.
        eval_categories = scenario_obj.get_applicable_eval_categories()
        eval_breakdown = config.annotation_targets.eval_breakdown

        eval_input_total = 0
        eval_output_total = 0
        for cat in eval_categories:
            cat_val = cat.value if hasattr(cat, "value") else cat
            target_count = eval_breakdown.get(cat_val, 0)
            if target_count == 0:
                continue
            # conversations text (~12k tokens) + summaries (~2k) + users desc (~200) + prompt (~500)
            eval_input_total += 12000 + 2000 + 200 + 500
            eval_output_total += target_count * 300

        question_model = models["question_gen"]
        _accum("question_gen", question_model, eval_input_total, eval_output_total)

        # Per-scenario cost
        def _cost(model_name: str, inp: int, out: int) -> float:
            p = MODEL_PRICING.get(model_name, {"input": 0.15, "output": 0.60})
            return (inp * p["input"] + out * p["output"]) / 1_000_000

        conv_cost = _cost(conv_model, conv_total_input, conv_total_output)
        sum_cost = _cost(summary_model, summary_total_input, summary_total_output)
        q_cost = _cost(question_model, eval_input_total, eval_output_total)
        scenario_cost = conv_cost + sum_cost + q_cost
        grand_total_cost += scenario_cost

        scenario_table.add_row(
            f"S{sid}: {config.name[:28]}",
            str(n_users),
            str(total_sessions),
            f"{actual_doc_tokens:,}",
            f"${conv_cost:.2f}",
            f"${sum_cost:.2f}",
            f"${q_cost:.2f}",
            f"${scenario_cost:.2f}",
        )

    # Totals row
    scenario_table.add_section()
    phase_costs = {}
    for phase, model_dict in phase_totals.items():
        phase_costs[phase] = 0.0
        for model_name, tokens in model_dict.items():
            p = MODEL_PRICING.get(model_name, {"input": 0.15, "output": 0.60})
            phase_costs[phase] += (
                tokens["input"] * p["input"] + tokens["output"] * p["output"]
            ) / 1_000_000

    scenario_table.add_row(
        "[bold]TOTAL[/bold]", "", "", "",
        f"[bold]${phase_costs.get('conversation', 0):.2f}[/bold]",
        f"[bold]${phase_costs.get('summary', 0):.2f}[/bold]",
        f"[bold]${phase_costs.get('question_gen', 0):.2f}[/bold]",
        f"[bold green]${grand_total_cost:.2f}[/bold green]",
    )

    console.print()
    console.print(scenario_table)

    # Token breakdown table
    token_table = Table(title="Token Breakdown by Phase")
    token_table.add_column("Phase", style="cyan")
    token_table.add_column("Model")
    token_table.add_column("Input Tokens", justify="right")
    token_table.add_column("Output Tokens", justify="right")
    token_table.add_column("Cost", justify="right", style="yellow")

    total_input_all = 0
    total_output_all = 0
    for phase in ["conversation", "summary", "question_gen"]:
        model_dict = phase_totals.get(phase, {})
        for model_name, tokens in model_dict.items():
            p = MODEL_PRICING.get(model_name, {"input": 0.15, "output": 0.60})
            cost = (tokens["input"] * p["input"] + tokens["output"] * p["output"]) / 1_000_000
            token_table.add_row(
                phase,
                model_name,
                f"{tokens['input']:,}",
                f"{tokens['output']:,}",
                f"${cost:.2f}",
            )
            total_input_all += tokens["input"]
            total_output_all += tokens["output"]

    token_table.add_section()
    token_table.add_row(
        "[bold]TOTAL[/bold]", "", f"[bold]{total_input_all:,}[/bold]",
        f"[bold]{total_output_all:,}[/bold]",
        f"[bold green]${grand_total_cost:.2f}[/bold green]",
    )

    console.print()
    console.print(token_table)
    console.print()
    console.print(f"[bold green]Total estimated cost ({mode} mode): ${grand_total_cost:.2f}[/bold green]")
    console.print()
    console.print("[dim]Notes:[/dim]")
    console.print("[dim]  - Document tokens are read from actual PDFs (falls back to YAML targets if PDFs missing)[/dim]")
    console.print("[dim]  - Conversation generation uses DeepSeek V3.2 via DeepInfra[/dim]")
    console.print("[dim]  - Question generation uses Gemini 2.5 Pro via Google AI[/dim]")


@main.command()
@click.option("--format", "-f", "fmt", default="huggingface", help="Export format")
@click.option("--output-dir", default="output", help="Output directory")
def export(fmt, output_dir):
    """Export benchmark to external format."""
    if fmt == "huggingface":
        try:
            from scripts.export_hf import export_to_huggingface
            export_to_huggingface(output_dir)
            console.print("[green]Exported to HuggingFace format.[/green]")
        except ImportError:
            console.print("[red]Install 'datasets' package: pip install datasets[/red]")
    else:
        console.print(f"[red]Unknown format: {fmt}[/red]")


if __name__ == "__main__":
    main()
