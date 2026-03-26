#!/usr/bin/env python3
"""Estimate API costs for benchmark generation — all pipeline phases."""

from rich.console import Console
from rich.table import Table

from datagen.llm.cost_tracker import MODEL_PRICING
from datagen.llm.token_counter import TokenCounter
from datagen.pipeline.phase1_document_prep import DocumentPreparer
from datagen.scenarios import create_scenario, load_scenario
from datagen.utils.logging import setup_logging

console = Console()


def main():
    setup_logging("WARNING")
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

    for mode, models in model_configs.items():
        table = Table(title=f"MUM Cost Estimate — {mode} mode")
        table.add_column("Scenario", style="cyan")
        table.add_column("Users", justify="right")
        table.add_column("Sessions", justify="right")
        table.add_column("Doc Tokens\n(actual)", justify="right")
        table.add_column("Conv Gen", justify="right", style="yellow")
        table.add_column("Summary", justify="right", style="yellow")
        table.add_column("Questions", justify="right", style="yellow")
        table.add_column("Scenario $", justify="right", style="bold green")

        grand_total = 0.0

        for sid in ["1", "2", "3", "4", "5"]:
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

            def _cost(model_name: str, inp: int, out: int) -> float:
                p = MODEL_PRICING.get(model_name, {"input": 0.15, "output": 0.60})
                return (inp * p["input"] + out * p["output"]) / 1_000_000

            # Phase 2a: Conversation Generation
            avg_prior_summary = int(sum(range(n_sessions)) * 270 / n_sessions)
            conv_overhead = 600 + 100 + 500 + avg_prior_summary + 50
            conv_in = total_sessions * (actual_doc_tokens + conv_overhead)
            conv_out = total_sessions * turns * 2 * 90
            conv_cost = _cost(models["conversation"], conv_in, conv_out)

            # Phase 2b: Session Summary
            turns_json_tokens = turns * 2 * 120
            sum_in = total_sessions * (350 + turns_json_tokens + avg_prior_summary)
            sum_out = total_sessions * 500
            sum_cost = _cost(models["summary"], sum_in, sum_out)

            # Phase 3: Question Generation
            eval_cats = scenario_obj.get_applicable_eval_categories()
            eval_breakdown = config.annotation_targets.eval_breakdown
            eval_in = 0
            eval_out = 0
            for cat in eval_cats:
                cat_val = cat.value if hasattr(cat, "value") else cat
                target = eval_breakdown.get(cat_val, 0)
                if target == 0:
                    continue
                eval_in += 12000 + 2000 + 200 + 500
                eval_out += target * 300

            q_cost = _cost(models["question_gen"], eval_in, eval_out)

            scenario_cost = conv_cost + sum_cost + q_cost
            grand_total += scenario_cost

            table.add_row(
                f"S{sid}: {config.name[:28]}",
                str(n_users),
                str(total_sessions),
                f"{actual_doc_tokens:,}",
                f"${conv_cost:.2f}",
                f"${sum_cost:.2f}",
                f"${q_cost:.2f}",
                f"${scenario_cost:.2f}",
            )

        table.add_section()
        table.add_row(
            "[bold]TOTAL[/bold]", "", "", "", "", "", "",
            f"[bold green]${grand_total:.2f}[/bold green]",
        )

        console.print()
        console.print(table)
        console.print(
            f"\n[bold green]Total ({mode}): ${grand_total:.2f}[/bold green]\n"
        )

    console.print("[dim]Notes:[/dim]")
    console.print("[dim]  - Document tokens are read from actual PDFs (falls back to YAML targets if missing)[/dim]")
    console.print("[dim]  - Conversation generation uses DeepSeek V3.2 via DeepInfra[/dim]")
    console.print("[dim]  - Question generation uses Gemini 2.5 Pro via Google AI[/dim]")


if __name__ == "__main__":
    main()
