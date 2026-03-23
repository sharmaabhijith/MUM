#!/usr/bin/env python3
"""Estimate API costs for benchmark generation — all pipeline phases."""

from rich.console import Console
from rich.table import Table

from src.llm.cost_tracker import MODEL_PRICING
from src.llm.token_counter import TokenCounter
from src.pipeline.phase1_document_prep import DocumentPreparer
from src.scenarios import create_scenario, load_scenario
from src.utils.logging import setup_logging

console = Console()


def main():
    setup_logging("WARNING")
    token_counter = TokenCounter()
    doc_preparer = DocumentPreparer(token_counter=token_counter)

    model_configs = {
        "iteration": {
            "conversation": "gpt-4o-mini",
            "summary": "gpt-4o-mini",
            "annotation": "gpt-4o-mini",
            "validation": "gpt-4o-mini",
        },
        "final": {
            "conversation": "gpt-4.1",
            "summary": "gpt-4o-mini",
            "annotation": "gpt-4o",
            "validation": "gpt-4o",
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
        table.add_column("Annotation", justify="right", style="yellow")
        table.add_column("Validation", justify="right", style="yellow")
        table.add_column("Scenario $", justify="right", style="bold green")

        grand_total = 0.0

        for sid in ["1", "2", "3", "4", "5"]:
            config = load_scenario(sid)
            scenario_obj = create_scenario(config)
            n_users = len(config.users)
            n_sessions = config.sessions_per_user
            turns = config.turns_per_session
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

            # Phase 3a: Memory Extraction
            avg_prior_mem = int(sum(range(n_sessions)) * 3.5 * 200 / n_sessions)
            mem_in = total_sessions * (760 + turns_json_tokens + avg_prior_mem + 750)
            mem_out = total_sessions * 700

            # Phase 3b: Conflict Detection
            total_memories = int(total_sessions * 3.5)
            n_conflicts = len(config.injected_conflicts)
            conf_in = 600 + (n_conflicts * 40) + (total_memories * 200) + 200
            conf_out = n_conflicts * 400

            # Phase 3c: Eval Question Generation
            eval_cats = scenario_obj.get_applicable_eval_categories()
            eval_breakdown = config.annotation_targets.eval_breakdown
            eval_in = 0
            eval_out = 0
            for cat in eval_cats:
                cat_val = cat.value if hasattr(cat, "value") else cat
                target = eval_breakdown.get(cat_val, 0)
                if target == 0:
                    continue
                eval_in += 4320
                eval_out += target * 300

            ann_cost = _cost(
                models["annotation"],
                mem_in + conf_in + eval_in,
                mem_out + conf_out + eval_out,
            )

            # Phase 4: Validation
            user_turns_tokens = (turns // 2) * 80
            mem_val = min(50, total_memories)
            ans_val = min(50, config.annotation_targets.eval_questions)
            per_val = min(50, total_sessions)

            val_in = (
                mem_val * (270 + user_turns_tokens)
                + ans_val * 1400
                + per_val * (550 + user_turns_tokens)
            )
            val_out = mem_val * 80 + ans_val * 100 + per_val * 80
            val_cost = _cost(models["validation"], val_in, val_out)

            scenario_cost = conv_cost + sum_cost + ann_cost + val_cost
            grand_total += scenario_cost

            table.add_row(
                f"S{sid}: {config.name[:28]}",
                str(n_users),
                str(total_sessions),
                f"{actual_doc_tokens:,}",
                f"${conv_cost:.2f}",
                f"${sum_cost:.2f}",
                f"${ann_cost:.2f}",
                f"${val_cost:.2f}",
                f"${scenario_cost:.2f}",
            )

        table.add_section()
        table.add_row(
            "[bold]TOTAL[/bold]", "", "", "", "", "", "", "",
            f"[bold green]${grand_total:.2f}[/bold green]",
        )

        console.print()
        console.print(table)
        console.print(
            f"\n[bold green]Total ({mode}): ${grand_total:.2f}[/bold green]\n"
        )

    console.print("[dim]Notes:[/dim]")
    console.print("[dim]  - Document tokens are read from actual PDFs (falls back to YAML targets if missing)[/dim]")
    console.print("[dim]  - OpenAI auto-caches prompt prefixes >= 1024 tokens at 50% off input[/dim]")
    console.print("[dim]  - Batch API gives 50% off all tokens[/dim]")
    console.print("[dim]  - To maximize cache hits, place document context at start of system prompts[/dim]")


if __name__ == "__main__":
    main()
