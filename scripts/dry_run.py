#!/usr/bin/env python3
"""Dry run: test full pipeline with mock LLM client."""

from rich.console import Console

from src.pipeline.orchestrator import PipelineOrchestrator
from src.utils.logging import setup_logging

console = Console()


def main():
    setup_logging("INFO")
    console.print("[bold]Running dry run with mock LLM...[/bold]\n")

    orchestrator = PipelineOrchestrator(
        model="gpt-4o-mini",
        dry_run=True,
        output_dir="output/dry_run",
    )

    try:
        output = orchestrator.run_scenario("1")
        console.print(f"\n[green]Dry run completed for scenario 1[/green]")
        console.print(f"  Conversations: {len(output.conversations)}")
        console.print(f"  Summaries: {len(output.session_summaries)}")
        console.print(f"  Memories: {len(output.memories)}")
        console.print(f"  Conflicts: {len(output.conflicts)}")
        console.print(f"  Eval questions: {len(output.eval_questions)}")
    except Exception as e:
        console.print(f"[red]Dry run failed: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
