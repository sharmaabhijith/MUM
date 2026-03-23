#!/usr/bin/env python3
"""Validate that all source PDFs exist and meet token targets."""

import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.llm.token_counter import TokenCounter
from src.scenarios import load_scenario

console = Console()


def validate_scenario_docs(scenario_id: str, token_counter: TokenCounter) -> bool:
    config = load_scenario(scenario_id)
    doc_dir = Path("documents") / f"scenario_{scenario_id}"

    table = Table(title=f"Scenario {scenario_id}: {config.name}")
    table.add_column("Document", style="cyan")
    table.add_column("File", style="white")
    table.add_column("Exists", style="green")
    table.add_column("Tokens", style="yellow")
    table.add_column("Target", style="yellow")
    table.add_column("Status", style="bold")

    all_ok = True
    for doc_cfg in config.documents:
        pdf_path = doc_dir / doc_cfg.filename
        exists = pdf_path.exists()
        tokens = "-"
        status = "MISSING"

        if exists:
            from src.utils.pdf_reader import extract_text_from_pdf

            text = extract_text_from_pdf(pdf_path)
            token_count = token_counter.count(text)
            tokens = str(token_count)
            lower = int(doc_cfg.target_tokens * 0.8)
            upper = int(doc_cfg.target_tokens * 1.2)
            if lower <= token_count <= upper:
                status = "OK"
            else:
                status = "OUT OF RANGE"
                all_ok = False
        else:
            all_ok = False

        table.add_row(
            doc_cfg.name,
            doc_cfg.filename,
            "Yes" if exists else "No",
            tokens,
            str(doc_cfg.target_tokens),
            status,
        )

    console.print(table)
    return all_ok


def main():
    token_counter = TokenCounter()
    scenario_ids = sys.argv[1:] if len(sys.argv) > 1 else ["1", "2", "3", "4", "5"]

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
        console.print("\n[green]All documents validated successfully.[/green]")
    else:
        console.print("\n[red]Some documents are missing or out of range.[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
