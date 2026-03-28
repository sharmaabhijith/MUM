"""MUMM-Core Report Generator.

Produces:
  1. Rich console table (model x category matrix)
  2. JSON report (mumm_report.json)
  3. Radar chart PNG + SVG (12-axis, one polygon per baseline)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from datagen.utils.io import read_json, write_json

from MUMBench.aggregator import aggregate_all
from MUMBench.config import CATEGORY_LABELS, CATEGORY_ORDER, MUMM_CONFIG

logger = logging.getLogger("mum.mummbench")
console = Console()

OUTPUT_DIR = Path("output") / "eval_results"
SCORES_DIR = OUTPUT_DIR / "scores"


# ── Console report ─────────────────────────────────────────────────────────────


def print_mumm_report(report: dict) -> None:
    """Print the MUMM report as a rich console table."""
    per_baseline = report.get("per_baseline", {})
    if not per_baseline:
        console.print("[red]No baseline results to display.[/red]")
        return

    baseline_names = sorted(per_baseline.keys())

    # Build column headers
    table = Table(
        title=f"MUMM-Core Evaluation Report\n{report.get('metadata', {}).get('total_questions', '?')} questions · {len(baseline_names)} baselines",
        show_lines=True,
    )
    table.add_column("Category", style="cyan", min_width=24)
    for bname in baseline_names:
        table.add_column(bname.replace("_", " ").title(), justify="right", min_width=12)

    # Category rows
    for cat in CATEGORY_ORDER:
        label = CATEGORY_LABELS.get(cat, cat)
        row = [label]
        for bname in baseline_names:
            cat_data = per_baseline.get(bname, {}).get("per_category", {}).get(cat, {})
            if cat_data and cat_data.get("n", 0) > 0:
                mean = cat_data["mean"]
                ci = cat_data.get("ci_95", [mean, mean])
                cell = f"{mean:.3f}"
                # Color coding
                if mean >= 0.7:
                    cell = f"[green]{cell}[/green]"
                elif mean >= 0.4:
                    cell = f"[yellow]{cell}[/yellow]"
                else:
                    cell = f"[red]{cell}[/red]"
                row.append(cell)
            else:
                row.append("[dim]-[/dim]")
        table.add_row(*row)

    # MUMM score row
    score_row = ["[bold]MUMM Score (overall)[/bold]"]
    for bname in baseline_names:
        mumm_score = per_baseline.get(bname, {}).get("mumm_score", 0.0)
        s = f"[bold]{mumm_score:.3f}[/bold]"
        if mumm_score >= 0.5:
            s = f"[bold green]{mumm_score:.3f}[/bold green]"
        elif mumm_score >= 0.25:
            s = f"[bold yellow]{mumm_score:.3f}[/bold yellow]"
        else:
            s = f"[bold red]{mumm_score:.3f}[/bold red]"
        score_row.append(s)
    table.add_row(*score_row)

    console.print()
    console.print(table)
    console.print()


def print_detailed_report(report: dict) -> None:
    """Print per-scenario breakdown."""
    per_baseline = report.get("per_baseline", {})
    for bname, bdata in sorted(per_baseline.items()):
        scen_table = Table(title=f"{bname} — Per-Scenario Scores", show_lines=False)
        scen_table.add_column("Scenario", style="cyan")
        scen_table.add_column("Mean Score", justify="right", style="green")
        scen_table.add_column("N", justify="right", style="dim")

        for sid, sdata in sorted(bdata.get("per_scenario", {}).items()):
            scen_table.add_row(
                f"Scenario {sid}",
                f"{sdata['mean']:.3f}",
                str(sdata["n"]),
            )
        console.print(scen_table)
        console.print()


# ── JSON report ────────────────────────────────────────────────────────────────


def build_json_report(
    scores_dir: Path = SCORES_DIR,
    n_bootstrap: int = 1000,
    metadata_override: dict | None = None,
) -> dict:
    """Build the full mumm_report.json structure."""
    aggregated = aggregate_all(scores_dir=scores_dir, n_bootstrap=n_bootstrap)

    # Count total questions and baselines from score files
    total_q = 0
    for baseline_dir in sorted(scores_dir.iterdir()) if scores_dir.exists() else []:
        if baseline_dir.is_dir():
            q_ids: set[str] = set()
            for f in baseline_dir.glob("*.json"):
                try:
                    for entry in read_json(f):
                        q_ids.add(entry.get("question_id", ""))
                except Exception:
                    pass
            total_q = max(total_q, len(q_ids))

    meta: dict = {
        "subset": "mumm_core",
        "total_questions": total_q or aggregated.get("total_questions", 0),
        "baselines": sorted(aggregated.get("per_baseline", {}).keys()),
        "judge_model": MUMM_CONFIG["judge_model"],
        "answer_model": MUMM_CONFIG["answer_model"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if metadata_override:
        meta.update(metadata_override)

    report = {
        "metadata": meta,
        **aggregated,
    }
    return report


def save_json_report(report: dict, output_dir: Path = OUTPUT_DIR) -> Path:
    """Save the report to mumm_report.json."""
    path = output_dir / "mumm_report.json"
    write_json(report, path)
    logger.info(f"JSON report saved to {path}")
    return path


# ── Radar chart ────────────────────────────────────────────────────────────────

# Category labels for radar axes (short form)
RADAR_LABELS = [
    "T1\nAttribution",
    "T2\nSynthesis",
    "T3\nConflict",
    "T4\nInfo Gap",
    "T5\nBriefing",
    "T6\nConfusion",
    "T7\nCoverage",
    "T8\nProvenance",
    "T9\nAuthority",
    "T10\nTemporal",
    "T11\nIsolation",
    "T12\nHandoff",
]

BASELINE_COLORS = {
    "no_memory":   "#d62728",   # red
    "rag":         "#1f77b4",   # blue
    "long_context": "#2ca02c",  # green
}

BASELINE_MARKERS = {
    "no_memory":   "o",
    "rag":         "s",
    "long_context": "^",
}


def generate_radar_chart(
    report: dict,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Generate a 12-axis radar chart and save as PNG + SVG."""
    try:
        import math

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib or numpy not installed. Skipping radar chart generation.")
        return

    diagnostic_vectors = report.get("diagnostic_vectors", {})
    if not diagnostic_vectors:
        logger.warning("No diagnostic vectors found. Skipping radar chart.")
        return

    n = len(CATEGORY_ORDER)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(polar=True))

    for baseline_name, vec in diagnostic_vectors.items():
        if len(vec) != n:
            logger.warning(f"Diagnostic vector for {baseline_name} has wrong length: {len(vec)}")
            continue
        values = vec + vec[:1]  # close the loop
        color = BASELINE_COLORS.get(baseline_name, "#888888")
        marker = BASELINE_MARKERS.get(baseline_name, "o")
        label = baseline_name.replace("_", " ").title()

        ax.plot(angles, values, color=color, linewidth=2, marker=marker, markersize=6, label=label)
        ax.fill(angles, values, color=color, alpha=0.1)

    # Axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_LABELS, size=9, fontweight="bold")

    # Radial ticks
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.0", "0.25", "0.50", "0.75", "1.0"], size=8, color="gray")
    ax.set_ylim(0, 1)

    # Grid
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # Title and legend
    ax.set_title(
        "MUMM-Core: 12-Dimensional Diagnostic Radar\n"
        "Baseline Memory Architecture Comparison",
        size=13,
        fontweight="bold",
        pad=25,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=11)

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "radar_chart.png"
    svg_path = output_dir / "radar_chart.svg"
    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(svg_path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info(f"Radar chart saved: {png_path}, {svg_path}")
    console.print(f"[green]Radar chart saved to {png_path}[/green]")


# ── Main report function ───────────────────────────────────────────────────────


def generate_report(
    scores_dir: Path = SCORES_DIR,
    output_dir: Path = OUTPUT_DIR,
    n_bootstrap: int = 1000,
    show_detailed: bool = False,
    skip_chart: bool = False,
) -> dict:
    """End-to-end report: build → print → save JSON → generate radar."""
    console.print("\n[bold]Building MUMM-Core Report...[/bold]")
    report = build_json_report(scores_dir=scores_dir, n_bootstrap=n_bootstrap)

    if not report.get("per_baseline"):
        console.print(
            "[yellow]No scored results found. "
            "Run 'mum-bench score' before generating the report.[/yellow]"
        )
        return report

    print_mumm_report(report)
    if show_detailed:
        print_detailed_report(report)

    report_path = save_json_report(report, output_dir)
    console.print(f"[green]JSON report saved to {report_path}[/green]")

    if not skip_chart:
        generate_radar_chart(report, output_dir)

    return report
