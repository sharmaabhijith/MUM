#!/usr/bin/env python3
"""Export benchmark to HuggingFace datasets format."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def export_to_huggingface(output_dir: str = "MUMBench") -> None:
    output_path = Path(output_dir)
    benchmark_path = output_path / "benchmark.json"

    if not benchmark_path.exists():
        console.print(f"[red]Benchmark file not found: {benchmark_path}[/red]")
        console.print("Run the full pipeline first: python -m datagen.cli generate --all")
        return

    with open(benchmark_path) as f:
        benchmark = json.load(f)

    try:
        from datasets import Dataset, DatasetDict
    except ImportError:
        console.print("[red]Install 'datasets': pip install datasets[/red]")
        return

    # Build dataset splits
    conversations_rows = []
    summaries_rows = []
    memories_rows = []
    conflicts_rows = []
    questions_rows = []

    for scenario in benchmark.get("scenarios", []):
        sid = scenario["scenario_id"]

        for conv in scenario.get("conversations", []):
            conversations_rows.append({
                "scenario_id": sid,
                "session_id": conv["session_id"],
                "user_id": conv["user_id"],
                "session_number": conv["session_number"],
                "session_timestamp": conv["session_timestamp"],
                "turns": json.dumps(conv["turns"]),
                "num_turns": len(conv["turns"]),
            })

        for summ in scenario.get("session_summaries", []):
            summaries_rows.append({
                "scenario_id": sid,
                "session_id": summ["session_id"],
                "user_id": summ["user_id"],
                "session_number": summ["session_number"],
                "summary": summ["summary"],
                "key_facts": json.dumps(summ["key_facts"]),
                "positions_taken": json.dumps(summ["positions_taken"]),
                "positions_changed": json.dumps(summ.get("positions_changed", [])),
            })

        for mem in scenario.get("memories", []):
            memories_rows.append({
                "scenario_id": sid,
                "memory_id": mem["memory_id"],
                "user_id": mem["user_id"],
                "session_id": mem["session_id"],
                "memory_type": mem["memory_type"],
                "content": mem["content"],
                "status": mem["status"],
                "superseded_by": mem.get("superseded_by", ""),
                "authority_level": mem["authority_level"],
                "timestamp": mem["timestamp"],
            })

        for conf in scenario.get("conflicts", []):
            conflicts_rows.append({
                "scenario_id": sid,
                "conflict_id": conf["conflict_id"],
                "users_involved": json.dumps(conf["users_involved"]),
                "topic": conf["topic"],
                "conflict_type": conf["conflict_type"],
                "positions": json.dumps(conf["positions"]),
                "resolution": conf["resolution"],
                "resolution_detail": conf["resolution_detail"],
                "evidence": json.dumps(conf["evidence"]),
            })

        for q in scenario.get("eval_questions", []):
            questions_rows.append({
                "scenario_id": sid,
                "question_id": q["question_id"],
                "category": q["category"],
                "question": q["question"],
                "gold_answer": q["gold_answer"],
                "evidence": json.dumps(q["evidence"]),
                "required_memories": json.dumps(q.get("required_memories", [])),
                "required_conflicts": json.dumps(q.get("required_conflicts", [])),
                "difficulty": q["difficulty"],
            })

    # Create datasets
    dataset_dict = DatasetDict({
        "conversations": Dataset.from_list(conversations_rows) if conversations_rows else Dataset.from_dict({}),
        "summaries": Dataset.from_list(summaries_rows) if summaries_rows else Dataset.from_dict({}),
        "memories": Dataset.from_list(memories_rows) if memories_rows else Dataset.from_dict({}),
        "conflicts": Dataset.from_list(conflicts_rows) if conflicts_rows else Dataset.from_dict({}),
        "eval_questions": Dataset.from_list(questions_rows) if questions_rows else Dataset.from_dict({}),
    })

    # Save to disk
    hf_output = output_path / "huggingface"
    dataset_dict.save_to_disk(str(hf_output))
    console.print(f"[green]Saved to {hf_output}[/green]")
    console.print(f"  Conversations: {len(conversations_rows)}")
    console.print(f"  Summaries: {len(summaries_rows)}")
    console.print(f"  Memories: {len(memories_rows)}")
    console.print(f"  Conflicts: {len(conflicts_rows)}")
    console.print(f"  Eval Questions: {len(questions_rows)}")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "MUMBench"
    export_to_huggingface(output_dir)
