"""MUMM-Core Subset Selector.

Reads the full ~1,200 questions from output/scenario_*/evaluation/eval_questions.json,
applies stratified-by-difficulty sampling according to CORE_BUDGET, and writes
a mumm_core.json manifest.

Usage:
    python -m MUMBench.subset_selector --output output/mumm_core.json
    python -m MUMBench.subset_selector --seed 42 --dry-run
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from MUMBench.config import ALL_SCENARIOS, CORE_BUDGET, DIFFICULTY_ORDER

logger = logging.getLogger("mum.mummbench")

BENCHMARK_DIR = Path("MUMBench")


# ── Data loading ───────────────────────────────────────────────────────────────


def load_scenario_questions(scenario_id: int, benchmark_dir: Path = BENCHMARK_DIR) -> list[dict]:
    """Load eval_questions.json for one scenario. Returns raw dicts."""
    path = benchmark_dir / f"scenario_{scenario_id}" / "evaluation" / "eval_questions.json"
    if not path.exists():
        logger.warning(f"eval_questions.json not found: {path}")
        return []
    with open(path) as f:
        questions = json.load(f)
    # Ensure scenario_id field is present as int/str consistent value
    for q in questions:
        q["scenario_id"] = str(scenario_id)
    logger.info(f"Loaded {len(questions)} questions from scenario {scenario_id}")
    return questions


# ── Stratified sampler ─────────────────────────────────────────────────────────


def _proportional_sample(
    questions: list[dict],
    budget: int,
    rng: random.Random,
) -> list[dict]:
    """Sample `budget` questions proportionally across difficulty strata.

    If a stratum is empty, its budget is redistributed to others.
    Ties go to medium first, then hard, then easy.
    """
    by_diff: dict[str, list[dict]] = {d: [] for d in DIFFICULTY_ORDER}
    for q in questions:
        diff = q.get("difficulty", "medium")
        if diff not in by_diff:
            diff = "medium"
        by_diff[diff].append(q)

    n = len(questions)
    if n == 0:
        return []

    # Compute proportional allocation
    alloc: dict[str, int] = {}
    remaining = budget
    fractions: dict[str, float] = {}
    for diff in DIFFICULTY_ORDER:
        if by_diff[diff]:
            frac = len(by_diff[diff]) / n * budget
            alloc[diff] = int(frac)
            fractions[diff] = frac - int(frac)
            remaining -= int(frac)
        else:
            alloc[diff] = 0
            fractions[diff] = 0.0

    # Distribute remainder by fractional priority (medium first as tiebreaker)
    priority_order = ["medium", "hard", "easy"]
    sorted_diffs = sorted(
        priority_order,
        key=lambda d: fractions.get(d, 0.0),
        reverse=True,
    )
    for diff in sorted_diffs:
        if remaining <= 0:
            break
        if by_diff[diff] and alloc[diff] < len(by_diff[diff]):
            alloc[diff] += 1
            remaining -= 1

    # If we still have remainder (due to small pool), give to any stratum
    for diff in priority_order:
        if remaining <= 0:
            break
        extra = min(remaining, len(by_diff[diff]) - alloc[diff])
        if extra > 0:
            alloc[diff] += extra
            remaining -= extra

    # Sample from each stratum
    sampled: list[dict] = []
    for diff in DIFFICULTY_ORDER:
        pool = by_diff[diff]
        k = min(alloc[diff], len(pool))
        if k > 0:
            sampled.extend(rng.sample(pool, k))

    return sampled


def select_core_subset(
    scenario_id: int,
    questions: list[dict],
    budget_entry: dict,
    rng: random.Random,
) -> list[dict]:
    """Select questions for one (category, scenario) pair from CORE_BUDGET entry.

    Returns sampled question dicts (augmented with category/scenario fields).
    """
    category = budget_entry["category"]
    per_scenario = budget_entry["per_scenario"]

    pool = [q for q in questions if q.get("category") == category]

    if not pool:
        logger.warning(
            f"No questions found for category={category} in scenario {scenario_id}. "
            f"Expected {per_scenario}."
        )
        return []

    if len(pool) < per_scenario:
        logger.warning(
            f"Insufficient questions for category={category} scenario={scenario_id}: "
            f"have {len(pool)}, want {per_scenario}. Using all."
        )
        return list(pool)

    sampled = _proportional_sample(pool, per_scenario, rng)
    logger.debug(
        f"  category={category} scenario={scenario_id}: "
        f"sampled {len(sampled)}/{len(pool)} (budget={per_scenario})"
    )
    return sampled


# ── Main entry point ───────────────────────────────────────────────────────────


def build_mumm_core(
    benchmark_dir: Path = BENCHMARK_DIR,
    output_path: Path | None = None,
    seed: int = 42,
    budget_override: dict | None = None,
    dry_run: bool = False,
) -> dict:
    """Build the MUMM-Core manifest.

    Args:
        benchmark_dir: Root of benchmark data (contains scenario_N/ dirs).
        output_path: Where to write mumm_core.json. Defaults to output/mumm_core.json.
        seed: Random seed for reproducibility.
        budget_override: Optional dict to override CORE_BUDGET.
        dry_run: If True, print stats but don't write file.

    Returns:
        The manifest dict (also written to output_path unless dry_run).
    """
    rng = random.Random(seed)
    budget = budget_override or CORE_BUDGET

    if output_path is None:
        output_path = Path("output") / "mumm_core.json"

    # Load all scenario data
    scenario_questions: dict[int, list[dict]] = {}
    for sid in ALL_SCENARIOS:
        scenario_questions[sid] = load_scenario_questions(sid, benchmark_dir)

    # Build per-scenario selection
    full_question_list: list[dict] = []
    per_scenario_manifest: dict[str, dict] = {}
    selected_ids: set[str] = set()

    for sid in ALL_SCENARIOS:
        questions_for_scenario = scenario_questions[sid]
        selected_for_scenario: list[dict] = []

        for cat, cat_budget in budget.items():
            applicable_scenarios = cat_budget.get("scenarios", ALL_SCENARIOS)
            if sid not in applicable_scenarios:
                continue

            entry = {"category": cat, "per_scenario": cat_budget["per_scenario"]}
            sampled = select_core_subset(sid, questions_for_scenario, entry, rng)
            selected_for_scenario.extend(sampled)

        # Build manifest for this scenario
        counts_by_cat: dict[str, int] = {}
        counts_by_diff: dict[str, int] = {}
        question_ids: list[str] = []

        for q in selected_for_scenario:
            qid = q["question_id"]
            if qid in selected_ids:
                logger.warning(f"Duplicate question ID {qid} — skipping")
                continue
            selected_ids.add(qid)
            question_ids.append(qid)

            cat = q["category"]
            diff = q.get("difficulty", "medium")
            counts_by_cat[cat] = counts_by_cat.get(cat, 0) + 1
            counts_by_diff[diff] = counts_by_diff.get(diff, 0) + 1

            # Build full question entry for manifest
            full_question_list.append({
                "question_id": qid,
                "scenario": sid,
                "category": cat,
                "difficulty": diff,
                "question": q["question"],
                "gold_answer": q["gold_answer"],
                "evidence_links": q.get("evidence", []),
            })

        per_scenario_manifest[f"scenario_{sid}"] = {
            "question_ids": question_ids,
            "counts_by_category": counts_by_cat,
            "counts_by_difficulty": counts_by_diff,
            "total": len(question_ids),
        }
        logger.info(
            f"Scenario {sid}: selected {len(question_ids)} questions "
            f"across {len(counts_by_cat)} categories"
        )

    total = len(full_question_list)
    manifest = {
        "version": "1.0",
        "total_questions": total,
        "sampling_strategy": "stratified_by_difficulty",
        "seed": seed,
        "benchmark_dir": str(benchmark_dir),
        "per_scenario": per_scenario_manifest,
        "full_question_list": full_question_list,
    }

    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info(f"MUMM-Core: {total} questions selected")
    # Category totals
    cat_totals: dict[str, int] = {}
    for q in full_question_list:
        c = q["category"]
        cat_totals[c] = cat_totals.get(c, 0) + 1
    for cat, count in sorted(cat_totals.items()):
        logger.info(f"  {cat}: {count}")
    logger.info(f"{'='*50}\n")

    if not dry_run:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logger.info(f"MUMM-Core manifest written to {output_path}")
    else:
        print(f"\n[dry-run] Would write {total} questions to {output_path}")
        for cat, count in sorted(cat_totals.items()):
            print(f"  {cat}: {count}")

    return manifest


# ── Loader for downstream components ──────────────────────────────────────────


def load_mumm_core(path: Path | str = "output/mumm_core.json") -> dict:
    """Load an existing MUMM-Core manifest from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MUMM-Core manifest not found: {p}")
    with open(p) as f:
        return json.load(f)


def get_questions_for_scenario(manifest: dict, scenario_id: int) -> list[dict]:
    """Return all MUMM-Core questions for a given scenario."""
    sid_str = str(scenario_id)
    ids_in_scenario = set(
        manifest["per_scenario"].get(f"scenario_{sid_str}", {}).get("question_ids", [])
    )
    return [q for q in manifest["full_question_list"] if q["question_id"] in ids_in_scenario]


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Build MUMM-Core subset")
    parser.add_argument("--benchmark-dir", default="MUMBench", help="Benchmark data dir")
    parser.add_argument("--output", default="output/mumm_core.json", help="Output path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    build_mumm_core(
        benchmark_dir=Path(args.benchmark_dir),
        output_path=Path(args.output),
        seed=args.seed,
        dry_run=args.dry_run,
    )
