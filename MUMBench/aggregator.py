"""MUMM-Core Score Aggregation.

Reads score files from output/eval_results/scores/ and produces:
  1. Per-question scores (raw)
  2. Per-category-per-scenario scores (mean ± std)
  3. Per-category scores across all applicable scenarios   ← PRIMARY
  4. Per-scenario scores
  5. Overall MUMM score (unweighted mean of 12 per-category scores)
  6. 12-dimensional diagnostic vector (one score per category) ← RADAR CHART
  7. Cost summary
  8. 95% confidence intervals via bootstrap (1000 resamples)
"""

from __future__ import annotations

import json
import logging
import random
import statistics
from pathlib import Path

from datagen.utils.io import read_json

from MUMBench.config import CATEGORY_ORDER, CORE_BUDGET, MUMM_CONFIG

logger = logging.getLogger("mum.mummbench")

SCORES_DIR = Path("output") / "eval_results" / "scores"
OUTPUT_DIR = Path("output") / "eval_results"


# ── Bootstrap CI ──────────────────────────────────────────────────────────────


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """95% confidence interval via bootstrap resampling."""
    if len(values) <= 1:
        v = values[0] if values else 0.0
        return v, v
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        sample = [rng.choice(values) for _ in range(len(values))]
        means.append(statistics.mean(sample))
    means.sort()
    lo_idx = int((1 - ci) / 2 * n_resamples)
    hi_idx = int((1 + ci) / 2 * n_resamples)
    return means[lo_idx], means[hi_idx]


def _safe_mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def _safe_std(vals: list[float]) -> float:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


# ── Score file loading ─────────────────────────────────────────────────────────


def load_score_files(
    baseline_name: str,
    scores_dir: Path = SCORES_DIR,
) -> list[dict]:
    """Load all score entries for a given baseline."""
    baseline_dir = scores_dir / baseline_name
    if not baseline_dir.exists():
        return []
    all_scores: list[dict] = []
    for f in sorted(baseline_dir.glob("scenario_*_scores.json")):
        try:
            data = read_json(f)
            all_scores.extend(data)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")
    return all_scores


def load_all_scores(scores_dir: Path = SCORES_DIR) -> dict[str, list[dict]]:
    """Load scores for all baselines. Returns {baseline_name: [score_dicts]}."""
    baselines: dict[str, list[dict]] = {}
    if not scores_dir.exists():
        return baselines
    for baseline_dir in sorted(scores_dir.iterdir()):
        if baseline_dir.is_dir():
            scores = load_score_files(baseline_dir.name, scores_dir)
            if scores:
                baselines[baseline_dir.name] = scores
    return baselines


# ── Per-category aggregation ───────────────────────────────────────────────────


def _applicable_scenarios(category: str) -> list[int]:
    """Return scenario IDs where this category is applicable."""
    budget_entry = CORE_BUDGET.get(category)
    if not budget_entry:
        return list(range(1, 6))
    return budget_entry.get("scenarios", list(range(1, 6)))


def aggregate_per_category(
    scores: list[dict],
    n_bootstrap: int = 1000,
) -> dict[str, dict]:
    """Aggregate scores per category, across all applicable scenarios.

    Returns:
        {category: {mean, std, n, ci_95_lo, ci_95_hi, per_scenario: {...}}}
    """
    from collections import defaultdict

    # Group by category
    by_cat: dict[str, list[float]] = defaultdict(list)
    by_cat_scenario: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for s in scores:
        cat = s.get("category", "")
        score_val = float(s.get("score", 0.0))
        scenario = str(s.get("scenario", ""))
        if cat:
            by_cat[cat].append(score_val)
            if scenario:
                by_cat_scenario[cat][scenario].append(score_val)

    result: dict[str, dict] = {}
    for cat in CATEGORY_ORDER:
        vals = by_cat.get(cat, [])
        if not vals:
            result[cat] = {
                "mean": 0.0, "std": 0.0, "n": 0,
                "ci_95": [0.0, 0.0], "per_scenario": {},
            }
            continue
        mean = _safe_mean(vals)
        std = _safe_std(vals)
        ci_lo, ci_hi = bootstrap_ci(vals, n_resamples=n_bootstrap)
        per_scenario: dict[str, dict] = {}
        for sid, svals in by_cat_scenario.get(cat, {}).items():
            per_scenario[sid] = {
                "mean": round(_safe_mean(svals), 4),
                "n": len(svals),
            }
        result[cat] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "n": len(vals),
            "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "per_scenario": per_scenario,
        }
    return result


def aggregate_per_scenario(scores: list[dict]) -> dict[str, dict]:
    """Aggregate scores per scenario (mean across all categories in that scenario)."""
    from collections import defaultdict

    by_scenario: dict[str, list[float]] = defaultdict(list)
    for s in scores:
        scenario = str(s.get("scenario", ""))
        score_val = float(s.get("score", 0.0))
        if scenario:
            by_scenario[scenario].append(score_val)

    return {
        sid: {"mean": round(_safe_mean(vals), 4), "n": len(vals)}
        for sid, vals in sorted(by_scenario.items())
    }


def compute_mumm_score(per_category: dict[str, dict]) -> float:
    """Overall MUMM score = unweighted mean of 12 per-category scores.

    Each category contributes equally regardless of sample size.
    """
    cat_means = [
        per_category[cat]["mean"]
        for cat in CATEGORY_ORDER
        if cat in per_category and per_category[cat]["n"] > 0
    ]
    return round(_safe_mean(cat_means), 4) if cat_means else 0.0


def compute_diagnostic_vector(per_category: dict[str, dict]) -> list[float]:
    """12-dimensional vector for radar chart. One entry per CATEGORY_ORDER."""
    return [
        round(per_category.get(cat, {}).get("mean", 0.0), 4)
        for cat in CATEGORY_ORDER
    ]


# ── Full aggregation ───────────────────────────────────────────────────────────


def aggregate_all(
    scores_dir: Path = SCORES_DIR,
    n_bootstrap: int = 1000,
) -> dict:
    """Aggregate scores across all baselines.

    Returns a full report dict (matches mumm_report.json structure).
    """
    all_scores = load_all_scores(scores_dir)

    if not all_scores:
        logger.warning("No score files found. Run scoring step first.")
        return {}

    per_baseline: dict[str, dict] = {}
    diagnostic_vectors: dict[str, list[float]] = {}

    for baseline_name, scores in all_scores.items():
        per_cat = aggregate_per_category(scores, n_bootstrap=n_bootstrap)
        per_scen = aggregate_per_scenario(scores)
        mumm_score = compute_mumm_score(per_cat)
        diag_vec = compute_diagnostic_vector(per_cat)

        per_baseline[baseline_name] = {
            "mumm_score": mumm_score,
            "per_category": per_cat,
            "per_scenario": per_scen,
        }
        diagnostic_vectors[baseline_name] = diag_vec

    return {
        "per_baseline": per_baseline,
        "diagnostic_vectors": diagnostic_vectors,
    }


# ── Score file builder (from answer files) ────────────────────────────────────


def build_score_files_from_answers(
    answer_dir: Path,
    scores_dir: Path,
    judge_model: str | None = None,
    factscore_model: str | None = None,
    resume: bool = True,
) -> None:
    """Run scoring on all answer files and write score files.

    This is the bridge between runner.py (answers) and aggregator.py (scores).
    Calls scorer.py and judge.py for each answer.
    """
    from datagen.llm.client import LLMClient
    from datagen.llm.cost_tracker import CostTracker

    from MUMBench.judge import MUMMJudge
    from MUMBench.scorer import score_question

    judge_mod = judge_model or MUMM_CONFIG["judge_model"]
    cost_tracker = CostTracker()
    judge = MUMMJudge(model=judge_mod, cost_tracker=cost_tracker)

    factscore_llm: LLMClient | None = None
    if factscore_model:
        factscore_llm = LLMClient(
            model=factscore_model, temperature=0.0, cost_tracker=cost_tracker
        )

    if not answer_dir.exists():
        logger.error(f"Answer directory not found: {answer_dir}")
        return

    for baseline_dir in sorted(answer_dir.iterdir()):
        if not baseline_dir.is_dir():
            continue
        baseline_name = baseline_dir.name
        scores_baseline_dir = scores_dir / baseline_name
        scores_baseline_dir.mkdir(parents=True, exist_ok=True)

        for answer_file in sorted(baseline_dir.glob("scenario_*_answers.json")):
            # Derive score file path
            score_filename = answer_file.name.replace("_answers.json", "_scores.json")
            score_path = scores_baseline_dir / score_filename

            # Resume: skip if score file exists and has same # entries
            answers: list[dict] = read_json(answer_file)
            if resume and score_path.exists():
                existing_scores: list[dict] = read_json(score_path)
                if len(existing_scores) >= len(answers):
                    logger.info(
                        f"[score] {baseline_name}/{score_filename}: already scored, skipping"
                    )
                    continue

            logger.info(f"[score] Scoring {baseline_name}/{answer_file.name}...")
            scored: list[dict] = []
            for ans in answers:
                q_dict = {
                    "question_id": ans["question_id"],
                    "category": ans["category"],
                    "difficulty": ans.get("difficulty", "medium"),
                    "question": ans["question"],
                    "gold_answer": ans["gold_answer"],
                    "scenario": ans.get("scenario"),
                }
                qs = score_question(
                    question_dict=q_dict,
                    predicted_answer=ans.get("predicted_answer", ""),
                    judge=judge,
                    factscore_llm=factscore_llm,
                )
                entry = qs.to_dict()
                entry["scenario"] = ans.get("scenario")
                scored.append(entry)

            from datagen.utils.io import write_json
            write_json(scored, score_path)
            logger.info(
                f"[score] {baseline_name}/{score_filename}: "
                f"{len(scored)} questions scored → {score_path}"
            )
