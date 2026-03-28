"""Tests for MUMBench.aggregator — aggregation math and CI computation."""

from __future__ import annotations

import statistics

import pytest

from MUMBench.aggregator import (
    aggregate_per_category,
    aggregate_per_scenario,
    bootstrap_ci,
    compute_diagnostic_vector,
    compute_mumm_score,
)
from MUMBench.config import CATEGORY_ORDER


# ── bootstrap_ci ───────────────────────────────────────────────────────────────


class TestBootstrapCI:
    def test_single_value(self):
        lo, hi = bootstrap_ci([0.5])
        assert lo == hi == 0.5

    def test_ci_within_data_range(self):
        vals = [0.1, 0.3, 0.5, 0.7, 0.9]
        lo, hi = bootstrap_ci(vals, n_resamples=200)
        assert lo <= statistics.mean(vals) <= hi
        assert 0.0 <= lo <= hi <= 1.0

    def test_ci_narrower_for_uniform_data(self):
        """More uniform data → narrower CI."""
        uniform = [0.5] * 100
        varied = [0.0, 0.1, 0.2, 0.8, 0.9, 1.0] * 16 + [0.5] * 4
        lo_u, hi_u = bootstrap_ci(uniform, n_resamples=500)
        lo_v, hi_v = bootstrap_ci(varied, n_resamples=500)
        width_uniform = hi_u - lo_u
        width_varied = hi_v - lo_v
        assert width_uniform <= width_varied

    def test_reproducibility(self):
        vals = [0.2, 0.4, 0.6, 0.8]
        lo1, hi1 = bootstrap_ci(vals, n_resamples=100, seed=42)
        lo2, hi2 = bootstrap_ci(vals, n_resamples=100, seed=42)
        assert lo1 == lo2 and hi1 == hi2


# ── aggregate_per_category ────────────────────────────────────────────────────


def _make_scores(
    category: str,
    scenarios: list[int],
    scores_per_scenario: list[float],
) -> list[dict]:
    """Build fake score records."""
    records = []
    for sid, score_val in zip(scenarios, scores_per_scenario):
        records.append({
            "question_id": f"q_{category}_{sid}",
            "category": category,
            "difficulty": "medium",
            "score": score_val,
            "scenario": sid,
            "metric": "exact_match",
        })
    return records


class TestAggregatePerCategory:
    def test_basic_aggregation(self):
        scores = _make_scores("user_attribution", [1, 2, 3, 4, 5], [0.8, 0.6, 0.7, 0.9, 0.5])
        result = aggregate_per_category(scores, n_bootstrap=100)
        cat_data = result["user_attribution"]
        assert cat_data["n"] == 5
        assert abs(cat_data["mean"] - statistics.mean([0.8, 0.6, 0.7, 0.9, 0.5])) < 0.001
        assert cat_data["ci_95"][0] <= cat_data["mean"] <= cat_data["ci_95"][1]

    def test_missing_category_returns_zero(self):
        scores = _make_scores("user_attribution", [1], [0.5])
        result = aggregate_per_category(scores, n_bootstrap=100)
        # authority_hierarchy not in scores → should return n=0, mean=0.0
        ah = result.get("authority_hierarchy", {})
        assert ah.get("n", 0) == 0
        assert ah.get("mean", 0.0) == 0.0

    def test_all_categories_present_in_output(self):
        scores = _make_scores("user_attribution", [1], [0.5])
        result = aggregate_per_category(scores, n_bootstrap=100)
        for cat in CATEGORY_ORDER:
            assert cat in result, f"Category {cat} missing from aggregation result"

    def test_per_scenario_breakdown(self):
        scores = (
            _make_scores("user_attribution", [1, 1], [0.8, 0.6]) +
            _make_scores("user_attribution", [2, 2], [0.4, 0.2])
        )
        result = aggregate_per_category(scores, n_bootstrap=100)
        per_scen = result["user_attribution"]["per_scenario"]
        assert "1" in per_scen
        assert "2" in per_scen
        assert abs(per_scen["1"]["mean"] - 0.7) < 0.001
        assert abs(per_scen["2"]["mean"] - 0.3) < 0.001


# ── aggregate_per_scenario ────────────────────────────────────────────────────


class TestAggregatePerScenario:
    def test_basic(self):
        scores = (
            _make_scores("user_attribution", [1, 1], [0.8, 0.6]) +
            _make_scores("conflict_resolution", [2], [0.4])
        )
        result = aggregate_per_scenario(scores)
        assert "1" in result
        assert "2" in result
        assert abs(result["1"]["mean"] - 0.7) < 0.001
        assert result["2"]["mean"] == 0.4


# ── compute_mumm_score ────────────────────────────────────────────────────────


class TestComputeMUMMScore:
    def test_equal_categories(self):
        """MUMM score = unweighted mean of category scores."""
        per_cat = {cat: {"mean": 0.5, "n": 10, "ci_95": [0.4, 0.6], "per_scenario": {}}
                   for cat in CATEGORY_ORDER}
        score = compute_mumm_score(per_cat)
        assert abs(score - 0.5) < 0.001

    def test_ignores_zero_n_categories(self):
        """Categories with n=0 are excluded from the mean."""
        per_cat = {cat: {"mean": 0.0, "n": 0, "ci_95": [0.0, 0.0], "per_scenario": {}}
                   for cat in CATEGORY_ORDER}
        per_cat["user_attribution"] = {"mean": 0.8, "n": 10, "ci_95": [0.7, 0.9], "per_scenario": {}}
        score = compute_mumm_score(per_cat)
        assert abs(score - 0.8) < 0.001

    def test_empty_gives_zero(self):
        per_cat = {cat: {"mean": 0.0, "n": 0, "ci_95": [0.0, 0.0], "per_scenario": {}}
                   for cat in CATEGORY_ORDER}
        score = compute_mumm_score(per_cat)
        assert score == 0.0


# ── compute_diagnostic_vector ─────────────────────────────────────────────────


class TestDiagnosticVector:
    def test_length_matches_category_order(self):
        per_cat = {cat: {"mean": 0.5, "n": 10, "ci_95": [0.4, 0.6], "per_scenario": {}}
                   for cat in CATEGORY_ORDER}
        vec = compute_diagnostic_vector(per_cat)
        assert len(vec) == len(CATEGORY_ORDER)

    def test_values_match_category_order(self):
        per_cat = {cat: {"mean": float(i) / len(CATEGORY_ORDER), "n": 10, "ci_95": [0.0, 1.0], "per_scenario": {}}
                   for i, cat in enumerate(CATEGORY_ORDER)}
        vec = compute_diagnostic_vector(per_cat)
        for i, cat in enumerate(CATEGORY_ORDER):
            expected = float(i) / len(CATEGORY_ORDER)
            assert abs(vec[i] - round(expected, 4)) < 0.001
