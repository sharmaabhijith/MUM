"""Tests for MUMBench.subset_selector."""

from __future__ import annotations

import random

import pytest

from MUMBench.config import CORE_BUDGET, ALL_SCENARIOS
from MUMBench.subset_selector import (
    _proportional_sample,
    select_core_subset,
    get_questions_for_scenario,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_questions(n: int, category: str, difficulties: list[str] | None = None) -> list[dict]:
    """Build n fake question dicts."""
    diffs = difficulties or ["easy", "medium", "hard"]
    return [
        {
            "question_id": f"q_{i}",
            "scenario_id": "1",
            "category": category,
            "question": f"Question {i}?",
            "gold_answer": f"Answer {i}.",
            "evidence": [],
            "difficulty": diffs[i % len(diffs)],
        }
        for i in range(n)
    ]


# ── _proportional_sample ──────────────────────────────────────────────────────


def test_proportional_sample_exact_count():
    """Sample returns exactly `budget` items."""
    rng = random.Random(42)
    questions = _make_questions(30, "user_attribution")
    sampled = _proportional_sample(questions, 8, rng)
    assert len(sampled) == 8


def test_proportional_sample_small_pool():
    """When pool is smaller than budget, returns all items."""
    rng = random.Random(42)
    questions = _make_questions(5, "user_attribution")
    sampled = _proportional_sample(questions, 8, rng)
    assert len(sampled) == 5


def test_proportional_sample_empty_pool():
    rng = random.Random(42)
    sampled = _proportional_sample([], 8, rng)
    assert sampled == []


def test_proportional_sample_respects_difficulties():
    """Sampled questions come from multiple difficulty strata."""
    rng = random.Random(42)
    # 10 easy, 10 medium, 10 hard → budget 12 should include all 3 strata
    questions = (
        _make_questions(10, "user_attribution", ["easy"]) +
        _make_questions(10, "user_attribution", ["medium"]) +
        _make_questions(10, "user_attribution", ["hard"])
    )
    sampled = _proportional_sample(questions, 12, rng)
    diffs = {q["difficulty"] for q in sampled}
    assert len(diffs) >= 2  # at least 2 difficulty levels


# ── select_core_subset ────────────────────────────────────────────────────────


def test_select_core_subset_correct_category():
    """Only questions from the matching category are selected."""
    rng = random.Random(42)
    questions = (
        _make_questions(20, "user_attribution") +
        _make_questions(20, "conflict_resolution")
    )
    budget = {"category": "user_attribution", "per_scenario": 8}
    sampled = select_core_subset(1, questions, budget, rng)
    assert all(q["category"] == "user_attribution" for q in sampled)
    assert len(sampled) == 8


def test_select_core_subset_missing_category():
    """Returns empty list when category not present in questions."""
    rng = random.Random(42)
    questions = _make_questions(10, "user_attribution")
    budget = {"category": "authority_hierarchy", "per_scenario": 8}
    sampled = select_core_subset(1, questions, budget, rng)
    assert sampled == []


# ── Budget totals ──────────────────────────────────────────────────────────────


def test_budget_total():
    """Total expected from CORE_BUDGET matches the plan's 448."""
    total = 0
    for cat, entry in CORE_BUDGET.items():
        scenarios = entry.get("scenarios", ALL_SCENARIOS)
        total += len(scenarios) * entry["per_scenario"]
    assert total == 448, f"Expected 448, got {total}"


# ── get_questions_for_scenario ────────────────────────────────────────────────


def test_get_questions_for_scenario():
    """Returns only questions whose IDs appear in the scenario manifest."""
    manifest = {
        "full_question_list": [
            {"question_id": "q1", "scenario": 1, "category": "user_attribution",
             "difficulty": "easy", "question": "Q?", "gold_answer": "A.", "evidence_links": []},
            {"question_id": "q2", "scenario": 2, "category": "user_attribution",
             "difficulty": "easy", "question": "Q?", "gold_answer": "A.", "evidence_links": []},
        ],
        "per_scenario": {
            "scenario_1": {"question_ids": ["q1"], "counts_by_category": {}, "counts_by_difficulty": {}, "total": 1},
            "scenario_2": {"question_ids": ["q2"], "counts_by_category": {}, "counts_by_difficulty": {}, "total": 1},
        },
    }
    result = get_questions_for_scenario(manifest, 1)
    assert len(result) == 1
    assert result[0]["question_id"] == "q1"
