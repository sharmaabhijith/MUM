"""Tests for MUMBench.scorer."""

from __future__ import annotations

import pytest

from MUMBench.scorer import (
    _parse_set_items,
    score_binary_accuracy,
    score_exact_match,
    score_factscore,
    score_question,
    score_set_prf1,
)


# ── score_exact_match ─────────────────────────────────────────────────────────


class TestExactMatch:
    def test_correct_student_attribution(self):
        assert score_exact_match(
            "Student A repeatedly confused these events.",
            "Student A repeatedly confused the two historical events.",
        ) == 1.0

    def test_wrong_student_is_partial(self):
        # Wrong student → partial (0.5) since entity detected but misattributed
        r = score_exact_match(
            "Student B did it.",
            "Student A repeatedly confused these events.",
        )
        assert r == 0.5

    def test_no_entity_match(self):
        assert score_exact_match(
            "I cannot determine the answer.",
            "Student A repeatedly confused these events.",
        ) == 0.0

    def test_empty_strings(self):
        assert score_exact_match("", "Student A did it.") == 0.0
        assert score_exact_match("Student A did it.", "") == 0.0

    def test_analyst_entity(self):
        assert score_exact_match(
            "Analyst 2 was responsible for the report.",
            "Analyst 2 filed the final report.",
        ) == 1.0


# ── score_binary_accuracy ─────────────────────────────────────────────────────


class TestBinaryAccuracy:
    def test_correct_rejection(self):
        r = score_binary_accuracy(
            "No, Student A said that. Student B actually claimed the opposite.",
            "No — Student A made this claim. Student B did not.",
        )
        assert r > 0

    def test_incorrect_agreement(self):
        r = score_binary_accuracy(
            "Yes, Student B did say that.",
            "No — Student B did not. Student A made this claim.",
        )
        assert r == 0.0

    def test_empty_predicted(self):
        assert score_binary_accuracy("", "No — Student A made this claim.") == 0.0

    def test_case_insensitive_rejection(self):
        r = score_binary_accuracy(
            "Incorrect — Student B did not make that claim.",
            "No — Student B did not. Student A did.",
        )
        assert r > 0


# ── score_set_prf1 ────────────────────────────────────────────────────────────


class TestSetPRF1:
    def test_perfect_match(self):
        gold = "Item 1: User A claims X\nItem 2: User B claims Y"
        pred = "Item 1: User A claims X\nItem 2: User B claims Y"
        result = score_set_prf1(pred, gold)
        assert result["f1"] > 0.8

    def test_partial_coverage(self):
        # Items must be semantically distinct to test partial recall.
        # Use domain-distinct items (different topics, minimal shared tokens).
        gold = (
            "GDP discrepancy between student_b and student_c\n"
            "Parliament structure confusion involving student_a\n"
            "Colonial history misattribution by student_d"
        )
        pred = "GDP discrepancy between student_b and student_c"
        result = score_set_prf1(pred, gold)
        # recall should be < 1.0: only 1 of 3 gold topics covered
        assert result["recall"] < 1.0
        assert result["f1"] < 1.0

    def test_empty_inputs(self):
        result = score_set_prf1("", "Something")
        assert result["f1"] == 0.0

    def test_f1_between_precision_recall(self):
        gold = "A; B; C"
        pred = "A; B; D; E"
        result = score_set_prf1(pred, gold)
        assert 0.0 <= result["f1"] <= 1.0
        assert 0.0 <= result["precision"] <= 1.0
        assert 0.0 <= result["recall"] <= 1.0


# ── score_factscore ────────────────────────────────────────────────────────────


class TestFactScore:
    def test_heuristic_returns_float(self):
        r = score_factscore(
            "Student A studied Ficlandia history.",
            "Student A studied Ficlandia history and culture extensively.",
        )
        assert isinstance(r, float)
        assert 0.0 <= r <= 1.0

    def test_empty_inputs(self):
        assert score_factscore("", "gold") == 0.0
        assert score_factscore("pred", "") == 0.0


# ── score_question routing ─────────────────────────────────────────────────────


class TestScoreQuestion:
    def _make_q(self, category: str, predicted: str = "Some answer.", gold: str = "Some gold.") -> tuple[dict, str]:
        return {
            "question_id": "q_test",
            "category": category,
            "difficulty": "medium",
            "question": "Who did X?",
            "gold_answer": gold,
        }, predicted

    def test_exact_match_routing(self):
        q, pred = self._make_q(
            "user_attribution",
            "Student A did it.",
            "Student A repeatedly did this.",
        )
        score = score_question(q, pred)
        assert score.metric == "exact_match"
        assert 0.0 <= score.score <= 1.0

    def test_binary_routing(self):
        q, pred = self._make_q(
            "adversarial_confusion",
            "No, Student A did it.",
            "No — Student A made this claim.",
        )
        score = score_question(q, pred)
        assert score.metric == "binary_accuracy"
        assert 0.0 <= score.score <= 1.0

    def test_set_prf1_routing(self):
        q, pred = self._make_q(
            "conflict_resolution",
            "Conflict 1: A vs B\nConflict 2: C vs D",
            "Conflict 1: A vs B\nConflict 2: C vs D",
        )
        score = score_question(q, pred)
        assert score.metric == "set_prf1"
        assert score.f1 is not None
        assert 0.0 <= score.score <= 1.0

    def test_factscore_routing(self):
        q, pred = self._make_q(
            "cross_user_synthesis",
            "Student A found X and Student B found Y.",
            "Student A found X. Student B also found Y.",
        )
        score = score_question(q, pred)
        assert score.metric == "factscore"
        assert 0.0 <= score.score <= 1.0

    def test_llm_judge_routing_without_judge(self):
        """Without a judge, falls back to token overlap heuristic."""
        q, pred = self._make_q(
            "information_gap",
            "User A knows about exports while User B does not.",
            "User A has information about exports that User B lacks.",
        )
        score = score_question(q, pred, judge=None)
        assert score.metric == "llm_judge"
        assert 0.0 <= score.score <= 1.0

    def test_score_to_dict(self):
        q, pred = self._make_q("user_attribution", "Student A did it.", "Student A did it.")
        score = score_question(q, pred)
        d = score.to_dict()
        assert "question_id" in d
        assert "score" in d
        assert "metric" in d
