"""Tests for MUMBench.judge — rubric loading and prompt building."""

from __future__ import annotations

import pytest

from MUMBench.judge import (
    RUBRIC_FILES,
    build_judge_messages,
    get_rubric,
)


# ── Rubric loading ─────────────────────────────────────────────────────────────


class TestRubricLoading:
    def test_all_rubric_files_exist(self):
        """Every category in RUBRIC_FILES has a loadable rubric."""
        for category, filename in RUBRIC_FILES.items():
            rubric = get_rubric(category)
            assert len(rubric) > 100, (
                f"Rubric for {category} ({filename}) is too short or missing"
            )

    def test_rubric_contains_scoring_formula(self):
        """Each rubric has a SCORING FORMULA section."""
        for category in RUBRIC_FILES:
            rubric = get_rubric(category)
            assert "SCORING FORMULA" in rubric, (
                f"Rubric for {category} missing SCORING FORMULA section"
            )

    def test_rubric_contains_dimensions(self):
        """Each rubric defines at least 2 scoring dimensions."""
        for category in RUBRIC_FILES:
            rubric = get_rubric(category)
            # Dimensions are defined with numbered lines like "1. dim_name (0-N)"
            dimension_count = rubric.count("points)")
            assert dimension_count >= 2, (
                f"Rubric for {category} has too few dimensions ({dimension_count})"
            )

    def test_unknown_category_returns_empty(self):
        rubric = get_rubric("nonexistent_category_xyz")
        assert rubric == ""

    def test_rubric_caching(self):
        """Second call returns same object (cached)."""
        r1 = get_rubric("information_gap")
        r2 = get_rubric("information_gap")
        assert r1 is r2


# ── build_judge_messages ───────────────────────────────────────────────────────


class TestBuildJudgeMessages:
    def test_returns_two_messages(self):
        msgs = build_judge_messages(
            category="information_gap",
            question="What does User A know that User B doesn't?",
            gold_answer="User A knows the export figures; User B does not.",
            predicted_answer="User A has more information about exports.",
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_system_contains_category(self):
        msgs = build_judge_messages(
            category="authority_hierarchy",
            question="Who has authority?",
            gold_answer="Commissioner.",
            predicted_answer="Commissioner.",
        )
        system = msgs[0]["content"]
        assert "authority hierarchy" in system.lower() or "T9" in system or "Authority" in system

    def test_user_message_contains_question(self):
        msgs = build_judge_messages(
            category="sequential_handoff",
            question="What happened at the handoff?",
            gold_answer="Agent 1 handed off to Agent 2 at session 3.",
            predicted_answer="There was a handoff.",
        )
        user_msg = msgs[1]["content"]
        assert "What happened at the handoff?" in user_msg
        assert "Agent 1 handed off" in user_msg
        assert "There was a handoff." in user_msg

    def test_authority_context_included(self):
        msgs = build_judge_messages(
            category="authority_hierarchy",
            question="Q?",
            gold_answer="A.",
            predicted_answer="A.",
            relationship_type="hierarchical",
            authority_context={"user_1": {"authority_level": "high", "authority_weight": 1.0}},
        )
        system = msgs[0]["content"]
        assert "hierarchical" in system

    def test_json_format_instruction(self):
        """System prompt must ask for JSON response."""
        msgs = build_judge_messages(
            category="temporal_correction",
            question="Q?",
            gold_answer="A.",
            predicted_answer="A.",
        )
        system = msgs[0]["content"]
        assert "JSON" in system
        assert "total_score" in system


# ── Rubric-specific content checks ────────────────────────────────────────────


class TestRubricContent:
    def test_t11_isolation_penalizes_leakage(self):
        rubric = get_rubric("adversarial_side_isolation")
        assert "leakage" in rubric.lower() or "leak" in rubric.lower()

    def test_t11_isolation_has_no_leakage_dimension(self):
        rubric = get_rubric("adversarial_side_isolation")
        assert "no_leakage" in rubric

    def test_t09_authority_has_correct_resolution_dimension(self):
        rubric = get_rubric("authority_hierarchy")
        assert "correct_resolution" in rubric

    def test_t10_temporal_has_current_state_dimension(self):
        rubric = get_rubric("temporal_correction")
        assert "current_state_correct" in rubric

    def test_t12_handoff_has_handoff_completeness(self):
        rubric = get_rubric("sequential_handoff")
        assert "handoff_completeness" in rubric
