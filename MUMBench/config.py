"""MUMM-Core evaluation configuration constants."""

from __future__ import annotations

# ── Model configuration ────────────────────────────────────────────────────────

MUMM_CONFIG = {
    "answer_model": "deepseek-ai/DeepSeek-V3.2",
    "judge_model": "google/gemini-2.5-pro",
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "rag_top_k": 5,
    "long_context_max_tokens": 128_000,
    "judge_temperature": 0.0,
    "answer_temperature": 0.3,
    "seed": 42,
    "default_baselines": ["rag", "long_context"],
}

# ── Category → metric mapping ──────────────────────────────────────────────────

CATEGORY_METRICS: dict[str, str] = {
    # Exact match / binary
    "user_attribution": "exact_match",
    "adversarial_confusion": "binary_accuracy",
    # Set-based P/R/F1
    "conflict_resolution": "set_prf1",
    "document_coverage": "set_prf1",
    # FactScore
    "cross_user_synthesis": "factscore",
    # LLM-as-judge (7 categories)
    "information_gap": "llm_judge",
    "role_appropriate_briefing": "llm_judge",
    "cross_user_provenance": "llm_judge",
    "authority_hierarchy": "llm_judge",
    "temporal_correction": "llm_judge",
    "adversarial_side_isolation": "llm_judge",
    "sequential_handoff": "llm_judge",
}

# ── MUMM-Core sampling budget ──────────────────────────────────────────────────
# Maps category → per-scenario budget + which scenarios include it.
# If "scenarios" is absent, the category appears in all 5 scenarios.

CORE_BUDGET: dict[str, dict] = {
    # Universal categories (T1–T8): 8 questions per scenario × 5 = 40 each
    "user_attribution":          {"per_scenario": 8},
    "cross_user_synthesis":      {"per_scenario": 8},
    "conflict_resolution":       {"per_scenario": 8},
    "information_gap":           {"per_scenario": 8},
    "role_appropriate_briefing": {"per_scenario": 8},
    "adversarial_confusion":     {"per_scenario": 8},
    "document_coverage":         {"per_scenario": 8},
    "cross_user_provenance":     {"per_scenario": 8},
    # Cluster categories (T9–T10)
    "authority_hierarchy": {
        "scenarios": [2, 3, 4],
        "per_scenario": 12,
    },  # → 36 total
    "temporal_correction": {
        "scenarios": [1, 2, 5],
        "per_scenario": 12,
    },  # → 36 total
    # Scenario-specific categories (T11–T12)
    "adversarial_side_isolation": {
        "scenarios": [4],
        "per_scenario": 28,
    },  # → 28 total
    "sequential_handoff": {
        "scenarios": [5],
        "per_scenario": 28,
    },  # → 28 total
}

# Grand total: (8 categories × 5 scenarios × 8) + (2 cluster × 3 scenarios × 12) + (28+28)
# = 320 + 72 + 56 = 448

ALL_SCENARIOS = [1, 2, 3, 4, 5]

# ── Difficulty distribution targets (for proportional sampling) ────────────────
DIFFICULTY_ORDER = ["easy", "medium", "hard"]

# ── Label → T-number mapping (for reporting) ──────────────────────────────────
CATEGORY_LABELS: dict[str, str] = {
    "user_attribution":          "T1  Attribution",
    "cross_user_synthesis":      "T2  Synthesis",
    "conflict_resolution":       "T3  Conflict",
    "information_gap":           "T4  Info Gap",
    "role_appropriate_briefing": "T5  Briefing",
    "adversarial_confusion":     "T6  Confusion",
    "document_coverage":         "T7  Coverage",
    "cross_user_provenance":     "T8  Provenance",
    "authority_hierarchy":       "T9  Authority",
    "temporal_correction":       "T10 Temporal",
    "adversarial_side_isolation": "T11 Isolation",
    "sequential_handoff":        "T12 Handoff",
}

CATEGORY_ORDER = list(CATEGORY_LABELS.keys())
