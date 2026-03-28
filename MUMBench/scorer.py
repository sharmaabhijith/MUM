"""MUMM-Core scoring pipeline.

Three scoring modes:
  - exact_match: soft exact match — gold entity appears in predicted text
  - binary_accuracy: binary correct/incorrect (for adversarial_confusion)
  - set_prf1: set-based precision/recall/F1 (for conflict_resolution, document_coverage)
  - factscore: atomic fact verification (for cross_user_synthesis)
  - llm_judge: category-specific rubric scoring (7 categories)

`score_all()` routes each question to the correct metric.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from datagen.llm.client import LLMClient
from datagen.llm.cost_tracker import CostTracker

from MUMBench.config import CATEGORY_METRICS

if TYPE_CHECKING:
    from MUMBench.judge import MUMMJudge

logger = logging.getLogger("mum.mummbench")


# ── Score dataclass ────────────────────────────────────────────────────────────


@dataclass
class QuestionScore:
    """Score for a single question."""

    question_id: str
    category: str
    difficulty: str
    metric: str
    score: float  # 0.0–1.0 primary score
    # For set_prf1
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    # For llm_judge
    dimensions: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    # Metadata
    predicted_answer: str = ""
    gold_answer: str = ""

    def to_dict(self) -> dict:
        d: dict = {
            "question_id": self.question_id,
            "category": self.category,
            "difficulty": self.difficulty,
            "metric": self.metric,
            "score": round(self.score, 4),
            "predicted_answer": self.predicted_answer,
            "gold_answer": self.gold_answer,
        }
        if self.precision is not None:
            d["precision"] = round(self.precision, 4)
            d["recall"] = round(self.recall, 4)
            d["f1"] = round(self.f1, 4)
        if self.dimensions:
            d["dimensions"] = {k: round(v, 4) for k, v in self.dimensions.items()}
        if self.reasoning:
            d["reasoning"] = self.reasoning
        return d


# ── Exact match ────────────────────────────────────────────────────────────────


def score_exact_match(predicted: str, gold: str) -> float:
    """Soft exact match: check if key entity from gold appears in predicted.

    Extracts user/role identifiers (e.g. 'Student A', 'user_b', 'Commissioner')
    and checks presence in predicted text (case-insensitive).
    Returns 1.0 if all gold entities found, 0.0 otherwise.
    """
    if not predicted or not gold:
        return 0.0

    pred_lower = predicted.lower()
    gold_lower = gold.lower()

    # Pattern: extract named entities (capitalized words, user IDs, role names)
    entity_patterns = [
        r"\b(student [a-e])\b",
        r"\b(user[_\s][a-e])\b",
        r"\b(analyst [1-4])\b",
        r"\b(officer [a-z]+)\b",
        r"\b(commissioner [a-z]+)\b",
        r"\b(dr\.?\s+[a-z]+)\b",
        r"\b(agent [a-z0-9]+)\b",
        r"\b(responder [a-z0-9]+)\b",
        r"\b(technician [a-z0-9]+)\b",
    ]

    gold_entities: set[str] = set()
    for pattern in entity_patterns:
        for match in re.finditer(pattern, gold_lower):
            gold_entities.add(match.group(0).strip())

    # Also try to extract the first few words before common verbs as the subject
    subject_match = re.match(r"^([A-Z][a-z]+(?: [A-Z][a-z]+)*)", gold)
    if subject_match:
        gold_entities.add(subject_match.group(0).lower())

    if not gold_entities:
        # Fallback: check if first sentence of gold appears in predicted
        first_sentence = gold.split(".")[0].strip().lower()
        # Take key words (nouns/names, filter stopwords)
        stopwords = {"the", "a", "an", "is", "was", "has", "have", "had", "and", "or", "in", "of", "to"}
        key_words = [w for w in first_sentence.split() if w not in stopwords and len(w) > 3]
        if not key_words:
            return 0.0
        # Check if most key words appear in predicted
        hits = sum(1 for w in key_words if w in pred_lower)
        return 1.0 if hits >= max(1, len(key_words) * 0.6) else 0.0

    # Check all gold entities present in prediction
    hits = sum(1 for e in gold_entities if e in pred_lower)
    return 1.0 if hits == len(gold_entities) else (0.5 if hits > 0 else 0.0)


# ── Binary accuracy ────────────────────────────────────────────────────────────


def score_binary_accuracy(predicted: str, gold: str) -> float:
    """Binary: 1.0 if predicted correctly rejects false attribution, else 0.0.

    Used for adversarial_confusion — gold answer always starts with 'No' (the
    model should reject the false attribution). We check:
    1. Predicted starts with No/Incorrect/False or contains rejection phrase.
    2. Predicted mentions the correct user from gold.
    """
    if not predicted:
        return 0.0

    pred_lower = predicted.lower().strip()
    gold_lower = gold.lower().strip()

    # Check rejection phrase
    rejection_phrases = ["no,", "no.", "no —", "no -", "incorrect", "false", "that is not", "that's not",
                         "actually", "rather", "in fact"]
    has_rejection = any(pred_lower.startswith(p) or pred_lower.startswith(p.lstrip(",. "))
                        for p in rejection_phrases)
    # Also accept if "no" appears in first 20 chars
    has_rejection = has_rejection or "no" in pred_lower[:20]

    if not has_rejection:
        return 0.0

    # Check if correct user entity from gold appears in predicted
    correct_score = score_exact_match(predicted, gold)
    return 1.0 if correct_score > 0 else 0.5  # partial credit for correct rejection


# ── Set-based P/R/F1 ──────────────────────────────────────────────────────────


def _parse_set_items(text: str) -> set[str]:
    """Parse a text into a set of normalized items for set-based scoring.

    Tries comma/semicolon/newline splitting, then normalizes each item.
    """
    # Try to split on common delimiters
    for delim in ["\n-", "\n•", "\n*", "\n", ";", ","]:
        parts = text.split(delim)
        if len(parts) > 1:
            items = {p.strip().lower() for p in parts if p.strip()}
            if len(items) > 1:
                return items

    # Fallback: split on sentence boundaries
    parts = re.split(r"(?<=[.!?])\s+", text)
    return {p.strip().lower() for p in parts if len(p.strip()) > 10}


def score_set_prf1(predicted: str, gold: str) -> dict[str, float]:
    """Set-based precision, recall, F1 for structured list answers.

    Used for conflict_resolution and document_coverage.
    Each item is a normalized string; matching uses substring containment
    (since paraphrasing is common).
    """
    if not predicted or not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    gold_items = _parse_set_items(gold)
    pred_items = _parse_set_items(predicted)

    if not gold_items:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not pred_items:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Soft matching: a predicted item "matches" a gold item if there's
    # substantial token overlap (Jaccard >= 0.3)
    def _tokens(s: str) -> set[str]:
        return set(re.findall(r"\b\w{3,}\b", s.lower()))

    def _jaccard(a: str, b: str) -> float:
        ta, tb = _tokens(a), _tokens(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    threshold = 0.25

    # True positives: gold items that have a matching pred item
    tp_gold = sum(
        1 for g in gold_items if any(_jaccard(g, p) >= threshold for p in pred_items)
    )
    # False positives: pred items with no matching gold item
    tp_pred = sum(
        1 for p in pred_items if any(_jaccard(p, g) >= threshold for g in gold_items)
    )

    precision = tp_pred / len(pred_items) if pred_items else 0.0
    recall = tp_gold / len(gold_items) if gold_items else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ── FactScore ─────────────────────────────────────────────────────────────────


def score_factscore(
    predicted: str,
    gold: str,
    llm: LLMClient | None = None,
    cost_tracker: CostTracker | None = None,
    model: str = "deepseek-ai/DeepSeek-V3.2",
) -> float:
    """Simplified FactScore for cross_user_synthesis.

    Steps:
      1. Decompose predicted into atomic facts (via LLM or heuristic).
      2. For each fact, check if it's supported by the gold answer.
      3. FactScore = supported_facts / total_facts.

    If no LLM is provided, falls back to token overlap heuristic.
    """
    if not predicted or not gold:
        return 0.0

    if llm is None:
        # Heuristic fallback: check what fraction of 5-word n-grams in predicted
        # appear in gold (rough proxy for fact support).
        def _ngrams(text: str, n: int = 5) -> set[tuple]:
            words = text.lower().split()
            return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}

        pred_grams = _ngrams(predicted, 5)
        gold_grams = _ngrams(gold, 5)
        if not pred_grams:
            return 0.0
        overlap = len(pred_grams & gold_grams) / len(pred_grams)
        # Scale: 0 overlap → 0, 100% overlap → 1. Typical expected: 0.1–0.3
        return min(1.0, overlap * 3)  # generous scale for partial overlap

    # LLM-based FactScore
    system = (
        "You are a fact-checking assistant. Given a predicted answer and a gold answer, "
        "decompose the predicted answer into atomic facts and determine which are supported "
        "by the gold answer.\n\n"
        "Respond with JSON:\n"
        "{\n"
        '  "total_facts": <int>,\n'
        '  "supported_facts": <int>,\n'
        '  "factscore": <float 0-1>,\n'
        '  "facts": [{"fact": "...", "supported": true/false}]\n'
        "}"
    )
    user_msg = (
        f"Gold answer:\n{gold}\n\n"
        f"Predicted answer:\n{predicted}\n\n"
        "Decompose the predicted answer into atomic facts. "
        "For each fact, check if it is supported by the gold answer. "
        "Compute factscore = supported/total. Respond with JSON only."
    )
    try:
        result = llm.generate_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=1024,
            phase="mumm_factscore",
        )
        return float(result.get("factscore", 0.0))
    except Exception as e:
        logger.warning(f"FactScore LLM call failed: {e}. Falling back to heuristic.")
        return score_factscore(predicted, gold, llm=None)


# ── Routing: score_all ─────────────────────────────────────────────────────────


def score_question(
    question_dict: dict,
    predicted_answer: str,
    judge: "MUMMJudge | None" = None,
    factscore_llm: LLMClient | None = None,
) -> QuestionScore:
    """Route a question to the appropriate scorer and return a QuestionScore.

    Args:
        question_dict: Entry from mumm_core full_question_list.
        predicted_answer: The baseline's generated answer.
        judge: MUMMJudge instance for llm_judge categories.
        factscore_llm: LLMClient for FactScore (optional; falls back to heuristic).
    """
    question_id = question_dict["question_id"]
    category = question_dict["category"]
    difficulty = question_dict.get("difficulty", "medium")
    gold_answer = question_dict["gold_answer"]
    question_text = question_dict["question"]

    metric = CATEGORY_METRICS.get(category, "llm_judge")

    base_kwargs = dict(
        question_id=question_id,
        category=category,
        difficulty=difficulty,
        metric=metric,
        predicted_answer=predicted_answer,
        gold_answer=gold_answer,
    )

    if metric == "exact_match":
        score = score_exact_match(predicted_answer, gold_answer)
        return QuestionScore(score=score, **base_kwargs)

    elif metric == "binary_accuracy":
        score = score_binary_accuracy(predicted_answer, gold_answer)
        return QuestionScore(score=score, **base_kwargs)

    elif metric == "set_prf1":
        prf = score_set_prf1(predicted_answer, gold_answer)
        return QuestionScore(
            score=prf["f1"],
            precision=prf["precision"],
            recall=prf["recall"],
            f1=prf["f1"],
            **base_kwargs,
        )

    elif metric == "factscore":
        score = score_factscore(predicted_answer, gold_answer, llm=factscore_llm)
        return QuestionScore(score=score, **base_kwargs)

    elif metric == "llm_judge":
        if judge is None:
            logger.warning(
                f"No judge provided for llm_judge category {category}. "
                "Using heuristic fallback."
            )
            # Rough fallback: check token overlap
            def _tok_overlap(a: str, b: str) -> float:
                if not a or not b:
                    return 0.0
                ta = set(a.lower().split())
                tb = set(b.lower().split())
                return len(ta & tb) / max(len(ta | tb), 1)

            score = _tok_overlap(predicted_answer, gold_answer)
            return QuestionScore(score=score, **base_kwargs)

        judge_result = judge.score(
            question_id=question_id,
            category=category,
            question=question_text,
            gold_answer=gold_answer,
            predicted_answer=predicted_answer,
        )
        return QuestionScore(
            score=judge_result["total_score"],
            dimensions=judge_result.get("dimensions", {}),
            reasoning=judge_result.get("reasoning", ""),
            **base_kwargs,
        )

    else:
        logger.warning(f"Unknown metric {metric} for category {category}. Score=0.")
        return QuestionScore(score=0.0, **base_kwargs)
