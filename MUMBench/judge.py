"""MUMM-Core LLM-as-Judge.

Category-specific rubric-based scoring for the 7 LLM-judge categories (T4, T5, T8–T12).
Each category loads a rubric from MUMBench/prompts/judge_rubrics/ and uses it to
produce a structured JSON score with per-dimension breakdown + chain-of-thought reasoning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from datagen.llm.client import LLMClient
from datagen.llm.cost_tracker import CostTracker

logger = logging.getLogger("mum.mummbench")

# Directory containing rubric .txt files
RUBRICS_DIR = Path(__file__).parent / "prompts" / "judge_rubrics"

# Map from category name to rubric filename
RUBRIC_FILES: dict[str, str] = {
    "information_gap":           "t04_information_gap.txt",
    "role_appropriate_briefing": "t05_briefing.txt",
    "cross_user_provenance":     "t08_provenance.txt",
    "authority_hierarchy":       "t09_authority.txt",
    "temporal_correction":       "t10_temporal.txt",
    "adversarial_side_isolation": "t11_isolation.txt",
    "sequential_handoff":        "t12_handoff.txt",
}

# ── Rubric loader ──────────────────────────────────────────────────────────────


def _load_rubric(category: str) -> str:
    """Load the rubric text for a given category."""
    filename = RUBRIC_FILES.get(category)
    if not filename:
        return ""
    path = RUBRICS_DIR / filename
    if not path.exists():
        logger.warning(f"Rubric file not found: {path}")
        return ""
    return path.read_text(encoding="utf-8")


# Cache loaded rubrics
_rubric_cache: dict[str, str] = {}


def get_rubric(category: str) -> str:
    if category not in _rubric_cache:
        _rubric_cache[category] = _load_rubric(category)
    return _rubric_cache[category]


# ── Judge prompt builder ───────────────────────────────────────────────────────

JUDGE_SYSTEM_TEMPLATE = """\
You are an expert evaluator for the MUMM benchmark (Multi-User Memory Management).

TASK: Score the predicted answer against the gold answer for a {category_name} question.

RELATIONSHIP TYPE: {relationship_type}
AUTHORITY CONTEXT: {authority_context}

RUBRIC:
{rubric_text}

Instructions:
- Score each dimension according to the rubric above.
- Apply any caps or penalties specified in the rubric (e.g. "if X=0, cap total at 0.3").
- Compute total_score as specified in the SCORING FORMULA.
- total_score must be in [0.0, 1.0].
- Be strict: 1.0 means the predicted answer is essentially equivalent to the gold answer.

Respond with ONLY valid JSON in this exact format:
{{
    "dimensions": {{"dim1": <int_or_float>, "dim2": <int_or_float>, ...}},
    "total_score": <float 0.0-1.0>,
    "reasoning": "<concise explanation of each dimension score and any caps applied>"
}}
"""

JUDGE_USER_TEMPLATE = """\
## QUESTION
{question}

## GOLD ANSWER (Reference)
{gold_answer}

## PREDICTED ANSWER
{predicted_answer}

Score the predicted answer using the rubric. Respond with JSON only.
"""


def build_judge_messages(
    category: str,
    question: str,
    gold_answer: str,
    predicted_answer: str,
    relationship_type: str = "",
    authority_context: dict | None = None,
) -> list[dict]:
    """Build the judge prompt messages for a given category."""
    rubric = get_rubric(category)
    if not rubric:
        rubric = f"Category: {category}\nScore how well the predicted answer matches the gold answer. Scale: 0.0 (completely wrong) to 1.0 (perfect)."

    auth_str = ""
    if authority_context:
        auth_str = "; ".join(f"{k}: {v}" for k, v in authority_context.items())
    if not auth_str:
        auth_str = "Not specified"

    system = JUDGE_SYSTEM_TEMPLATE.format(
        category_name=category.replace("_", " ").title(),
        relationship_type=relationship_type or "Not specified",
        authority_context=auth_str,
        rubric_text=rubric,
    )
    user_msg = JUDGE_USER_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]


# ── MUMM Judge ────────────────────────────────────────────────────────────────


@dataclass
class JudgeResult:
    """Result from the MUMM LLM judge."""

    question_id: str
    category: str
    total_score: float
    dimensions: dict[str, float]
    reasoning: str
    raw_response: dict

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "category": self.category,
            "total_score": round(self.total_score, 4),
            "dimensions": {k: round(float(v), 4) for k, v in self.dimensions.items()},
            "reasoning": self.reasoning,
        }


class MUMMJudge:
    """LLM-as-judge for MUMM category-specific rubric scoring.

    Loads category rubrics from MUMBench/prompts/judge_rubrics/ and calls
    the judge LLM with structured JSON response format.
    """

    def __init__(
        self,
        model: str = "google/gemini-2.5-pro",
        cost_tracker: CostTracker | None = None,
        temperature: float = 0.0,
        relationship_type: str = "",
        authority_context: dict | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.relationship_type = relationship_type
        self.authority_context = authority_context or {}
        self.cost_tracker = cost_tracker or CostTracker()
        self.llm = LLMClient(
            model=model,
            temperature=temperature,
            cost_tracker=self.cost_tracker,
        )

    def score(
        self,
        question_id: str,
        category: str,
        question: str,
        gold_answer: str,
        predicted_answer: str,
    ) -> dict:
        """Score a single predicted answer and return a dict with total_score + dimensions.

        Returns a dict compatible with QuestionScore.dimensions / .score fields.
        On failure, returns a safe fallback (score=0.0).
        """
        messages = build_judge_messages(
            category=category,
            question=question,
            gold_answer=gold_answer,
            predicted_answer=predicted_answer,
            relationship_type=self.relationship_type,
            authority_context=self.authority_context,
        )
        try:
            result = self.llm.generate_json(
                messages=messages,
                temperature=self.temperature,
                max_tokens=1024,
                phase="mumm_judge",
            )
            total_score = float(result.get("total_score", 0.0))
            total_score = max(0.0, min(1.0, total_score))
            dimensions = {k: float(v) for k, v in result.get("dimensions", {}).items()}
            reasoning = result.get("reasoning", "")
            logger.debug(
                f"Judge: {question_id} | {category} | score={total_score:.3f}"
            )
            return {
                "total_score": total_score,
                "dimensions": dimensions,
                "reasoning": reasoning,
                "raw": result,
            }
        except Exception as e:
            logger.warning(f"Judge failed for {question_id} ({category}): {e}")
            return {
                "total_score": 0.0,
                "dimensions": {},
                "reasoning": f"Judge error: {e}",
                "raw": {},
            }

    def score_batch(
        self,
        questions: list[dict],
        predicted_answers: dict[str, str],
    ) -> list[dict]:
        """Score a batch of questions. predicted_answers maps question_id → answer."""
        results = []
        for q in questions:
            qid = q["question_id"]
            predicted = predicted_answers.get(qid, "")
            result = self.score(
                question_id=qid,
                category=q["category"],
                question=q["question"],
                gold_answer=q["gold_answer"],
                predicted_answer=predicted,
            )
            result["question_id"] = qid
            results.append(result)
        return results
