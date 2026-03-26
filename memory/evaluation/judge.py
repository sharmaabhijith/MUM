"""LLM-as-judge scoring for evaluation answers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from datagen.llm.client import LLMClient
from datagen.llm.cost_tracker import CostTracker

from memory.prompts.judge import build_binary_judge_prompt, build_judge_prompt

logger = logging.getLogger("mum.memory")

# Categories where binary (correct/incorrect) evaluation is appropriate
BINARY_CATEGORIES = {"adversarial_confusion"}


@dataclass
class JudgeScore:
    """Score from the LLM judge for a single question."""

    question_id: str
    correctness: float
    completeness: float
    attribution: float
    hallucination: float
    reasoning: str
    is_binary: bool = False
    binary_correct: bool | None = None
    _weights: dict[str, float] | None = None

    @property
    def overall(self) -> float:
        """Weighted average score (0-1 scale)."""
        if self.is_binary:
            return 1.0 if self.binary_correct else 0.0
        w = self._weights or {
            "correctness": 0.35,
            "completeness": 0.25,
            "attribution": 0.25,
            "hallucination": 0.15,
        }
        return (
            w["correctness"] * self.correctness
            + w["completeness"] * self.completeness
            + w["attribution"] * self.attribution
            + w["hallucination"] * self.hallucination
        ) / 5.0

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "correctness": self.correctness,
            "completeness": self.completeness,
            "attribution": self.attribution,
            "hallucination": self.hallucination,
            "overall": round(self.overall, 4),
            "reasoning": self.reasoning,
            "is_binary": self.is_binary,
            "binary_correct": self.binary_correct,
        }


DEFAULT_SCORING_WEIGHTS = {
    "correctness": 0.35,
    "completeness": 0.25,
    "attribution": 0.25,
    "hallucination": 0.15,
}


class LLMJudge:
    """Evaluates predicted answers against gold answers using an LLM."""

    def __init__(
        self,
        model: str = "google/gemini-2.5-pro",
        cost_tracker: CostTracker | None = None,
        temperature: float = 0.1,
        scoring_weights: dict[str, float] | None = None,
    ):
        self.cost_tracker = cost_tracker or CostTracker()
        self.scoring_weights = scoring_weights or DEFAULT_SCORING_WEIGHTS
        self.temperature = temperature
        self.llm = LLMClient(
            model=model,
            temperature=temperature,
            cost_tracker=self.cost_tracker,
        )

    def score(
        self,
        question_id: str,
        question: str,
        gold_answer: str,
        predicted_answer: str,
        category: str,
    ) -> JudgeScore:
        """Score a single predicted answer."""
        if category in BINARY_CATEGORIES:
            return self._score_binary(question_id, question, gold_answer, predicted_answer)
        return self._score_dimensional(
            question_id, question, gold_answer, predicted_answer, category
        )

    def _score_dimensional(
        self,
        question_id: str,
        question: str,
        gold_answer: str,
        predicted_answer: str,
        category: str,
    ) -> JudgeScore:
        messages = build_judge_prompt(question, gold_answer, predicted_answer, category)
        try:
            result = self.llm.generate_json(
                messages=messages,
                temperature=self.temperature,
                max_tokens=1024,
                phase="eval_judge",
            )
            return JudgeScore(
                question_id=question_id,
                correctness=float(result.get("correctness", 1)),
                completeness=float(result.get("completeness", 1)),
                attribution=float(result.get("attribution", 1)),
                hallucination=float(result.get("hallucination", 1)),
                reasoning=result.get("reasoning", ""),
                _weights=self.scoring_weights,
            )
        except Exception as e:
            logger.warning(f"Judge failed for {question_id}: {e}")
            return JudgeScore(
                question_id=question_id,
                correctness=1.0,
                completeness=1.0,
                attribution=1.0,
                hallucination=1.0,
                reasoning=f"Judge error: {e}",
            )

    def _score_binary(
        self,
        question_id: str,
        question: str,
        gold_answer: str,
        predicted_answer: str,
    ) -> JudgeScore:
        messages = build_binary_judge_prompt(question, gold_answer, predicted_answer)
        try:
            result = self.llm.generate_json(
                messages=messages,
                temperature=self.temperature,
                max_tokens=512,
                phase="eval_judge",
            )
            correct = bool(result.get("correct", False))
            score_val = 5.0 if correct else 1.0
            return JudgeScore(
                question_id=question_id,
                correctness=score_val,
                completeness=score_val,
                attribution=score_val,
                hallucination=score_val,
                reasoning=result.get("reasoning", ""),
                is_binary=True,
                binary_correct=correct,
            )
        except Exception as e:
            logger.warning(f"Binary judge failed for {question_id}: {e}")
            return JudgeScore(
                question_id=question_id,
                correctness=1.0,
                completeness=1.0,
                attribution=1.0,
                hallucination=1.0,
                reasoning=f"Judge error: {e}",
                is_binary=True,
                binary_correct=False,
            )
