from __future__ import annotations

import logging
from typing import Callable

from rouge_score import rouge_scorer

from src.llm.client import LLMClient
from src.models.enums import EvalQuestionCategory
from src.models.schemas import EvalQuestion

logger = logging.getLogger("mum")


def exact_match(predicted: str, gold: str) -> float:
    """Binary exact match (case-insensitive, stripped)."""
    return 1.0 if predicted.strip().lower() == gold.strip().lower() else 0.0


def exact_match_contains(predicted: str, gold: str) -> float:
    """Check if gold answer appears in predicted response."""
    return 1.0 if gold.strip().lower() in predicted.strip().lower() else 0.0


def compute_rouge(predicted: str, gold: str) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(gold, predicted)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def compute_f1(
    predictions: list, gold: list, matcher: Callable | None = None
) -> dict[str, float]:
    """Compute precision, recall, F1 between predicted and gold sets."""
    if matcher is None:
        matcher = lambda p, g: p.strip().lower() == g.strip().lower()

    matched = 0
    for g in gold:
        if any(matcher(p, g) for p in predictions):
            matched += 1

    precision = matched / len(predictions) if predictions else 0
    recall = matched / len(gold) if gold else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}


def llm_judge_score(
    predicted: str,
    gold: str,
    question: EvalQuestion,
    criteria: str,
    llm_client: LLMClient,
) -> dict:
    """Use LLM-as-judge to score a response."""
    prompt = f"""You are an expert evaluator. Score the predicted answer against the gold answer.

## Question
{question.question}

## Gold Answer
{gold}

## Predicted Answer
{predicted}

## Scoring Criteria
{criteria}

## Output Format
Return JSON: {{"score": 0.0-1.0, "reasoning": "explanation"}}

Score 1.0 = perfect match in meaning and completeness.
Score 0.0 = completely wrong or missing.
Partial scores for partially correct answers."""

    messages = [{"role": "user", "content": prompt}]
    response = llm_client.generate_json(
        messages=messages,
        temperature=0.0,
        max_tokens=512,
        phase="eval_scoring",
    )
    return {
        "score": response.get("score", 0.0),
        "reasoning": response.get("reasoning", ""),
    }


def factscore(
    predicted: str,
    gold: str,
    llm_client: LLMClient,
) -> dict:
    """Decompose into atomic facts and check precision/recall."""
    prompt = f"""You are an expert evaluator computing FactScore.

Step 1: Decompose the GOLD answer into atomic facts.
Step 2: Decompose the PREDICTED answer into atomic facts.
Step 3: For each gold fact, check if it's present in predicted (recall).
Step 4: For each predicted fact, check if it's correct per gold (precision).

## Gold Answer
{gold}

## Predicted Answer
{predicted}

## Output Format
Return JSON:
{{
  "gold_facts": ["fact1", "fact2", ...],
  "predicted_facts": ["fact1", "fact2", ...],
  "precision": 0.0-1.0,
  "recall": 0.0-1.0,
  "f1": 0.0-1.0
}}"""

    messages = [{"role": "user", "content": prompt}]
    response = llm_client.generate_json(
        messages=messages,
        temperature=0.0,
        max_tokens=2048,
        phase="eval_factscore",
    )
    return response


# Map categories to their primary metric
CATEGORY_METRICS: dict[str, str] = {
    EvalQuestionCategory.USER_ATTRIBUTION: "exact_match",
    EvalQuestionCategory.CROSS_USER_SYNTHESIS: "factscore",
    EvalQuestionCategory.CONFLICT_RESOLUTION: "llm_judge",
    EvalQuestionCategory.INFORMATION_GAP: "llm_judge",
    EvalQuestionCategory.ROLE_APPROPRIATE_BRIEFING: "llm_judge",
    EvalQuestionCategory.ADVERSARIAL_CONFUSION: "exact_match",
    EvalQuestionCategory.DOCUMENT_COVERAGE: "f1",
    EvalQuestionCategory.CROSS_USER_PROVENANCE: "llm_judge",
    EvalQuestionCategory.AUTHORITY_HIERARCHY: "llm_judge",
    EvalQuestionCategory.TEMPORAL_CORRECTION: "llm_judge",
    EvalQuestionCategory.ADVERSARIAL_SIDE_ISOLATION: "llm_judge",
    EvalQuestionCategory.SEQUENTIAL_HANDOFF: "llm_judge",
}


CATEGORY_CRITERIA: dict[str, str] = {
    EvalQuestionCategory.USER_ATTRIBUTION: "Is the correct user identified?",
    EvalQuestionCategory.CROSS_USER_SYNTHESIS: "Does the answer include all relevant facts from all users?",
    EvalQuestionCategory.CONFLICT_RESOLUTION: "Are both sides fairly represented? Is the resolution correct for the relationship type?",
    EvalQuestionCategory.INFORMATION_GAP: "Is the identified gap genuine and actionable?",
    EvalQuestionCategory.ROLE_APPROPRIATE_BRIEFING: "Is the briefing relevant, role-appropriate, and free of information leakage?",
    EvalQuestionCategory.ADVERSARIAL_CONFUSION: "Does the answer correctly reject the misattribution and provide the right correction?",
    EvalQuestionCategory.DOCUMENT_COVERAGE: "Are the (user, section) pairs accurate? Are blind spots identified?",
    EvalQuestionCategory.CROSS_USER_PROVENANCE: "Is the provenance chain complete and temporally accurate?",
    EvalQuestionCategory.AUTHORITY_HIERARCHY: "Is the correct authority identified and applied appropriately?",
    EvalQuestionCategory.TEMPORAL_CORRECTION: "Is the correction chain complete, temporally accurate, and is the current state correct?",
    EvalQuestionCategory.ADVERSARIAL_SIDE_ISOLATION: "Is there no information leakage? Are positions preserved per side?",
    EvalQuestionCategory.SEQUENTIAL_HANDOFF: "Is the handoff complete? Is customer communication accurate? Is temporal ordering correct?",
}


def score_question(
    question: EvalQuestion,
    predicted: str,
    llm_client: LLMClient,
) -> dict:
    """Score a single question using the appropriate metric for its category."""
    category = question.category
    metric_type = CATEGORY_METRICS.get(category, "llm_judge")
    gold = question.gold_answer

    if metric_type == "exact_match":
        score = exact_match_contains(predicted, gold)
        return {"score": score, "metric": "exact_match"}
    elif metric_type == "factscore":
        result = factscore(predicted, gold, llm_client)
        return {"score": result.get("f1", 0.0), "metric": "factscore", "details": result}
    elif metric_type == "f1":
        rouge = compute_rouge(predicted, gold)
        return {"score": rouge["rougeL"], "metric": "rouge", "details": rouge}
    else:  # llm_judge
        criteria = CATEGORY_CRITERIA.get(category, "Score the quality of the answer.")
        result = llm_judge_score(predicted, gold, question, criteria, llm_client)
        return {"score": result["score"], "metric": "llm_judge", "details": result}
