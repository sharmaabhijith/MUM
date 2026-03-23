from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict

from src.llm.client import LLMClient
from src.models.schemas import (
    BenchmarkDataset,
    ConversationSession,
    EvalQuestion,
)
from eval.metrics import score_question

logger = logging.getLogger("mum")


class MemorySystemInterface(ABC):
    """Abstract interface for memory systems to implement."""

    @abstractmethod
    def process_conversations(
        self, conversations: list[ConversationSession], documents: dict[str, str]
    ) -> None:
        """Process all conversations and build internal memory representation."""
        ...

    @abstractmethod
    def answer_question(self, question: str) -> str:
        """Answer a question using the memory system's internal representation."""
        ...


class EvalRunner:
    def __init__(
        self,
        benchmark: BenchmarkDataset,
        llm_client: LLMClient,
    ):
        self.benchmark = benchmark
        self.llm_client = llm_client

    def evaluate_system(
        self,
        system: MemorySystemInterface,
        scenario_id: str | None = None,
    ) -> dict:
        """Evaluate a memory system against the benchmark."""
        scenarios = self.benchmark.scenarios
        if scenario_id:
            scenarios = [s for s in scenarios if s.scenario_id == scenario_id]

        all_results = {}
        for scenario_output in scenarios:
            sid = scenario_output.scenario_id
            logger.info(f"Evaluating scenario {sid}")

            # Feed conversations to the system
            doc_texts = {
                d.name: "" for d in scenario_output.config.documents
            }
            system.process_conversations(scenario_output.conversations, doc_texts)

            # Score each question
            results = self._score_questions(
                system, scenario_output.eval_questions
            )
            all_results[sid] = results

        return self._aggregate_results(all_results)

    def evaluate_baseline(
        self,
        baseline_name: str,
        scenario_id: str | None = None,
    ) -> dict:
        """Evaluate a baseline by name."""
        if baseline_name == "full_context":
            from eval.baselines.full_context import FullContextBaseline
            system = FullContextBaseline(llm_client=self.llm_client)
        elif baseline_name == "rag_summaries":
            from eval.baselines.rag_summaries import RAGSummariesBaseline
            system = RAGSummariesBaseline(llm_client=self.llm_client)
        else:
            raise ValueError(f"Unknown baseline: {baseline_name}")

        return self.evaluate_system(system, scenario_id)

    def _score_questions(
        self,
        system: MemorySystemInterface,
        questions: list[EvalQuestion],
    ) -> dict:
        """Score all questions for a scenario."""
        per_category: dict[str, list[dict]] = defaultdict(list)

        for q in questions:
            predicted = system.answer_question(q.question)
            result = score_question(q, predicted, self.llm_client)
            result["question_id"] = q.question_id
            result["category"] = q.category
            result["difficulty"] = q.difficulty
            per_category[q.category].append(result)

        # Compute per-category averages
        category_scores = {}
        for cat, results in per_category.items():
            scores = [r["score"] for r in results]
            category_scores[cat] = {
                "mean_score": sum(scores) / len(scores) if scores else 0,
                "count": len(scores),
                "scores": scores,
            }

        return {
            "per_category": category_scores,
            "per_question": {
                cat: results for cat, results in per_category.items()
            },
        }

    def _aggregate_results(self, all_results: dict) -> dict:
        """Aggregate results across scenarios."""
        overall_scores = defaultdict(list)

        for sid, results in all_results.items():
            for cat, data in results["per_category"].items():
                overall_scores[cat].extend(data.get("scores", []))

        overall_category = {}
        for cat, scores in overall_scores.items():
            overall_category[cat] = {
                "mean_score": sum(scores) / len(scores) if scores else 0,
                "count": len(scores),
            }

        all_scores = [s for scores in overall_scores.values() for s in scores]
        overall_mean = sum(all_scores) / len(all_scores) if all_scores else 0

        return {
            "overall_score": overall_mean,
            "per_category": overall_category,
            "per_scenario": all_results,
        }
