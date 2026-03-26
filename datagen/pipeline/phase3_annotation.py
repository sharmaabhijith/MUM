from __future__ import annotations

import logging

from datagen.llm.client import LLMClient
from datagen.llm.cost_tracker import CostTracker
from datagen.models.enums import EvalQuestionCategory
from datagen.models.schemas import (
    ConversationSession,
    EvalQuestion,
    EvidenceLink,
    SessionSummary,
)
from datagen.pipeline.phase1_document_prep import DocumentContext
from datagen.prompts.eval_question_gen import build_eval_question_prompt
from datagen.scenarios.base import BaseScenario

logger = logging.getLogger("mum")


class QuestionGenerator:
    """Generates evaluation questions directly from conversations."""

    def __init__(
        self,
        llm_client: LLMClient,
        cost_tracker: CostTracker | None = None,
    ):
        self.llm_client = llm_client
        self.cost_tracker = cost_tracker or llm_client.cost_tracker

    def generate_questions(
        self,
        scenario: BaseScenario,
        conversations: list[ConversationSession],
        summaries: list[SessionSummary],
        doc_context: DocumentContext,
    ) -> list[EvalQuestion]:
        config = scenario.config
        applicable_categories = scenario.get_applicable_eval_categories()
        eval_breakdown = config.annotation_targets.eval_breakdown

        # Build a users description for the prompt
        users_description = self._build_users_description(scenario)

        all_questions: list[EvalQuestion] = []

        for category in applicable_categories:
            cat_val = category.value if hasattr(category, "value") else category
            target_count = eval_breakdown.get(cat_val, 0)
            if target_count == 0:
                continue

            logger.info(f"  Generating {target_count} questions for {cat_val}")

            prompt = build_eval_question_prompt(
                scenario_id=config.scenario_id,
                category=(
                    category
                    if isinstance(category, EvalQuestionCategory)
                    else EvalQuestionCategory(category)
                ),
                target_count=target_count,
                conversations=conversations,
                summaries=summaries,
                doc_context=doc_context,
                relationship_type=config.relationship_type,
                authority_context=scenario.get_authority_context(),
                users_description=users_description,
            )
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.generate_json(
                messages=messages,
                temperature=0.2,
                max_tokens=8192,
                phase="eval_question_generation",
            )

            for q_data in response.get("questions", []):
                # Parse evidence links
                evidence = []
                for e in q_data.get("evidence", []):
                    evidence.append(
                        EvidenceLink(
                            user_id=e.get("user_id", ""),
                            session_id=e.get("session_id", ""),
                        )
                    )
                q_data["evidence"] = evidence
                q_data.setdefault("scenario_id", config.scenario_id)
                # Remove any extra fields the LLM might generate
                q_data.pop("required_memories", None)
                q_data.pop("required_conflicts", None)

                try:
                    question = EvalQuestion.model_validate(q_data)
                    all_questions.append(question)
                except Exception as e:
                    logger.warning(f"  Failed to parse eval question: {e}")

        logger.info(
            f"  Total eval questions: {len(all_questions)} "
            f"(target: {config.annotation_targets.eval_questions})"
        )
        return all_questions

    def _build_users_description(self, scenario: BaseScenario) -> str:
        lines = []
        for user in scenario.config.users:
            parts = [
                f"- {user.display_name} ({user.user_id}): "
                f"authority={user.authority_level}, "
                f"expertise={user.expertise}"
            ]
            if user.domain_authority:
                parts.append(f", domain_authority={user.domain_authority}")
            if user.side:
                parts.append(f", side={user.side}")
            lines.append("".join(parts))
        return "\n".join(lines)
