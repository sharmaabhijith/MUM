from __future__ import annotations

import json
import logging

from src.llm.client import LLMClient
from src.llm.cost_tracker import CostTracker
from src.models.enums import EvalQuestionCategory
from src.models.schemas import (
    ConflictAnnotation,
    ConversationSession,
    EvalQuestion,
    EvidenceLink,
    ExtractedMemory,
    SessionSummary,
)
from src.pipeline.phase1_document_prep import DocumentContext
from src.prompts.conflict_detection import build_conflict_detection_prompt
from src.prompts.eval_question_gen import build_eval_question_prompt
from src.prompts.memory_extraction import build_memory_extraction_prompt
from src.scenarios.base import BaseScenario

logger = logging.getLogger("mum")


class AnnotationPipeline:
    def __init__(
        self,
        llm_client: LLMClient,
        cost_tracker: CostTracker | None = None,
    ):
        self.llm_client = llm_client
        self.cost_tracker = cost_tracker or llm_client.cost_tracker

    def annotate_scenario(
        self,
        scenario: BaseScenario,
        conversations: list[ConversationSession],
        summaries: list[SessionSummary],
        doc_context: DocumentContext,
    ) -> tuple[list[ExtractedMemory], list[ConflictAnnotation], list[EvalQuestion]]:
        config = scenario.config

        # Phase 3a: Memory extraction (per-user, per-session)
        logger.info(f"Extracting memories for scenario {config.scenario_id}")
        all_memories = self._extract_memories(scenario, conversations, doc_context)
        logger.info(f"Extracted {len(all_memories)} memories")

        # Phase 3b: Conflict detection (cross-user)
        logger.info(f"Detecting conflicts for scenario {config.scenario_id}")
        conflicts = self._detect_conflicts(scenario, all_memories)
        logger.info(f"Detected {len(conflicts)} conflicts")

        # Phase 3c: Eval question generation (per-category)
        logger.info(f"Generating eval questions for scenario {config.scenario_id}")
        eval_questions = self._generate_eval_questions(
            scenario, conversations, summaries, all_memories, conflicts, doc_context
        )
        logger.info(f"Generated {len(eval_questions)} eval questions")

        return all_memories, conflicts, eval_questions

    def _extract_memories(
        self,
        scenario: BaseScenario,
        conversations: list[ConversationSession],
        doc_context: DocumentContext,
    ) -> list[ExtractedMemory]:
        all_memories: list[ExtractedMemory] = []

        for user in scenario.config.users:
            user_sessions = sorted(
                [c for c in conversations if c.user_id == user.user_id],
                key=lambda c: c.session_number,
            )
            prior_memories: list[ExtractedMemory] = []

            for session in user_sessions:
                logger.info(
                    f"  Extracting memories: {user.display_name} session {session.session_number}"
                )

                # Add authority_level to session metadata for the prompt
                session.metadata["authority_level"] = user.authority_level

                prompt = build_memory_extraction_prompt(
                    session=session,
                    prior_memories=prior_memories,
                    doc_context=doc_context,
                )
                messages = [{"role": "user", "content": prompt}]
                response = self.llm_client.generate_json(
                    messages=messages,
                    temperature=0.2,
                    max_tokens=4096,
                    phase="memory_extraction",
                )

                memories_data = response.get("memories", [])
                session_memories = []
                for m_data in memories_data:
                    # Clean up fields
                    m_data.pop("supersession_notes", None)
                    # Ensure required fields
                    m_data.setdefault("scenario_id", session.scenario_id)
                    m_data.setdefault("user_id", session.user_id)
                    m_data.setdefault("session_id", session.session_id)
                    m_data.setdefault("timestamp", session.session_timestamp)
                    m_data.setdefault("authority_level", user.authority_level)
                    m_data.setdefault("status", "active")

                    try:
                        memory = ExtractedMemory.model_validate(m_data)
                        session_memories.append(memory)
                    except Exception as e:
                        logger.warning(f"  Failed to parse memory: {e}")

                # Handle supersession
                self._process_supersession(prior_memories, session_memories)

                prior_memories.extend(session_memories)
                all_memories.extend(session_memories)

        return all_memories

    def _process_supersession(
        self,
        prior_memories: list[ExtractedMemory],
        new_memories: list[ExtractedMemory],
    ) -> None:
        """Mark prior memories as superseded if new memories replace them."""
        for new_mem in new_memories:
            if new_mem.superseded_by:
                # This memory references what it supersedes — find it
                for prior_mem in prior_memories:
                    if prior_mem.memory_id == new_mem.superseded_by:
                        prior_mem.status = "superseded"
                        prior_mem.superseded_by = new_mem.memory_id
                        break

    def _detect_conflicts(
        self,
        scenario: BaseScenario,
        all_memories: list[ExtractedMemory],
    ) -> list[ConflictAnnotation]:
        config = scenario.config

        prompt = build_conflict_detection_prompt(
            all_memories=all_memories,
            injected_conflicts=config.injected_conflicts,
            relationship_type=config.relationship_type,
            authority_context=scenario.get_authority_context(),
            scenario_id=config.scenario_id,
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.generate_json(
            messages=messages,
            temperature=0.2,
            max_tokens=4096,
            phase="conflict_detection",
        )

        conflicts = []
        for c_data in response.get("conflicts", []):
            # Parse evidence links
            evidence = []
            for e in c_data.get("evidence", []):
                evidence.append(
                    EvidenceLink(
                        user_id=e.get("user_id", ""),
                        session_id=e.get("session_id", ""),
                    )
                )
            c_data["evidence"] = evidence
            c_data.setdefault("scenario_id", config.scenario_id)

            try:
                conflict = ConflictAnnotation.model_validate(c_data)
                conflicts.append(conflict)
            except Exception as e:
                logger.warning(f"  Failed to parse conflict: {e}")

        return conflicts

    def _generate_eval_questions(
        self,
        scenario: BaseScenario,
        conversations: list[ConversationSession],
        summaries: list[SessionSummary],
        memories: list[ExtractedMemory],
        conflicts: list[ConflictAnnotation],
        doc_context: DocumentContext,
    ) -> list[EvalQuestion]:
        config = scenario.config
        applicable_categories = scenario.get_applicable_eval_categories()
        eval_breakdown = config.annotation_targets.eval_breakdown

        # Build conversations summary for the prompt
        conversations_summary = self._build_conversations_summary(summaries)

        all_questions: list[EvalQuestion] = []

        for category in applicable_categories:
            target_count = eval_breakdown.get(category.value if hasattr(category, 'value') else category, 0)
            if target_count == 0:
                continue

            logger.info(f"  Generating {target_count} questions for {category.value}")

            prompt = build_eval_question_prompt(
                scenario_id=config.scenario_id,
                category=category if isinstance(category, EvalQuestionCategory) else EvalQuestionCategory(category),
                target_count=target_count,
                conversations_summary=conversations_summary,
                memories=memories,
                conflicts=conflicts,
                doc_context=doc_context,
                relationship_type=config.relationship_type,
                authority_context=scenario.get_authority_context(),
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

    def _build_conversations_summary(
        self, summaries: list[SessionSummary]
    ) -> str:
        lines = []
        for s in sorted(summaries, key=lambda x: (x.user_id, x.session_number)):
            lines.append(
                f"[{s.user_id} - Session {s.session_number}] {s.summary}"
            )
        return "\n\n".join(lines)
