from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from datagen.models.enums import EvalQuestionCategory
from datagen.models.schemas import InjectedConflict, ScenarioConfig, UserProfile
from datagen.pipeline.phase1_document_prep import DocumentContext

logger = logging.getLogger("mum")

# Universal categories apply to all scenarios
UNIVERSAL_CATEGORIES = [
    EvalQuestionCategory.USER_ATTRIBUTION,
    EvalQuestionCategory.CROSS_USER_SYNTHESIS,
    EvalQuestionCategory.CONFLICT_RESOLUTION,
    EvalQuestionCategory.INFORMATION_GAP,
    EvalQuestionCategory.ROLE_APPROPRIATE_BRIEFING,
    EvalQuestionCategory.ADVERSARIAL_CONFUSION,
    EvalQuestionCategory.DOCUMENT_COVERAGE,
    EvalQuestionCategory.CROSS_USER_PROVENANCE,
]


class BaseScenario(ABC):
    def __init__(self, config: ScenarioConfig, doc_context: DocumentContext | None = None):
        self.config = config
        self.doc_context = doc_context

    def get_user_by_id(self, user_id: str) -> UserProfile:
        for user in self.config.users:
            if user.user_id == user_id:
                return user
        raise ValueError(f"User {user_id} not found in scenario {self.config.scenario_id}")

    def get_conflicts_for_session(
        self, user_id: str, session_number: int
    ) -> list[InjectedConflict]:
        return [
            c
            for c in self.config.injected_conflicts
            if user_id in c.users and session_number in c.target_sessions
        ]

    def get_session_timestamp(self, user_id: str, session_number: int) -> str:
        schedule = self.config.timeline.session_schedule.get(user_id, [])
        for entry in schedule:
            if entry.get("session") == session_number:
                return entry.get("date", self.config.timeline.start_date)
        return self.config.timeline.start_date

    def get_applicable_eval_categories(self) -> list[EvalQuestionCategory]:
        categories = list(UNIVERSAL_CATEGORIES)
        categories.extend(self._get_extra_eval_categories())
        return categories

    @abstractmethod
    def _get_extra_eval_categories(self) -> list[EvalQuestionCategory]:
        """Return scenario-cluster and scenario-specific categories."""
        ...

    @abstractmethod
    def get_authority_context(self) -> str:
        """Return a description of the authority structure for prompts."""
        ...
