"""Abstract base class for MUMM memory baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod

from datagen.models.schemas import ConversationSession, SessionSummary


class MUMMBaseline(ABC):
    """Abstract baseline that ingests scenario data and answers questions.

    Lifecycle:
        1. ingest() — called once per scenario to build internal memory representation
        2. answer() — called per question using only the built memory
        3. reset() — clear state between scenarios
    """

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.config = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        """Baseline identifier string, e.g. 'no_memory', 'rag', 'long_context'."""
        ...

    @abstractmethod
    def ingest(
        self,
        scenario_id: int,
        conversations: list[ConversationSession],
        summaries: list[SessionSummary],
        relationship_type: str = "",
        authority_context: dict | None = None,
    ) -> None:
        """Build memory from scenario data. Called once per scenario."""
        ...

    @abstractmethod
    def answer(self, question: str) -> tuple[str, dict]:
        """Answer a question using only the built memory.

        Returns:
            (answer_text, metadata) where metadata may include token counts,
            retrieval info, truncation stats, etc.
        """
        ...

    def reset(self) -> None:
        """Clear state between scenarios. Override if needed."""
        pass
