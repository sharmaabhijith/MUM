"""Base class for all memory management methods."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from datagen.models.schemas import ConversationSession, SessionSummary

logger = logging.getLogger("mum.memory")


@dataclass
class MemoryContext:
    """Context provided to the LLM when answering an eval question."""

    method_name: str
    context_text: str
    token_count: int
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseMemoryMethod(ABC):
    """Abstract base class for memory management strategies.

    Each method implements how conversation history is stored and retrieved
    when answering evaluation questions. The lifecycle is:
      1. ingest() — process all conversations for a scenario
      2. retrieve() — given a question, return relevant context
      3. reset() — clear state for the next scenario
    """

    name: str = "base"

    def __init__(self, model: str = "google/gemini-2.5-pro", **kwargs):
        self.model = model
        self.config = kwargs

    @abstractmethod
    def ingest(
        self,
        conversations: list[ConversationSession],
        summaries: list[SessionSummary],
    ) -> None:
        """Ingest all conversations and summaries for a scenario."""
        ...

    @abstractmethod
    def retrieve(self, question: str, user_id: str | None = None) -> MemoryContext:
        """Retrieve relevant context for answering an eval question.

        Args:
            question: The evaluation question text.
            user_id: Optional user ID to scope retrieval (e.g., for role-appropriate briefing).

        Returns:
            MemoryContext with the assembled context text.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored state (between scenarios)."""
        ...

    def get_method_description(self) -> str:
        """Human-readable description for reports."""
        return f"{self.name}: {self.__class__.__doc__ or 'No description'}"
