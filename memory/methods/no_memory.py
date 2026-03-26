"""Baseline: LLM with no memory — only the question itself."""

from __future__ import annotations

from datagen.models.schemas import ConversationSession, SessionSummary

from memory.methods.base import BaseMemoryMethod, MemoryContext


class NoMemory(BaseMemoryMethod):
    """No memory baseline — the LLM sees only the question, with no conversation history."""

    name = "no_memory"

    def ingest(
        self,
        conversations: list[ConversationSession],
        summaries: list[SessionSummary],
    ) -> None:
        # Nothing to store
        pass

    def retrieve(self, question: str, user_id: str | None = None) -> MemoryContext:
        return MemoryContext(
            method_name=self.name,
            context_text="",
            token_count=0,
            metadata={"note": "No memory provided — answer from parametric knowledge only."},
        )

    def reset(self) -> None:
        pass
