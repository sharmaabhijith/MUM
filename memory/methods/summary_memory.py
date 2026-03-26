"""Summary-based memory — use session summaries instead of raw conversations."""

from __future__ import annotations

import logging

import tiktoken

from datagen.models.schemas import ConversationSession, SessionSummary

from memory.methods.base import BaseMemoryMethod, MemoryContext

logger = logging.getLogger("mum.memory")


class SummaryMemory(BaseMemoryMethod):
    """Summary memory — uses pre-generated session summaries as the memory store."""

    name = "summary"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._summaries: list[SessionSummary] = []
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def ingest(
        self,
        conversations: list[ConversationSession],
        summaries: list[SessionSummary],
    ) -> None:
        self._summaries = sorted(
            summaries, key=lambda s: (s.user_id, s.session_number)
        )

    def retrieve(self, question: str, user_id: str | None = None) -> MemoryContext:
        summaries = self._summaries
        if user_id:
            summaries = [s for s in summaries if s.user_id == user_id]

        blocks = []
        for summ in summaries:
            block = (
                f"=== User: {summ.user_id} | Session {summ.session_number} ===\n"
                f"Summary: {summ.summary}\n"
                f"Key Facts:\n"
            )
            for fact in summ.key_facts:
                block += f"  - {fact}\n"
            if summ.positions_taken:
                block += "Positions:\n"
                for pos in summ.positions_taken:
                    block += f"  - {pos}\n"
            if summ.positions_changed:
                block += "Positions Changed:\n"
                for change in summ.positions_changed:
                    block += f"  - {change}\n"
            blocks.append(block)

        context_text = "\n".join(blocks)
        token_count = len(self._encoder.encode(context_text))

        return MemoryContext(
            method_name=self.name,
            context_text=context_text,
            token_count=token_count,
            metadata={
                "num_summaries": len(summaries),
                "scoped_to_user": user_id,
            },
        )

    def reset(self) -> None:
        self._summaries = []
