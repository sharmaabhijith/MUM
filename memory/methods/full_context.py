"""Full context baseline — dump all conversations into the prompt."""

from __future__ import annotations

import logging

import tiktoken

from datagen.models.schemas import ConversationSession, SessionSummary

from memory.methods.base import BaseMemoryMethod, MemoryContext

logger = logging.getLogger("mum.memory")


class FullContext(BaseMemoryMethod):
    """Full context window — all conversations concatenated into the prompt (oracle upper-bound)."""

    name = "full_context"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._conversations: list[ConversationSession] = []
        self._summaries: list[SessionSummary] = []
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def ingest(
        self,
        conversations: list[ConversationSession],
        summaries: list[SessionSummary],
    ) -> None:
        self._conversations = sorted(
            conversations, key=lambda c: (c.user_id, c.session_number)
        )
        self._summaries = summaries

    def retrieve(self, question: str, user_id: str | None = None) -> MemoryContext:
        blocks = []
        for conv in self._conversations:
            header = (
                f"=== User: {conv.user_id} | Session {conv.session_number} "
                f"({conv.session_timestamp}) ==="
            )
            turns_text = "\n".join(
                f"[{t.role}]: {t.content}" for t in conv.turns
            )
            blocks.append(f"{header}\n{turns_text}")

        context_text = "\n\n".join(blocks)
        token_count = len(self._encoder.encode(context_text))

        return MemoryContext(
            method_name=self.name,
            context_text=context_text,
            token_count=token_count,
            metadata={
                "num_conversations": len(self._conversations),
                "num_turns": sum(len(c.turns) for c in self._conversations),
            },
        )

    def reset(self) -> None:
        self._conversations = []
        self._summaries = []
