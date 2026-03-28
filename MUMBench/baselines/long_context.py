"""Long-context baseline — concatenate ALL conversation sessions into one prompt.

Prepends relationship type, authority context, and user roster.
Truncates to model context window if needed and logs truncation %.
This is the most expensive baseline (~80K+ tokens per call).
"""

from __future__ import annotations

import logging

import tiktoken

from datagen.llm.client import LLMClient
from datagen.llm.cost_tracker import CostTracker
from datagen.models.schemas import ConversationSession, SessionSummary

from MUMBench.baselines.base import MUMMBaseline
from MUMBench.config import MUMM_CONFIG

logger = logging.getLogger("mum.mummbench")

_ENCODER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """Truncate text to max_tokens. Returns (truncated_text, was_truncated)."""
    tokens = _ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return text, False
    truncated = _ENCODER.decode(tokens[:max_tokens])
    return truncated, True


class LongContextBaseline(MUMMBaseline):
    """Full-context: concatenate all conversations chronologically."""

    @property
    def name(self) -> str:
        return "long_context"

    def __init__(
        self,
        model: str,
        max_context_tokens: int | None = None,
        cost_tracker: CostTracker | None = None,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.max_context_tokens = max_context_tokens or MUMM_CONFIG["long_context_max_tokens"]
        self.cost_tracker = cost_tracker or CostTracker()
        self._full_context: str = ""
        self._context_tokens: int = 0
        self._truncated: bool = False
        self._truncation_pct: float = 0.0
        self._relationship_type: str = ""
        self._llm = LLMClient(
            model=model,
            temperature=self.config.get("temperature", 0.3),
            cost_tracker=self.cost_tracker,
        )

    def ingest(
        self,
        scenario_id: int,
        conversations: list[ConversationSession],
        summaries: list[SessionSummary],
        relationship_type: str = "",
        authority_context: dict | None = None,
    ) -> None:
        """Concatenate all conversations in chronological order."""
        self._relationship_type = relationship_type

        # Sort conversations: by session_timestamp then session_number
        sorted_convs = sorted(
            conversations,
            key=lambda c: (c.session_timestamp or "", c.session_number or 0),
        )

        # Build header
        header_parts = [f"=== SCENARIO {scenario_id} CONVERSATION HISTORY ==="]
        if relationship_type:
            header_parts.append(f"Relationship type: {relationship_type}")
        if authority_context:
            auth_str = "\n".join(f"  {k}: {v}" for k, v in authority_context.items())
            header_parts.append(f"Authority context:\n{auth_str}")

        user_ids = sorted({c.user_id for c in conversations})
        header_parts.append(f"Users: {', '.join(user_ids)}")
        header = "\n".join(header_parts)

        # Build conversation blocks
        blocks = [header]
        for conv in sorted_convs:
            block_header = (
                f"\n[User: {conv.user_id} | Session: {conv.session_number} "
                f"| Date: {conv.session_timestamp}]"
            )
            turns_text = "\n".join(
                f"  [{t.role.upper()}]: {t.content}" for t in conv.turns
            )
            blocks.append(f"{block_header}\n{turns_text}\n---")

        full_text = "\n".join(blocks)
        total_tokens = _count_tokens(full_text)

        # Reserve space for question prompt (~500 tokens)
        available = self.max_context_tokens - 500
        if total_tokens > available:
            self._full_context, self._truncated = _truncate_to_tokens(full_text, available)
            self._truncation_pct = (1.0 - available / total_tokens) * 100
            logger.warning(
                f"[long_context] Scenario {scenario_id}: truncated context from "
                f"{total_tokens:,} to {available:,} tokens "
                f"({self._truncation_pct:.1f}% lost)"
            )
        else:
            self._full_context = full_text
            self._truncated = False
            self._truncation_pct = 0.0

        self._context_tokens = _count_tokens(self._full_context)
        logger.info(
            f"[long_context] Scenario {scenario_id}: "
            f"{len(conversations)} conversations, "
            f"{self._context_tokens:,} tokens "
            f"({'truncated' if self._truncated else 'full'})"
        )

    def answer(self, question: str) -> tuple[str, dict]:
        system = (
            "You are answering questions about a multi-user AI memory system. "
            "You have access to the COMPLETE conversation history below. "
            "Answer based ONLY on the conversations provided."
        )
        user_msg = (
            f"=== FULL CONVERSATION HISTORY ===\n{self._full_context}\n\n"
            f"=== QUESTION ===\n{question}\n\n"
            "Answer based only on the conversation history above. "
            "Be specific and cite the relevant user and session when possible."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

        answer_text, _ = self._llm.generate(
            messages=messages,
            max_tokens=self.config.get("max_tokens", 1024),
            phase="mumm_answer",
        )
        return answer_text, {
            "method": "long_context",
            "context_tokens": self._context_tokens,
            "truncated": self._truncated,
            "truncation_pct": round(self._truncation_pct, 1),
        }

    def reset(self) -> None:
        self._full_context = ""
        self._context_tokens = 0
        self._truncated = False
        self._truncation_pct = 0.0
        self._relationship_type = ""
