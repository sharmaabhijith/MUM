"""No-memory baseline — answers questions with zero conversation history.

Establishes the parametric-knowledge floor. Any score here reflects
what the model knows about the fictional scenario without memory access
(should be ~0 since scenarios use invented countries/events).
"""

from __future__ import annotations

import logging

from datagen.llm.client import LLMClient
from datagen.llm.cost_tracker import CostTracker
from datagen.models.schemas import ConversationSession, SessionSummary

from MUMBench.baselines.base import MUMMBaseline

logger = logging.getLogger("mum.mummbench")


class NoMemoryBaseline(MUMMBaseline):
    """No memory: answer using only the question + relationship context."""

    @property
    def name(self) -> str:
        return "no_memory"

    def __init__(self, model: str, cost_tracker: CostTracker | None = None, **kwargs):
        super().__init__(model, **kwargs)
        self.cost_tracker = cost_tracker or CostTracker()
        self._relationship_type: str = ""
        self._user_list: list[str] = []
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
        """Store minimal context: relationship type and user list only."""
        self._relationship_type = relationship_type
        self._user_list = sorted({c.user_id for c in conversations})
        logger.info(
            f"[no_memory] ingest scenario {scenario_id}: "
            f"relationship={relationship_type}, users={self._user_list}"
        )

    def answer(self, question: str) -> tuple[str, dict]:
        """Answer using only question + minimal context — no conversation data."""
        context_parts = []
        if self._relationship_type:
            context_parts.append(f"Relationship type: {self._relationship_type}")
        if self._user_list:
            context_parts.append(f"Users in scenario: {', '.join(self._user_list)}")
        context = "\n".join(context_parts)

        system = (
            "You are answering questions about a multi-user AI memory system. "
            "You have NO access to any conversation history or user sessions. "
            "Answer based only on the minimal context provided."
        )
        user_msg = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer based only on the context above. "
            "If you cannot determine the answer, say so explicitly."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

        answer_text, _ = self._llm.generate(
            messages=messages,
            max_tokens=self.config.get("max_tokens", 512),
            phase="mumm_answer",
        )
        return answer_text, {"method": "no_memory", "context_tokens": len(context.split())}

    def reset(self) -> None:
        self._relationship_type = ""
        self._user_list = []
