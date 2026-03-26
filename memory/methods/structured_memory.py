"""Structured multi-user memory — user-partitioned memory with cross-user indexing.

This method models how an ideal multi-user memory system would work:
  - Per-user fact stores extracted from summaries
  - Cross-user conflict tracking
  - Temporal correction chains
  - Authority-aware retrieval
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import tiktoken

from datagen.models.schemas import ConversationSession, SessionSummary

from memory.methods.base import BaseMemoryMethod, MemoryContext

logger = logging.getLogger("mum.memory")


@dataclass
class UserMemoryStore:
    """Per-user memory partition."""

    user_id: str
    facts: list[dict] = field(default_factory=list)
    positions: list[dict] = field(default_factory=list)
    corrections: list[dict] = field(default_factory=list)


class StructuredMemory(BaseMemoryMethod):
    """Structured multi-user memory with per-user partitions and cross-user indexing.

    Organizes memory by user, tracks facts, positions, and corrections explicitly.
    Retrieval assembles context relevant to the question type.
    """

    name = "structured"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._user_stores: dict[str, UserMemoryStore] = {}
        self._cross_user_facts: list[dict] = []
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def ingest(
        self,
        conversations: list[ConversationSession],
        summaries: list[SessionSummary],
    ) -> None:
        sorted_summaries = sorted(summaries, key=lambda s: (s.user_id, s.session_number))

        for summ in sorted_summaries:
            uid = summ.user_id
            if uid not in self._user_stores:
                self._user_stores[uid] = UserMemoryStore(user_id=uid)

            store = self._user_stores[uid]

            # Extract facts
            for fact in summ.key_facts:
                entry = {
                    "content": fact,
                    "session": summ.session_number,
                    "session_id": summ.session_id,
                    "user_id": uid,
                }
                store.facts.append(entry)

            # Extract positions
            for pos in summ.positions_taken:
                entry = {
                    "content": pos,
                    "session": summ.session_number,
                    "session_id": summ.session_id,
                    "user_id": uid,
                }
                store.positions.append(entry)

            # Track corrections
            for change in summ.positions_changed:
                entry = {
                    "content": change,
                    "session": summ.session_number,
                    "session_id": summ.session_id,
                    "user_id": uid,
                }
                store.corrections.append(entry)

        # Build cross-user fact index (facts that appear across multiple users)
        fact_users: dict[str, list[str]] = defaultdict(list)
        for uid, store in self._user_stores.items():
            for fact in store.facts:
                # Use first 80 chars as a rough dedup key
                key = fact["content"][:80].lower().strip()
                fact_users[key].append(uid)

        for key, users in fact_users.items():
            if len(set(users)) > 1:
                self._cross_user_facts.append({
                    "fact_key": key,
                    "users": list(set(users)),
                })

        logger.info(
            f"Structured memory: {len(self._user_stores)} users, "
            f"{sum(len(s.facts) for s in self._user_stores.values())} facts, "
            f"{sum(len(s.corrections) for s in self._user_stores.values())} corrections, "
            f"{len(self._cross_user_facts)} cross-user facts"
        )

    def retrieve(self, question: str, user_id: str | None = None) -> MemoryContext:
        blocks = []
        q_lower = question.lower()

        # Determine retrieval strategy based on question keywords
        is_attribution = any(w in q_lower for w in ["who ", "which user", "which student", "which analyst"])
        is_cross_user = any(w in q_lower for w in ["combining", "across", "all users", "complete picture", "cross"])
        is_correction = any(w in q_lower for w in ["correct", "error", "mistake", "revise", "change", "update"])
        is_conflict = any(w in q_lower for w in ["disagree", "conflict", "contradict", "dispute"])
        is_gap = any(w in q_lower for w in ["gap", "missing", "doesn't know", "not aware", "lacks"])
        is_handoff = any(w in q_lower for w in ["handoff", "shift", "handover", "transfer"])

        if user_id:
            # Scoped retrieval for a specific user
            blocks.append(self._format_user_store(user_id))
        else:
            # Multi-user retrieval
            for uid in sorted(self._user_stores.keys()):
                blocks.append(self._format_user_store(uid))

        # Add cross-user section for relevant question types
        if is_cross_user or is_conflict or is_gap or is_attribution:
            cross_block = self._format_cross_user_section()
            if cross_block:
                blocks.append(cross_block)

        # Add corrections section for temporal questions
        if is_correction or is_handoff:
            corr_block = self._format_corrections_section()
            if corr_block:
                blocks.append(corr_block)

        context_text = "\n\n".join(blocks)
        token_count = len(self._encoder.encode(context_text))

        return MemoryContext(
            method_name=self.name,
            context_text=context_text,
            token_count=token_count,
            metadata={
                "num_users": len(self._user_stores),
                "retrieval_signals": {
                    "attribution": is_attribution,
                    "cross_user": is_cross_user,
                    "correction": is_correction,
                    "conflict": is_conflict,
                    "gap": is_gap,
                    "handoff": is_handoff,
                },
                "scoped_to_user": user_id,
            },
        )

    def _format_user_store(self, user_id: str) -> str:
        store = self._user_stores.get(user_id)
        if not store:
            return f"=== User: {user_id} ===\n[No data]\n"

        lines = [f"=== User: {user_id} ==="]

        lines.append("\n[Facts]")
        for fact in store.facts:
            lines.append(f"  (S{fact['session']}) {fact['content']}")

        if store.positions:
            lines.append("\n[Positions]")
            for pos in store.positions:
                lines.append(f"  (S{pos['session']}) {pos['content']}")

        if store.corrections:
            lines.append("\n[Corrections/Changes]")
            for corr in store.corrections:
                lines.append(f"  (S{corr['session']}) {corr['content']}")

        return "\n".join(lines)

    def _format_cross_user_section(self) -> str:
        if not self._cross_user_facts:
            return ""
        lines = ["=== Cross-User Shared Facts ==="]
        for entry in self._cross_user_facts[:30]:  # Cap at 30
            users_str = ", ".join(entry["users"])
            lines.append(f"  [{users_str}] {entry['fact_key']}")
        return "\n".join(lines)

    def _format_corrections_section(self) -> str:
        all_corrections = []
        for uid, store in self._user_stores.items():
            for corr in store.corrections:
                all_corrections.append(corr)

        if not all_corrections:
            return ""

        all_corrections.sort(key=lambda c: (c["user_id"], c["session"]))
        lines = ["=== Correction/Change Timeline ==="]
        for corr in all_corrections:
            lines.append(
                f"  {corr['user_id']} (S{corr['session']}): {corr['content']}"
            )
        return "\n".join(lines)

    def reset(self) -> None:
        self._user_stores = {}
        self._cross_user_facts = []
