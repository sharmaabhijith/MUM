from __future__ import annotations

from src.llm.client import LLMClient
from src.models.schemas import ConversationSession
from eval.run_eval import MemorySystemInterface


class FullContextBaseline(MemorySystemInterface):
    """Baseline: feed all conversations as raw context. No memory system."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.context: str = ""

    def process_conversations(
        self, conversations: list[ConversationSession], documents: dict[str, str]
    ) -> None:
        parts = []

        # Add documents
        for name, text in documents.items():
            if text:
                parts.append(f"=== DOCUMENT: {name} ===\n{text}\n=== END ===\n")

        # Add all conversations
        for conv in sorted(conversations, key=lambda c: (c.user_id, c.session_number)):
            parts.append(
                f"\n--- {conv.user_id} / Session {conv.session_number} ---"
            )
            for turn in conv.turns:
                parts.append(f"[{turn.role.upper()}] {turn.content}")

        self.context = "\n".join(parts)

    def answer_question(self, question: str) -> str:
        # Truncate context if too long
        max_context = 120000  # ~120K chars
        context = self.context[:max_context]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are answering questions about multi-user conversations. "
                    "Multiple users independently interacted with an AI assistant. "
                    "Answer based only on the provided conversations and documents."
                ),
            },
            {
                "role": "user",
                "content": f"## Conversations and Documents\n{context}\n\n## Question\n{question}",
            },
        ]
        return self.llm_client.generate(
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
            phase="eval_baseline_full_context",
        )
