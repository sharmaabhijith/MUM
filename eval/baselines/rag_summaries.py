from __future__ import annotations

import numpy as np

from src.llm.client import LLMClient
from src.models.schemas import ConversationSession
from eval.run_eval import MemorySystemInterface


class RAGSummariesBaseline(MemorySystemInterface):
    """Baseline: RAG over session summaries using OpenAI embeddings."""

    def __init__(
        self,
        llm_client: LLMClient,
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 10,
    ):
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.summaries: list[str] = []
        self.embeddings: list[list[float]] = []
        self.documents: dict[str, str] = {}

    def process_conversations(
        self, conversations: list[ConversationSession], documents: dict[str, str]
    ) -> None:
        self.documents = documents

        # Build summaries from conversations (simple extraction)
        self.summaries = []
        for conv in sorted(conversations, key=lambda c: (c.user_id, c.session_number)):
            # Create a summary-like text from the conversation
            turns_text = "\n".join(
                f"[{t.role}] {t.content}" for t in conv.turns
            )
            summary = (
                f"User: {conv.user_id} | Session: {conv.session_number} | "
                f"Timestamp: {conv.session_timestamp}\n{turns_text[:2000]}"
            )
            self.summaries.append(summary)

        # Embed all summaries
        self.embeddings = self._embed_texts(self.summaries)

    def answer_question(self, question: str) -> str:
        # Embed the question
        question_embedding = self._embed_texts([question])[0]

        # Retrieve top-k summaries by cosine similarity
        similarities = [
            self._cosine_similarity(question_embedding, emb)
            for emb in self.embeddings
        ]
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        retrieved = [self.summaries[i] for i in top_indices]

        context = "\n\n---\n\n".join(retrieved)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are answering questions about multi-user conversations. "
                    "Answer based only on the retrieved session summaries below."
                ),
            },
            {
                "role": "user",
                "content": f"## Retrieved Sessions\n{context}\n\n## Question\n{question}",
            },
        ]
        return self.llm_client.generate(
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
            phase="eval_baseline_rag",
        )

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenAI embeddings API."""
        if not texts:
            return []
        try:
            response = self.llm_client.client.embeddings.create(
                model=self.embedding_model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception:
            # Fallback: random embeddings for testing
            return [np.random.randn(256).tolist() for _ in texts]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        a_arr = np.array(a)
        b_arr = np.array(b)
        dot = np.dot(a_arr, b_arr)
        norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        return float(dot / norm) if norm > 0 else 0.0
