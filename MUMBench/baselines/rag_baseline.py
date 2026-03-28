"""RAG baseline — embed session summaries and retrieve top-k for each question.

Embeds each of the 140 session summaries (28 per scenario) using ChromaDB's
default embedding model, retrieves top-k by cosine similarity, and builds a
prompt from retrieved summaries + relationship context.
"""

from __future__ import annotations

import logging

from datagen.llm.client import LLMClient
from datagen.llm.cost_tracker import CostTracker
from datagen.models.schemas import ConversationSession, SessionSummary

from MUMBench.baselines.base import MUMMBaseline
from MUMBench.config import MUMM_CONFIG

logger = logging.getLogger("mum.mummbench")


class RAGBaseline(MUMMBaseline):
    """RAG over session summaries: embed summaries, retrieve top-k per question."""

    @property
    def name(self) -> str:
        return "rag"

    def __init__(
        self,
        model: str,
        top_k: int | None = None,
        cost_tracker: CostTracker | None = None,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.top_k = top_k or MUMM_CONFIG["rag_top_k"]
        self.cost_tracker = cost_tracker or CostTracker()
        self._relationship_type: str = ""
        self._authority_context: dict = {}
        self._collection = None
        self._chroma_client = None
        self._collection_name: str = ""
        self._llm = LLMClient(
            model=model,
            temperature=self.config.get("temperature", 0.3),
            cost_tracker=self.cost_tracker,
        )

    def _init_collection(self, scenario_id: int) -> None:
        import chromadb

        self._chroma_client = chromadb.EphemeralClient()
        self._collection_name = f"mumm_rag_s{scenario_id}"
        self._collection = self._chroma_client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest(
        self,
        scenario_id: int,
        conversations: list[ConversationSession],
        summaries: list[SessionSummary],
        relationship_type: str = "",
        authority_context: dict | None = None,
    ) -> None:
        """Embed session summaries into a ChromaDB ephemeral collection."""
        self._relationship_type = relationship_type
        self._authority_context = authority_context or {}
        self._init_collection(scenario_id)

        if not summaries:
            logger.warning(f"[rag] No summaries for scenario {scenario_id}")
            return

        docs = []
        ids = []
        metadatas = []

        for summ in summaries:
            # Build rich text for embedding
            parts = [
                f"User: {summ.user_id} | Session {summ.session_number}",
                summ.summary,
            ]
            if summ.key_facts:
                parts.append("Key facts: " + "; ".join(summ.key_facts))
            if summ.positions_taken:
                parts.append("Positions: " + "; ".join(summ.positions_taken))
            if summ.positions_changed:
                parts.append("Corrections: " + "; ".join(summ.positions_changed))
            text = "\n".join(parts)

            doc_id = f"{summ.user_id}_session_{summ.session_number}"
            docs.append(text)
            ids.append(doc_id)
            metadatas.append({
                "user_id": summ.user_id,
                "session_number": summ.session_number,
                "session_id": summ.session_id,
            })

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(docs), batch_size):
            self._collection.upsert(
                ids=ids[i : i + batch_size],
                documents=docs[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

        logger.info(
            f"[rag] Ingested {len(docs)} session summaries for scenario {scenario_id}"
        )

    def answer(self, question: str) -> tuple[str, dict]:
        if self._collection is None or self._collection.count() == 0:
            return (
                "No memory available.",
                {"method": "rag", "error": "empty collection", "retrieved": 0},
            )

        k = min(self.top_k, self._collection.count())
        results = self._collection.query(query_texts=[question], n_results=k)

        retrieved_docs = results["documents"][0] if results["documents"] else []
        retrieved_meta = results["metadatas"][0] if results["metadatas"] else []

        # Build context
        blocks = []
        for doc, meta in zip(retrieved_docs, retrieved_meta):
            header = (
                f"[Session Summary] User: {meta.get('user_id', '?')} | "
                f"Session: {meta.get('session_number', '?')}"
            )
            blocks.append(f"--- {header} ---\n{doc}")
        context = "\n\n".join(blocks)

        system_parts = [
            "You are answering questions about a multi-user AI memory system.",
            "You have access to session summaries retrieved from a vector index.",
        ]
        if self._relationship_type:
            system_parts.append(f"Relationship type: {self._relationship_type}")
        if self._authority_context:
            auth_str = "; ".join(
                f"{k}: {v}" for k, v in self._authority_context.items()
            )
            system_parts.append(f"Authority context: {auth_str}")
        system_parts.append(
            "Answer using ONLY the retrieved context below. "
            "If the answer cannot be determined from the context, say so."
        )
        system = "\n".join(system_parts)

        user_msg = (
            f"Retrieved context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer based only on the retrieved context above."
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
            "method": "rag",
            "retrieved": len(retrieved_docs),
            "top_k": self.top_k,
            "context_chars": len(context),
        }

    def reset(self) -> None:
        if self._chroma_client is not None and self._collection is not None:
            try:
                self._chroma_client.delete_collection(self._collection_name)
            except Exception:
                pass
        self._collection = None
        self._chroma_client = None
        self._relationship_type = ""
        self._authority_context = {}
