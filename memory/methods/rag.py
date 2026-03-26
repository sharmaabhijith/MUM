"""RAG baseline — chunk conversations and retrieve via embedding similarity."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import tiktoken

from datagen.models.schemas import ConversationSession, SessionSummary

from memory.methods.base import BaseMemoryMethod, MemoryContext

logger = logging.getLogger("mum.memory")

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_TOP_K = 20


class RAGMemory(BaseMemoryMethod):
    """RAG (Retrieval-Augmented Generation) — chunk conversations, embed, retrieve top-k by similarity."""

    name = "rag"

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
        persist_dir: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.persist_dir = persist_dir
        self._encoder = tiktoken.get_encoding("cl100k_base")
        self._collection = None
        self._client = None

    def _get_collection(self, scenario_id: str = "default"):
        """Lazily initialize ChromaDB client and collection."""
        import chromadb

        if self._client is None:
            if self.persist_dir:
                Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=self.persist_dir)
            else:
                self._client = chromadb.EphemeralClient()

        collection_name = f"mum_rag_{scenario_id}"
        # ChromaDB collection names must be 3-63 chars, alphanumeric + underscores
        collection_name = collection_name[:63]
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def _chunk_text(self, text: str, metadata: dict) -> list[dict]:
        """Split text into overlapping token-based chunks."""
        tokens = self._encoder.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self._encoder.decode(chunk_tokens)
            chunk_id = hashlib.md5(
                f"{metadata.get('user_id', '')}_{metadata.get('session_number', '')}_{start}".encode()
            ).hexdigest()
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {**metadata, "token_start": start, "token_end": end},
            })
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def ingest(
        self,
        conversations: list[ConversationSession],
        summaries: list[SessionSummary],
    ) -> None:
        if not conversations:
            return

        scenario_id = conversations[0].scenario_id
        collection = self._get_collection(scenario_id)

        all_chunks = []

        # Chunk conversations
        for conv in conversations:
            conv_text = "\n".join(
                f"[{t.role}]: {t.content}" for t in conv.turns
            )
            full_text = (
                f"User: {conv.user_id} | Session {conv.session_number} "
                f"({conv.session_timestamp})\n{conv_text}"
            )
            chunks = self._chunk_text(full_text, {
                "user_id": conv.user_id,
                "session_number": conv.session_number,
                "session_id": conv.session_id,
                "source_type": "conversation",
            })
            all_chunks.extend(chunks)

        # Chunk summaries
        for summ in summaries:
            summ_text = (
                f"User: {summ.user_id} | Session {summ.session_number} Summary\n"
                f"{summ.summary}\n"
                f"Key facts: {'; '.join(summ.key_facts)}\n"
                f"Positions: {'; '.join(summ.positions_taken)}"
            )
            if summ.positions_changed:
                summ_text += f"\nPositions changed: {'; '.join(summ.positions_changed)}"
            chunks = self._chunk_text(summ_text, {
                "user_id": summ.user_id,
                "session_number": summ.session_number,
                "session_id": summ.session_id,
                "source_type": "summary",
            })
            all_chunks.extend(chunks)

        # Batch upsert into ChromaDB
        batch_size = 500
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            collection.upsert(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
            )

        logger.info(
            f"RAG: ingested {len(all_chunks)} chunks from "
            f"{len(conversations)} conversations + {len(summaries)} summaries"
        )

    def retrieve(self, question: str, user_id: str | None = None) -> MemoryContext:
        if self._collection is None or self._collection.count() == 0:
            return MemoryContext(
                method_name=self.name,
                context_text="",
                token_count=0,
                metadata={"error": "No data ingested"},
            )

        where_filter = None
        if user_id:
            where_filter = {"user_id": user_id}

        results = self._collection.query(
            query_texts=[question],
            n_results=min(self.top_k, self._collection.count()),
            where=where_filter,
        )

        retrieved_docs = results["documents"][0] if results["documents"] else []
        retrieved_meta = results["metadatas"][0] if results["metadatas"] else []

        # Build context with source attribution
        blocks = []
        for doc, meta in zip(retrieved_docs, retrieved_meta):
            source = f"[{meta.get('source_type', '?')}] User: {meta.get('user_id', '?')} Session: {meta.get('session_number', '?')}"
            blocks.append(f"--- {source} ---\n{doc}")

        context_text = "\n\n".join(blocks)
        token_count = len(self._encoder.encode(context_text))

        return MemoryContext(
            method_name=self.name,
            context_text=context_text,
            token_count=token_count,
            metadata={
                "num_chunks_retrieved": len(retrieved_docs),
                "top_k": self.top_k,
                "chunk_size": self.chunk_size,
            },
        )

    def reset(self) -> None:
        if self._client is not None and self._collection is not None:
            try:
                self._client.delete_collection(self._collection.name)
            except Exception:
                pass
        self._collection = None
        self._client = None
