from memory.methods.base import BaseMemoryMethod
from memory.methods.no_memory import NoMemory
from memory.methods.full_context import FullContext
from memory.methods.rag import RAGMemory
from memory.methods.summary_memory import SummaryMemory
from memory.methods.structured_memory import StructuredMemory

METHODS: dict[str, type[BaseMemoryMethod]] = {
    "no_memory": NoMemory,
    "full_context": FullContext,
    "rag": RAGMemory,
    "summary": SummaryMemory,
    "structured": StructuredMemory,
}

__all__ = [
    "BaseMemoryMethod",
    "NoMemory",
    "FullContext",
    "RAGMemory",
    "SummaryMemory",
    "StructuredMemory",
    "METHODS",
]
