"""MUMM baseline memory architectures."""

from MUMBench.baselines.base import MUMMBaseline
from MUMBench.baselines.no_memory import NoMemoryBaseline
from MUMBench.baselines.rag_baseline import RAGBaseline
from MUMBench.baselines.long_context import LongContextBaseline

BASELINES: dict[str, type[MUMMBaseline]] = {
    "no_memory": NoMemoryBaseline,
    "rag": RAGBaseline,
    "long_context": LongContextBaseline,
}

__all__ = [
    "MUMMBaseline",
    "NoMemoryBaseline",
    "RAGBaseline",
    "LongContextBaseline",
    "BASELINES",
]
