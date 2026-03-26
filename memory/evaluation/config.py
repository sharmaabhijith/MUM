"""Load evaluation configuration from config/evaluation.yaml."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger("mum.memory")

DEFAULT_CONFIG_PATH = Path("config/evaluation.yaml")


@dataclass
class EvalConfig:
    """Parsed evaluation configuration."""

    eval_models: list[str] = field(default_factory=list)
    judge_model: str = "google/gemini-2.5-pro"
    methods: dict[str, dict] = field(default_factory=dict)
    judge_temperature: float = 0.1
    judge_max_tokens: int = 1024
    answer_temperature: float = 0.3
    answer_max_tokens: int = 2048
    correctness_weight: float = 0.35
    completeness_weight: float = 0.25
    attribution_weight: float = 0.25
    hallucination_weight: float = 0.15

    @property
    def method_names(self) -> list[str]:
        return list(self.methods.keys()) if self.methods else []

    def rag_kwargs(self) -> dict:
        """Return RAG-specific kwargs from the config."""
        rag_cfg = self.methods.get("rag", {})
        return {
            "chunk_size": rag_cfg.get("chunk_size", 512),
            "chunk_overlap": rag_cfg.get("chunk_overlap", 64),
            "top_k": rag_cfg.get("top_k", 20),
        }


def load_eval_config(path: Path | str | None = None) -> EvalConfig:
    """Load evaluation config from YAML, falling back to defaults."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        logger.warning(f"Eval config not found at {config_path}, using defaults")
        return EvalConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    judge_params = raw.get("judge", {})
    answer_params = raw.get("answer", {})
    scoring = raw.get("scoring", {})

    return EvalConfig(
        eval_models=raw.get("eval_models", []),
        judge_model=raw.get("judge_model", "google/gemini-2.5-pro"),
        methods=raw.get("methods", {}),
        judge_temperature=judge_params.get("temperature", 0.1),
        judge_max_tokens=judge_params.get("max_tokens", 1024),
        answer_temperature=answer_params.get("temperature", 0.3),
        answer_max_tokens=answer_params.get("max_tokens", 2048),
        correctness_weight=scoring.get("correctness_weight", 0.35),
        completeness_weight=scoring.get("completeness_weight", 0.25),
        attribution_weight=scoring.get("attribution_weight", 0.25),
        hallucination_weight=scoring.get("hallucination_weight", 0.15),
    )
