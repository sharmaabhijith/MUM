from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from datagen.llm.token_counter import TokenCounter
from datagen.models.schemas import ScenarioConfig
from datagen.utils.pdf_reader import extract_text_from_pdf

logger = logging.getLogger("mum")

DOCUMENTS_DIR = Path("MUMBench/documents")


@dataclass
class DocumentContext:
    scenario_id: str
    documents: dict[str, str]  # {doc_name: extracted_text}
    token_counts: dict[str, int]
    total_tokens: int
    context_block: str  # Concatenated text for prompts
    warnings: list[str] = field(default_factory=list)


class DocumentPreparer:
    def __init__(self, token_counter: TokenCounter | None = None, documents_dir: Path | str = DOCUMENTS_DIR):
        self.token_counter = token_counter or TokenCounter()
        self.documents_dir = Path(documents_dir)

    def prepare_scenario(self, scenario_config: ScenarioConfig) -> DocumentContext:
        scenario_dir = self.documents_dir / f"scenario_{scenario_config.scenario_id}"
        documents: dict[str, str] = {}
        token_counts: dict[str, int] = {}
        warnings: list[str] = []

        for doc_cfg in scenario_config.documents:
            pdf_path = scenario_dir / doc_cfg.filename
            if not pdf_path.exists():
                warnings.append(f"PDF not found: {pdf_path}")
                logger.warning(f"PDF not found: {pdf_path}")
                continue

            text = extract_text_from_pdf(pdf_path)
            tokens = self.token_counter.count(text)
            documents[doc_cfg.name] = text
            token_counts[doc_cfg.name] = tokens

            # Validate token count within ±20%
            target = doc_cfg.target_tokens
            lower = int(target * 0.8)
            upper = int(target * 1.2)
            if tokens < lower or tokens > upper:
                msg = (
                    f"{doc_cfg.name}: {tokens} tokens (target: {target}, "
                    f"range: {lower}-{upper})"
                )
                warnings.append(msg)
                logger.warning(f"Token count out of range: {msg}")
            else:
                logger.info(f"{doc_cfg.name}: {tokens} tokens (target: {target}) ✓")

        context_block = self._build_context_block(documents)
        total_tokens = sum(token_counts.values())

        logger.info(
            f"Scenario {scenario_config.scenario_id}: "
            f"{len(documents)} docs, {total_tokens} total tokens"
        )

        return DocumentContext(
            scenario_id=scenario_config.scenario_id,
            documents=documents,
            token_counts=token_counts,
            total_tokens=total_tokens,
            context_block=context_block,
            warnings=warnings,
        )

    def _build_context_block(self, documents: dict[str, str]) -> str:
        blocks = []
        for name, text in documents.items():
            blocks.append(
                f"\n\n=== DOCUMENT: {name} ===\n{text}\n=== END DOCUMENT ===\n"
            )
        return "\n".join(blocks)
