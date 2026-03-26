from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger("mum")


def extract_text_from_pdf(path: Path | str) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(str(path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append(f"--- Page {i + 1} ---\n{text}")
    doc.close()
    return "\n\n".join(pages)


def extract_text_from_directory(dir_path: Path | str) -> dict[str, str]:
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    results = {}
    for pdf_file in sorted(dir_path.glob("*.pdf")):
        logger.info(f"Extracting text from {pdf_file.name}")
        results[pdf_file.name] = extract_text_from_pdf(pdf_file)
    return results
