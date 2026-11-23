"""Extract source text while preserving document and page boundaries."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import pymupdf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Source:
    """One text file or one extracted PDF page."""

    doc_id: str
    filename: str
    page: int | None
    text: str


def load_sources(root: Path) -> Iterator[Source]:
    """Yield supported documents in stable order with one-based PDF pages."""
    if not root.is_dir():
        raise ValueError(f"Document directory does not exist: {root}")
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            continue
        suffix = path.suffix.lower()
        if suffix not in {".pdf", ".md", ".txt"}:
            continue
        filename = path.relative_to(root).as_posix()
        doc_id = quote(filename, safe="/")
        if suffix != ".pdf":
            yield Source(doc_id, filename, None, path.read_bytes().decode("utf-8-sig"))
            continue
        with pymupdf.open(path) as document:
            for number, page in enumerate(document, start=1):
                text = page.get_text(sort=True)
                if text.strip():
                    yield Source(doc_id, filename, number, text)
                else:
                    logger.warning(
                        "No extractable text in %s, page %d", filename, number
                    )
