"""Resolve answer citations to the chunks actually supplied to the model."""

import re
from collections.abc import Sequence
from typing import Literal

from rag.ingest.chunker import Chunk

ABSTENTION = "I don't know based on the provided documents"
CITATION = re.compile(r"\[([^\[\]\s:]+:[0-9]+)\]")
CitationAbstention = Literal[
    "model_said_unknown", "no_citations", "invalid_citation_id"
]


def format_citations(answer: str, chunks: Sequence[Chunk]) -> tuple[str, list[Chunk]]:
    """Deduplicate valid citations in order and abstain on missing or unknown IDs."""
    answer, citations, _ = check_citations(answer, chunks)
    return answer, citations


def check_citations(
    answer: str, chunks: Sequence[Chunk]
) -> tuple[str, list[Chunk], CitationAbstention | None]:
    """Resolve citations and return the reason for any abstention."""
    answer = answer.strip()
    if answer.rstrip(".").casefold() == ABSTENTION.casefold():
        return ABSTENTION, [], "model_said_unknown"
    by_id = {chunk.citation_id: chunk for chunk in chunks}
    ids = list(dict.fromkeys(CITATION.findall(answer)))
    if not ids:
        return ABSTENTION, [], "no_citations"
    if any(citation not in by_id for citation in ids):
        return ABSTENTION, [], "invalid_citation_id"
    return answer, [by_id[citation] for citation in ids], None
