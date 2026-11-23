import pytest

from rag.generate.citations import ABSTENTION, format_citations
from rag.ingest.chunker import Chunk

PASSAGE = Chunk("guide%20a.md", "guide a.md", None, 0, 0, 4, "fact")


def test_resolves_citations_in_order_without_duplicates() -> None:
    answer = "One [guide%20a.md:0]. Two [guide%20a.md:0]."
    assert format_citations(answer, [PASSAGE]) == (answer, [PASSAGE])


@pytest.mark.parametrize(
    "answer", ["No citation", "[wrong:0]", "[guide%20a.md:0] [wrong:0]"]
)
def test_invalid_citation_abstains(answer: str) -> None:
    assert format_citations(answer, [PASSAGE]) == (ABSTENTION, [])
