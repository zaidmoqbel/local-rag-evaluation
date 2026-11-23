from rag.config import Chunking
from rag.ingest.chunker import chunk_sources
from rag.ingest.loaders import Source


def test_overlapping_page_local_offsets_and_document_wide_indices() -> None:
    sources = [
        Source("a.pdf", "a.pdf", 1, "one two\nred blue"),
        Source("a.pdf", "a.pdf", 2, "xyz"),
    ]
    config = Chunking(strategy="fixed", chunk_size=9, chunk_overlap=4)
    chunks = list(chunk_sources(sources, config))
    assert [(c.text, c.page, c.char_start, c.char_end) for c in chunks] == [
        ("one two\n", 1, 0, 8),
        ("two\nred ", 1, 4, 12),
        ("red blue", 1, 8, 16),
        ("xyz", 2, 0, 3),
    ]
    assert [c.chunk_index for c in chunks] == list(range(4))


def test_heading_strategy_respects_section_offsets() -> None:
    source = Source("a.md", "a.md", None, "# One\nalpha\n## Two\nbeta")
    settings = Chunking(strategy="heading", chunk_size=50, chunk_overlap=2)
    chunks = list(chunk_sources([source], settings))
    assert [chunk.text for chunk in chunks] == ["# One\nalpha\n", "## Two\nbeta"]
    assert chunks[1].char_start == 12


def test_heading_strategy_supports_setext_and_ignores_fenced_headings() -> None:
    text = "Title\r\n=====\r\n\r\n```\r\n# Not a heading\r\n```\r\n\r\n## Real\r\nBody"
    source = Source("a.md", "a.md", None, text)
    settings = Chunking(strategy="heading", chunk_size=200, chunk_overlap=0)
    chunks = list(chunk_sources([source], settings))
    assert len(chunks) == 2
    assert "# Not a heading" in chunks[0].text
    assert chunks[1].text == "## Real\r\nBody"
    assert chunks[1].char_start == text.index("## Real")


def test_heading_strategy_supports_python_documentation_underlines() -> None:
    text = "Overview\n========\n\nBody.\n\nDetails\n~~~~~~~\n\nMore body."
    source = Source("a.txt", "a.txt", None, text)
    settings = Chunking(strategy="heading", chunk_size=200, chunk_overlap=0)
    chunks = list(chunk_sources([source], settings))
    assert [chunk.char_start for chunk in chunks] == [0, text.index("Details")]
    assert all(
        chunk.text == text[chunk.char_start : chunk.char_end] for chunk in chunks
    )


def test_token_windows_preserve_original_text() -> None:
    def offsets(_: str) -> list[tuple[int, int]]:
        return [(0, 2), (4, 6), (7, 9)]

    source = Source("a.txt", "a.txt", None, "ab  cd ef")
    settings = Chunking(strategy="fixed", unit="token", chunk_size=2, chunk_overlap=1)
    chunks = list(chunk_sources([source], settings, offsets))
    assert [chunk.text for chunk in chunks] == ["ab  cd", "cd ef"]


def test_overlong_word_is_preserved_and_windows_advance() -> None:
    source = Source("a.txt", "a.txt", None, "a extraordinarily b")
    settings = Chunking(strategy="fixed", chunk_size=4, chunk_overlap=3)
    chunks = list(chunk_sources([source], settings))
    assert [chunk.text.strip() for chunk in chunks] == ["a", "extraordinarily", "b"]
    assert all(
        chunk.text == source.text[chunk.char_start : chunk.char_end] for chunk in chunks
    )


def test_subword_offsets_snap_across_whitespace() -> None:
    def offsets(_: str) -> list[tuple[int, int]]:
        return [(0, 2), (2, 5), (6, 8), (9, 12)]

    source = Source("a.txt", "a.txt", None, "alpha bb cat")
    settings = Chunking(strategy="fixed", unit="token", chunk_size=3, chunk_overlap=2)
    chunks = list(chunk_sources([source], settings, offsets))
    assert [chunk.text for chunk in chunks] == ["alpha bb", "bb cat"]
