"""Split source text into overlapping, citable windows."""

import re
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import PurePath

from markdown_it import MarkdownIt

from rag.config import Chunking
from rag.ingest.loaders import Source

TokenOffsets = Callable[[str], list[tuple[int, int]]]
LINE_END = re.compile(r"\r\n|\r|\n")
RST_UNDERLINE = re.compile(r"([=~^*+`:#_-])\1{2,}")
MARKDOWN = MarkdownIt("commonmark")


@dataclass(frozen=True)
class Chunk:
    """A source span with a document-wide chunk index and local offsets."""

    doc_id: str
    filename: str
    page: int | None
    chunk_index: int
    char_start: int
    char_end: int
    text: str

    @property
    def citation_id(self) -> str:
        """Return the identifier used in prompts and citations."""
        return f"{self.doc_id}:{self.chunk_index}"


def _word_breaks(text: str, spans: list[tuple[int, int]] | None) -> list[int]:
    if spans is None:
        return [
            0,
            *(
                i
                for i in range(1, len(text))
                if text[i - 1].isspace() or text[i].isspace()
            ),
            len(text),
        ]
    return [
        0,
        *(
            i
            for i in range(1, len(spans))
            if any(
                char.isspace() for char in text[spans[i - 1][1] - 1 : spans[i][0] + 1]
            )
        ),
        len(spans),
    ]


def _fixed_windows(breaks: list[int], settings: Chunking) -> Iterator[tuple[int, int]]:
    start = 0
    last = len(breaks) - 1
    while start < last:
        limit = breaks[start] + settings.chunk_size
        end = max(start + 1, bisect_right(breaks, limit) - 1)
        yield breaks[start], breaks[end]
        if end == last:
            break
        target = breaks[end] - settings.chunk_overlap
        desired = bisect_right(breaks, target) - 1
        # Reduce overlap when it would prevent the next window from adding a word.
        earliest = bisect_left(breaks, breaks[end + 1] - settings.chunk_size)
        start = min(end, max(start + 1, desired, earliest))


def _windows(
    text: str, settings: Chunking, offsets: TokenOffsets | None
) -> Iterator[tuple[int, int]]:
    spans = None
    if settings.unit == "token":
        if offsets is None:
            raise ValueError("Token chunking requires tokenizer offsets")
        spans = [(start, end) for start, end in offsets(text) if start < end]
    size = len(text) if spans is None else len(spans)
    if not size:
        return
    for start, end in _fixed_windows(_word_breaks(text, spans), settings):
        if spans is None:
            yield start, end
        else:
            yield spans[start][0], spans[end - 1][1]


def _section_starts(source: Source) -> list[int]:
    suffix = PurePath(source.filename).suffix.lower()
    if suffix not in {".md", ".txt"}:
        return []
    line_offsets = [0, *(match.end() for match in LINE_END.finditer(source.text))]
    tokens = MARKDOWN.parse(source.text)
    starts = {
        token.map[0]
        for token in tokens
        if token.type == "heading_open" and token.level == 0 and token.map
    }
    if suffix == ".txt":
        lines = LINE_END.split(source.text)
        code_lines = {
            line
            for token in tokens
            if token.type in {"fence", "code_block"} and token.map
            for line in range(*token.map)
        }
        for i, (title, marker) in enumerate(pairwise(lines)):
            if (
                title
                and not title[0].isspace()
                and i not in code_lines
                and RST_UNDERLINE.fullmatch(marker.rstrip())
                and len(marker.rstrip()) >= len(title.rstrip())
                and not RST_UNDERLINE.fullmatch(title.rstrip())
            ):
                starts.add(i)
    return [line_offsets[line] for line in sorted(starts) if line > 0]


def chunk_sources(
    sources: Iterable[Source], settings: Chunking, offsets: TokenOffsets | None = None
) -> Iterator[Chunk]:
    """Yield chunks with half-open offsets into the original file or PDF page text."""
    next_index: dict[str, int] = {}
    for source in sources:
        boundaries = [0]
        if settings.strategy == "heading":
            boundaries.extend(_section_starts(source))
        boundaries.append(len(source.text))
        for first, last in pairwise(boundaries):
            for start, end in _windows(source.text[first:last], settings, offsets):
                text = source.text[first + start : first + end]
                if text.strip():
                    index = next_index.get(source.doc_id, 0)
                    yield Chunk(
                        source.doc_id,
                        source.filename,
                        source.page,
                        index,
                        first + start,
                        first + end,
                        text,
                    )
                    next_index[source.doc_id] = index + 1
