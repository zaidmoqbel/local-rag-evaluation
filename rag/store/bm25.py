"""Build and query a lexical index over an indexed Chroma snapshot."""

import re
from collections import defaultdict

from rank_bm25 import BM25Plus

from rag.ingest.chunker import Chunk
from rag.store.chroma import Match

TERM = re.compile(r"[^\W_]+")


def tokenize(text: str) -> list[str]:
    """Split case-folded text on punctuation, whitespace, and underscores."""
    return TERM.findall(text.casefold())


class BM25Index:
    """Rank only passages that share a term with the query."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks: list[Chunk] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        corpus = []
        for chunk in sorted(chunks, key=lambda item: item.citation_id):
            tokens = tokenize(chunk.text)
            if not tokens:
                continue
            for term in set(tokens):
                self.postings[term].append(len(corpus))
            corpus.append(tokens)
            self.chunks.append(chunk)
        self.model = BM25Plus(corpus) if corpus else None

    def search(self, question: str, top_k: int) -> list[Match]:
        """Return the highest BM25+ scores with stable tie ordering."""
        terms = list(dict.fromkeys(tokenize(question)))
        ids = sorted({index for term in terms for index in self.postings.get(term, ())})
        if self.model is None or not ids:
            return []
        scores = self.model.get_batch_scores(terms, ids)
        ranked = sorted(zip(ids, scores, strict=True), key=lambda item: -item[1])
        return [
            Match(self.chunks[index], float(score)) for index, score in ranked[:top_k]
        ]
