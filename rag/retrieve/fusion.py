"""Combine independent ranked lists using weighted reciprocal rank fusion."""

from collections.abc import Sequence

from rag.ingest.chunker import Chunk
from rag.store.chroma import Match


def fuse(
    dense: Sequence[Match],
    sparse: Sequence[Match],
    dense_weight: float,
    rank_constant: int,
) -> list[Match]:
    """Fuse by citation ID, keeping dense list order on equal fused scores."""
    passages: dict[str, Chunk] = {}
    scores: dict[str, float] = {}
    for ranking, weight in ((dense, dense_weight), (sparse, 1 - dense_weight)):
        if not weight:
            continue
        seen: set[str] = set()
        for rank, match in enumerate(ranking, 1):
            key = match.chunk.citation_id
            if key in seen:
                continue
            seen.add(key)
            passages[key] = match.chunk
            scores[key] = scores.get(key, 0.0) + weight / (rank_constant + rank)
    return [
        Match(passages[key], scores[key])
        for key in sorted(scores, key=scores.__getitem__, reverse=True)
    ]
