"""Compute ranked retrieval metrics over annotated evidence IDs."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalScores:
    """Per-question contributions to hit@k, recall@k, and MRR@k."""

    hit: float
    recall: float
    reciprocal_rank: float


def score_retrieval(
    ranked_ids: Sequence[str], source_ids: Sequence[str], k: int
) -> RetrievalScores:
    """Match exact chunk IDs or their document IDs within the first k ranks."""
    if k < 1 or not source_ids:
        raise ValueError(
            "Retrieval scoring requires positive k and at least one source ID"
        )
    expected = set(source_ids)
    found: set[str] = set()
    reciprocal_rank = 0.0
    for rank, chunk_id in enumerate(ranked_ids[:k], 1):
        matches = expected.intersection({chunk_id, chunk_id.rsplit(":", 1)[0]})
        if matches and not reciprocal_rank:
            reciprocal_rank = 1.0 / rank
        found.update(matches)
    return RetrievalScores(
        float(bool(found)), len(found) / len(expected), reciprocal_rank
    )
