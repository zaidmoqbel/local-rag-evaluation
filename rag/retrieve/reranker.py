"""Cross-encoder rescoring for a small retrieval candidate pool."""

from rag.config import Reranking
from rag.store.chroma import Match


class Reranker:
    """Load a local cross-encoder once and reuse it across queries."""

    def __init__(self, settings: Reranking) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            settings.model_name,
            device=settings.device,
            max_length=settings.max_length,
            local_files_only=settings.local_files_only,
        )
        self.batch_size = settings.batch_size

    def rank(self, question: str, matches: list[Match]) -> list[Match]:
        """Score query–passage pairs and sort by decreasing relevance."""
        if not matches:
            return []
        pairs = [(question, match.chunk.text) for match in matches]
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        rescored = [
            Match(match.chunk, float(score))
            for match, score in zip(matches, scores, strict=True)
        ]
        return sorted(rescored, key=lambda match: -match.score)
