"""Dense cosine retrieval from the local vector index."""

from rag.config import Retrieval
from rag.embed.encoder import Encoder
from rag.store.chroma import Match, Store


def retrieve(
    question: str,
    settings: Retrieval,
    encoder: Encoder,
    store: Store,
    *,
    top_k: int | None = None,
) -> list[Match]:
    """Embed a question and retrieve top-k cosine matches."""
    return store.search(
        encoder.query(question), settings.top_k if top_k is None else top_k
    )
