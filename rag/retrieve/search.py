"""Select retrieval and reranking stages using the experiment config."""

from rag.config import Config
from rag.embed.encoder import Encoder
from rag.retrieve.dense import retrieve
from rag.retrieve.fusion import fuse
from rag.retrieve.reranker import Reranker
from rag.store.bm25 import BM25Index
from rag.store.chroma import Match, Store


class Retriever:
    """Reuse retrieval models and a lexical index across questions."""

    def __init__(self, config: Config, encoder: Encoder | None = None) -> None:
        self.config = config
        self.store = Store(config)
        self.encoder = encoder
        self.reranker: Reranker | None = None
        self.lexical: BM25Index | None = None
        self.lexical_generation: str | None = None

    def _sparse(self, question: str, top_k: int, generation: str) -> list[Match]:
        if self.lexical is None or self.lexical_generation != generation:
            self.lexical = BM25Index(self.store.read_chunks())
            self.lexical_generation = generation
        return self.lexical.search(question, top_k)

    def search(self, question: str) -> list[Match]:
        """Retrieve final top-k passages without invoking the LLM."""
        if not question.strip():
            raise ValueError("Question cannot be blank")
        settings = self.config.retrieval
        hybrid = settings.mode == "hybrid"
        use_dense = settings.mode == "dense" or (hybrid and settings.hybrid_weight > 0)
        use_sparse = settings.mode == "bm25" or (hybrid and settings.hybrid_weight < 1)
        rerank = self.config.reranking.enabled
        depth = (
            max(settings.top_k, settings.candidate_k)
            if hybrid or rerank
            else settings.top_k
        )
        self.store.index_id()
        if use_dense and self.encoder is None:
            self.encoder = Encoder(self.config.embedding)
        with self.store.lock:
            generation = self.store.index_id()
            dense = (
                retrieve(question, settings, self.encoder, self.store, top_k=depth)
                if use_dense and self.encoder is not None
                else []
            )
            sparse = self._sparse(question, depth, generation) if use_sparse else []
        matches = (
            fuse(dense, sparse, settings.hybrid_weight, settings.rrf_k)
            if hybrid
            else dense
            if use_dense
            else sparse
        )
        if rerank and matches:
            if self.reranker is None:
                self.reranker = Reranker(self.config.reranking)
            matches = self.reranker.rank(question, matches[:depth])
        return matches[: settings.top_k]
