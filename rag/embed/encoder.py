"""Local sentence-transformer embeddings and tokenizer offsets."""

from rag.config import Embedding


class Encoder:
    """Reuse one encoder for passages, queries, and token offsets."""

    def __init__(self, config: Embedding) -> None:
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.model = SentenceTransformer(
            config.model_name,
            device=config.device,
            local_files_only=config.local_files_only,
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Return normalized passage vectors."""
        return self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def query(self, question: str) -> list[float]:
        """Embed a query with its configured instruction prefix."""
        return self.encode([self.config.query_prefix + question])[0]

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Get character offsets from the embedding model's fast tokenizer."""
        if not getattr(self.model.tokenizer, "is_fast", False):
            raise ValueError("Token chunking requires a fast embedding tokenizer")
        encoded = self.model.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
            verbose=False,
        )
        return list(encoded["offset_mapping"])
