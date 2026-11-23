"""Publish complete persistent Chroma indexes and read their nearest neighbors."""

import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import islice
from uuid import uuid4

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from filelock import FileLock

from rag.config import Config
from rag.embed.encoder import Encoder
from rag.ingest.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Match:
    """A retrieved passage and its score from the last ranking stage."""

    chunk: Chunk
    score: float


class Store:
    """Serialize access to local Chroma and a pointer to the active collection."""

    def __init__(self, config: Config) -> None:
        self.config = config
        root = config.paths.vector_store_dir
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.active = root / "active-index"
        self.lock = FileLock(str(root / "index.lock"), timeout=30)
        self.metadata = {
            "schema": 2,
            "embedding_model": config.embedding.model_name,
            "chunking": config.chunking.model_dump_json(),
        }
        if config.chunking.strategy == "heading":
            self.metadata["heading_schema"] = 1

    def rebuild(self, chunks: Iterable[Chunk], encoder: Encoder) -> int:
        """Batch a new index and publish it only after successful ingestion."""
        with self.lock:
            client = self._client()
            previous = (
                self.active.read_text(encoding="utf-8")
                if self.active.exists()
                else None
            )
            name = f"passages-{uuid4().hex}"
            collection = client.create_collection(
                name,
                embedding_function=None,
                metadata=self.metadata,
                configuration={"hnsw": {"space": "cosine"}},
            )
            staged = self.active.with_suffix(".tmp")
            size = min(self.config.embedding.batch_size, client.get_max_batch_size())
            iterator = iter(chunks)
            count = 0
            try:
                while batch := list(islice(iterator, size)):
                    metadata = [asdict(chunk) for chunk in batch]
                    texts = [item.pop("text") for item in metadata]
                    collection.add(
                        ids=[chunk.citation_id for chunk in batch],
                        documents=texts,
                        embeddings=encoder.encode(texts),
                        metadatas=[
                            {
                                key: value
                                for key, value in item.items()
                                if value is not None
                            }
                            for item in metadata
                        ],
                    )
                    count += len(batch)
                if not count:
                    raise ValueError("No extractable text found in supported documents")
                staged.write_text(name, encoding="utf-8")
                staged.replace(self.active)
            except Exception:
                self._discard(client, name)
                raise
            if previous:
                self._discard(client, previous)
            return count

    @staticmethod
    def _discard(client: chromadb.ClientAPI, name: str) -> None:
        try:
            client.delete_collection(name)
        except Exception:
            logger.warning(
                "Could not remove inactive Chroma collection %s", name, exc_info=True
            )

    def _client(self) -> chromadb.ClientAPI:
        return chromadb.PersistentClient(
            path=str(self.root), settings=Settings(anonymized_telemetry=False)
        )

    def search(self, vector: list[float], top_k: int) -> list[Match]:
        """Return the top cosine matches from the active compatible index."""
        with self.lock:
            collection = self._collection()
            size = min(top_k, collection.count())
            if not size:
                return []
            result = collection.query(
                query_embeddings=[vector],
                n_results=size,
                include=["documents", "metadatas", "distances"],
            )
        assert result["documents"] and result["metadatas"] and result["distances"]
        return [
            Match(Chunk(text=text, **{"page": None, **metadata}), 1.0 - distance)
            for text, metadata, distance in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]

    def index_id(self) -> str:
        """Return the active index identifier."""
        if not self.active.is_file():
            raise ValueError("No index found. Run the index command first")
        return self.active.read_text(encoding="utf-8")

    def _collection(self) -> chromadb.Collection:
        try:
            collection = self._client().get_collection(
                self.index_id(), embedding_function=None
            )
        except NotFoundError as error:
            raise ValueError(
                "The active index is missing. Rebuild it with index"
            ) from error
        if collection.metadata != self.metadata:
            raise ValueError("Index settings changed. Rebuild the index")
        return collection

    def read_chunks(self) -> list[Chunk]:
        """Read indexed passages and source metadata without accessing raw files."""
        with self.lock:
            result = self._collection().get(include=["documents", "metadatas"])
        return [
            Chunk(text=text, **{"page": None, **metadata})
            for text, metadata in zip(
                result["documents"] or [], result["metadatas"] or [], strict=True
            )
        ]
