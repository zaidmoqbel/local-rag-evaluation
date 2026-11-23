"""Request and response schemas shared by the API and local UI."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints


class AskRequest(BaseModel):
    """A question over the currently indexed corpus."""

    model_config = ConfigDict(extra="forbid")
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Passage(BaseModel):
    """A ranked chunk with its source location and final retrieval score."""

    citation_id: str
    doc_id: str
    filename: str
    page: int | None
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    rank: int
    score: float


class AskResponse(BaseModel):
    """An answer, cited passages, ranked candidates, and request timing."""

    answer: str
    citations: list[Passage]
    retrieved: list[Passage]
    score_type: Literal["cosine", "bm25", "rrf", "cross_encoder"]
    abstention_reason: str | None
    latency_ms: float
    request_id: str


class IndexResponse(BaseModel):
    """Number of chunks published by a completed rebuild."""

    indexed_chunks: int


class HealthResponse(BaseModel):
    """API liveness and lightweight readiness checks."""

    status: Literal["ok", "degraded"]
    index_present: bool
    ollama_reachable: bool
    model_available: bool
