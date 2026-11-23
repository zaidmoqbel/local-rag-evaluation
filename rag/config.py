"""Validated configuration for the local pipeline."""

import os
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)


class Settings(BaseModel):
    """Reject unknown settings so misspellings fail early."""

    model_config = ConfigDict(extra="forbid")


class Paths(Settings):
    raw_data_dir: Path
    vector_store_dir: Path


class Chunking(Settings):
    strategy: Literal["fixed", "heading"]
    unit: Literal["character", "token"] = "character"
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> Self:
        """Require each window to advance."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class Embedding(Settings):
    model_name: str = Field(min_length=1)
    batch_size: int = Field(gt=0)
    device: str = Field(min_length=1)
    query_prefix: str = ""
    local_files_only: bool = True


class Retrieval(Settings):
    mode: Literal["dense", "bm25", "hybrid"]
    top_k: int = Field(gt=0)
    hybrid_weight: float = Field(ge=0, le=1)
    candidate_k: int = Field(default=20, gt=0)
    rrf_k: int = Field(default=60, gt=0)


class Reranking(Settings):
    """Local cross-encoder settings for optional candidate rescoring."""

    enabled: bool = False
    model_name: str = Field(default="BAAI/bge-reranker-base", min_length=1)
    device: str = Field(default="cpu", min_length=1)
    batch_size: int = Field(default=8, gt=0)
    max_length: int = Field(default=512, gt=0)
    local_files_only: bool = True


class LLM(Settings):
    base_url: HttpUrl
    model_name: str = Field(min_length=1)
    temperature: float = Field(ge=0, le=2)
    max_tokens: int = Field(gt=0)
    context_window: int = Field(ge=4096, le=8192)
    timeout_seconds: float = Field(gt=0)
    think: bool = False
    seed: int = Field(ge=0)

    @field_validator("base_url")
    @classmethod
    def validate_host(cls, url: HttpUrl) -> HttpUrl:
        """Keep document generation on this machine."""
        if url.host not in {"localhost", "127.0.0.1", "[::1]"}:
            raise ValueError("Ollama must use a loopback address")
        return url

    @model_validator(mode="after")
    def validate_budget(self) -> Self:
        """Reserve space for the question and passages."""
        if self.max_tokens >= self.context_window:
            raise ValueError("max_tokens must be smaller than context_window")
        return self


class Evaluation(Settings):
    """Paths and scoring options for a reproducible evaluation run."""

    dataset_path: Path = Path("eval/dataset/questions.jsonl")
    results_dir: Path = Path("eval/results")
    judge_model: str | None = Field(default=None, min_length=1)
    judge_attempts: int = Field(default=3, ge=1)
    judge_temperature: float = Field(default=0.0, ge=0, le=2)


class Web(Settings):
    """Loopback API location and client timeouts for the local interface."""

    api_url: HttpUrl = HttpUrl("http://127.0.0.1:8000")
    request_timeout_seconds: float = Field(default=360, gt=0)
    health_timeout_seconds: float = Field(default=2, gt=0)

    @field_validator("api_url")
    @classmethod
    def validate_host(cls, url: HttpUrl) -> HttpUrl:
        """Restrict document queries to a local API."""
        if url.host not in {"localhost", "127.0.0.1", "[::1]"}:
            raise ValueError("The API must use a loopback address")
        return url


class Config(Settings):
    paths: Paths
    chunking: Chunking
    embedding: Embedding
    retrieval: Retrieval
    llm: LLM
    evaluation: Evaluation = Field(default_factory=Evaluation)
    reranking: Reranking = Field(default_factory=Reranking)
    web: Web = Field(default_factory=Web)


def load_config(path: Path) -> Config:
    """Load YAML and resolve relative paths against its directory."""
    with path.open(encoding="utf-8") as file:
        config = Config.model_validate(yaml.safe_load(file))
    root = path.resolve().parent
    config.paths.raw_data_dir = (root / config.paths.raw_data_dir).resolve()
    config.paths.vector_store_dir = (root / config.paths.vector_store_dir).resolve()
    config.evaluation.dataset_path = (root / config.evaluation.dataset_path).resolve()
    config.evaluation.results_dir = (root / config.evaluation.results_dir).resolve()
    if config.paths.vector_store_dir.is_relative_to(config.paths.raw_data_dir):
        raise ValueError("vector_store_dir must be outside raw_data_dir")
    return config


def load_web_config() -> Config:
    """Load the API and UI config selected through RAG_CONFIG."""
    return load_config(Path(os.getenv("RAG_CONFIG", "config.yaml")))
