"""FastAPI entry point for running the document assistant locally."""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from threading import Lock
from time import perf_counter
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException
from filelock import Timeout

from rag.api_models import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IndexResponse,
    Passage,
)
from rag.config import Config, load_web_config
from rag.generate.ollama import OllamaError
from rag.pipeline import AnswerDetails, ask_with_details, rebuild_index
from rag.retrieve.search import Retriever

logger = logging.getLogger("uvicorn.error.rag")


class RagService:
    """Serialize model work and retain query models between requests."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.lock = Lock()
        self.retriever: Retriever | None = None

    def ask(self, question: str) -> AnswerDetails:
        """Answer a question through a shared retriever."""
        with self.lock:
            if self.retriever is None:
                self.retriever = Retriever(self.config)
            return ask_with_details(question, self.config, retriever=self.retriever)

    def index(self) -> int:
        """Rebuild from configured files and release previously cached query models."""
        with self.lock:
            self.retriever = None
            return rebuild_index(self.config)

    def health(self) -> HealthResponse:
        """Probe Ollama and the active-index pointer without loading models."""
        config = self.config
        index_present = (config.paths.vector_store_dir / "active-index").is_file()
        reachable = available = False
        try:
            with httpx.Client(
                timeout=config.web.health_timeout_seconds, trust_env=False
            ) as client:
                response = client.get(
                    f"{str(config.llm.base_url).rstrip('/')}/api/tags"
                )
                response.raise_for_status()
                models = response.json()["models"]
                name = config.llm.model_name
                if ":" not in name.rsplit("/", 1)[-1]:
                    name += ":latest"
                available = any(model["name"] == name for model in models)
                reachable = True
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            pass
        return HealthResponse(
            status="ok" if index_present and available else "degraded",
            index_present=index_present,
            ollama_reachable=reachable,
            model_available=available,
        )


def _elapsed(start: float) -> float:
    return round((perf_counter() - start) * 1000, 2)


def _answer_response(
    result: AnswerDetails, config: Config, start: float, request_id: str
) -> AskResponse:
    retrieved = [
        Passage(
            **asdict(match.chunk),
            citation_id=match.chunk.citation_id,
            rank=rank,
            score=match.score,
        )
        for rank, match in enumerate(result.matches, 1)
    ]
    by_id = {chunk.citation_id: chunk for chunk in retrieved}
    score_types = {"dense": "cosine", "bm25": "bm25", "hybrid": "rrf"}
    return AskResponse(
        answer=result.answer,
        citations=[by_id[chunk.citation_id] for chunk in result.citations],
        retrieved=retrieved,
        score_type="cross_encoder"
        if config.reranking.enabled
        else score_types[config.retrieval.mode],
        abstention_reason=result.abstention,
        latency_ms=_elapsed(start),
        request_id=request_id,
    )


def _handle_ask(service: RagService, body: AskRequest) -> AskResponse:
    start = perf_counter()
    request_id = uuid4().hex
    status = 500
    payload = None
    try:
        result = service.ask(body.question)
        payload = _answer_response(result, service.config, start, request_id)
        status = 200
        return payload
    except OllamaError as error:
        status = 502
        raise HTTPException(
            status, "Ollama could not answer. Check that the model is running."
        ) from error
    except Timeout as error:
        status = 503
        raise HTTPException(
            status, "The index is busy. Retry after indexing finishes."
        ) from error
    except ValueError as error:
        status = 409
        raise HTTPException(status, str(error)) from error
    except (OSError, RuntimeError) as error:
        status = 503
        raise HTTPException(
            status, "Unable to load the index or retrieval model."
        ) from error
    finally:
        logger.info(
            "%s",
            json.dumps(
                {
                    "event": "rag_query",
                    "request_id": request_id,
                    "status": status,
                    "latency_ms": payload.latency_ms if payload else _elapsed(start),
                    "retrieval_mode": service.config.retrieval.mode,
                    "retrieved_count": len(payload.retrieved) if payload else None,
                    "citation_count": len(payload.citations) if payload else None,
                    "abstention_reason": payload.abstention_reason if payload else None,
                }
            ),
        )


def create_app(config: Config | None = None) -> FastAPI:
    """Create an API using the supplied config or the shared web config."""
    service: RagService

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        nonlocal service
        service = RagService(config if config is not None else load_web_config())
        yield

    app = FastAPI(title="Local RAG", lifespan=lifespan)

    @app.get("/health")
    def health() -> HealthResponse:
        return service.health()

    @app.post("/ask")
    def ask(body: AskRequest) -> AskResponse:
        return _handle_ask(service, body)

    @app.post("/index")
    def index() -> IndexResponse:
        try:
            return IndexResponse(indexed_chunks=service.index())
        except (Timeout, OSError, RuntimeError) as error:
            raise HTTPException(
                503, "Unable to rebuild the index. Check the files and model cache."
            ) from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    return app


app = create_app()
