"""Orchestrate indexing and grounded question answering."""

from dataclasses import dataclass
from typing import Literal

from rag.config import Config
from rag.embed.encoder import Encoder
from rag.generate.citations import ABSTENTION, CitationAbstention, check_citations
from rag.generate.ollama import generate
from rag.generate.prompt import SYSTEM, build_prompt
from rag.ingest.chunker import Chunk, chunk_sources
from rag.ingest.loaders import load_sources
from rag.retrieve.search import Retriever
from rag.store.chroma import Match, Store


@dataclass(frozen=True)
class AnswerDetails:
    """Answer and diagnostics from one pass through the pipeline."""

    answer: str
    citations: list[Chunk]
    matches: list[Match]
    prompt_tokens: int = 0
    raw_output: str | None = None
    abstention: CitationAbstention | Literal["no_chunks_retrieved"] | None = None


def rebuild_index(config: Config) -> int:
    """Extract, chunk, embed, and publish a fresh index."""
    if not config.paths.raw_data_dir.is_dir():
        raise ValueError(
            f"Document directory does not exist: {config.paths.raw_data_dir}"
        )
    encoder = Encoder(config.embedding)
    sources = load_sources(config.paths.raw_data_dir)
    chunks = chunk_sources(sources, config.chunking, encoder.offsets)
    return Store(config).rebuild(chunks, encoder)


def retrieve_matches(
    question: str, config: Config, encoder: Encoder | None = None
) -> list[Match]:
    """Query the configured retrieval pipeline without calling the generator."""
    return Retriever(config, encoder).search(question)


def ask_with_details(
    question: str,
    config: Config,
    *,
    encoder: Encoder | None = None,
    retriever: Retriever | None = None,
) -> AnswerDetails:
    """Answer once and retain retrieval, prompt, and generation diagnostics."""
    matches = (
        retriever.search(question)
        if retriever is not None
        else retrieve_matches(question, config, encoder)
    )
    if not matches:
        return AnswerDetails(ABSTENTION, [], matches, abstention="no_chunks_retrieved")
    prompt, selected = build_prompt(
        question, (match.chunk for match in matches), config.llm
    )
    if not selected:
        return AnswerDetails(ABSTENTION, [], matches, abstention="no_chunks_retrieved")
    prompt_tokens = (len(SYSTEM.encode("utf-8")) + len(prompt.encode("utf-8")) + 3) // 4
    raw = generate(prompt, config.llm)
    answer, citations, reason = check_citations(raw, selected)
    return AnswerDetails(answer, citations, matches, prompt_tokens, raw, reason)


def ask(question: str, config: Config) -> tuple[str, list[Chunk]]:
    """Return the grounded response with its citations."""
    result = ask_with_details(question, config)
    return result.answer, result.citations
