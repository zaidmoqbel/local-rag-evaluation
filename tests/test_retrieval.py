import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
from test_config import CONFIG

from eval.run_experiment import run_experiment
from rag.config import load_config
from rag.embed.encoder import Encoder
from rag.ingest.chunker import Chunk
from rag.pipeline import rebuild_index
from rag.retrieve.fusion import fuse
from rag.retrieve.search import Retriever
from rag.store.bm25 import BM25Index
from rag.store.chroma import Match


def passage(name: str, text: str) -> Chunk:
    return Chunk(name, name, None, 0, 0, len(text), text)


def test_bm25_returns_keyword_matches_and_skips_empty_queries() -> None:
    numbers = passage("a.txt", "Array array array")
    regex = passage("b.txt", "regex FULL_MATCH pattern")
    index = BM25Index([numbers, regex, passage("c.txt", "!!!")])
    assert [match.chunk for match in index.search("full_match", 5)] == [regex]
    assert [match.chunk for match in index.search("ARRAY", 5)] == [numbers]
    assert index.search("unseen", 5) == index.search("...", 5) == []


def test_weighted_fusion_deduplicates_and_uses_ranks() -> None:
    first = Match(passage("first", "one"), 0.1)
    shared = Match(passage("shared", "two"), -0.2)
    last = Match(passage("last", "three"), 999)
    dense = [first, shared]
    sparse = [last, shared]
    result = fuse(dense, sparse, 0.75, 10)
    assert [match.chunk for match in result] == [shared.chunk, first.chunk, last.chunk]
    assert [match.score for match in result] == pytest.approx(
        [1 / 12, 0.75 / 11, 0.25 / 11]
    )
    assert [match.chunk for match in fuse(dense, sparse, 0, 10)] == [
        last.chunk,
        shared.chunk,
    ]


@pytest.fixture
def experiment_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_bytes(CONFIG.read_bytes())
    config = load_config(cfg_path)
    raw = config.paths.raw_data_dir
    raw.mkdir(parents=True)
    (raw / "a.md").write_text(
        "# Main\nneedle needle needle\n\nAppendix\n========\nunneeded", encoding="utf-8"
    )
    (raw / "b.txt").write_text("needle decoy", encoding="utf-8")
    dataset = config.evaluation.dataset_path
    dataset.parent.mkdir(parents=True)
    dataset.write_text(
        json.dumps(
            {
                "id": "any",
                "question": "needle",
                "reference_answer": "needle",
                "source_ids": ["a.md"],
                "type": "factual",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def encode(self: Encoder, texts: list[str]) -> list[list[float]]:
        return [[1, 0] if "needle" in item else [0, 1] for item in texts]

    monkeypatch.setattr(
        Encoder, "__init__", lambda self, config: setattr(self, "config", config)
    )
    monkeypatch.setattr(Encoder, "encode", encode)
    return cfg_path


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
@pytest.mark.parametrize("strategy", ["fixed", "heading"])
@pytest.mark.parametrize("reranking", [False, True])
def test_experiment_matrix(
    experiment_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: Literal["dense", "bm25", "hybrid"],
    strategy: Literal["fixed", "heading"],
    reranking: bool,
) -> None:
    config = load_config(experiment_config)
    config.retrieval.mode = mode
    config.retrieval.top_k = 1
    config.chunking.strategy = strategy
    config.reranking.enabled = reranking
    assert rebuild_index(config) == (3 if strategy == "heading" else 2)

    if mode == "bm25":

        def unexpected(*_: object) -> None:
            pytest.fail("BM25 query loaded the embedding model")

        monkeypatch.setattr(Encoder, "__init__", unexpected)

    class FakeCrossEncoder:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            assert model_name == config.reranking.model_name
            assert kwargs["device"] == "cpu"

        def predict(self, pairs: list[tuple[str, str]], **_: object) -> list[float]:
            assert len(pairs) >= 2
            return [10.0 if "Main" in text else -10.0 for _, text in pairs]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    monkeypatch.setattr("rag.pipeline.generate", lambda *_: "Needle [a.md:0].")
    monkeypatch.setattr(
        "eval.judge.generate", lambda *_, **__: '{"correctness":5,"faithfulness":5}'
    )

    report = run_experiment(config, "example")
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["config"]["retrieval"]["mode"] == mode
    assert result["config"]["reranking"]["enabled"] == reranking
    retrieved = result["per_question"][0]["retrieved"]
    assert len(retrieved) == 1
    if reranking:
        assert retrieved == [{"id": "a.md:0", "score": 10.0}]


def test_bm25_cache_tracks_index_rebuild(experiment_config: Path) -> None:
    config = load_config(experiment_config)
    config.retrieval.mode = "bm25"
    rebuild_index(config)
    retriever = Retriever(config)
    assert retriever.search("needle")
    (config.paths.raw_data_dir / "a.md").write_text("replacement", encoding="utf-8")
    rebuild_index(config)
    assert retriever.search("replacement")[0].chunk.doc_id == "a.md"
