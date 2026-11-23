from pathlib import Path

import httpx
import pymupdf
import pytest
from test_config import CONFIG

from rag.cli import main
from rag.config import load_config
from rag.embed.encoder import Encoder
from rag.generate.citations import ABSTENTION
from rag.generate.ollama import generate
from rag.generate.prompt import build_prompt
from rag.ingest.chunker import Chunk
from rag.ingest.loaders import load_sources
from rag.pipeline import ask, rebuild_index
from rag.retrieve.dense import retrieve
from rag.store.chroma import Match, Store


@pytest.fixture
def sample_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_bytes(CONFIG.read_bytes())
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "guide.md").write_text(
        "# Access\nThe access code is violet.", encoding="utf-8"
    )
    (raw / "notes.txt").write_text("The break room is upstairs.", encoding="utf-8")
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "The parking lot closes at noon.")
        pdf.save(raw / "parking.pdf")
    return path


def test_loads_pdf_markdown_and_text(sample_config: Path) -> None:
    sources = list(load_sources(load_config(sample_config).paths.raw_data_dir))
    assert [source.page for source in sources] == [None, None, 1]
    assert "parking lot" in sources[2].text


def test_cli_and_rebuild_failure_preserves_index(
    sample_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def encode(self: Encoder, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "violet" in text else [0.0, 1.0] for text in texts]

    def respond(prompt: str, settings: object) -> str:
        assert "violet" in prompt
        return "The code is violet [guide.md:0]."

    monkeypatch.setattr(
        Encoder, "__init__", lambda self, config: setattr(self, "config", config)
    )
    monkeypatch.setattr(Encoder, "encode", encode)
    monkeypatch.setattr("rag.pipeline.generate", respond)
    args = ["--config", str(sample_config)]
    assert main([*args, "index"]) == 0
    config = load_config(sample_config)
    store = Store(config)
    with pytest.raises(ValueError, match="No extractable text"):
        store.rebuild([], Encoder(config.embedding))
    matches = retrieve(
        "violet code", config.retrieval, Encoder(config.embedding), store
    )
    assert matches[0].score == pytest.approx(1)
    assert matches[0].chunk.filename == "guide.md"
    capsys.readouterr()
    with monkeypatch.context() as mock:
        mock.setattr(
            "rag.pipeline.generate", lambda *_: pytest.fail("retrieve called Ollama")
        )
        assert main([*args, "retrieve", "violet code"]) == 0
    output = capsys.readouterr().out
    assert "1. score=1.000000 [guide.md:0] guide.md (0:36)" in output
    assert "The access code is violet." in output
    assert main([*args, "ask", "What is the violet code?"]) == 0
    assert "[guide.md:0] guide.md" in capsys.readouterr().out
    assert rebuild_index(config) == 3
    assert ask("violet code", config)[1][0].filename == "guide.md"


def test_prompt_and_ollama_request(
    sample_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_config(sample_config).llm
    big = Chunk("big", "big", None, 0, 0, 8192, "a" * 8192)
    small = Chunk("small", "small", 1, 0, 0, 4, "fact")
    prompt, used = build_prompt("question", [big, small], settings)
    assert used == [small]
    original_client = httpx.Client

    def reply(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        assert b"num_ctx" in request.content and b"fact" in request.content
        return httpx.Response(200, json={"response": "Fact [small:0]."})

    def client(**kwargs: object) -> httpx.Client:
        return original_client(transport=httpx.MockTransport(reply), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    assert generate(prompt, settings) == "Fact [small:0]."


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (None, "no_chunks_retrieved"),
        (ABSTENTION + ".", "model_said_unknown"),
        ("Fact.", "no_citations"),
        ("Fact [missing.md:0].", "invalid_citation_id"),
        ("Fact [guide.md:0].", None),
    ],
)
def test_ask_debug(
    sample_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    raw: str | None,
    reason: str | None,
) -> None:
    passage = Chunk("guide.md", "guide.md", None, 0, 0, 4, "fact")
    matches = [Match(passage, 0.75)] if raw is not None else []
    monkeypatch.setattr("rag.pipeline.retrieve_matches", lambda *_: matches)

    def respond(*_: object) -> str:
        assert raw is not None, "Ollama must not run without context"
        return raw

    monkeypatch.setattr("rag.pipeline.generate", respond)
    assert main(["--config", str(sample_config), "ask", "--debug", "question"]) == 0
    output, debug = capsys.readouterr()
    assert "Retrieved chunks" in debug
    assert "Final prompt size (approx tokens):" in debug
    if raw is None:
        assert "tokens): 0" in debug and "(model not called)" in debug
    else:
        assert "score=0.750000" in debug and f"Raw model output:\n{raw}" in debug
    if reason:
        assert f"Abstention reason: {reason}" in debug and output.strip() == ABSTENTION
    else:
        assert "Abstention reason" not in debug and "[guide.md:0]" in output
