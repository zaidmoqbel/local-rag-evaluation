import json
import logging
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from test_config import CONFIG

from rag.api import create_app
from rag.config import Config, load_config
from rag.embed.encoder import Encoder
from rag.generate.citations import ABSTENTION
from rag.generate.ollama import OllamaError


@pytest.fixture
def api_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    path = tmp_path / "config.yaml"
    path.write_bytes(CONFIG.read_bytes())
    config = load_config(path)
    config.paths.raw_data_dir.mkdir(parents=True)
    (config.paths.raw_data_dir / "guide.txt").write_text(
        "The locker code is 7319.", encoding="utf-8"
    )
    monkeypatch.setattr(
        Encoder, "__init__", lambda self, config: setattr(self, "config", config)
    )
    monkeypatch.setattr(
        Encoder, "encode", lambda self, texts: [[1.0, 0.0] for _ in texts]
    )
    return config


def test_index_ask_health_and_structured_logging(
    api_config: Config,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def model_tags(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200, json={"models": [{"name": api_config.llm.model_name}]}
        )

    original_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(model_tags), **kwargs
        ),
    )
    monkeypatch.setattr(
        "rag.pipeline.generate", lambda *_: "The code is 7319 [guide.txt:0]."
    )
    caplog.set_level(logging.INFO, logger="uvicorn.error.rag")
    with TestClient(create_app(api_config)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "degraded",
            "index_present": False,
            "ollama_reachable": True,
            "model_available": True,
        }
        assert client.post("/ask", json={"question": "  "}).status_code == 422
        missing = client.post("/ask", json={"question": "code?"})
        assert missing.status_code == 409
        assert "No index found" in missing.json()["detail"]

        indexed = client.post("/index")
        assert indexed.json() == {"indexed_chunks": 1}
        assert client.get("/health").json()["status"] == "ok"
        queried = client.post("/ask", json={"question": "What is the locker code?"})
        assert queried.status_code == 200
        result = queried.json()
        assert result["score_type"] == "cosine"
        assert result["citations"] == result["retrieved"]
        assert result["citations"][0]["citation_id"] == "guide.txt:0"
        assert result["citations"][0]["text"] == "The locker code is 7319."
        assert result["citations"][0]["score"] == pytest.approx(1.0)
        assert result["latency_ms"] >= 0

    events = [
        json.loads(event.message)
        for event in caplog.records
        if event.name == "uvicorn.error.rag"
    ]
    assert [event["status"] for event in events] == [409, 200]
    assert events[1]["request_id"] == result["request_id"]
    assert events[1]["latency_ms"] == result["latency_ms"]
    assert "7319" not in json.dumps(events) and "locker code" not in json.dumps(events)


def test_abstention_and_ollama_failure(
    api_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_: object) -> str:
        raise OllamaError("private upstream error")

    with TestClient(create_app(api_config)) as client:
        assert client.post("/index").status_code == 200
        monkeypatch.setattr("rag.pipeline.generate", lambda *_: ABSTENTION)
        result = client.post("/ask", json={"question": "unknown?"}).json()
        assert result["abstention_reason"] == "model_said_unknown"
        assert result["citations"] == [] and len(result["retrieved"]) == 1

        monkeypatch.setattr("rag.pipeline.generate", fail)
        failed = client.post("/ask", json={"question": "unknown?"})
        assert failed.status_code == 502
        assert "private upstream error" not in failed.text
