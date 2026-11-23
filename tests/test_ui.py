import httpx
import pytest
from streamlit.testing.v1 import AppTest

from app.ui import request_answer
from rag.api_models import AskResponse, Passage
from rag.config import Web


@pytest.fixture
def answer() -> AskResponse:
    chunk = Passage(
        citation_id="guide.txt:0",
        doc_id="guide.txt",
        filename="guide.txt",
        page=None,
        chunk_index=0,
        char_start=0,
        char_end=10,
        text="Blue kiosk",
        rank=1,
        score=0.91,
    )
    return AskResponse(
        answer="The blue kiosk [guide.txt:0].",
        citations=[chunk],
        retrieved=[chunk],
        score_type="cosine",
        abstention_reason=None,
        latency_ms=500,
        request_id="test",
    )


def test_api_client(answer: AskResponse, monkeypatch: pytest.MonkeyPatch) -> None:
    original_client = httpx.Client

    def reply(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ask" and b"Where?" in request.content
        return httpx.Response(200, json=answer.model_dump())

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(reply), **kwargs
        ),
    )
    assert request_answer("Where?", Web()) == answer


def test_question_renders_citations_and_scores(
    answer: AskResponse, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.ui.request_answer", lambda *_: answer)

    def run_ui() -> None:
        from app.ui import main

        main()

    app = AppTest.from_function(run_ui, default_timeout=30).run()
    app.text_area[0].set_value("Where is the kiosk?")
    app.button[0].click().run()
    assert not app.exception
    assert app.text[0].value == answer.answer
    assert app.expander[0].label == "guide.txt · chunk 0"
    assert app.expander[1].label == "Retrieval scores · cosine similarity"
    assert app.table[0].value.iloc[0]["Score"] == pytest.approx(0.91)
