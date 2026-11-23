import json
from pathlib import Path

import httpx
import pytest
from test_config import CONFIG

from eval.compare import main as compare_main
from eval.dataset.loader import load_questions
from eval.judge import score_answer
from eval.metrics import RetrievalScores, score_retrieval
from eval.run_experiment import main as run_main
from rag.config import load_config
from rag.generate.citations import ABSTENTION
from rag.ingest.chunker import Chunk
from rag.pipeline import AnswerDetails
from rag.store.chroma import Match

SAMPLE = Path(__file__).parent / "fixtures" / "eval-questions.jsonl"


def test_dataset_schema_and_duplicate_ids(tmp_path: Path) -> None:
    questions = load_questions(SAMPLE)
    assert len(questions) == 5
    assert {question.type for question in questions} == {
        "factual",
        "multi_chunk",
        "unanswerable",
    }
    dataset = tmp_path / "bad.jsonl"
    row = questions[0].model_dump_json()
    dataset.write_text(f"{row}\n{row}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=":2: duplicate question ID"):
        load_questions(dataset)
    dataset.write_text(row.replace('"operations.md:0"', "12"), encoding="utf-8")
    with pytest.raises(ValueError, match=":1:"):
        load_questions(dataset)


def test_retrieval_matches_exact_and_document_labels() -> None:
    ranked = ["irrelevant:0", "doc-a:1", "doc-a:2", "doc-b:0"]
    expected = ["doc-a", "doc-b:0"]
    assert score_retrieval(ranked, expected, 1) == RetrievalScores(0, 0, 0)
    assert score_retrieval(ranked, expected, 3) == RetrievalScores(1, 0.5, 0.5)
    assert score_retrieval(ranked, expected, 4) == RetrievalScores(1, 1, 0.5)
    assert score_retrieval(ranked, ["doc-a:0"], 4) == RetrievalScores(0, 0, 0)
    assert score_retrieval([], expected, 4) == RetrievalScores(0, 0, 0)


def test_judge_retries_invalid_json_and_uses_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG)
    config.evaluation.judge_attempts = 4
    question = load_questions(SAMPLE)[0]
    chunk = Chunk("operations.md", "operations.md", None, 0, 0, 4, "7319")
    result = AnswerDetails("7319 [operations.md:0]", [chunk], [])
    responses = iter(
        [
            "{",
            '{"correctness":"5","faithfulness":5}',
            '{"correctness":5,"faithfulness":6,"extra":0}',
            '{"correctness":5,"faithfulness":4}',
        ]
    )
    original = httpx.Client

    def reply(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["format"]["additionalProperties"] is False
        assert body["format"]["properties"]["correctness"]["type"] == "integer"
        assert "7319" in body["prompt"] and body["keep_alive"] == 0
        return httpx.Response(200, json={"response": next(responses, "invalid")})

    def client(**kwargs: object) -> httpx.Client:
        return original(transport=httpx.MockTransport(reply), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    scores = score_answer(question, result, config)
    assert (scores.correctness, scores.faithfulness) == (5, 4)
    with pytest.raises(ValueError, match="after 4 attempts"):
        score_answer(question, result, config)


def test_runner_and_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            "eval/dataset/questions.jsonl", "questions.jsonl"
        ),
        encoding="utf-8",
    )
    (tmp_path / "questions.jsonl").write_bytes(SAMPLE.read_bytes())
    index = tmp_path / "vector_store"
    index.mkdir()
    (index / "active-index").write_text("test-index", encoding="utf-8")
    chunk = Chunk("operations.md", "operations.md", None, 0, 0, 4, "7319")
    cases = {question.question: question.type for question in load_questions(SAMPLE)}

    def answer(question: str, *_: object, **__: object) -> AnswerDetails:
        if cases[question] == "unanswerable":
            return AnswerDetails(ABSTENTION, [], [], abstention="model_said_unknown")
        return AnswerDetails("7319 [operations.md:0]", [chunk], [Match(chunk, 1)])

    monkeypatch.setattr("eval.run_experiment.Retriever", lambda *_: object())
    monkeypatch.setattr("eval.run_experiment.ask_with_details", answer)
    monkeypatch.setattr(
        "eval.judge.generate", lambda *_, **__: '{"correctness":5,"faithfulness":4}'
    )
    assert run_main(["--config", str(config_path), "--name", "baseline"]) == 0
    report = tmp_path / "eval" / "results" / "baseline.json"
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["aggregate"] == {
        "question_count": 5,
        "answerable_count": 4,
        "unanswerable_count": 1,
        "hit_at_k": 0.25,
        "recall_at_k": 0.25,
        "mrr_at_k": 0.25,
        "correctness": 5,
        "faithfulness": 4,
        "abstention_accuracy": 1,
    }
    assert result["config"]["retrieval"]["top_k"] == 5
    assert result["per_question"][-1]["retrieval"] is None
    assert result["per_question"][-1]["judge"] is None
    assert compare_main([str(report), str(report)]) == 0
    assert capsys.readouterr().out.count("| baseline | 5 | 5 | 0.250 |") == 2
