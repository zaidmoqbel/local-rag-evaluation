"""Run a benchmark against the local RAG pipeline and save a JSON report."""

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from tempfile import NamedTemporaryFile

import yaml
from filelock import Timeout

from eval.dataset.loader import Question, load_questions
from eval.judge import JudgeScores, correctly_abstained, score_answer
from eval.metrics import RetrievalScores, score_retrieval
from rag.config import Config, load_config
from rag.pipeline import ask_with_details
from rag.retrieve.search import Retriever

logger = logging.getLogger(__name__)
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


@dataclass(frozen=True)
class QuestionResult:
    """Inputs, pipeline outputs, and scores for one benchmark question."""

    question: Question
    answer: str
    citation_ids: list[str]
    retrieved: dict[str, float]
    abstention_reason: str | None
    retrieval: RetrievalScores | None
    judge: JudgeScores | None
    correctly_abstained: bool | None

    def to_json(self) -> dict[str, object]:
        """Serialize the record with explicit retrieval rank order."""
        record = asdict(self)
        record["question"] = self.question.model_dump(mode="json")
        record["judge"] = self.judge.model_dump(mode="json") if self.judge else None
        record["retrieved"] = [
            {"id": citation_id, "score": score}
            for citation_id, score in self.retrieved.items()
        ]
        return record


def evaluate(
    question: Question, config: Config, retriever: Retriever
) -> QuestionResult:
    """Run retrieval and generation once, then score the delivered result."""
    result = ask_with_details(question.question, config, retriever=retriever)
    ranked = {match.chunk.citation_id: match.score for match in result.matches}
    answerable = question.type != "unanswerable"
    return QuestionResult(
        question=question,
        answer=result.answer,
        citation_ids=[chunk.citation_id for chunk in result.citations],
        retrieved=ranked,
        abstention_reason=result.abstention,
        retrieval=score_retrieval(
            list(ranked), question.source_ids, config.retrieval.top_k
        )
        if answerable
        else None,
        judge=score_answer(question, result, config) if answerable else None,
        correctly_abstained=None if answerable else correctly_abstained(result),
    )


def aggregate(rows: list[QuestionResult]) -> dict[str, int | float | None]:
    """Macro-average answerable scores and report unanswerable abstention accuracy."""
    retrieval = [row.retrieval for row in rows if row.retrieval is not None]
    judged = [row.judge for row in rows if row.judge is not None]
    abstention = [
        row.correctly_abstained for row in rows if row.correctly_abstained is not None
    ]
    return {
        "question_count": len(rows),
        "answerable_count": len(retrieval),
        "unanswerable_count": len(abstention),
        "hit_at_k": mean(item.hit for item in retrieval) if retrieval else None,
        "recall_at_k": mean(item.recall for item in retrieval) if retrieval else None,
        "mrr_at_k": mean(item.reciprocal_rank for item in retrieval)
        if retrieval
        else None,
        "correctness": mean(item.correctness for item in judged) if judged else None,
        "faithfulness": mean(item.faithfulness for item in judged) if judged else None,
        "abstention_accuracy": mean(abstention) if abstention else None,
    }


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, delete=False, encoding="utf-8", mode="w"
    ) as file:
        temporary = Path(file.name)
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_experiment(config: Config, name: str) -> Path:
    """Score all dataset questions and atomically publish a completed report."""
    if not RUN_NAME.fullmatch(name) or Path(f"{name}.json").is_reserved():
        raise ValueError(
            "Use a valid run name with letters, digits, underscores, or hyphens"
        )
    questions = load_questions(config.evaluation.dataset_path)
    active_index = config.paths.vector_store_dir / "active-index"
    if not active_index.is_file():
        raise ValueError("No index found. Run the index command first")
    index_id = active_index.read_text(encoding="utf-8")
    retriever = Retriever(config)
    rows = []
    for question in questions:
        logger.info("Evaluating %s", question.id)
        rows.append(evaluate(question, config, retriever))
    if active_index.read_text(encoding="utf-8") != index_id:
        raise ValueError(
            "The index changed during evaluation. Run the experiment again"
        )
    output = config.evaluation.results_dir / f"{name}.json"
    _write_report(
        output,
        {
            "name": name,
            "top_k": config.retrieval.top_k,
            "index_id": index_id,
            "config": config.model_dump(mode="json"),
            "aggregate": aggregate(rows),
            "per_question": [row.to_json() for row in rows],
        },
    )
    return output


def main(argv: list[str] | None = None) -> int:
    """Run one named experiment from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        output = run_experiment(load_config(args.config), args.name)
    except (OSError, RuntimeError, Timeout, ValueError, yaml.YAMLError) as error:
        logger.error("%s", error)
        return 1
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
