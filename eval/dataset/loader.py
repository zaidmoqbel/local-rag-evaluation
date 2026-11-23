"""Load and validate evaluation questions from JSON Lines."""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    ValidationError,
    model_validator,
)

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SourceID = Annotated[str, StringConstraints(pattern=r"^[^\s:\[\]]+(?::[0-9]+)?$")]


class Question(BaseModel):
    """A benchmark question and its expected evidence."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: Text
    question: Text
    reference_answer: str
    source_ids: list[SourceID]
    type: Literal["factual", "multi_chunk", "unanswerable"]

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        """Require evidence and an answer for answerable rows only."""
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("source_ids must be unique")
        if self.type == "unanswerable":
            if self.source_ids:
                raise ValueError("unanswerable questions must have no source_ids")
        elif not self.source_ids or not self.reference_answer.strip():
            raise ValueError(
                "answerable questions require source_ids and a reference_answer"
            )
        return self


def load_questions(path: Path) -> list[Question]:
    """Return validated rows in file order, with line numbers on errors."""
    questions = []
    seen = set()
    with path.open(encoding="utf-8-sig") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                question = Question.model_validate_json(line)
            except ValidationError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            if question.id in seen:
                raise ValueError(
                    f"{path}:{number}: duplicate question ID {question.id!r}"
                )
            seen.add(question.id)
            questions.append(question)
    if not questions:
        raise ValueError(f"No evaluation questions in {path}")
    return questions
