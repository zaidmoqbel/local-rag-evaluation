"""Compare completed experiment reports in a Markdown table."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

COLUMNS = (
    "hit_at_k",
    "recall_at_k",
    "mrr_at_k",
    "correctness",
    "faithfulness",
    "abstention_accuracy",
)


def compare(paths: Sequence[Path]) -> str:
    """Render runs in the requested order, showing each run's k."""
    lines = [
        "| Run | k | Questions | Hit@k | Recall@k | MRR@k | Correctness | Faithfulness | Abstention accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for path in paths:
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
            aggregate = run["aggregate"]
            scores = [aggregate[column] for column in COLUMNS]
            name = run["name"].replace("|", "\\|").replace("\n", " ")
            fields = [name, str(run["top_k"]), str(aggregate["question_count"])]
            fields.extend(
                "N/A" if score is None else f"{score:.3f}" for score in scores
            )
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise ValueError(f"Invalid evaluation report {path}: {error}") from error
        lines.append("| " + " | ".join(fields) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print a Markdown table for one or more report paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        print(compare(args.reports))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
