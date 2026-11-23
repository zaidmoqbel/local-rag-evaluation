"""Command-line entry point for indexing and asking local documents."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import yaml
from filelock import Timeout

from rag.config import load_config

if TYPE_CHECKING:
    from rag.store.chroma import Match

logger = logging.getLogger(__name__)


def _print_matches(matches: list[Match], stream: TextIO) -> None:
    print(f"Retrieved chunks ({len(matches)})", file=stream)
    for rank, match in enumerate(matches, start=1):
        chunk = match.chunk
        page = f", page {chunk.page}" if chunk.page is not None else ""
        preview = " ".join(chunk.text.split())
        if len(preview) > 160:
            preview = preview[:157] + "..."
        print(
            f"{rank}. score={match.score:.6f} [{chunk.citation_id}] "
            f"{chunk.filename}{page} ({chunk.char_start}:{chunk.char_end})",
            file=stream,
        )
        print(f"   {preview}", file=stream)


def main(argv: list[str] | None = None) -> int:
    """Dispatch a CLI command and report actionable failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index", help="Rebuild the persistent index")
    query = commands.add_parser("ask", help="Ask a question over the index")
    query.add_argument("question")
    query.add_argument(
        "--debug", action="store_true", help="Show diagnostics on stderr"
    )
    search = commands.add_parser(
        "retrieve", help="Inspect retrieval without generation"
    )
    search.add_argument("question")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        config = load_config(args.config)
        from rag.pipeline import ask_with_details, rebuild_index, retrieve_matches

        if args.command == "index":
            print(f"Indexed {rebuild_index(config)} chunks.")
            return 0
        if args.command == "retrieve":
            _print_matches(retrieve_matches(args.question, config), sys.stdout)
            return 0
        result = ask_with_details(args.question, config)
    except (OSError, RuntimeError, Timeout, ValueError, yaml.YAMLError) as error:
        logger.error("%s", error)
        return 1
    if args.debug:
        _print_matches(result.matches, sys.stderr)
        print(
            f"Final prompt size (approx tokens): {result.prompt_tokens}",
            file=sys.stderr,
        )
        raw = (
            result.raw_output if result.raw_output is not None else "(model not called)"
        )
        print(f"Raw model output:\n{raw}", file=sys.stderr)
        if result.abstention:
            print(f"Abstention reason: {result.abstention}", file=sys.stderr)
    print(result.answer)
    if result.citations:
        print("\nSources")
    for chunk in result.citations:
        page = f", page {chunk.page}" if chunk.page is not None else ""
        print(
            f"[{chunk.citation_id}] {chunk.filename}{page} ({chunk.char_start}:{chunk.char_end})"
        )
        print(chunk.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
