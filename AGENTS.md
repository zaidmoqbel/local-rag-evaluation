# AGENTS.md

## Project

Local RAG system with an evaluation suite. Everything runs on a single Windows machine
(RTX 5060 8 GB VRAM, 16 GB RAM). No cloud services required.

## Goals

- Answer questions over a local document collection with citations.
- Provide a reproducible evaluation pipeline that measures retrieval and answer quality.
- Make every design choice (chunking, embeddings, retrieval, reranking, LLM) configurable
  and comparable through experiments.

## Tech stack

- Python 3.11+, `uv` for dependency management
- Ollama for the LLM (default: qwen3:8b Q4) and, optionally, embeddings
- Embeddings: sentence-transformers (bge-small-en-v1.5 default; bge-m3 optional)
- Vector store: ChromaDB (persistent, local)
- Keyword search: rank_bm25
- Reranker (optional): bge-reranker-base via sentence-transformers
- API: FastAPI. UI: Streamlit
- Tests: pytest. Lint/format: ruff

## Constraints

- 8 GB VRAM: keep LLM context to 4K-8K tokens; never load multiple large models at once.
- Do not use LangChain or LlamaIndex. Implement the pipeline directly.
- All behavior controlled via `config.yaml`; no hardcoded paths, model names, or parameters.
- Fixed random seeds where applicable so experiments are reproducible.

## Code conventions

- Type hints everywhere, docstrings on public functions.
- Small modules with a single responsibility; no function over ~50 lines.
- Use `logging`, not `print`, outside CLI output.
- Every module gets at least one unit test. Run `ruff check` and `pytest` before finishing a task.

## Repo structure

```text
rag/
  config.py        # load/validate config.yaml
  ingest/          # loaders, parsers, chunkers
  embed/           # embedding backends
  store/           # vector store + BM25 index
  retrieve/        # dense, sparse, hybrid, reranker
  generate/        # prompt building, LLM client, citation formatting
  pipeline.py      # end-to-end query pipeline
  api.py           # FastAPI app
eval/
  dataset/         # questions.jsonl (question, reference_answer, source_ids, type)
  metrics.py       # hit@k, MRR, recall@k, answer scoring
  judge.py         # LLM-as-judge scoring
  run_experiment.py
  results/         # one JSON/CSV per experiment run
app/ui.py          # Streamlit UI
data/raw/          # source documents (gitignored)
tests/
config.yaml
README.md
```

## Working rules

- Work milestone by milestone. Finish and verify one before starting the next.
- After each milestone: summarize what you changed, how to run it, and how you tested it.
- If a requirement is ambiguous, state your assumption in the summary rather than guessing silently.
- Never modify files in `data/raw/` or `eval/dataset/` unless asked.
