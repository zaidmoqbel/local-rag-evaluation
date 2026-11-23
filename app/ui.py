"""Streamlit interface for querying the local FastAPI service."""

import httpx
import streamlit as st
from pydantic import ValidationError
from yaml import YAMLError

from rag.api_models import AskResponse
from rag.config import Web, load_web_config

SCORE_NAMES = {
    "cosine": "cosine similarity",
    "bm25": "BM25+",
    "rrf": "reciprocal rank fusion",
    "cross_encoder": "cross-encoder",
}


def request_answer(question: str, settings: Web) -> AskResponse:
    """Submit a question over loopback and validate the returned data."""
    try:
        with httpx.Client(
            timeout=settings.request_timeout_seconds, trust_env=False
        ) as client:
            response = client.post(
                f"{str(settings.api_url).rstrip('/')}/ask", json={"question": question}
            )
            response.raise_for_status()
            return AskResponse.model_validate_json(response.content)
    except httpx.HTTPStatusError as error:
        try:
            detail = error.response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        message = (
            detail
            if isinstance(detail, str)
            else f"API request failed ({error.response.status_code})."
        )
        raise RuntimeError(message) from error
    except httpx.TimeoutException as error:
        raise RuntimeError(
            "The local API took too long to answer. Retry or check its timeout settings."
        ) from error
    except httpx.HTTPError as error:
        raise RuntimeError(
            "Could not reach the local API. Make sure FastAPI is running."
        ) from error
    except ValidationError as error:
        raise RuntimeError("The local API returned an unexpected response.") from error


def show_answer(result: AskResponse) -> None:
    """Display the answer, cited chunks, and ranked retrieval scores."""
    st.subheader("Answer")
    st.text(result.answer)
    st.caption(f"{result.latency_ms / 1000:.2f} s")
    if result.citations:
        st.subheader("Cited sources")
        for source in result.citations:
            page = f" · page {source.page}" if source.page is not None else ""
            label = f"{source.filename}{page} · chunk {source.chunk_index}"
            with st.expander(label):
                st.text(
                    f"[{source.citation_id}] · characters {source.char_start}–{source.char_end}"
                )
                st.caption(f"Rank {source.rank} · score {source.score:.4f}")
                st.text(source.text)
    if result.retrieved:
        with st.expander(f"Retrieval scores · {SCORE_NAMES[result.score_type]}"):
            st.table(
                [
                    {
                        "Rank": chunk.rank,
                        "Source": chunk.citation_id,
                        "Score": chunk.score,
                    }
                    for chunk in result.retrieved
                ]
            )


def main() -> None:
    """Render a query form and keep the last answer for this browser session."""
    st.set_page_config(page_title="Local RAG", page_icon="📚")
    st.title("Ask your documents")
    try:
        settings = load_web_config().web
    except (OSError, ValueError, YAMLError) as error:
        st.error(f"Could not load the local configuration. {error}")
        return

    with st.form("ask"):
        question = st.text_area(
            "Question", placeholder="What do the documents say about…?"
        )
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted:
        st.session_state.pop("answer", None)
        if not question.strip():
            st.warning("Enter a question.")
        else:
            try:
                with st.spinner("Searching the index and generating an answer…"):
                    st.session_state["answer"] = request_answer(
                        question.strip(), settings
                    )
            except RuntimeError as error:
                st.error(str(error))
    if "answer" in st.session_state:
        show_answer(st.session_state["answer"])


if __name__ == "__main__":
    main()
