"""Streamlit inspector UI — the differentiator surface for guideline-gpt.

Layout: a two-column split with chat on the left and the five-stage pipeline
inspector on the right. The sidebar exposes settings (LLM provider/model,
top_k sliders, timing toggle, clear chat) and a document upload + re-ingest
workflow so users can run the system against their own PDFs without leaving
the browser.

This module contains no business logic. It composes Streamlit components and
calls :class:`QueryPipeline`.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import streamlit as st

from guideline_gpt.config import Settings, get_settings
from guideline_gpt.ingestion.pipeline import ingest as run_ingest
from guideline_gpt.logging_setup import configure_logging
from guideline_gpt.pipeline import QueryPipeline
from guideline_gpt.retrieval.bm25 import BM25Retriever
from guideline_gpt.retrieval.rerank import CrossEncoderReranker
from guideline_gpt.retrieval.vector import VectorRetriever
from guideline_gpt.ui.components.chat import (
    ChatMessage,
    render_assistant_message,
    render_history,
    render_user_message,
)
from guideline_gpt.ui.components.pipeline_inspector import (
    render_empty_state,
    render_inspector,
)

# ----------------------------- session helpers -----------------------------


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("show_timing", False)
    st.session_state.setdefault("pending_prompt", None)
    base = get_settings()
    st.session_state.setdefault("provider", base.llm_provider)
    st.session_state.setdefault("anthropic_model", base.anthropic_model)
    st.session_state.setdefault("openai_model", base.openai_model)
    st.session_state.setdefault("retrieval_top_k", base.retrieval_top_k)
    st.session_state.setdefault("rerank_top_k", base.rerank_top_k)


def _current_settings() -> Settings:
    """Build a Settings instance reflecting the sidebar overrides."""
    base = get_settings()
    return base.model_copy(
        update={
            "llm_provider": st.session_state.provider,
            "anthropic_model": st.session_state.anthropic_model,
            "openai_model": st.session_state.openai_model,
            "retrieval_top_k": st.session_state.retrieval_top_k,
            "rerank_top_k": st.session_state.rerank_top_k,
        }
    )


# Heavy components (Chroma client, BM25 pickle, cross-encoder weights) are
# invariant w.r.t. LLM provider/top_k, so we cache them once per process. The
# LLM client is cheap and re-created from current settings on every query.
@st.cache_resource(show_spinner="Loading retrievers and reranker…")
def _get_components(
    persist_dir: str,
    embedding_model: str,
    openai_api_key: str,
    collection_name: str,
    rerank_model: str,
) -> tuple[VectorRetriever, BM25Retriever, CrossEncoderReranker]:
    cfg = get_settings().model_copy(
        update={
            "chroma_persist_dir": Path(persist_dir),
            "embedding_model": embedding_model,
            "openai_api_key": openai_api_key,
            "collection_name": collection_name,
            "rerank_model": rerank_model,
        }
    )
    return VectorRetriever(cfg), BM25Retriever(cfg), CrossEncoderReranker(cfg)


def _build_pipeline(settings: Settings) -> QueryPipeline:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required (it powers embeddings).")
    vector, bm25, reranker = _get_components(
        str(settings.chroma_persist_dir),
        settings.embedding_model,
        settings.openai_api_key,
        settings.collection_name,
        settings.rerank_model,
    )
    return QueryPipeline(settings, vector=vector, bm25=bm25, reranker=reranker)


# ------------------------------- sidebar -----------------------------------


def _render_sidebar() -> None:
    """Sidebar holds only the runtime knobs. Document management lives up top."""
    with st.sidebar:
        st.subheader("Settings")
        st.caption("Changes apply to the **next** query.")

        st.radio(
            "Provider",
            options=["anthropic", "openai"],
            key="provider",
            horizontal=True,
        )
        if st.session_state.provider == "anthropic":
            st.text_input("Model", key="anthropic_model")
        else:
            st.text_input("Model", key="openai_model")

        st.slider("retrieval_top_k", 5, 50, key="retrieval_top_k")
        st.slider("rerank_top_k", 1, 15, key="rerank_top_k")
        st.toggle("Show timing breakdown", key="show_timing")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("latest_trace", None)
            st.rerun()


def _render_documents_bar(settings: Settings) -> None:
    """Compact corpus-management strip at the top of the main area."""
    docs = sorted(settings.documents_dir.glob("*.pdf")) if settings.documents_dir.exists() else []
    label = f"Corpus — {len(docs)} PDF(s) indexed-ready"
    with st.expander(label, expanded=False):
        uploaded = st.file_uploader(
            "Upload PDFs",
            type="pdf",
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            settings.documents_dir.mkdir(parents=True, exist_ok=True)
            for upload in uploaded:
                dest = settings.documents_dir / upload.name
                dest.write_bytes(upload.getbuffer())
            st.success(f"Saved {len(uploaded)} file(s) to `{settings.documents_dir}/`.")

        cols = st.columns([4, 1])
        with cols[0]:
            if docs:
                st.caption(" · ".join(p.name for p in docs[:6]) + (" …" if len(docs) > 6 else ""))
            else:
                st.caption("No PDFs yet — upload some above, or drop them into `documents/`.")
        with cols[1]:
            if st.button("Re-ingest", use_container_width=True, disabled=not docs):
                with st.spinner("Embedding and indexing…"):
                    try:
                        report = run_ingest(settings.documents_dir, settings)
                    except Exception as exc:  # noqa: BLE001 - surface failure to the user
                        st.error(f"Ingest failed: {exc}")
                        return
                _get_components.clear()
                st.success(
                    f"{report.files} file(s), {report.pages} page(s) → "
                    f"{report.chunks:,} chunks in {report.elapsed_seconds}s."
                )


# ------------------------------- query flow --------------------------------


def _set_pending(prompt: str) -> None:
    """Schedule a prompt to be processed on the next rerun (used by example buttons)."""
    st.session_state.pending_prompt = prompt


def _process_prompt(prompt: str, settings: Settings) -> ChatMessage | None:
    user_message: ChatMessage = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    render_user_message(prompt)

    with st.spinner("Retrieving and generating…"):
        try:
            pipeline = _build_pipeline(settings)
            response = pipeline.query(prompt)
        except Exception as exc:  # noqa: BLE001 - any failure should land in the UI
            st.error(f"Query failed: {exc}")
            return None

    assistant: ChatMessage = {
        "role": "assistant",
        "content": response.answer or "_No answer was generated._",
        "citations": list(response.citations),
    }
    st.session_state.messages.append(assistant)
    st.session_state["latest_trace"] = response.trace
    render_assistant_message(assistant["content"], list(response.citations))
    return assistant


# --------------------------------- main ------------------------------------


def main() -> None:
    st.set_page_config(page_title="guideline-gpt", layout="wide")
    _init_state()
    configure_logging(get_settings().log_level)
    settings = _current_settings()

    _render_sidebar()

    st.title("guideline-gpt")
    st.caption(
        "Educational demo only — not a medical device, not for clinical "
        "decision-making. Public documents only."
    )

    _render_documents_bar(settings)

    chat_col, inspector_col = st.columns([3, 2], gap="medium")

    # Resolve a pending example-button prompt before rendering history so the
    # new message lands at the bottom of the chat column in order.
    pending: str | None = st.session_state.pop("pending_prompt", None)

    with chat_col:
        st.markdown("#### Chat")
        render_history(cast(list[ChatMessage], st.session_state.messages))
        if pending:
            _process_prompt(pending, settings)

    with inspector_col:
        st.markdown("#### Pipeline inspector")
        latest = st.session_state.get("latest_trace")
        if latest is None:
            render_empty_state(on_example=_set_pending)
        else:
            render_inspector(latest, show_timing=st.session_state.show_timing)

    # Chat input pins to the bottom of the page.
    prompt = st.chat_input("Ask a question about the ingested guidelines…")
    if prompt:
        with chat_col:
            _process_prompt(prompt, settings)
        st.rerun()


main()
