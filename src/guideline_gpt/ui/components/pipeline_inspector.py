"""The five-stage pipeline inspector — the differentiator UI of this project."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from guideline_gpt.types import QueryTrace
from guideline_gpt.ui.components.retrieval_viewer import (
    render_hits_table,
    render_reranked_rows,
)
from guideline_gpt.ui.components.trace_viewer import render_llm_metrics, render_prompt


def render_empty_state(on_example: Callable[[str], None]) -> None:
    """Initial inspector content: a short explainer plus example questions."""
    st.subheader("How this works")
    st.write(
        "Every question you ask flows through five stages: **retrieval** (vector + "
        "keyword), **fusion** (RRF), **reranking** (cross-encoder), **prompt assembly**, "
        "and **generation**. After you ask something, you'll see each stage's inputs "
        "and outputs here."
    )
    st.divider()
    st.subheader("Try one of these")
    for example in (
        "What is the recommended PEEP setting for moderate ARDS?",
        "How are COPD exacerbations classified by severity?",
        "What tidal volume is recommended for lung-protective ventilation?",
    ):
        if st.button(example, use_container_width=True, key=f"example_{hash(example)}"):
            on_example(example)


def _retrieval_counts(trace: QueryTrace) -> str:
    return (
        f"Vector: {len(trace.vector_hits)} · "
        f"BM25: {len(trace.bm25_hits)} · "
        f"Fused: {len(trace.fused_hits)} unique"
    )


def render_inspector(trace: QueryTrace, *, show_timing: bool = False) -> None:
    """Render the five inspector stages for one query trace."""
    # Stage 1 — Query (always expanded)
    with st.expander("1. Query", expanded=True):
        st.write(f"`{trace.query}`")
        st.caption(trace.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        if trace.stage_errors:
            st.warning(f"Stage errors: {trace.stage_errors}")

    # Stage 2 — Retrieval (vector / bm25 / fused tabs)
    with st.expander(f"2. Retrieval — {_retrieval_counts(trace)}", expanded=False):
        vec_tab, bm_tab, fused_tab = st.tabs(["Vector", "BM25", "Fused (RRF)"])
        with vec_tab:
            render_hits_table(trace.vector_hits, score_label="similarity")
        with bm_tab:
            render_hits_table(trace.bm25_hits, score_label="bm25")
        with fused_tab:
            render_hits_table(trace.fused_hits, score_label="rrf")

    # Stage 3 — Rerank (the chunks actually sent to the LLM) — expanded: this is
    # the most informative single view of why the answer is what it is.
    with st.expander(f"3. Rerank — top {len(trace.reranked_hits)} sent to LLM", expanded=True):
        render_reranked_rows(trace.reranked_hits)

    # Stage 4 — Prompt (verbose; collapsed)
    with st.expander("4. Prompt sent to LLM", expanded=False):
        render_prompt(trace)

    # Stage 5 — LLM response (expanded for the metric tiles)
    with st.expander("5. LLM response", expanded=True):
        render_llm_metrics(trace, show_timing=show_timing)
