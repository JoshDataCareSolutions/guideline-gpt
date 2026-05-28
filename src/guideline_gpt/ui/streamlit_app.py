"""Minimal Streamlit chat UI (M2).

A thin layer over :class:`QueryPipeline`: it submits questions, renders the
cited answer, and lists the source chunks. The full pipeline inspector (the
five-stage view, settings drawer, cost display) arrives in M5.
"""

from __future__ import annotations

import streamlit as st

from guideline_gpt.config import get_settings
from guideline_gpt.pipeline import QueryPipeline, estimate_cost
from guideline_gpt.types import QueryResponse


@st.cache_resource
def _get_pipeline() -> QueryPipeline:
    """Build the query pipeline once per session (cached across reruns)."""
    return QueryPipeline(get_settings())


def _render_sources(response: QueryResponse) -> None:
    with st.expander(f"Sources ({len(response.citations)})", expanded=False):
        for index, chunk in enumerate(response.citations, start=1):
            meta = chunk.metadata
            st.markdown(f"**[{index}] {meta.source_name}.pdf — p. {meta.page_number}**")
            st.caption(chunk.text)


def _render_metrics(response: QueryResponse) -> None:
    trace = response.trace
    cost = estimate_cost(trace.llm_model, trace.llm_input_tokens, trace.llm_output_tokens)
    cols = st.columns(4)
    cols[0].metric("Model", trace.llm_model or "—")
    cols[1].metric("Latency", f"{trace.llm_latency_ms:,} ms")
    cols[2].metric("Tokens", f"{trace.llm_input_tokens:,} in / {trace.llm_output_tokens:,} out")
    cols[3].metric("Est. cost", f"${cost:.4f}")


def main() -> None:
    st.set_page_config(page_title="guideline-gpt", layout="wide")
    st.title("guideline-gpt")
    st.caption(
        "⚠️ Educational demo only — not a medical device and not for clinical "
        "decision-making. Public documents only."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about the ingested guidelines…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"), st.spinner("Retrieving and generating…"):
        try:
            response = _get_pipeline().query(prompt)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            st.error(f"Query failed: {exc}")
            return

        answer = response.answer or "_No answer was generated._"
        st.markdown(answer)
        if response.trace.stage_errors:
            st.warning(f"Some stages failed: {response.trace.stage_errors}")
        _render_metrics(response)
        _render_sources(response)
        st.session_state.messages.append({"role": "assistant", "content": answer})


main()
