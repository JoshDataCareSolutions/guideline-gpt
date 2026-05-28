"""Streamlit renderers for the prompt and LLM-metrics inspector stages."""

from __future__ import annotations

import streamlit as st

from guideline_gpt.pipeline import estimate_cost
from guideline_gpt.types import QueryTrace


def render_prompt(trace: QueryTrace) -> None:
    """Show the exact system + user prompt the LLM received."""
    if not trace.prompt_assembled:
        st.caption("_prompt was not assembled (a prior stage failed)_")
        return
    st.code(trace.prompt_assembled, language="text")


def render_llm_metrics(trace: QueryTrace, *, show_timing: bool = False) -> None:
    """Show provider/model/latency/tokens/cost for the LLM call."""
    cost = estimate_cost(trace.llm_model, trace.llm_input_tokens, trace.llm_output_tokens)
    cols = st.columns(4)
    cols[0].metric("Provider", trace.llm_provider or "—")
    cols[1].metric("Model", trace.llm_model or "—")
    cols[2].metric(
        "Tokens",
        f"{trace.llm_input_tokens:,} in / {trace.llm_output_tokens:,} out",
    )
    cols[3].metric("Est. cost", f"${cost:.4f}")

    if show_timing:
        st.caption(f"LLM latency: **{trace.llm_latency_ms:,} ms**")

    if trace.answer:
        with st.expander("Raw LLM response (pre-parse)", expanded=False):
            st.code(trace.answer, language="text")
