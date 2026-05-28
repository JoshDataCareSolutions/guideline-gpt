"""Streamlit renderers for retrieval-stage hit tables and rerank rows."""

from __future__ import annotations

import streamlit as st

from guideline_gpt.types import RetrievalHit

_PREVIEW_CHARS = 200


def _preview(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= _PREVIEW_CHARS else flat[:_PREVIEW_CHARS].rstrip() + "…"


def render_hits_table(hits: list[RetrievalHit], score_label: str = "score") -> None:
    """Render a compact table of retrieval hits."""
    if not hits:
        st.caption("_no hits_")
        return
    st.dataframe(
        {
            "rank": [h.rank for h in hits],
            score_label: [round(h.score, 4) for h in hits],
            "source": [h.chunk.metadata.source_name for h in hits],
            "page": [h.chunk.metadata.page_number for h in hits],
            "preview": [_preview(h.chunk.text) for h in hits],
        },
        hide_index=True,
        use_container_width=True,
    )


def render_reranked_rows(hits: list[RetrievalHit]) -> None:
    """Render the post-rerank list with expandable full text and citation anchors.

    Each row is anchored as ``cite-N`` (1-indexed by position) so that the
    matching ``[N]`` citation marker in the answer can link to it.
    """
    if not hits:
        st.caption("_no chunks made it through rerank_")
        return
    for index, hit in enumerate(hits, start=1):
        meta = hit.chunk.metadata
        # Anchor target for clickable citations from the chat answer.
        st.markdown(f"<a id='cite-{index}'></a>", unsafe_allow_html=True)
        header = (
            f"**[{index}]** &nbsp; score `{hit.score:+.3f}` &nbsp; "
            f"{meta.source_name}.pdf · p. {meta.page_number}"
        )
        st.markdown(header, unsafe_allow_html=True)
        with st.expander("Show chunk text", expanded=False):
            st.write(hit.chunk.text)
