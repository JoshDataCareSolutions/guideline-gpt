"""Chat-message rendering: bordered Q&A blocks with clickable [n] citations.

We deliberately do NOT use ``st.chat_message`` — its avatar slot can't be
suppressed and a robot/person icon reads as generic chatbot UI. A bordered
container with a small role label looks more like a research notebook.
"""

from __future__ import annotations

import re
from typing import TypedDict

import streamlit as st

from guideline_gpt.types import Chunk

_CITATION_RE = re.compile(r"\[(\d+)\]")


class ChatMessage(TypedDict, total=False):
    role: str  # "user" | "assistant"
    content: str
    citations: list[Chunk]


def _link_citations(text: str) -> str:
    """Turn ``[n]`` markers in answer text into anchor links to the inspector row."""
    return _CITATION_RE.sub(
        lambda m: (
            f"<a href='#cite-{m.group(1)}' "
            "style='text-decoration:none;font-weight:600'>"
            f"[{m.group(1)}]</a>"
        ),
        text,
    )


def render_user_message(content: str) -> None:
    """Render a user question as a bordered, captioned block."""
    with st.container(border=True):
        st.caption("Question")
        st.markdown(content)


def render_assistant_message(content: str, citations: list[Chunk] | None = None) -> None:
    """Render an assistant answer with linkified citations and a sources expander."""
    with st.container(border=True):
        st.caption("Answer")
        st.markdown(_link_citations(content), unsafe_allow_html=True)
        if citations:
            with st.expander(f"Sources ({len(citations)})", expanded=False):
                for index, chunk in enumerate(citations, start=1):
                    meta = chunk.metadata
                    st.markdown(f"**[{index}] {meta.source_name}.pdf — p. {meta.page_number}**")
                    st.caption(chunk.text)


def render_message(message: ChatMessage) -> None:
    """Render a single chat message by role."""
    role = message.get("role", "assistant")
    content = message.get("content", "")
    if role == "user":
        render_user_message(content)
    else:
        render_assistant_message(content, message.get("citations"))


def render_history(messages: list[ChatMessage]) -> None:
    """Render the full chat history."""
    for message in messages:
        render_message(message)
