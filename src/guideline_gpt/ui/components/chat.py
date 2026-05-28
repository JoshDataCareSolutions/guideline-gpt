"""Chat-message rendering with clickable [n] citation anchors."""

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


def render_message(message: ChatMessage) -> None:
    """Render a single chat message, linkifying citation markers for the assistant."""
    role = message.get("role", "assistant")
    content = message.get("content", "")
    with st.chat_message(role):
        if role == "assistant":
            st.markdown(_link_citations(content), unsafe_allow_html=True)
            cited = message.get("citations") or []
            if cited:
                with st.expander(f"Sources ({len(cited)})", expanded=False):
                    for index, chunk in enumerate(cited, start=1):
                        meta = chunk.metadata
                        st.markdown(f"**[{index}] {meta.source_name}.pdf — p. {meta.page_number}**")
                        st.caption(chunk.text)
        else:
            st.markdown(content)


def render_history(messages: list[ChatMessage]) -> None:
    """Render the full chat history."""
    for message in messages:
        render_message(message)
