"""System and user prompt templates for grounded, cited answers."""

from __future__ import annotations

from guideline_gpt.types import Chunk

SYSTEM_PROMPT = """\
You are guideline-gpt, an assistant that answers questions using ONLY the \
provided guideline excerpts.

Rules:
- Answer strictly from the numbered excerpts below. Do not use outside knowledge.
- Cite every claim with bracketed markers like [1] or [2] that refer to the \
excerpt numbers. Place citations inline, immediately after the claim they support.
- If the excerpts do not contain enough information to answer, say so plainly \
("The provided guidelines do not cover this.") and do not guess.
- Be concise and clinical in tone. Do not give medical advice or address an \
individual patient; you summarize published guidelines only.
"""


def format_context(chunks: list[Chunk]) -> str:
    """Render chunks as a numbered, source-attributed context block.

    Each chunk is prefixed with its 1-indexed citation marker and provenance,
    e.g. ``[1] (Source: GOLD-COPD-2024.pdf, p. 47)``.
    """
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        header = f"[{index}] (Source: {meta.source_name}.pdf, p. {meta.page_number})"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_user_prompt(query: str, chunks: list[Chunk]) -> str:
    """Assemble the user prompt from retrieved chunks and the question."""
    context = format_context(chunks)
    return (
        "Guideline excerpts:\n\n"
        f"{context}\n\n"
        "---\n"
        f"Question: {query}\n\n"
        "Answer using only the excerpts above, citing them with [n] markers."
    )
