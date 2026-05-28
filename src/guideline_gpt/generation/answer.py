"""Answer generation: assemble prompts, call the LLM, parse citations."""

from __future__ import annotations

import re

from guideline_gpt.generation.llm_client import LLMClient
from guideline_gpt.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from guideline_gpt.types import Chunk

# Matches citation markers like [1] or [12] in the model's answer.
_CITATION_RE = re.compile(r"\[(\d+)\]")


def parse_citations(text: str) -> list[str]:
    """Extract the distinct citation markers used in an answer, in order.

    Args:
        text: The model's answer.

    Returns:
        Citation numbers as strings (e.g. ``["1", "3"]``), de-duplicated while
        preserving first-seen order.
    """
    seen: dict[str, None] = {}
    for match in _CITATION_RE.finditer(text):
        seen.setdefault(match.group(1), None)
    return list(seen)


def generate_answer(query: str, chunks: list[Chunk], client: LLMClient) -> tuple[str, list[str]]:
    """Generate a cited answer for a query against retrieved chunks.

    Args:
        query: The user's question.
        chunks: The context chunks to ground the answer in.
        client: The configured LLM client.

    Returns:
        A tuple of (answer_text, citation_markers_used).
    """
    user_prompt = build_user_prompt(query, chunks)
    result = client.complete(SYSTEM_PROMPT, user_prompt)
    return result.text, parse_citations(result.text)
