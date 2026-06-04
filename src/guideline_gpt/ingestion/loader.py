"""Load PDFs into per-page text with provenance metadata.

Uses ``pypdf`` (pure Python, no system dependencies) as the primary extractor
and falls back to ``pymupdf`` for documents that ``pypdf`` cannot read text from
(e.g. some scanned or unusually encoded PDFs).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from math import ceil
from pathlib import Path

from guideline_gpt.logging_setup import get_logger
from guideline_gpt.types import DocumentMetadata

log = get_logger(__name__)

# Pages with less text than this are skipped (cover pages, blanks, dividers).
MIN_PAGE_CHARS = 50

# Boilerplate detection is unreliable on very short documents (a line appearing
# on "most" of 2 pages means nothing), so we only run it past this page count,
# and never strip a line seen on fewer than this many pages.
_MIN_PAGES_FOR_BOILERPLATE = 5
_MIN_BOILERPLATE_PAGES = 3

_WHITESPACE_RE = re.compile(r"\s+")
# A running header/footer often ends (or starts) with the page number, which
# differs on every page. Stripping that edge number lets the rest of the line
# match across pages.
_EDGE_PAGE_NUM_RE = re.compile(r"^\d+\s+|\s+\d+$")


def _normalize_line(line: str) -> str:
    """Canonicalize a line for cross-page comparison.

    Collapses internal whitespace and drops a leading/trailing bare page number,
    so that a running header like ``"… Disclaimer        50"`` compares equal to
    the same header on every other page. Returns ``""`` for lines that are only a
    page number (which carry no content and are never treated as boilerplate).
    """
    collapsed = _WHITESPACE_RE.sub(" ", line.strip())
    if collapsed.isdigit():  # a bare page number — no content
        return ""
    return _EDGE_PAGE_NUM_RE.sub("", collapsed).strip()


def _find_boilerplate(pages: list[str], page_fraction: float) -> set[str]:
    """Identify running header/footer lines repeated across a document's pages.

    A line whose normalized form (see :func:`_normalize_line`) appears on at
    least ``page_fraction`` of the pages is almost certainly boilerplate — a
    running header, footer, or copyright notice — rather than content: real prose
    rarely repeats verbatim on half a document. Each page contributes a given
    line at most once, so a phrase repeated many times on a single page does not
    count as document-wide boilerplate.

    Returns the set of normalized lines to strip (empty for short documents).
    """
    if len(pages) < _MIN_PAGES_FOR_BOILERPLATE:
        return set()
    counts: Counter[str] = Counter()
    for text in pages:
        normalized = {_normalize_line(ln) for ln in text.splitlines() if ln.strip()}
        normalized.discard("")  # page-number-only lines carry no content
        for line in normalized:
            counts[line] += 1
    threshold = max(_MIN_BOILERPLATE_PAGES, ceil(len(pages) * page_fraction))
    return {line for line, count in counts.items() if count >= threshold}


def _strip_boilerplate(text: str, boilerplate: set[str]) -> str:
    """Drop any line whose normalized form is in ``boilerplate``."""
    if not boilerplate:
        return text
    return "\n".join(ln for ln in text.splitlines() if _normalize_line(ln) not in boilerplate)


class LoaderError(RuntimeError):
    """Raised when a PDF cannot be read by any available extractor."""


def _extract_with_pypdf(path: Path) -> list[str]:
    """Extract per-page text using pypdf. Returns one string per page."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


def _extract_with_pymupdf(path: Path) -> list[str]:
    """Extract per-page text using pymupdf (fitz). Returns one string per page."""
    import fitz  # pymupdf

    with fitz.open(str(path)) as doc:
        return [page.get_text() for page in doc]


def _extract_pages(path: Path) -> list[str]:
    """Extract per-page text, falling back to pymupdf when pypdf yields nothing.

    Args:
        path: Path to the PDF file.

    Returns:
        A list of page text strings, one per page.

    Raises:
        LoaderError: If neither extractor can read the file.
    """
    try:
        pages = _extract_with_pypdf(path)
    except Exception as exc:  # noqa: BLE001 - we fall back regardless of cause
        log.warning("pypdf_failed", file=path.name, error=str(exc))
        pages = []

    if sum(len(p.strip()) for p in pages) > 0:
        return pages

    log.info("falling_back_to_pymupdf", file=path.name)
    try:
        return _extract_with_pymupdf(path)
    except Exception as exc:  # noqa: BLE001
        raise LoaderError(f"Could not extract text from {path.name}: {exc}") from exc


def load_pdfs(
    documents_dir: Path,
    *,
    boilerplate_page_fraction: float = 0.5,
) -> Iterator[tuple[str, DocumentMetadata]]:
    """Yield (page_text, metadata) pairs for every page of every PDF in a folder.

    Running headers/footers repeated across most of a document's pages are
    stripped before yielding (see :func:`_find_boilerplate`). Pages with fewer
    than :data:`MIN_PAGE_CHARS` characters of *content* remaining are skipped.

    Args:
        documents_dir: Directory containing ``*.pdf`` files (searched recursively).
        boilerplate_page_fraction: A line on at least this fraction of a
            document's pages is treated as boilerplate and removed.

    Yields:
        Tuples of cleaned page text and its :class:`DocumentMetadata`.
    """
    pdf_paths = sorted(documents_dir.rglob("*.pdf"))
    if not pdf_paths:
        log.warning("no_pdfs_found", documents_dir=str(documents_dir))
        return

    for path in pdf_paths:
        pages = _extract_pages(path)
        boilerplate = _find_boilerplate(pages, boilerplate_page_fraction)
        kept = 0
        total_chars = 0
        for page_index, raw in enumerate(pages, start=1):
            text = _strip_boilerplate(raw, boilerplate)
            stripped = text.strip()
            if len(stripped) < MIN_PAGE_CHARS:
                continue
            kept += 1
            total_chars += len(stripped)
            yield (
                text,
                DocumentMetadata(
                    source_path=str(path),
                    source_name=path.stem,
                    page_number=page_index,
                ),
            )
        log.info(
            "loaded_pdf",
            file=path.name,
            pages_total=len(pages),
            pages_kept=kept,
            chars=total_chars,
            boilerplate_lines=len(boilerplate),
        )
