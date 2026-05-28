"""Load PDFs into per-page text with provenance metadata.

Uses ``pypdf`` (pure Python, no system dependencies) as the primary extractor
and falls back to ``pymupdf`` for documents that ``pypdf`` cannot read text from
(e.g. some scanned or unusually encoded PDFs).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from guideline_gpt.logging_setup import get_logger
from guideline_gpt.types import DocumentMetadata

log = get_logger(__name__)

# Pages with less text than this are skipped (cover pages, blanks, dividers).
MIN_PAGE_CHARS = 50


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


def load_pdfs(documents_dir: Path) -> Iterator[tuple[str, DocumentMetadata]]:
    """Yield (page_text, metadata) pairs for every page of every PDF in a folder.

    Pages with fewer than :data:`MIN_PAGE_CHARS` characters of text are skipped.

    Args:
        documents_dir: Directory containing ``*.pdf`` files (searched recursively).

    Yields:
        Tuples of page text and its :class:`DocumentMetadata`.
    """
    pdf_paths = sorted(documents_dir.rglob("*.pdf"))
    if not pdf_paths:
        log.warning("no_pdfs_found", documents_dir=str(documents_dir))
        return

    for path in pdf_paths:
        pages = _extract_pages(path)
        kept = 0
        total_chars = 0
        for page_index, text in enumerate(pages, start=1):
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
        )
