"""Tests for PDF loading helpers, focused on boilerplate (header/footer) removal."""

from __future__ import annotations

from guideline_gpt.ingestion.loader import (
    _find_boilerplate,
    _normalize_line,
    _strip_boilerplate,
)


def _doc(n_pages: int, *, header: str, body: str) -> list[str]:
    """Build a synthetic document: every page shares ``header``, plus unique body."""
    return [f"{header}\n{body} page {i}\nUnique line {i}." for i in range(n_pages)]


def test_finds_line_repeated_across_most_pages() -> None:
    pages = _doc(10, header="COPYRIGHT MATERIAL - DO NOT DISTRIBUTE", body="Content")
    boilerplate = _find_boilerplate(pages, page_fraction=0.5)
    assert "COPYRIGHT MATERIAL - DO NOT DISTRIBUTE" in boilerplate
    # Per-page unique lines must not be flagged.
    assert "Unique line 3." not in boilerplate


def test_short_documents_are_left_alone() -> None:
    # Under the minimum page count, repetition is not a reliable signal.
    pages = ["SAMPLE GUIDELINE\nIntro.", "SAMPLE GUIDELINE\nMore."]
    assert _find_boilerplate(pages, page_fraction=0.5) == set()


def test_line_below_threshold_is_kept() -> None:
    # A line on only 2 of 10 pages is content, not boilerplate.
    pages = _doc(10, header="Real content header", body="x")
    pages[0] = "Occasional note\n" + pages[0]
    pages[1] = "Occasional note\n" + pages[1]
    assert "Occasional note" not in _find_boilerplate(pages, page_fraction=0.5)


def test_repeats_within_one_page_do_not_count() -> None:
    # A phrase repeated many times on a SINGLE page is not document-wide boilerplate.
    pages = ["Buy now\nBuy now\nBuy now\nReal content here." for _ in range(2)] + [
        f"Different content {i}." for i in range(8)
    ]
    assert "Buy now" not in _find_boilerplate(pages, page_fraction=0.5)


def test_strip_boilerplate_removes_only_matching_lines() -> None:
    notice = "COPYRIGHT MATERIAL - DO NOT DISTRIBUTE"
    # Boilerplate may carry surrounding whitespace; matching is on the stripped form.
    text = f"{notice}\nActual clinical guidance.\n  {notice}  "
    cleaned = _strip_boilerplate(text, {notice})
    assert cleaned == "Actual clinical guidance."


def test_strip_boilerplate_noop_when_empty() -> None:
    text = "Line one\nLine two"
    assert _strip_boilerplate(text, set()) == text


def test_header_with_trailing_page_number_is_detected_and_stripped() -> None:
    # Real GOLD pattern: a running header that ends with the page number, so the
    # raw line differs on every page but normalizes to the same thing.
    header = "Important Purpose & Liability Disclaimer"
    pages = [f"{header}        {i}\nClinical content for page {i}." for i in range(10)]
    boilerplate = _find_boilerplate(pages, page_fraction=0.5)
    assert header in boilerplate
    cleaned = _strip_boilerplate(pages[3], boilerplate)
    assert cleaned == "Clinical content for page 3."


def test_normalize_line_collapses_space_and_edge_numbers() -> None:
    assert _normalize_line("Header   text     50") == "Header text"
    assert _normalize_line("12   Leading number header") == "Leading number header"
    # A bare page number normalizes to empty (never boilerplate).
    assert _normalize_line("  42  ") == ""
