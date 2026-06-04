#!/usr/bin/env python
"""Download the default public-guideline showcase corpus into ``documents/``.

This populates the corpus referenced by the Quickstart. The default documents
are freely available public clinical strategy reports (respiratory / critical
care), matching the showcase described in the README. The system is
corpus-agnostic — drop your own PDFs into ``documents/`` to query a different
domain.

Behavior:
- Idempotent: a file that already exists on disk is skipped, not re-downloaded.
- Resilient: if a URL has moved or the network is unavailable, that document is
  reported and skipped — the run does not abort. The repo also ships a tiny
  ``sample-guideline.pdf`` so ``guideline-gpt ingest`` works even if every
  download fails.

Usage:
    python scripts/download_corpus.py
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

# documents/ lives at the repository root, one level up from scripts/.
DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"

# Network timeout (seconds) per download.
TIMEOUT = 60

# Some hosts reject the default urllib User-Agent; present a browser-like one.
_HEADERS = {"User-Agent": "Mozilla/5.0 (guideline-gpt corpus downloader)"}

# Destination filename -> source URL. Every entry is a public, freely
# downloadable clinical guideline. Add or replace entries to curate your own
# default corpus.
CORPUS: dict[str, str] = {
    "GOLD-COPD-2024.pdf": (
        "https://goldcopd.org/wp-content/uploads/2024/02/"
        "GOLD-2024_v1.2-11Jan24_WMV.pdf"
    ),
    "GINA-Asthma-2024.pdf": (
        "https://ginasthma.org/wp-content/uploads/2024/05/"
        "GINA-2024-Strategy-Report-24_05_22_WMS.pdf"
    ),
}


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest``, raising on any HTTP/network error."""
    request = urllib.request.Request(url, headers=_HEADERS)  # noqa: S310 - https URLs only
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310
        dest.write_bytes(response.read())


def main() -> int:
    """Download every corpus document that is not already present.

    Returns:
        Process exit code: ``0`` if at least one PDF is available in
        ``documents/`` afterwards, ``1`` if the corpus is empty (every download
        failed and no PDFs were already there).
    """
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed: list[str] = []

    for filename, url in CORPUS.items():
        dest = DOCUMENTS_DIR / filename
        if dest.exists():
            print(f"  skip   {filename} (already present)")
            skipped += 1
            continue
        print(f"  fetch  {filename} <- {url}")
        try:
            _download(url, dest)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            # Clean up any partial file so a re-run retries from scratch.
            dest.unlink(missing_ok=True)
            print(f"  FAILED {filename}: {exc}", file=sys.stderr)
            failed.append(filename)
            continue
        downloaded += 1

    present = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    print(
        f"\nDone: {downloaded} downloaded, {skipped} already present, "
        f"{len(failed)} failed. {len(present)} PDF(s) now in {DOCUMENTS_DIR}/."
    )
    if failed:
        print(
            "Some documents could not be fetched (URLs move over time). You can "
            "drop any PDFs into documents/ manually and re-run `guideline-gpt "
            "ingest`.",
            file=sys.stderr,
        )

    if not present:
        print(
            "No PDFs are available to ingest. Add your own to documents/ and try "
            "again.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
