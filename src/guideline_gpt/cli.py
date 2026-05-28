"""Typer command-line interface for guideline-gpt.

Commands are thin wrappers: they parse arguments, configure logging, and call
into the library. The ``ingest`` and full query commands are added as the
corresponding pipeline stages land (see milestones M1+).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from guideline_gpt import __version__
from guideline_gpt.config import get_settings
from guideline_gpt.logging_setup import configure_logging, get_logger

app = typer.Typer(
    name="guideline-gpt",
    help="A production-pattern RAG system with a pipeline inspector UI.",
    no_args_is_help=True,
    add_completion=False,
)

log = get_logger(__name__)


@app.command()
def version() -> None:
    """Print the installed version and exit."""
    typer.echo(f"guideline-gpt {__version__}")


@app.command()
def ingest(
    documents_dir: Annotated[
        Path | None,
        typer.Argument(help="Folder of PDFs to ingest. Defaults to DOCUMENTS_DIR from config."),
    ] = None,
) -> None:
    """Ingest PDFs into the vector store and BM25 index."""
    from guideline_gpt.ingestion.pipeline import ingest as run_ingest

    settings = get_settings()
    configure_logging(settings.log_level)
    target = documents_dir or settings.documents_dir
    report = run_ingest(target, settings)
    typer.echo(
        f"Ingested {report.files} file(s), {report.pages} page(s) -> "
        f"{report.chunks:,} chunks ({report.total_tokens:,} tokens) "
        f"in {report.elapsed_seconds}s."
    )


@app.command()
def serve(
    port: Annotated[int, typer.Option(help="Port for the Streamlit server.")] = 8501,
) -> None:
    """Launch the Streamlit inspector UI."""
    settings = get_settings()
    configure_logging(settings.log_level)
    app_path = Path(__file__).parent / "ui" / "streamlit_app.py"
    log.info("starting_ui", app_path=str(app_path), port=port)
    subprocess.run(  # noqa: S603 - args are not user-controlled
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            str(port),
        ],
        check=True,
    )


if __name__ == "__main__":
    app()
