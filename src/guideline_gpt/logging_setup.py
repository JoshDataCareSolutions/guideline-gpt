"""structlog configuration for the package.

Call :func:`configure_logging` once at process startup (CLI entrypoint, UI
bootstrap). All modules then obtain a logger via ``structlog.get_logger()`` and
never use :func:`print`.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog to emit structured logs to stderr.

    Args:
        level: Standard logging level name, e.g. ``"INFO"`` or ``"DEBUG"``.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=numeric_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, optionally named.

    Args:
        name: Optional logger name, typically ``__name__``.

    Returns:
        A bound logger instance.
    """
    return structlog.get_logger(name)  # type: ignore[no-any-return]
