"""Logging helpers shared across the package."""

from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


class SilentLogger:
    """A no-op logger used to silence ``yt-dlp`` output.

    ``yt-dlp`` accepts any object exposing ``debug``/``warning``/``error``; this one
    simply discards everything.
    """

    def debug(self, msg: str) -> None:  # noqa: D102 - trivial no-op
        pass

    def warning(self, msg: str) -> None:  # noqa: D102 - trivial no-op
        pass

    def error(self, msg: str) -> None:  # noqa: D102 - trivial no-op
        pass


def get_logger(name: str = "somali_foodsec_radio", level: int = logging.INFO) -> logging.Logger:
    """Return a console logger.

    Idempotent: calling it repeatedly for the same *name* will not attach duplicate
    handlers. Configuration happens here (in a function) rather than at import time so
    that ``import somali_foodsec_radio`` has no side effects.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
