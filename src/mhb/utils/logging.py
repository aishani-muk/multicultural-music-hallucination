"""Unified logger for the package."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_DEFAULT_FMT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


def get_logger(
    name: str = "mhb",
    level: str = "INFO",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if getattr(logger, "_mhb_configured", False):
        return logger

    logger.setLevel(level)
    logger.handlers.clear()

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter(_DEFAULT_FMT, datefmt=_DEFAULT_DATEFMT))
    logger.addHandler(sh)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(_DEFAULT_FMT, datefmt=_DEFAULT_DATEFMT))
        logger.addHandler(fh)

    logger._mhb_configured = True  # type: ignore[attr-defined]
    logger.propagate = False
    return logger
