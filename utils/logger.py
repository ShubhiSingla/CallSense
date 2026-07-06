"""
utils/logger.py
---------------
Centralised logging configuration for CallSense-AI.

Every module should obtain its logger via:

    from utils.logger import get_logger
    logger = get_logger(__name__)

This ensures consistent formatting, log levels, and handler setup
across the entire application without repeating boilerplate.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #
_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "outputs"
_LOG_FILE = _LOG_DIR / "callsense.log"


def _ensure_log_dir() -> None:
    """Create the log output directory if it does not exist."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def _build_console_handler(level: int) -> logging.StreamHandler:
    """Return a stderr StreamHandler with the standard formatter."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def _build_file_handler(level: int) -> logging.FileHandler:
    """Return a rotating-friendly FileHandler writing to data/outputs/callsense.log."""
    _ensure_log_dir()
    handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def configure_root_logger(level: str = "INFO", enable_file: bool = True) -> None:
    """
    Configure the root logger once at application startup.

    Parameters
    ----------
    level:
        String log level, e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``.
    enable_file:
        When ``True`` a file handler writing to ``data/outputs/callsense.log``
        is added in addition to the console handler.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()

    # Avoid adding duplicate handlers on repeated calls (e.g. Streamlit reruns)
    if root.handlers:
        return

    root.setLevel(numeric_level)
    root.addHandler(_build_console_handler(numeric_level))

    if enable_file:
        root.addHandler(_build_file_handler(numeric_level))


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.
        Falls back to the root logger when ``None``.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)
