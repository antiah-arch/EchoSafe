"""
utils.py
--------
Shared console output utilities for EchoSafe.

All user-facing messages should go through these functions so styling,
stream routing (stdout vs stderr), flush behaviour, and file logging are
consistent across the entire project.

Log file
--------
All output (including verbose debug messages) is also written to a rotating
log file at LOG_PATH (default: echosafe.log). The file is capped at 1 MB
with 3 backups kept, so it never fills the disk.

To change the log path set the ECHOSAFE_LOG environment variable:
    ECHOSAFE_LOG=/tmp/echosafe.log python ai_classifier.py

Colours require colorama (pip install colorama). If it isn't installed
all console output falls back to plain text — no crash.
"""

import logging
import logging.handlers
import os
import sys
from typing import NoReturn

# ── Log file setup ────────────────────────────────────────────────────────────

LOG_PATH = os.environ.get("ECHOSAFE_LOG", "echosafe.log")

_file_handler = logging.handlers.RotatingFileHandler(
    LOG_PATH,
    maxBytes=1_000_000,   # 1 MB per file
    backupCount=3,        # keep echosafe.log, echosafe.log.1, .2, .3
    encoding="utf-8",
    delay=True,           # don't create file until first message is written
)
_file_handler.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)

_logger = logging.getLogger("echosafe")
_logger.setLevel(logging.DEBUG)
_logger.addHandler(_file_handler)
_logger.propagate = False   # don't bubble up to root logger


# ── Colour setup ──────────────────────────────────────────────────────────────

try:
    from colorama import Fore, Style, init as _colorama_init
    _colorama_init(autoreset=True)
    _COLOUR = True
except ImportError:
    _COLOUR = False


def _style(msg: str, fore: str = "", dim: bool = False) -> str:
    """Apply colour and style if colorama is available, otherwise return plain text."""
    if not _COLOUR:
        return msg
    prefix = fore + (Style.DIM if dim else "")  # type: ignore[name-defined]
    return f"{prefix}{msg}{Style.RESET_ALL}"     # type: ignore[name-defined]


# ── Public API ────────────────────────────────────────────────────────────────

def error(msg: str) -> NoReturn:
    """Print a red error message to stderr, log it, and exit with code 1."""
    _logger.error(msg)
    print(_style(f"Error: {msg}", fore=Fore.RED if _COLOUR else ""),   # type: ignore[name-defined]
          file=sys.stderr, flush=True)
    sys.exit(1)


def warning(msg: str) -> None:
    """Print a yellow warning to stderr and log it."""
    _logger.warning(msg)
    print(_style(f"Warning: {msg}", fore=Fore.YELLOW if _COLOUR else ""),  # type: ignore[name-defined]
          file=sys.stderr, flush=True)


def success(msg: str) -> None:
    """Print a green success message to stdout and log it."""
    _logger.info(msg)
    print(_style(msg, fore=Fore.GREEN if _COLOUR else ""), flush=True)  # type: ignore[name-defined]


def info(msg: str) -> None:
    """Print a cyan informational message to stdout and log it."""
    _logger.info(msg)
    print(_style(msg, fore=Fore.CYAN if _COLOUR else ""), flush=True)   # type: ignore[name-defined]


def debug(msg: str, verbose: bool = False) -> None:
    """
    Verbose/debug message — always written to log file, only printed to
    console when verbose=True. Dim white to visually distinguish from
    normal output.
    """
    _logger.debug(msg)   # always log to file regardless of verbose flag
    if not verbose:
        return
    print(_style(msg, fore=Fore.WHITE if _COLOUR else "", dim=True),    # type: ignore[name-defined]
          flush=True)


def subtext(msg: str) -> None:
    """Kept for backward compatibility — delegates to info()."""
    info(msg)