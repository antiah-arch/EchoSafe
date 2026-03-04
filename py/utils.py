"""
utils.py
--------
Shared console output utilities for EchoSafe.

All user-facing messages should go through these functions so styling,
stream routing (stdout vs stderr), and flush behaviour are consistent
across the entire project.

Colours require colorama (pip install colorama). If it isn't installed
all output falls back to plain text — no crash.
"""

import sys
from typing import NoReturn

# FIX 1: switched from `colored` (version-dependent, not installed) to
# `colorama` which is already present, has a stable API, and works on Windows
try:
    from colorama import Fore, Style, init as _colorama_init
    _colorama_init(autoreset=True)   # resets colour after every print automatically
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
    """Print a red error message to stderr and exit with code 1."""
    # FIX 2 / FIX 7: stderr + flush=True so it appears immediately before exit
    print(_style(f"Error: {msg}", fore=Fore.RED if _COLOUR else ""),   # type: ignore[name-defined]
          file=sys.stderr, flush=True)
    sys.exit(1)


def warning(msg: str) -> None:
    """Print a yellow warning message to stderr."""
    # FIX 2: warnings are diagnostic — they belong on stderr, not stdout,
    # so they don't corrupt piped output in the CLI pipeline
    # FIX 7: flush=True so warnings appear immediately before any crash
    print(_style(f"Warning: {msg}", fore=Fore.YELLOW if _COLOUR else ""),  # type: ignore[name-defined]
          file=sys.stderr, flush=True)


def success(msg: str) -> None:
    """Print a green success message to stdout."""
    # FIX 7: flush=True so real-time detection output appears without buffering
    print(_style(msg, fore=Fore.GREEN if _COLOUR else ""), flush=True)  # type: ignore[name-defined]


def info(msg: str) -> None:
    """
    FIX 6: general informational message to stdout (cyan).
    Use for status updates that aren't errors or successes —
    e.g. 'Connecting to COM3...' or 'Loading model...'.
    """
    print(_style(msg, fore=Fore.CYAN if _COLOUR else ""), flush=True)   # type: ignore[name-defined]


def debug(msg: str, verbose: bool = False) -> None:
    """
    FIX 6: verbose/debug message — only printed when verbose=True.
    Replaces the scattered `if verbose: print(f'[verbose] ...')` pattern
    across listener.py, recording.py, and ai_classifier.py.
    Output is dim white to visually distinguish it from normal output.
    """
    if not verbose:
        return
    print(_style(msg, fore=Fore.WHITE if _COLOUR else "", dim=True),    # type: ignore[name-defined]
          flush=True)


def subtext(msg: str) -> None:
    """
    FIX 3: kept for backward compatibility but now delegates to info().
    Callers should migrate to info() or debug() as appropriate.
    """
    info(msg)