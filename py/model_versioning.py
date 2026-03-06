"""
model_versioning.py
-------------------
Versioned model saving for EchoSafe.

Instead of silently overwriting clap_model.pkl and cnn_sound_model.h5 on
every save, this module:
  1. Saves a timestamped copy alongside the main file
     e.g.  clap_model.pkl  +  clap_model_20240415_143022.pkl
  2. Keeps only the N most recent backups (default 3), deleting older ones
  3. Provides load_latest() to find the most recent versioned file if the
     canonical path is missing

Usage:
    from model_versioning import versioned_save, load_latest

    # Save sklearn model
    versioned_save(path="clap_model.pkl", save_fn=lambda p: joblib.dump(model, p))

    # Save Keras model
    versioned_save(path="cnn_sound_model.h5", save_fn=lambda p: model.save(p))

    # Find latest backup
    latest = load_latest("clap_model.pkl")  # returns path or None
"""

import glob
import os
import time


# How many timestamped backups to keep per model file
KEEP_VERSIONS = 3


def versioned_save(path: str, save_fn, keep: int = KEEP_VERSIONS) -> str:
    """
    Save a model to `path` (canonical) AND a timestamped backup.

    Parameters
    ----------
    path    : canonical model path, e.g. "clap_model.pkl"
    save_fn : callable(dest_path) that writes the model to dest_path
    keep    : number of timestamped backups to keep (oldest are pruned)

    Returns the path of the timestamped backup that was written.
    """
    # 1. Write to the canonical path (used by all loaders)
    save_fn(path)

    # 2. Write a timestamped copy
    ts        = time.strftime("%Y%m%d_%H%M%S")
    stem, ext = os.path.splitext(path)
    backup    = f"{stem}_{ts}{ext}"
    save_fn(backup)

    # 3. Prune old backups — keep only the `keep` most recent
    _prune(path, keep)

    return backup


def _prune(canonical: str, keep: int) -> None:
    """Delete all but the `keep` most recent timestamped backups for `canonical`."""
    stem, ext   = os.path.splitext(canonical)
    pattern     = f"{stem}_????????_??????{ext}"   # matches YYYYMMDD_HHMMSS
    backups     = sorted(glob.glob(pattern))       # alphabetical = chronological

    to_delete = backups[: max(0, len(backups) - keep)]
    for old in to_delete:
        try:
            os.remove(old)
        except OSError:
            pass   # not worth crashing over a stale backup


def load_latest(canonical: str) -> str | None:
    """
    Return the path of the most recent versioned backup for `canonical`,
    or None if no backups exist.

    Useful as a fallback when the canonical file is missing:
        path = canonical if os.path.exists(canonical) else load_latest(canonical)
    """
    stem, ext = os.path.splitext(canonical)
    pattern   = f"{stem}_????????_??????{ext}"
    backups   = sorted(glob.glob(pattern))
    return backups[-1] if backups else None


def list_versions(canonical: str) -> list[str]:
    """Return all timestamped backups for `canonical`, oldest first."""
    stem, ext = os.path.splitext(canonical)
    pattern   = f"{stem}_????????_??????{ext}"
    return sorted(glob.glob(pattern))