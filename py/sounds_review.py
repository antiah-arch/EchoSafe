"""
sounds_review.py
----------------
Review, play back, and manage sounds recorded by ai_classifier.py.

Features:
  - Browse all .wav files saved in sounds_db/
  - Play back any recording
  - Relabel or delete recordings
  - Export a clean CSV of reviewed detections for use with trainer.py

Usage:
    python sounds_review.py
    python sounds_review.py --dir my_sounds_db
    python sounds_review.py --export reviewed.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

from config import SOUNDS_DB_DIR, PC_SAMPLE_RATE as SAMPLE_RATE


# ── HELPERS ───────────────────────────────────────────────────────────────────

def list_wav_files(directory: str) -> list[Path]:
    """Return all .wav files in directory sorted by modification time (newest last)."""
    d = Path(directory)
    if not d.exists():
        print(f"Error: directory {directory!r} does not exist.")
        sys.exit(1)
    files = sorted(d.glob("*.wav"), key=lambda p: p.stat().st_mtime)
    return files


def load_detection_log(csv_path: str) -> dict[str, dict]:
    """
    Load detections.csv into a dict keyed by wav filename stem
    e.g. {"clap_1712345678": {"label": "clap", "peak": "512.0", ...}}
    """
    log: dict[str, dict] = {}
    if not os.path.exists(csv_path):
        return log
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            # best-effort match: csv rows don't store filename, match by timestamp
            log[row.get("time", "")] = row
    return log


def play_wav(path: Path) -> None:
    """Play a wav file through the default output device."""
    if not AUDIO_AVAILABLE:
        print("  (playback unavailable — install sounddevice and soundfile)")
        return
    try:
        data, sr = sf.read(str(path))
        print(f"  ▶ Playing {path.name}  ({len(data)/sr:.2f}s @ {sr}Hz)...")
        sd.play(data, sr)
        sd.wait()
    except Exception as e:
        print(f"  Playback error: {e}")


def rename_wav(path: Path, new_label: str) -> Path:
    """Rename a wav file to reflect a corrected label, preserving the timestamp."""
    # Extract timestamp from original name e.g. "clap_1712345678.wav" → "1712345678"
    parts = path.stem.split("_")
    timestamp = parts[-1] if parts[-1].isdigit() else path.stem
    safe_label = new_label.strip().replace(" ", "_").replace("/", "-")
    new_name = path.parent / f"{safe_label}_{timestamp}.wav"
    path.rename(new_name)
    return new_name


def export_csv(wav_files: list[Path], labels: dict[str, str], out_path: str) -> None:
    """Export a CSV of reviewed wav files and their labels for use with trainer.py."""
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "sample_rate"])
        writer.writeheader()
        for wav in wav_files:
            label = labels.get(str(wav), wav.stem.rsplit("_", 1)[0])
            writer.writerow({
                "filepath": str(wav),
                "label": label,
                "sample_rate": SAMPLE_RATE,
            })
    print(f"\n✅ Exported {len(wav_files)} entries to {out_path!r}")


# ── REVIEW LOOP ───────────────────────────────────────────────────────────────

def review(directory: str, export_path: str | None) -> None:
    wav_files = list_wav_files(directory)

    if not wav_files:
        print(f"No .wav files found in {directory!r}.")
        return

    print(f"\nFound {len(wav_files)} recording(s) in {directory!r}")
    print("Commands:  [ENTER] play & keep   [r] relabel   [d] delete   [s] skip (keep without playing)   [q] quit\n")

    # Track any label changes made during review
    reviewed_labels: dict[str, str] = {}
    kept: list[Path] = []
    i = 0

    while i < len(wav_files):
        wav = wav_files[i]
        # Infer current label from filename stem e.g. "clap_1712345678"
        current_label = wav.stem.rsplit("_", 1)[0]

        print(f"[{i+1}/{len(wav_files)}]  {wav.name}  label={current_label}")
        play_wav(wav)   # auto-play on display

        cmd = input("  > ").strip().lower()

        if cmd == "":
            # ENTER = keep as-is (sound already played above)
            reviewed_labels[str(wav)] = current_label
            kept.append(wav)
            i += 1

        elif cmd == "s":
            # Skip — keep without playing again
            reviewed_labels[str(wav)] = current_label
            kept.append(wav)
            i += 1

        elif cmd == "r":
            new_label = input("  New label: ").strip()
            if new_label:
                new_path = rename_wav(wav, new_label)
                print(f"  Renamed to {new_path.name}")
                wav_files[i] = new_path   # update list so export uses new path
                reviewed_labels[str(new_path)] = new_label
                kept.append(new_path)
            else:
                print("  No label entered — keeping original.")
                reviewed_labels[str(wav)] = current_label
                kept.append(wav)
            i += 1

        elif cmd == "d":
            confirm = input(f"  Delete {wav.name}? (y/N): ").strip().lower()
            if confirm == "y":
                wav.unlink()
                print(f"  Deleted {wav.name}")
                wav_files.pop(i)
                # Don't increment i — next file is now at same index
            else:
                print("  Cancelled.")

        elif cmd == "q":
            print("  Quitting review early.")
            break

        else:
            print("  Unknown command. Use ENTER, p, r, d, or q.")

    print(f"\nReview complete. {len(kept)} file(s) kept.")

    if export_path:
        export_csv(kept, reviewed_labels, export_path)
    else:
        save = input("\nExport reviewed labels to CSV for training? (y/N): ").strip().lower()
        if save == "y":
            default_out = os.path.join(directory, "reviewed.csv")
            out = input(f"Output path [{default_out}]: ").strip() or default_out
            export_csv(kept, reviewed_labels, out)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sounds_review",
        description="Review and manage sounds recorded by ai_classifier.py",
    )
    parser.add_argument(
        "--dir",
        default=SOUNDS_DB_DIR,
        help=f"directory containing .wav files (default: {SOUNDS_DB_DIR})",
    )
    parser.add_argument(
        "--export",
        default=None,
        metavar="CSV_PATH",
        help="automatically export reviewed labels to this CSV path after review",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    review(args.dir, args.export)