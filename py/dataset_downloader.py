"""
dataset_downloader.py
---------------------
Downloads and prepares labelled sound data from ESC-50 for use with
EchoSafe's trainer.py.

ESC-50 is a dataset of 2000 labelled environmental sounds across 50 classes.
This script:
  1. Downloads and extracts ESC-50
  2. Filters to classes relevant to EchoSafe (claps, noise, speech, etc.)
  3. Converts all audio to 16kHz mono and maps to Arduino ADC range (0-1023)
  4. Writes two CSVs incrementally (memory safe) ready for trainer.py:
       <output_dir>/sound_data_label1.csv  (clap examples)
       <output_dir>/sound_data_label0.csv  (noise/background examples)

Usage:
    python dataset_downloader.py
    python dataset_downloader.py --output-dir my_data
    python dataset_downloader.py --list-classes
    python dataset_downloader.py --clap-classes clapping --noise-classes wind rain
"""

import argparse
import csv
import os
import sys
import tarfile
import urllib.request  # FIX 1: kept — used in download_esc50()

import librosa
import numpy as np

# FIX 1: removed unused imports: shutil, pathlib.Path, soundfile
# FIX 5: import shared constants from config instead of redefining them
from config import SAMPLE_RATE, SOUNDS_DB_DIR as OUTPUT_DIR, CLAP_CSV, NOISE_CSV


# ── Constants ──────────────────────────────────────────────────────────────────

ESC50_URL     = "https://github.com/karoldvl/ESC-50/archive/master.tar.gz"
ESC50_DIR     = "ESC-50-master"
ESC50_ARCHIVE = "esc50.tar.gz"

# ESC-50 metadata URL — used by --list-classes without downloading the full archive
ESC50_META_URL = (
    "https://raw.githubusercontent.com/karoldvl/ESC-50/master/meta/esc50.csv"
)

# ESC-50 category names → label 1 (clap)
CLAP_CLASSES: set[str] = {
    "clapping",
    "hand_clapping",
}

# ESC-50 category names → label 0 (noise / background)
NOISE_CLASSES: set[str] = {
    "wind",
    "rain",
    "thunderstorm",
    "water_drops",
    "sea_waves",
    "crackling_fire",
    "crickets",
    "birds",
    "silence",
}

CSV_FIELDS = ["mic_value", "label", "source_file"]


# ── Download & extract ─────────────────────────────────────────────────────────

def download_esc50(dest: str = ESC50_ARCHIVE) -> None:
    if os.path.exists(dest):
        print(f"Archive already exists: {dest} — skipping download.")
        return
    print("Downloading ESC-50 (~600 MB)...")

    def _progress(block: int, block_size: int, total: int) -> None:
        done = block * block_size
        pct = min(done / total * 100, 100) if total > 0 else 0
        print(f"  {pct:.1f}%", end="\r")

    urllib.request.urlretrieve(ESC50_URL, dest, reporthook=_progress)
    print("\nDownload complete.")


def extract_esc50(archive: str = ESC50_ARCHIVE, dest: str = ".") -> str:
    extracted_path = os.path.join(dest, ESC50_DIR)
    if os.path.exists(extracted_path):
        print(f"Already extracted: {extracted_path} — skipping.")
        return extracted_path
    print("Extracting archive...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest, filter="data")   # FIX 3: filter="data" for Python 3.12+
    print("Extraction complete.")
    return extracted_path


# ── Audio helpers ──────────────────────────────────────────────────────────────

def load_and_resample(filepath: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load any audio file and resample to target_sr mono."""
    waveform, _ = librosa.load(filepath, sr=target_sr, mono=True)
    return waveform


def waveform_to_mic_values(waveform: np.ndarray) -> list[int]:
    """
    Convert a float32 waveform to Arduino ADC integer range (0–1023).
    FIX 4: clips waveform to -1..1 first so out-of-range samples from
    librosa resampling don't produce values outside 0–1023.
    """
    clipped = np.clip(waveform, -1.0, 1.0)        # FIX 4: clamp before scaling
    normalised = (clipped + 1.0) / 2.0             # shift -1..1 → 0..1
    return (normalised * 1023).astype(int).tolist()


# ── CSV helpers ────────────────────────────────────────────────────────────────

def open_csv_writer(path: str):
    """
    FIX 8: open a CSV for incremental writing so rows are flushed to disk
    as they are produced rather than accumulated in memory.
    Returns (file_handle, csv.DictWriter) — caller must close the file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    f = open(path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    writer.writeheader()
    return f, writer


# ── List classes ───────────────────────────────────────────────────────────────

def list_classes() -> None:
    """
    FIX 7: fetch ESC-50 metadata and print all unique category names
    so the user knows what values are valid for --clap-classes / --noise-classes.
    Downloads only the small metadata CSV, not the full archive.
    """
    print("Fetching ESC-50 class list...")
    try:
        with urllib.request.urlopen(ESC50_META_URL) as resp:
            lines = resp.read().decode("utf-8").splitlines()
    except Exception as e:
        print(f"Error fetching class list: {e}")
        sys.exit(1)

    reader = csv.DictReader(lines)
    categories: dict[str, int] = {}
    for row in reader:
        cat = row["category"].lower().replace(" ", "_")
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\nESC-50 contains {len(categories)} unique categories:\n")
    for cat, count in sorted(categories.items()):
        marker = ""
        if cat in CLAP_CLASSES:
            marker = "  ← default clap class"
        elif cat in NOISE_CLASSES:
            marker = "  ← default noise class"
        print(f"  {cat:<30} ({count} files){marker}")
    print(
        f"\nPass category names to --clap-classes or --noise-classes to customise.\n"
        f"Example: --clap-classes clapping --noise-classes wind rain thunderstorm"
    )


# ── Build dataset ──────────────────────────────────────────────────────────────

def build_dataset(
    esc50_root: str,
    clap_classes: set[str],
    noise_classes: set[str],
    output_dir: str,
) -> None:
    meta_path = os.path.join(esc50_root, "meta", "esc50.csv")
    if not os.path.exists(meta_path):
        print(f"Error: could not find ESC-50 metadata at {meta_path}")
        sys.exit(1)

    audio_dir = os.path.join(esc50_root, "audio")

    with open(meta_path, newline="") as f:
        entries = list(csv.DictReader(f))

    # FIX 6: check which files are already in the output CSVs so we can skip them
    # Use config paths so filenames stay consistent with trainer.py
    clap_out  = CLAP_CSV if output_dir == OUTPUT_DIR else os.path.join(output_dir, "sound_data_label1.csv")
    noise_out = NOISE_CSV if output_dir == OUTPUT_DIR else os.path.join(output_dir, "sound_data_label0.csv")
    already_done = _load_processed_files(clap_out) | _load_processed_files(noise_out)
    if already_done:
        print(f"Resuming — {len(already_done)} file(s) already processed, skipping.")

    os.makedirs(output_dir, exist_ok=True)

    # FIX 8: open both CSVs for incremental writing
    clap_mode  = "a" if os.path.exists(clap_out)  else "w"
    noise_mode = "a" if os.path.exists(noise_out) else "w"
    clap_file  = open(clap_out,  clap_mode,  newline="")
    noise_file = open(noise_out, noise_mode, newline="")
    clap_writer  = csv.DictWriter(clap_file,  fieldnames=CSV_FIELDS)
    noise_writer = csv.DictWriter(noise_file, fieldnames=CSV_FIELDS)
    if clap_mode  == "w": clap_writer.writeheader()
    if noise_mode == "w": noise_writer.writeheader()

    skipped = 0
    clap_count = 0
    noise_count = 0
    warnings: list[str] = []   # FIX 9: collect warnings, print after progress line

    print(f"Processing {len(entries)} ESC-50 files...")

    try:
        for i, entry in enumerate(entries):
            category = entry["category"].lower().replace(" ", "_")
            filename  = entry["filename"]
            filepath  = os.path.join(audio_dir, filename)

            # FIX 9: print progress on its own line, warnings collected separately
            print(f"  [{i+1}/{len(entries)}] {filename} ({category})          ", end="\r")

            if category not in clap_classes and category not in noise_classes:
                skipped += 1
                continue

            # FIX 6: skip files already written in a previous run
            if filename in already_done:
                skipped += 1
                continue

            if not os.path.exists(filepath):
                warnings.append(f"  Warning: file not found: {filepath}")
                skipped += 1
                continue

            try:
                waveform   = load_and_resample(filepath)
                mic_values = waveform_to_mic_values(waveform)   # FIX 4: clipped
                label      = 1 if category in clap_classes else 0
                writer     = clap_writer if label == 1 else noise_writer

                for val in mic_values:
                    writer.writerow({
                        "mic_value":   val,
                        "label":       label,
                        "source_file": filename,
                    })

                # FIX 8: flush after each file so data isn't lost on crash
                (clap_file if label == 1 else noise_file).flush()

                if label == 1:
                    clap_count += len(mic_values)
                else:
                    noise_count += len(mic_values)

            except Exception as e:
                warnings.append(f"  Warning: could not process {filename}: {e}")
                skipped += 1

    finally:
        clap_file.close()
        noise_file.close()

    # FIX 9: print all collected warnings after the progress line is done
    print()  # newline after final \r progress
    for w in warnings:
        print(w)

    print(f"Done. Skipped {skipped} file(s) (not in selected classes or already done).")

    if clap_count == 0:
        print(
            "Warning: no clap samples written. "
            f"Selected clap classes were: {sorted(clap_classes)}\n"
            "Run with --list-classes to see available categories."
        )
    if noise_count == 0:
        print("Warning: no noise samples written.")

    print(
        f"\n✅ Dataset ready.\n"
        f"   Clap samples : {clap_count:,}  → {clap_out}\n"
        f"   Noise samples: {noise_count:,}  → {noise_out}\n"
        f"\nRun trainer.py to train your model."
    )


def _load_processed_files(csv_path: str) -> set[str]:
    """Return the set of source_file values already written to a CSV."""
    if not os.path.exists(csv_path):
        return set()
    done: set[str] = set()
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                src = row.get("source_file")
                if src:
                    done.add(src)
    except Exception:
        pass
    return done


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare ESC-50 data for EchoSafe trainer.py"
    )
    # FIX 7: --list-classes flag
    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="fetch and print all ESC-50 category names, then exit",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"directory to write label CSVs (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="keep the downloaded .tar.gz after extraction",
    )
    parser.add_argument(
        "--clap-classes",
        nargs="+",
        default=sorted(CLAP_CLASSES),
        help="ESC-50 category names to treat as clap / label 1",
    )
    parser.add_argument(
        "--noise-classes",
        nargs="+",
        default=sorted(NOISE_CLASSES),
        help="ESC-50 category names to treat as noise / label 0",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # FIX 7: handle --list-classes early, before any download
    if args.list_classes:
        list_classes()
        sys.exit(0)

    download_esc50()
    esc50_root = extract_esc50()
    build_dataset(
        esc50_root=esc50_root,
        clap_classes=set(args.clap_classes),
        noise_classes=set(args.noise_classes),
        output_dir=args.output_dir,
    )

    if not args.keep_archive and os.path.exists(ESC50_ARCHIVE):
        os.remove(ESC50_ARCHIVE)
        print(f"Removed archive {ESC50_ARCHIVE}")