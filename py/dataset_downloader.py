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
from config import PC_SAMPLE_RATE as SAMPLE_RATE, SOUNDS_DB_DIR as OUTPUT_DIR, TRAINING_DATA_DIR


# ── Constants ──────────────────────────────────────────────────────────────────

ESC50_URL     = "https://github.com/karoldvl/ESC-50/archive/master.tar.gz"
ESC50_DIR     = "ESC-50-master"
ESC50_ARCHIVE = "esc50.tar.gz"

# ESC-50 metadata URL — used by --list-classes without downloading the full archive
ESC50_META_URL = (
    "https://raw.githubusercontent.com/karoldvl/ESC-50/master/meta/esc50.csv"
)

# Default ESC-50 categories to download if --classes / --label-map are not given.
# {echosafe_label: {esc50_category, ...}}
DEFAULT_CLASS_MAP: dict[str, set[str]] = {
    "clap":  {"clapping", "hand_clapping"},
    "knock": {"door_knock", "knocking_on_door"},
    "noise": {"wind", "rain", "thunderstorm", "water_drops",
              "sea_waves", "crackling_fire", "crickets", "birds", "silence"},
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
    esc_to_label: dict[str, str] = {}
    for lbl, cats in DEFAULT_CLASS_MAP.items():
        for c in cats:
            esc_to_label[c] = lbl
    for cat, count in sorted(categories.items()):
        lbl = esc_to_label.get(cat, "")
        marker = f"  ← default label: {lbl!r}" if lbl else ""
        print(f"  {cat:<30} ({count} files){marker}")
    print(
        "\nUse --classes ESC50_CAT [..] or --label-map LABEL=CAT[,..] to customise."
    )


# ── Build dataset ──────────────────────────────────────────────────────────────

def build_dataset(
    esc50_root: str,
    class_map: dict[str, set[str]],
    output_dir: str,
) -> None:
    """Convert ESC-50 audio → one label_<n>.csv per sound class."""
    meta_path = os.path.join(esc50_root, "meta", "esc50.csv")
    if not os.path.exists(meta_path):
        print(f"Error: ESC-50 metadata not found at {meta_path}")
        sys.exit(1)
    audio_dir = os.path.join(esc50_root, "audio")
    with open(meta_path, newline="") as f:
        entries = list(csv.DictReader(f))

    # Build reverse map: esc50_category → echosafe_label
    cat_to_label: dict[str, str] = {}
    for label_name, cats in class_map.items():
        for cat in cats:
            cat_to_label[cat.lower().replace(" ", "_")] = label_name

    all_labels = sorted(class_map.keys())
    out_paths  = {name: os.path.join(output_dir, f"label_{name}.csv") for name in all_labels}

    # Resume: collect already-processed filenames
    already_done: set[str] = set()
    for path in out_paths.values():
        already_done |= _load_processed_files(path)
    if already_done:
        print(f"Resuming — {len(already_done)} file(s) already processed, skipping.")

    os.makedirs(output_dir, exist_ok=True)

    # Open one CSV per label
    open_files: dict[str, object] = {}
    writers: dict[str, csv.DictWriter] = {}
    for name, path in out_paths.items():
        mode = "a" if os.path.exists(path) else "w"
        fh = open(path, mode, newline="")
        open_files[name] = fh
        writers[name] = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if mode == "w":
            writers[name].writeheader()

    skipped = 0
    counts: dict[str, int] = {name: 0 for name in all_labels}
    warnings_list: list[str] = []

    print(f"Processing {len(entries)} ESC-50 files → labels: {all_labels}")

    try:
        for i, entry in enumerate(entries):
            category = entry["category"].lower().replace(" ", "_")
            filename  = entry["filename"]
            filepath  = os.path.join(audio_dir, filename)
            print(f"  [{i+1}/{len(entries)}] {filename} ({category})          ", end="\r")

            if category not in cat_to_label:
                skipped += 1
                continue
            if filename in already_done:
                skipped += 1
                continue
            if not os.path.exists(filepath):
                warnings_list.append(f"  Warning: file not found: {filepath}")
                skipped += 1
                continue

            try:
                lbl_name   = cat_to_label[category]
                waveform   = load_and_resample(filepath)
                mic_values = waveform_to_mic_values(waveform)
                for val in mic_values:
                    writers[lbl_name].writerow({
                        "mic_value":   val,
                        "label":       lbl_name,
                        "source_file": filename,
                    })
                open_files[lbl_name].flush()
                counts[lbl_name] += len(mic_values)
            except Exception as e:
                warnings_list.append(f"  Warning: could not process {filename}: {e}")
                skipped += 1

    finally:
        for fh in open_files.values():
            fh.close()

    print()
    for w in warnings_list:
        print(w)
    print(f"Done. Skipped {skipped} file(s).")
    for name in all_labels:
        if counts[name] == 0:
            print(f"  ⚠️  No samples for label {name!r}. Check --list-classes.")
        else:
            print(f"  ✅ {name}: {counts[name]:,} samples → {out_paths[name]}")
    print("\nRun trainer.py to train your model.")

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
        "--classes",
        nargs="+", default=None, metavar="CAT",
        help="ESC-50 category names to download (each becomes its own label).",
    )
    parser.add_argument(
        "--label-map", nargs="+", default=None, metavar="LABEL=CAT[,CAT]",
        help="explicit label→category mapping e.g. knock=door_knock noise=wind,rain.",
    )
    return parser.parse_args()


def _build_class_map(args) -> dict[str, set[str]]:
    if args.label_map:
        cm: dict[str, set[str]] = {}
        for token in args.label_map:
            label, _, cats_str = token.partition("=")
            if not cats_str:
                print(f"Error: --label-map token {token!r} must be label=cat[,cat2]")
                sys.exit(1)
            cm[label.strip()] = {c.strip() for c in cats_str.split(",")}
        return cm
    if args.classes:
        return {cat.lower().replace(" ", "_"): {cat} for cat in args.classes}
    return DEFAULT_CLASS_MAP


if __name__ == "__main__":
    args = parse_args()
    if args.list_classes:
        list_classes()
        sys.exit(0)
    class_map = _build_class_map(args)
    print(f"Label map: { {k: sorted(v) for k, v in class_map.items()} }")
    download_esc50()
    esc50_root = extract_esc50()
    build_dataset(esc50_root=esc50_root, class_map=class_map, output_dir=args.output_dir)
    if not args.keep_archive and os.path.exists(ESC50_ARCHIVE):
        os.remove(ESC50_ARCHIVE)
        print(f"Removed archive {ESC50_ARCHIVE}")