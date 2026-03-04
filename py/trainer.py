"""
trainer.py
----------
Trains and evaluates the EchoSafe logistic regression clap detector.

Can be used two ways:
  1. Standalone: python trainer.py [options]
  2. Imported:   from trainer import train, extract_features

Usage:
    python trainer.py
    python trainer.py --clap-csv sounds_db/sound_data_label1.csv
    python trainer.py --model my_model.pkl --window-size 128 --verbose
"""

import argparse
import os
import sys
from collections import deque          # FIX 1: deque for bounded buffer
from collections.abc import Iterator

import joblib
import numpy as np
import pandas as pd
from numpy.fft import rfft
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from source import DataEntry
from config import (
    FEATURE_COUNT,
    CLI_WINDOW_SIZE as WINDOW_SIZE,
    CLAP_MODEL_PATH,
    SOUNDS_DB_DIR,
    CLAP_CSV,    # FIX 5 / CSV redirect: paths now point inside sounds_db/
    NOISE_CSV,
)

# FIX 9: initialize_model moved to where TFLite loading actually belongs.
# trainer.py only produces sklearn models — TFLite is listener.py's concern.
# Kept here as a thin compatibility shim in case other files still import it.
def initialize_model(model_path: str):
    """
    Load a trained model from disk.
    .tflite  → TFLite interpreter (for listener.py inference)
    .pkl     → sklearn model via joblib (for clap detection)
    """
    import tensorflow as tf
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path!r}. "
            "Run trainer.py first to produce the model."
        )
    if model_path.endswith(".tflite"):
        interp = tf.lite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        return interp
    return joblib.load(model_path)


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(signal: np.ndarray, feature_count: int | None = None) -> np.ndarray:
    # FIX 4: don't capture FEATURE_COUNT as a default arg at definition time —
    # read it directly so runtime config changes are always respected
    fc = feature_count if feature_count is not None else FEATURE_COUNT
    fft_vals = np.abs(rfft(signal))
    return np.array(
        [np.mean(fft_vals[i::fc]) for i in range(fc)],
        dtype=np.float32,
    )


# ── Online training (called from main.py CLI pipeline) ────────────────────────

def train(
    model,
    output,
    window_size: int,
    feature_count: int,
    source: Iterator[DataEntry],
    model_out: str = CLAP_MODEL_PATH,   # FIX 3: caller can specify output path
) -> None:
    """
    Online training pass: consume DataEntry records from source, extract
    features in sliding windows, and fit the model.

    Args:
        model:        sklearn model with a .fit() method
        output:       binary-writable stream for status messages
        window_size:  samples per FFT window
        feature_count: number of FFT features per window
        source:       iterator of DataEntry records
        model_out:    path to save the fitted model
    """
    # FIX 1: deque caps memory — buffer never grows beyond window_size entries
    buffer: deque[int] = deque(maxlen=window_size)
    X: list[np.ndarray] = []
    y: list[int] = []

    for entry in source:
        buffer.append(entry.microphone)
        if len(buffer) == window_size:
            window = np.array(buffer, dtype=float)
            X.append(extract_features(window, feature_count))

            # FIX 7: warn when confidence is borderline rather than silently
            # converting a fractional score to the wrong binary label
            conf = entry.clap_confidence
            if 0.4 <= conf <= 0.6:
                _write(output, f"Warning: borderline confidence {conf:.2f} "
                               f"labelled as {'clap' if conf > 0.5 else 'noise'}\n")
            y.append(1 if conf > 0.5 else 0)

    if not X:
        _write(output, "Training produced no windows — source may have too few samples.\n")
        return

    X_arr = np.array(X)
    y_arr = np.array(y)

    if hasattr(model, "fit"):
        model.fit(X_arr, y_arr)
        # FIX 3: save to caller-specified path, not hardcoded MODEL_OUT
        os.makedirs(os.path.dirname(os.path.abspath(model_out)), exist_ok=True)
        joblib.dump(model, model_out)
        _write(output, f"Training complete. Model saved to {model_out}\n")
    else:
        _write(output, "Model type does not support .fit() — training skipped.\n")


def _write(output, msg: str) -> None:
    """
    FIX 2: single consistent write helper — removes the broken
    hasattr(output, 'mode') conditional (always True for binary streams)
    and the inconsistent encode(errors='ignore') vs encode() usage.
    """
    output.write(msg.encode(errors="replace"))


# ── Standalone CSV training ────────────────────────────────────────────────────

def _train_from_csvs(
    clap_csv: str = CLAP_CSV,       # FIX 5: default paths now in sounds_db/
    noise_csv: str = NOISE_CSV,
    model_out: str = CLAP_MODEL_PATH,
    window_size: int = WINDOW_SIZE,
    feature_count: int = FEATURE_COUNT,
    verbose: bool = False,
) -> None:
    # Ensure output directory exists
    os.makedirs(SOUNDS_DB_DIR, exist_ok=True)

    for path in (clap_csv, noise_csv):
        if not os.path.exists(path):
            print(f"Error: training data file not found: {path!r}")
            print(f"Expected files in {SOUNDS_DB_DIR}/")
            print("Run recording.py or dataset_downloader.py first.")
            sys.exit(1)

    # FIX 8: load and process each CSV separately to halve peak memory usage
    print(f"Loading {clap_csv}...")
    clap_df  = pd.read_csv(clap_csv)
    print(f"Loading {noise_csv}...")
    noise_df = pd.read_csv(noise_csv)

    X: list[np.ndarray] = []
    y: list[int] = []

    for label, df in ((1, clap_df), (0, noise_df)):
        if "mic_value" not in df.columns:
            print(f"Error: CSV missing 'mic_value' column: {clap_csv if label == 1 else noise_csv}")
            sys.exit(1)
        subset = df["mic_value"].values
        n_windows = max(0, len(subset) - window_size)
        if verbose:
            print(f"  label={label}: {len(subset)} samples → {n_windows} windows")
        for i in range(n_windows):
            window = subset[i : i + window_size]
            X.append(extract_features(window, feature_count))
            y.append(label)

    if not X:
        print(
            f"Error: not enough data to create any windows. "
            f"Need at least {window_size} samples per class."
        )
        sys.exit(1)

    X_arr = np.array(X)
    y_arr = np.array(y)

    if verbose:
        print(f"\nTotal windows: {len(X_arr)}  "
              f"(clap={sum(y_arr==1)}, noise={sum(y_arr==0)})")

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
    )

    print("\nTraining logistic regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"✅ Accuracy: {accuracy * 100:.2f}%")

    if verbose:
        print("\nClassification report:")
        print(classification_report(y_test, preds, target_names=["noise", "clap"]))

    joblib.dump(model, model_out)
    print(f"💾 Saved model to {model_out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    # FIX 6: argparse so all settings can be overridden at runtime
    parser = argparse.ArgumentParser(
        prog="trainer",
        description="Train EchoSafe clap detection model from labelled CSV data",
    )
    parser.add_argument(
        "--clap-csv",
        default=CLAP_CSV,
        help=f"CSV of clap examples, label=1 (default: {CLAP_CSV})",
    )
    parser.add_argument(
        "--noise-csv",
        default=NOISE_CSV,
        help=f"CSV of noise examples, label=0 (default: {NOISE_CSV})",
    )
    parser.add_argument(
        "--model",
        default=CLAP_MODEL_PATH,
        help=f"output model path (default: {CLAP_MODEL_PATH})",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help=f"FFT window size in samples (default: {WINDOW_SIZE})",
    )
    parser.add_argument(
        "--feature-count",
        type=int,
        default=FEATURE_COUNT,
        help=f"number of FFT features per window (default: {FEATURE_COUNT})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="print window counts and full classification report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _train_from_csvs(
        clap_csv=args.clap_csv,
        noise_csv=args.noise_csv,
        model_out=args.model,
        window_size=args.window_size,
        feature_count=args.feature_count,
        verbose=args.verbose,
    )
    return 0


# FIX 10: propagate exit code to shell
if __name__ == "__main__":
    sys.exit(main())