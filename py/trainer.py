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
import glob
import os
import sys
from collections import deque          # FIX 1: deque for bounded buffer
from collections.abc import Iterator

import joblib
from model_versioning import versioned_save
import numpy as np
import pandas as pd
from numpy.fft import rfft
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from source import DataEntry
from config import (
    FEATURE_COUNT,
    CLI_WINDOW_SIZE as WINDOW_SIZE,
    CLAP_MODEL_PATH,
    SOUND_MODEL_PATH,
    CLAP_CONFIDENCE_BORDER_LOW,
    CLAP_CONFIDENCE_BORDER_HIGH,
    SOUNDS_DB_DIR,
    TRAINING_DATA_DIR,
    CLAP_CSV,
    NOISE_CSV,
    ARDUINO_SAMPLE_RATE,
    FREQ_BAND_LOW_HZ, FREQ_BAND_MID_HZ, FREQ_BAND_HIGH_HZ,
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

def _hz_to_bin(hz: float, sample_rate: int, window: int) -> int:
    """Convert a frequency in Hz to an FFT bin index."""
    return int(round(hz * window / sample_rate))


def extract_features(signal: np.ndarray, feature_count: int | None = None,
                     window: int | None = None) -> np.ndarray:
    """
    Extract the feature vector used for both training and inference.

    Features (in order):
        1. FEATURE_COUNT normalised FFT bin averages  — spectral shape
        2. 3 frequency band energy ratios             — distance-invariant alarm cues
           (low 0–500 Hz, mid 500–1500 Hz, high 1500–4000 Hz)

    Normalisation: the FFT bins are divided by their total so the vector
    represents spectral *shape* rather than *loudness*.  The band ratios are
    fractions of total energy and are therefore also volume-invariant.

    This means a smoke alarm at 5 metres through a wall produces nearly the
    same feature vector as one at 1 metre — only the amplitude changes, not
    the shape.  The classifier learns shape, so it fires correctly at distance.
    """
    fc  = feature_count if feature_count is not None else FEATURE_COUNT
    win = window if window is not None else WINDOW_SIZE
    sr  = ARDUINO_SAMPLE_RATE

    fft_raw = np.abs(rfft(signal))

    # Binned averages, normalised by sum
    bins = np.array([np.mean(fft_raw[i::fc]) for i in range(fc)], dtype=np.float32)
    bins = bins / (bins.sum() + 1e-9)

    # Frequency band energy ratios
    energy = fft_raw ** 2
    total  = energy.sum() + 1e-9

    def band(lo_hz: float, hi_hz: float) -> float:
        lo = _hz_to_bin(lo_hz, sr, win)
        hi = _hz_to_bin(hi_hz, sr, win)
        return float(energy[lo:hi].sum() / total)

    bands = np.array([
        band(*FREQ_BAND_LOW_HZ),
        band(*FREQ_BAND_MID_HZ),
        band(*FREQ_BAND_HIGH_HZ),
    ], dtype=np.float32)

    return np.concatenate([bins, bands])


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
            if CLAP_CONFIDENCE_BORDER_LOW <= conf <= CLAP_CONFIDENCE_BORDER_HIGH:
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
        versioned_save(model_out, lambda p, m=model: joblib.dump(m, p))
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

def scan_training_dir(training_dir: str) -> dict[str, str]:
    """
    Scan a directory for label_*.csv files and return a dict of
    {label_name: csv_path}.

    Convention: label_clap.csv → label 'clap'
                label_knock.csv → label 'knock'
                label_noise.csv → label 'noise'

    Also accepts the legacy sound_data_label0.csv / sound_data_label1.csv
    filenames for backward compatibility.
    """
    found: dict[str, str] = {}

    # New convention: label_<name>.csv
    for path in sorted(glob.glob(os.path.join(training_dir, 'label_*.csv'))):
        name = os.path.basename(path)[len('label_'):-len('.csv')]
        found[name] = path

    # Legacy: sound_data_label0.csv (noise) / sound_data_label1.csv (clap)
    legacy = {
        'noise': os.path.join(training_dir, 'sound_data_label0.csv'),
        'clap':  os.path.join(training_dir, 'sound_data_label1.csv'),
    }
    for name, path in legacy.items():
        if os.path.exists(path) and name not in found:
            found[name] = path

    return found


def _train_from_csvs(
    training_dir: str = TRAINING_DATA_DIR,
    model_out: str = SOUND_MODEL_PATH,
    window_size: int = WINDOW_SIZE,
    feature_count: int = FEATURE_COUNT,
    verbose: bool = False,
    dry_run: bool = False,
    export_tflite_path: str | None = None,
) -> None:
    # Ensure output directory exists
    os.makedirs(training_dir, exist_ok=True)

    label_csvs = scan_training_dir(training_dir)
    if not label_csvs:
        print(f"Error: no label_*.csv files found in {training_dir!r}")
        print("Run: python recording.py --label <name> --duration 60")
        sys.exit(1)

    # Build label → integer index map (sorted for reproducibility)
    label_names = sorted(label_csvs.keys())
    label_index = {name: i for i, name in enumerate(label_names)}
    print(f"Found {len(label_names)} class(es): {label_names}")

    X: list[np.ndarray] = []
    y: list[int] = []

    for name in label_names:
        path = label_csvs[name]
        print(f"Loading {path}...")
        df = pd.read_csv(path)
        if "mic_value" not in df.columns:
            print(f"Error: CSV missing 'mic_value' column: {path}")
            sys.exit(1)
        subset = df["mic_value"].values
        n_windows = max(0, len(subset) - window_size)
        if verbose:
            print(f"  {name}: {len(subset)} samples → {n_windows} windows")
        for i in range(n_windows):
            window = subset[i : i + window_size]
            X.append(extract_features(window, feature_count))
            y.append(label_index[name])

    if not X:
        print(
            f"Error: not enough data to create any windows. "
            f"Need at least {window_size} samples per class."
        )
        sys.exit(1)

    X_arr = np.array(X)
    y_arr = np.array(y)

    if verbose:
        counts = {name: int(sum(y_arr == i)) for name, i in label_index.items()}
        print(f"\nTotal windows: {len(X_arr)}  {counts}")

    # Class balance check — warn if any class outnumbers another by >3:1
    counts_list = [int(sum(y_arr == i)) for i in range(len(label_names))]
    max_count = max(counts_list)
    min_count = max(min(counts_list), 1)
    ratio = max_count / min_count
    if ratio > 3.0:
        dominant   = label_names[counts_list.index(max_count)]
        underrepresented = label_names[counts_list.index(min(counts_list))]
        print(f"\n⚠️  Class imbalance (ratio {ratio:.1f}:1): '{dominant}' dominates. "
              f"Consider collecting more '{underrepresented}' samples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
    )

    print("\nTraining logistic regression...")
    n_classes = len(label_names)
    model = LogisticRegression(
        max_iter=1000,
        multi_class='ovr',     # one-vs-rest for N-class classification
        C=1.0,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"✅ Accuracy: {accuracy * 100:.2f}%")

    # Confusion matrix — always shown
    cm = confusion_matrix(y_test, preds, labels=list(range(n_classes)))
    col_w = max(len(n) for n in label_names) + 2
    print(f"\n   Confusion matrix (rows=actual, cols=predicted):")
    header = ''.join(f'{n:>{col_w}}' for n in label_names)
    print(f"   {'':>{col_w}}{header}")
    for i, row_name in enumerate(label_names):
        row = ''.join(f'{cm[i][j]:>{col_w}}' for j in range(n_classes))
        print(f"   {row_name:>{col_w}}{row}")
    # Highlight any off-diagonal values
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i][j] > 0:
                print(f"   ⚠️  {cm[i][j]} '{label_names[i]}' sample(s) "
                      f"misclassified as '{label_names[j]}'")

    if verbose:
        print("\nClassification report:")
        print(classification_report(y_test, preds, target_names=label_names))

    if dry_run:
        print("\n🔍 Dry run — model NOT saved. Remove --dry-run to save.")
    else:
        # Save model bundled with label_names so classifier knows class names
        bundle = {'model': model, 'label_names': label_names}
        versioned_save(model_out, lambda p, b=bundle: joblib.dump(b, p))
        print(f"💾 Saved model to {model_out}  (classes: {label_names})")

    if export_tflite_path and not dry_run:
        _export_sklearn_tflite(model, X_train.shape[1], export_tflite_path, label_names)


def _export_sklearn_tflite(model, n_features: int, out_path: str, label_names: list[str] | None = None) -> None:
    """
    Wrap a trained sklearn LogisticRegression as a TFLite model so
    listener.py can use it without scikit-learn at runtime.
    """
    try:
        import tensorflow as tf
        import numpy as np
        # Build a tiny Keras model that replicates the LR decision boundary
        inp = tf.keras.Input(shape=(n_features,))
        out = tf.keras.layers.Dense(
            1, activation='sigmoid',
            kernel_initializer=tf.keras.initializers.Constant(model.coef_),
            bias_initializer=tf.keras.initializers.Constant(model.intercept_),
            trainable=False,
        )(inp)
        keras_model = tf.keras.Model(inp, out)
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_bytes = converter.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_bytes)
        print(f"✅ TFLite model exported: {out_path} ({len(tflite_bytes)//1024} KB)")
    except Exception as e:
        print(f"Warning: TFLite export failed: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    # FIX 6: argparse so all settings can be overridden at runtime
    parser = argparse.ArgumentParser(
        prog="trainer",
        description="Train EchoSafe clap detection model from labelled CSV data",
    )
    parser.add_argument(
        "--training-dir",
        default=TRAINING_DATA_DIR,
        help=f"directory containing label_*.csv files (default: {TRAINING_DATA_DIR})",
    )
    parser.add_argument(
        "--model",
        default=SOUND_MODEL_PATH,
        help=f"output model path (default: {SOUND_MODEL_PATH})",
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
        "--export-tflite",
        metavar="PATH",
        default=None,
        help="also export model as TFLite to PATH (for use with listener.py)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="evaluate model without saving — shows confusion matrix and accuracy only",
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
        training_dir=args.training_dir,
        model_out=args.model,
        window_size=args.window_size,
        feature_count=args.feature_count,
        verbose=args.verbose,
        dry_run=args.dry_run,
        export_tflite_path=args.export_tflite,
    )
    return 0


# FIX 10: propagate exit code to shell
if __name__ == "__main__":
    sys.exit(main())