import argparse
import os
import csv
import json
import time
import joblib
import sys
import numpy as np
import librosa
import tensorflow as tf

# Access keras through tf.keras to avoid Pylance import resolution errors on TF 2.16+
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Conv2D = tf.keras.layers.Conv2D
Flatten = tf.keras.layers.Flatten
MaxPooling2D = tf.keras.layers.MaxPooling2D
Adam = tf.keras.optimizers.Adam

from collections import deque

from config import (
    COM_PORT, BAUD_RATE,
    SPEC_H, SPEC_W,
    COOLDOWN, CLAP_WINDOW, FEATURE_COUNT,
    TRIGGER_MULT, SILENCE_MULT,
    MIN_SOUND_SAMPLES, MAX_SOUND_SAMPLES,
    CNN_CONFIDENCE_THRESHOLD,
    ROLLING_NOISE_LEN, BASELINE_UPDATE_EVERY,
    CLAP_MODEL_PATH, CNN_MODEL_FILE, LABEL_MAP_FILE,
    SOUNDS_DB_DIR, LOG_CSV, SAVE_WAV,
    HISTORY_LEN, TRAIN_EPOCHS, AUTOSAVE_INTERVAL,
)

from serial_helper import open_serial, close_serial, reconnect_serial, send

os.makedirs(SOUNDS_DB_DIR, exist_ok=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai_classifier",
        description="EchoSafe real-time sound classifier",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="print debug info: raw mic level, FFT scores, CNN probabilities per class",
    )
    parser.add_argument(
        "--port",
        default=COM_PORT,
        help=f"serial port to use (default: {COM_PORT})",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CNN_CONFIDENCE_THRESHOLD,
        metavar="THRESHOLD",
        help=f"CNN confidence threshold 0-1 (default: {CNN_CONFIDENCE_THRESHOLD})",
    )
    return parser.parse_args()


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_clap_model(path: str):
    if not os.path.exists(path):
        print(f"Error: clap model not found at {path!r}. Run trainer.py first.")
        sys.exit(1)
    return joblib.load(path)


def load_label_map(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            print(f"Loaded label map ({len(data)} labels): {path}")
            return data
    return {}


def save_label_map(label_map: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(label_map, f, indent=2)


def load_or_build_cnn(model_file: str, num_classes: int) -> tf.keras.Model:
    if os.path.exists(model_file):
        print(f"Loading CNN model: {model_file}")
        return tf.keras.models.load_model(model_file)
    print("No saved CNN model found — building a new one.")
    return _build_cnn(num_classes)


def _build_cnn(num_classes: int) -> tf.keras.Model:
    model = Sequential([
        Conv2D(16, (3, 3), activation="relu", input_shape=(SPEC_H, SPEC_W, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def batch_predict(cnn_model, specs_aug: list) -> np.ndarray:
    batch = np.array(specs_aug).reshape(len(specs_aug), SPEC_H, SPEC_W, 1)
    return cnn_model.predict(batch, verbose=0)


def make_spectrograms(waveform: np.ndarray) -> list:
    try:
        w_pitch = librosa.effects.pitch_shift(waveform, sr=16000, n_steps=2)
        w_stretch = librosa.effects.time_stretch(waveform, rate=1.1)
        raw_waves = [waveform, w_pitch, w_stretch]
    except Exception:
        raw_waves = [waveform]

    specs = []
    for w in raw_waves:
        w = w + np.random.normal(0, 5.0, len(w))
        w_norm = w / (np.max(np.abs(w)) + 1e-6)
        spec = librosa.feature.melspectrogram(y=w_norm.astype(float), sr=16000, n_mels=SPEC_H)
        spec_db = librosa.power_to_db(spec)
        spec_resized = librosa.util.fix_length(spec_db, size=SPEC_W, axis=1)
        specs.append(spec_resized.reshape(SPEC_H, SPEC_W, 1))
    return specs


def log_detection(path: str, label: str, peak: float, duration: float,
                  energy: float, sharpness: float) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["time", "label", "peak", "duration", "energy", "sharpness"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "time": round(time.time(), 3),
            "label": label,
            "peak": round(peak, 2),
            "duration": round(duration, 3),
            "energy": round(energy, 2),
            "sharpness": round(sharpness, 4),
        })


def save_wav(waveform: np.ndarray, label: str) -> str | None:
    """Save detection waveform as a .wav file. Returns saved path or None."""
    try:
        import soundfile as sf
        ts = int(time.time())
        safe_label = label.replace(" ", "_").replace("/", "-")
        path = os.path.join(SOUNDS_DB_DIR, f"{safe_label}_{ts}.wav")
        sf.write(path, waveform / 1023.0, 16000)   # normalise Arduino 0-1023 → -1..1
        return path
    except Exception as e:
        print(f"Warning: could not save wav: {e}")
        return None


def shutdown(cnn_model, label_map: dict, ser, session_counts: dict) -> None:
    print("\n\n── Shutting down ──────────────────────────────")
    try:
        if label_map:
            cnn_model.save(CNN_MODEL_FILE)
            save_label_map(label_map, LABEL_MAP_FILE)
            print("💾 Model and label map saved.")
    except Exception as e:
        print(f"Warning: could not save model on exit: {e}")
    try:
        if ser and ser.is_open:
            send(ser, b"0")
            close_serial(ser)
    except Exception:
        pass

    total = sum(session_counts.values())
    print(f"\n📊 Session summary — {total} sound(s) detected:")
    if session_counts:
        for label, count in sorted(session_counts.items(), key=lambda x: -x[1]):
            print(f"   {label}: {count}")
    else:
        print("   (none labelled)")
    print("──────────────────────────────────────────────")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import serial as _serial_mod

    args = parse_args()
    verbose: bool = args.verbose
    confidence_threshold: float = args.confidence
    serial_port: str = args.port

    # ── Load models & state ───────────────────────────────────────────────────
    clap_model = load_clap_model(CLAP_MODEL_PATH)
    label_map: dict = load_label_map(LABEL_MAP_FILE)

    num_classes = max(len(label_map), 2)
    cnn_model = load_or_build_cnn(CNN_MODEL_FILE, num_classes)

    # ── Serial + calibration ──────────────────────────────────────────────────
    ser = open_serial(serial_port, BAUD_RATE)

    print("Calibrating noise floor (3 s) — stay quiet...")
    cal: list[int] = []
    rolling_noise: deque = deque(maxlen=ROLLING_NOISE_LEN)
    start = time.time()
    while time.time() - start < 3.0:
        raw = ser.readline()
        if raw:
            decoded = raw.decode(errors="ignore").strip()
            if decoded.isdigit():
                val = int(decoded)
                cal.append(val)
                rolling_noise.append(val)

    if not cal:
        print("Error: calibration received no data. Check Arduino connection.")
        sys.exit(1)

    baseline_noise = float(np.mean(cal))
    print(f"Baseline: {baseline_noise:.1f}")

    if verbose:
        print(f"[verbose] confidence threshold : {confidence_threshold}")
        print(f"[verbose] trigger level        : {baseline_noise * TRIGGER_MULT:.1f}")
        print(f"[verbose] silence level        : {baseline_noise * SILENCE_MULT:.1f}")

    # ── Runtime state ─────────────────────────────────────────────────────────
    fft_buffer: deque = deque(maxlen=CLAP_WINDOW)
    pred_history: deque = deque(maxlen=HISTORY_LEN)
    current_sound: list[int] = []
    recording = False
    last_save = time.time()
    last_clap = time.time()
    dynamic_baseline = baseline_noise
    baseline_update_counter = 0
    fft_sample_count = 0
    session_counts: dict[str, int] = {}

    print("🎧 Listening...  (Ctrl+C to stop)\n")

    try:
        while True:
            # ── Serial read with reconnection ─────────────────────────────────
            try:
                raw = ser.readline()
            except (_serial_mod.SerialException, OSError):
                ser = reconnect_serial(serial_port, BAUD_RATE)
                if ser is None:
                    break
                rolling_noise.clear()
                continue

            if not raw:
                continue
            decoded = raw.decode(errors="ignore").strip()
            if not decoded.isdigit():
                continue
            mic = int(decoded)

            rolling_noise.append(mic)

            baseline_update_counter += 1
            if baseline_update_counter >= BASELINE_UPDATE_EVERY:
                dynamic_baseline = float(np.mean(rolling_noise))
                baseline_update_counter = 0

            if verbose:
                print(f"[verbose] mic={mic:4d}  baseline={dynamic_baseline:.1f}", end="\r")

            # ── FFT CLAP DETECTION ────────────────────────────────────────────
            clap_detected = False
            fft_buffer.append(mic)
            fft_sample_count += 1

            if fft_sample_count >= CLAP_WINDOW and len(fft_buffer) == CLAP_WINDOW:
                fft_sample_count = 0
                fft_feat = np.abs(np.fft.rfft(np.array(fft_buffer)))
                fft_feat = np.array(
                    [np.mean(fft_feat[i::FEATURE_COUNT]) for i in range(FEATURE_COUNT)]
                ).reshape(1, -1)
                pred = clap_model.predict(fft_feat)[0]

                if verbose:
                    print(f"\n[verbose] FFT window pred={pred}")

                if pred == 1 and time.time() - last_clap > COOLDOWN:
                    print("👏 CLAP detected")
                    send(ser, b"1")
                    last_clap = time.time()
                    clap_detected = True

            # ── SOUND RECORDING ───────────────────────────────────────────────
            if not recording and mic > dynamic_baseline * TRIGGER_MULT:
                recording = True
                current_sound = [mic]
                if verbose:
                    print(f"\n[verbose] Recording started  "
                          f"mic={mic}  threshold={dynamic_baseline * TRIGGER_MULT:.1f}")

            elif recording:
                current_sound.append(mic)

                sound_ended = (
                    mic < dynamic_baseline * SILENCE_MULT
                    and len(current_sound) > MIN_SOUND_SAMPLES
                ) or len(current_sound) >= MAX_SOUND_SAMPLES

                if sound_ended:
                    recording = False
                    waveform = np.array(current_sound, dtype=float)

                    if verbose:
                        print(f"[verbose] Recording ended  samples={len(current_sound)}")

                    specs_aug = make_spectrograms(waveform)

                    # ── CNN prediction ────────────────────────────────────────
                    all_probs = batch_predict(cnn_model, specs_aug)
                    probs_avg = all_probs.mean(axis=0)
                    pred_idx = int(probs_avg.argmax())
                    confidence = float(probs_avg.max())

                    if verbose:
                        reverse_map = {v: k for k, v in label_map.items()}
                        prob_strs = "  ".join(
                            f"{reverse_map.get(i, str(i))}={p:.0%}"
                            for i, p in enumerate(probs_avg)
                        )
                        print(f"[verbose] CNN probs  {prob_strs}")

                    pred_history.clear()
                    pred_history.append(pred_idx)
                    pred_idx_smooth = max(set(pred_history), key=pred_history.count)

                    reverse = {v: k for k, v in label_map.items()}
                    if confidence < confidence_threshold:
                        cnn_label = f"uncertain ({confidence:.0%})"
                    else:
                        cnn_label = reverse.get(pred_idx_smooth, "unknown")

                    final_label = "clap" if clap_detected else cnn_label

                    peak = float(max(current_sound))
                    energy = float(sum(abs(v - baseline_noise) for v in current_sound))
                    duration = len(current_sound) * 0.01
                    sharpness = peak / (baseline_noise + 1)

                    print(f"Detected: {final_label}  "
                          f"(confidence={confidence:.0%}, peak={peak:.0f}, dur={duration:.2f}s)")

                    log_detection(LOG_CSV, final_label, peak, duration, energy, sharpness)
                    session_counts[final_label] = session_counts.get(final_label, 0) + 1

                    if SAVE_WAV:
                        wav_path = save_wav(waveform, final_label)
                        if wav_path and verbose:
                            print(f"[verbose] Saved wav: {wav_path}")

                    # ── Live training ─────────────────────────────────────────
                    label_input = input("Label this sound (or ENTER to skip): ").strip()
                    if label_input:
                        if label_input not in label_map:
                            label_map[label_input] = len(label_map)

                        if len(label_map) > cnn_model.output_shape[-1]:
                            print(f"ℹ️  New label count ({len(label_map)}) — rebuilding CNN.")
                            cnn_model = _build_cnn(len(label_map))

                        y_val = label_map[label_input]
                        batch_arr = np.array(specs_aug).reshape(
                            len(specs_aug), SPEC_H, SPEC_W, 1
                        )
                        y_batch = np.array([y_val] * len(specs_aug))
                        cnn_model.fit(batch_arr, y_batch, epochs=TRAIN_EPOCHS, verbose=0)
                        cnn_model.save(CNN_MODEL_FILE)
                        save_label_map(label_map, LABEL_MAP_FILE)
                        print(f"💾 Sound '{label_input}' trained and saved")

            # ── AUTOSAVE ──────────────────────────────────────────────────────
            if time.time() - last_save > AUTOSAVE_INTERVAL:
                if label_map:
                    cnn_model.save(CNN_MODEL_FILE)
                    save_label_map(label_map, LABEL_MAP_FILE)
                    print("💾 CNN model autosaved")
                last_save = time.time()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        shutdown(cnn_model, label_map, ser, session_counts)


if __name__ == "__main__":
    main()