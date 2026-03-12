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
    MIN_TRAINING_CONFIDENCE,
    SPEC_H, SPEC_W,
    COOLDOWN, CLAP_WINDOW, FFT_STRIDE, FEATURE_COUNT,
    DERIVATIVE_WINDOW, SPIKE_THRESHOLD_MULT, SPIKE_COOLDOWN, SPIKE_MIN_GAP,
    TRIGGER_MULT, SILENCE_MULT,
    MIN_SOUND_SAMPLES, MAX_SOUND_SAMPLES,
    CNN_CONFIDENCE_THRESHOLD,
    ROLLING_NOISE_LEN, BASELINE_UPDATE_EVERY,
    CLAP_MODEL_PATH, SOUND_MODEL_PATH, CNN_MODEL_FILE, LABEL_MAP_FILE,
    SOUNDS_DB_DIR, LOG_CSV, SAVE_WAV,
    HISTORY_LEN, TRAIN_EPOCHS, AUTOSAVE_INTERVAL,
    ARDUINO_SAMPLE_RATE, ARDUINO_SAMPLE_INTERVAL, PC_SAMPLE_RATE,
)

from serial_helper import open_serial, close_serial, reconnect_serial, send, auto_detect_port
from model_versioning import versioned_save
from label_ui import LabelWorker, SoundResult

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
        "--simulate",
        metavar="CSV",
        default=None,
        help="replay a recorded CSV instead of live Arduino serial"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CNN_CONFIDENCE_THRESHOLD,
        metavar="THRESHOLD",
        help=f"CNN confidence threshold 0-1 (default: {CNN_CONFIDENCE_THRESHOLD}) "
             "applies to both FFT and CNN classifiers",
    )
    parser.add_argument(
        "--export-tflite",
        metavar="PATH",
        default=None,
        help="after session ends, export CNN to TFLite at this path "
             f"(default: {None} — no export)"
    )
    return parser.parse_args()


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_sound_model(path: str) -> tuple:
    """
    Load the multi-class sound model bundle.
    Returns (sklearn_model, label_names) where label_names is a list of
    class name strings in index order.

    Handles both:
      - New bundle format: {'model': model, 'label_names': [...]}
      - Legacy format: raw joblib model (binary clap/noise)
    """
    # Try new path first, fall back to legacy clap_model.pkl
    for try_path in [path, CLAP_MODEL_PATH]:
        if os.path.exists(try_path):
            obj = joblib.load(try_path)
            if isinstance(obj, dict) and 'model' in obj:
                return obj['model'], obj.get('label_names', ['noise', 'sound'])
            # Legacy: raw model, assume binary noise/clap
            print(f"Warning: legacy model format at {try_path!r} "
                  "— classes assumed to be ['noise', 'clap']")
            return obj, ['noise', 'clap']
    print(f"Error: sound model not found at {path!r}. Run trainer.py first.")
    sys.exit(1)


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


def expand_cnn_output(model: tf.keras.Model, new_num_classes: int) -> tf.keras.Model:
    """
    Add a new output neuron to the existing CNN rather than rebuilding from scratch.
    Preserves all learned weights in the conv and dense layers — only the final
    softmax layer is replaced, with existing class weights copied over.
    This avoids catastrophic forgetting when a new label is added.
    """
    # Build a fresh model with the new output size
    new_model = _build_cnn(new_num_classes)

    # Copy weights from every layer except the final output layer
    for old_layer, new_layer in zip(model.layers[:-1], new_model.layers[:-1]):
        new_layer.set_weights(old_layer.get_weights())

    # Copy existing output weights into the first N neurons of the new output layer
    old_w, old_b = model.layers[-1].get_weights()   # shape: (64, old_n), (old_n,)
    new_w, new_b = new_model.layers[-1].get_weights()  # shape: (64, new_n), (new_n,)
    old_n = old_w.shape[1]
    new_w[:, :old_n] = old_w
    new_b[:old_n]    = old_b
    new_model.layers[-1].set_weights([new_w, new_b])

    return new_model


def batch_predict(cnn_model, specs_aug: list) -> np.ndarray:
    batch = np.array(specs_aug).reshape(len(specs_aug), SPEC_H, SPEC_W, 1)
    return cnn_model.predict(batch, verbose=0)


def make_spectrograms(waveform: np.ndarray) -> list:
    try:
        w_pitch = librosa.effects.pitch_shift(waveform, sr=PC_SAMPLE_RATE, n_steps=2)
        w_stretch = librosa.effects.time_stretch(waveform, rate=1.1)
        raw_waves = [waveform, w_pitch, w_stretch]
    except Exception:
        raw_waves = [waveform]

    specs = []
    for w in raw_waves:
        w = w + np.random.normal(0, 5.0, len(w))
        w_norm = w / (np.max(np.abs(w)) + 1e-6)
        spec = librosa.feature.melspectrogram(y=w_norm.astype(float), sr=PC_SAMPLE_RATE, n_mels=SPEC_H)
        spec_db = librosa.power_to_db(spec)
        spec_resized = librosa.util.fix_length(spec_db, size=SPEC_W, axis=1)
        specs.append(spec_resized.reshape(SPEC_H, SPEC_W, 1))
    return specs


def log_detection(path: str, label: str, peak: float, duration: float,
                  energy: float, sharpness: float,
                  num_samples: int = 0, baseline: float = 0.0) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["time", "label", "peak", "duration", "energy",
                           "sharpness", "num_samples", "baseline"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "time":        round(time.time(), 3),
            "label":       label,
            "peak":        round(peak, 2),
            "duration":    round(duration, 3),
            "energy":      round(energy, 2),
            "sharpness":   round(sharpness, 4),
            "num_samples": num_samples,
            "baseline":    round(baseline, 2),
        })


def save_wav(waveform: np.ndarray, label: str) -> str | None:
    """Save detection waveform as a .wav file. Returns saved path or None."""
    try:
        import soundfile as sf
        ts = int(time.time())
        safe_label = label.replace(" ", "_").replace("/", "-")
        path = os.path.join(SOUNDS_DB_DIR, f"{safe_label}_{ts}.wav")
        sf.write(path, waveform / 1023.0, ARDUINO_SAMPLE_RATE)   # normalise Arduino 0-1023 → -1..1
        return path
    except Exception as e:
        print(f"Warning: could not save wav: {e}")
        return None


def _simulate_source(csv_path: str):
    """
    Replay a recorded CSV as a fake serial source for --simulate mode.
    Yields integers at ARDUINO_SAMPLE_RATE so timing mirrors a live session.
    """
    import csv as _csv
    if not os.path.exists(csv_path):
        print(f"Error: simulate file not found: {csv_path!r}")
        sys.exit(1)
    with open(csv_path, newline='') as f:
        reader = _csv.DictReader(f)
        if 'mic_value' not in (reader.fieldnames or []):
            print(f"Error: CSV has no 'mic_value' column: {csv_path!r}")
            sys.exit(1)
        for row in reader:
            val = row['mic_value'].strip()
            if val.isdigit():
                yield int(val)
            time.sleep(ARDUINO_SAMPLE_INTERVAL)


def export_tflite(keras_model: tf.keras.Model, out_path: str) -> None:
    """
    Convert a trained Keras CNN to TFLite so listener.py can use it.
    Saves to out_path (e.g. 'sound_model.tflite').
    """
    print(f"Converting CNN to TFLite → {out_path}...")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]   # float16 quantisation
    tflite_model = converter.convert()
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"✅ TFLite model saved: {out_path} ({size_kb:.1f} KB)")


def shutdown(cnn_model, label_map: dict, ser, session_counts: dict) -> None:
    print("\n\n── Shutting down ──────────────────────────────")
    try:
        if label_map:
            versioned_save(CNN_MODEL_FILE, lambda p, m=cnn_model: m.save(p))
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
    sound_model, fft_label_names = load_sound_model(SOUND_MODEL_PATH)
    label_map: dict = load_label_map(LABEL_MAP_FILE)

    num_classes = max(len(label_map), 2)
    cnn_model = load_or_build_cnn(CNN_MODEL_FILE, num_classes)

    # ── Label worker (must start before serial so it's ready immediately) ────
    worker = LabelWorker(label_map, verbose=verbose)
    worker.start()

    # ── Serial + calibration ──────────────────────────────────────────────────
    # ── Simulate mode or live serial ─────────────────────────────────────
    simulate_iter = None
    if args.simulate:
        print(f"🔁 Simulate mode — replaying {args.simulate!r}")
        simulate_iter = _simulate_source(args.simulate)
        ser = None
    else:
        if serial_port == COM_PORT:
            detected = auto_detect_port(BAUD_RATE)
            if detected and detected != serial_port:
                print(f"Auto-detected Arduino on {detected} (overrides config COM_PORT)")
                serial_port = detected
        ser = open_serial(serial_port, BAUD_RATE, timeout=2.0)

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
    fft_buffer: deque     = deque(maxlen=CLAP_WINDOW)
    # Derivative spike detector state
    deriv_buf: deque      = deque(maxlen=DERIVATIVE_WINDOW + 1)  # +1 for diff
    spike_count           = 0       # spikes detected in current sound event
    last_spike_time       = 0.0
    last_spike_sample     = -SPIKE_MIN_GAP  # sample index of last spike
    sample_index          = 0       # global sample counter for spike gap check
    pred_history: deque = deque(maxlen=HISTORY_LEN)
    current_sound: list[int] = []
    recording = False
    last_save = time.time()
    last_clap = time.time()
    dynamic_baseline = baseline_noise
    baseline_update_counter = 0
    fft_sample_count     = 0       # counts samples since last FFT window was run
    session_counts: dict[str, int] = {}

    print("🎧 Listening...  (Ctrl+C to stop)\n")

    try:
        while True:
            # ── Read next sample — serial or simulate ────────────────────────
            if simulate_iter is not None:
                try:
                    mic = next(simulate_iter)
                except StopIteration:
                    print("\n🔁 Simulation complete.")
                    break
            else:
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

            # ── DERIVATIVE SPIKE DETECTOR ─────────────────────────────────────
            # Per-sample transient onset detector. A spike fires when the
            # smoothed first-derivative exceeds baseline * SPIKE_THRESHOLD_MULT.
            # A second spike while already recording = probable simultaneous event.
            fft_detected_label: str | None = None
            sample_index += 1
            deriv_buf.append(mic)

            if len(deriv_buf) == deriv_buf.maxlen:
                diffs = [abs(deriv_buf[i+1] - deriv_buf[i])
                         for i in range(len(deriv_buf) - 1)]
                smooth_deriv = sum(diffs) / len(diffs)
                spike_threshold = dynamic_baseline * SPIKE_THRESHOLD_MULT

                if (smooth_deriv > spike_threshold
                        and time.time() - last_spike_time > SPIKE_COOLDOWN
                        and sample_index - last_spike_sample > SPIKE_MIN_GAP):
                    spike_count += 1
                    last_spike_time   = time.time()
                    last_spike_sample = sample_index
                    if verbose:
                        print(f"\n[verbose] SPIKE #{spike_count}  "
                              f"deriv={smooth_deriv:.1f}  "
                              f"threshold={spike_threshold:.1f}")
                    if recording and spike_count >= 2:
                        print(f"⚡ Simultaneous event detected "
                              f"(spike #{spike_count} mid-recording — "
                              "two sounds may be overlapping)")

            if not recording:
                spike_count = 0

            # ── FFT SOUND DETECTION (overlapping windows) ─────────────────────
            # Each window advances by FFT_STRIDE samples (< CLAP_WINDOW),
            # giving CLAP_WINDOW/FFT_STRIDE-fold overlap so sounds at window
            # boundaries still receive a full-window classification.
            fft_buffer.append(mic)
            fft_sample_count += 1

            if fft_sample_count >= FFT_STRIDE and len(fft_buffer) == CLAP_WINDOW:
                fft_sample_count = 0
                fft_feat = np.abs(np.fft.rfft(np.array(fft_buffer)))
                fft_feat = np.array(
                    [np.mean(fft_feat[i::FEATURE_COUNT]) for i in range(FEATURE_COUNT)]
                ).reshape(1, -1)
                fft_pred_idx = int(sound_model.predict(fft_feat)[0])
                fft_probs    = sound_model.predict_proba(fft_feat)[0]
                fft_conf     = float(fft_probs[fft_pred_idx])
                fft_label    = (fft_label_names[fft_pred_idx]
                                if fft_pred_idx < len(fft_label_names) else 'unknown')

                if verbose:
                    prob_str = '  '.join(
                        f'{fft_label_names[i] if i < len(fft_label_names) else i}='
                        f'{fft_probs[i]:.0%}'
                        for i in range(len(fft_probs))
                    )
                    print(f"\n[verbose] FFT: {fft_label} ({fft_conf:.0%})  {prob_str}")

                # Trigger LED for any non-noise sound above confidence threshold
                is_noise = fft_label in ('noise', 'background', 'silence')
                if (not is_noise
                        and fft_conf >= confidence_threshold
                        and time.time() - last_clap > COOLDOWN):
                    print(f"🔊 FFT detected: {fft_label} ({fft_conf:.0%})")
                    send(ser, b"1")
                    last_clap = time.time()
                    fft_detected_label = fft_label

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

                    # FFT label takes precedence if it fired on this window;
                    # otherwise use CNN label
                    final_label = fft_detected_label if fft_detected_label else cnn_label

                    peak = float(max(current_sound))
                    energy = float(sum(abs(v - baseline_noise) for v in current_sound))
                    duration = len(current_sound) * ARDUINO_SAMPLE_INTERVAL
                    sharpness = peak / (baseline_noise + 1)

                    print(f"Detected: {final_label}  "
                          f"(confidence={confidence:.0%}, peak={peak:.0f}, dur={duration:.2f}s)")

                    log_detection(LOG_CSV, final_label, peak, duration, energy, sharpness,
                                  num_samples=len(current_sound), baseline=dynamic_baseline)
                    session_counts[final_label] = session_counts.get(final_label, 0) + 1

                    wav_path = None
                    if SAVE_WAV and not wav_saving_disabled:
                        wav_path = save_wav(waveform, final_label)
                        if wav_path and verbose:
                            print(f"[verbose] Saved wav: {wav_path}")
                        elif wav_path is None:
                            # save_wav() failed — check if disk is full
                            import shutil
                            free = shutil.disk_usage(SOUNDS_DB_DIR).free
                            if free < 10 * 1024 * 1024:  # < 10 MB
                                wav_saving_disabled = True
                                print("\n❌ Disk full — wav saving disabled for this session.")
                                print("   Free up space and restart to re-enable.")

                    # ── Confidence floor warning ──────────────────────────────
                    if confidence < 0.4 and not fft_detected_label:
                        print(f"   ⚠️  Very low confidence ({confidence:.0%}) — "
                              "labelling this sound may hurt model accuracy.")

                    # ── Submit to label worker (non-blocking) ─────────────────
                    worker.submit(SoundResult(
                        waveform=waveform,
                        specs_aug=specs_aug,
                        final_label=final_label,
                        confidence=confidence,
                        peak=peak,
                        duration=duration,
                        wav_path=wav_path,
                    ))

            # ── Process label results from worker ──────────────────────────────
            result = worker.poll()
            if result is not None:
                user_label = result.user_label
                sound      = result.sound

                # Sync label_map — worker may have added a new label during prompt
                label_map = worker.label_map

                if len(label_map) > cnn_model.output_shape[-1]:
                    print(f"ℹ️  New label '{user_label}' — expanding CNN output layer "
                          f"({cnn_model.output_shape[-1]} → {len(label_map)} classes).")
                    cnn_model = expand_cnn_output(cnn_model, len(label_map))

                y_val     = label_map[user_label]
                batch_arr = np.array(sound.specs_aug).reshape(
                    len(sound.specs_aug), SPEC_H, SPEC_W, 1
                )
                y_batch = np.array([y_val] * len(sound.specs_aug))
                cnn_model.fit(batch_arr, y_batch, epochs=TRAIN_EPOCHS, verbose=0)
                versioned_save(CNN_MODEL_FILE, lambda p, m=cnn_model: m.save(p))
                save_label_map(label_map, LABEL_MAP_FILE)
                print(f"💾 Sound '{user_label}' trained and saved")

            # ── AUTOSAVE ──────────────────────────────────────────────────────
            if time.time() - last_save > AUTOSAVE_INTERVAL:
                if label_map:
                    versioned_save(CNN_MODEL_FILE, lambda p, m=cnn_model: m.save(p))
                    save_label_map(label_map, LABEL_MAP_FILE)
                    print("💾 CNN model autosaved")
                last_save = time.time()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        print("Waiting for label worker to finish...")
        worker.stop()
        shutdown(cnn_model, label_map, ser, session_counts)
        if args.export_tflite and label_map:
            try:
                export_tflite(cnn_model, args.export_tflite)
            except Exception as e:
                print(f"Warning: TFLite export failed: {e}")


if __name__ == "__main__":
    main()