import argparse
import atexit
import os
import time
from collections import deque

import numpy as np
import tensorflow as tf
from numpy.fft import rfft

from config import (
    COM_PORT, BAUD_RATE,
    CLAP_WINDOW,                # FIX 1: was WINDOW_SIZE (512) — clap model needs 64
    FEATURE_COUNT,              # FIX 2: was NUM_FEATURES (13) — clap model needs 16
    COOLDOWN,
    CLAP_CONFIDENCE_THRESHOLD,  # FIX 4: threshold so weak predictions don't trigger
    LISTENER_MODEL_PATH,        # FIX 3: model path from config, not hardcoded string
)
from serial_helper import open_serial, close_serial, reconnect_serial, send


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="listener",
        description="EchoSafe lightweight TFLite clap detector",
    )
    # FIX 6: --port and --verbose flags
    parser.add_argument(
        "--port",
        default=COM_PORT,
        help=f"serial port to listen on (default: {COM_PORT})",
    )
    parser.add_argument(
        "--model",
        default=LISTENER_MODEL_PATH,
        help=f"TFLite model file to use (default: {LISTENER_MODEL_PATH})",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CLAP_CONFIDENCE_THRESHOLD,
        metavar="THRESHOLD",
        help=f"minimum clap probability to trigger (default: {CLAP_CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="print raw prediction probabilities and buffer fill level",
    )
    return parser.parse_args()


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(signal: list[int]) -> np.ndarray:
    # FIX 7: reference FEATURE_COUNT directly in the body rather than via a
    # default argument evaluated at definition time, so config changes are
    # always picked up at runtime
    fft_vals = np.abs(rfft(np.array(signal, dtype=float)))
    features = np.array(
        [np.mean(fft_vals[i::FEATURE_COUNT]) for i in range(FEATURE_COUNT)],
        dtype=np.float32,
    )
    return features.reshape(1, -1)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_interpreter(model_path: str):
    # FIX 3: clear error if model file is missing
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"TFLite model not found: {model_path!r}. "
            "Export the model before running listener.py."
        )
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()



# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import serial as _serial_mod

    args = parse_args()
    verbose: bool       = args.verbose
    port: str           = args.port
    confidence_threshold: float = args.confidence

    interp, input_details, output_details = load_interpreter(args.model)

    # FIX 1 & 2: buffer sized to CLAP_WINDOW (64), not WINDOW_SIZE (512)
    buffer: deque = deque(maxlen=CLAP_WINDOW)

    ser = open_serial(port, BAUD_RATE)
    atexit.register(close_serial, ser)

    last_trigger = time.time()

    print(f"🎧 Listening on {port}...  (Ctrl+C to stop)")
    if verbose:
        print(f"[verbose] window={CLAP_WINDOW}  features={FEATURE_COUNT}  "
              f"threshold={confidence_threshold}")

    try:
        while True:
            # FIX 5: catch serial disconnects and attempt reconnection
            try:
                raw = ser.readline()
            except (_serial_mod.SerialException, OSError):
                ser = reconnect_serial(port, BAUD_RATE)
                if ser is None:
                    break
                buffer.clear()
                atexit.unregister(close_serial)
                atexit.register(close_serial, ser)
                continue

            line = raw.decode(errors="ignore").strip()
            if not line.isdigit():
                continue

            buffer.append(int(line))

            if verbose:
                print(f"[verbose] buffer={len(buffer)}/{CLAP_WINDOW}  "
                      f"sample={line}", end="\r")

            if len(buffer) < CLAP_WINDOW:
                continue

            # ── Inference ────────────────────────────────────────────────────
            X = extract_features(list(buffer))
            interp.set_tensor(input_details[0]["index"], X)
            interp.invoke()
            output = interp.get_tensor(output_details[0]["index"])[0]

            # FIX 8: verbose shows full probability vector per class
            if verbose:
                prob_str = "  ".join(f"class{i}={p:.0%}" for i, p in enumerate(output))
                print(f"\n[verbose] probs  {prob_str}")

            prediction = int(np.argmax(output))
            clap_prob  = float(output[1]) if len(output) > 1 else float(output[0])

            # FIX 4: only trigger if confidence exceeds threshold
            if (
                prediction == 1
                and clap_prob >= confidence_threshold
                and (time.time() - last_trigger) > COOLDOWN
            ):
                print(f" CLAP detected  (confidence={clap_prob:.0%})")
                send(ser, b"1", verbose=verbose)
                last_trigger = time.time()
            elif verbose and prediction == 1:
                print(f"[verbose] Clap suppressed — confidence {clap_prob:.0%} "
                      f"below threshold {confidence_threshold:.0%}")

    except KeyboardInterrupt:
        print("\nExiting...")
        # atexit handles close_serial cleanly


if __name__ == "__main__":
    main()