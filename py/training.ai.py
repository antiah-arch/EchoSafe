import argparse
import os
import time
from collections import deque

import serial
import numpy as np
import tensorflow as tf
from numpy.fft import rfft # ignore 

# -----------------------------
# SETTINGS (defaults, overridable via CLI)
# -----------------------------
COM_PORT = "COM3"   # Change to your Arduino port or pass --com
WINDOW_SIZE = 256   # Must match feature extraction window
NUM_FEATURES = 20
COOLDOWN = 0.3  # seconds between motor triggers

ser = serial.Serial("COM3", 9600, timeout=1)
time.sleep(2)  # Arduino reset

ser.write(b'1')  # Turn LED on


def extract_features(signal, num_features=NUM_FEATURES):
    """Convert a 1D time-domain `signal` into `num_features` FFT-band features.

    Returns a numpy array shaped `(1, num_features)` and dtype float32 (model input).
    """
    fft_vals = np.abs(rfft(signal))
    features = np.array([np.mean(fft_vals[i::num_features]) for i in range(num_features)])
    return np.array([features], dtype=np.float32)


def load_interpreter(model_path):
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()


def parse_int_line(line):
    """Robustly parse an integer from a line. Returns None on failure."""
    try:
        return int(line.strip())
    except Exception:
        return None


def replay_csv_lines(path):
    """Yield integer-like lines from a CSV or plain text file for simulation."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            # Try to extract the first numeric token from the line
            token = raw.strip().split(',')[0]
            val = parse_int_line(token)
            if val is not None:
                yield str(val)


def main(args=None):
    parser = argparse.ArgumentParser(description="Arduino -> TFLite inference bridge")
    parser.add_argument("--com", default=COM_PORT, help="Serial COM port (e.g. COM3)")
    parser.add_argument("--model", default="sound_model.tflite", help="Path to TFLite model")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--num-features", type=int, default=NUM_FEATURES)
    parser.add_argument("--simulate", help="Path to CSV/text file to replay instead of serial")
    parsed = parser.parse_args(args=args)

    window_size = parsed.window_size
    num_features = parsed.num_features

    # Load model
    interp, input_details, output_details = load_interpreter(parsed.model)

    # Prepare buffer as a bounded deque to avoid unbounded growth
    buffer = deque(maxlen=window_size)

    ser = None
    line_source = None

    if parsed.simulate:
        if not os.path.exists(parsed.simulate):
            raise SystemExit(f"Simulation file not found: {parsed.simulate}")
        print(f"🔁 Running in simulate mode, reading: {parsed.simulate}")
        line_source = replay_csv_lines(parsed.simulate)
    else:
        # Connect to Arduino
        ser = serial.Serial(parsed.com, 9600, timeout=1)
        print(f"✅ Connected to Arduino on {parsed.com}. Listening for audio...")
        # iterator that yields stripped lines until an empty string sentinel
        line_source = iter(lambda: ser.readline().decode(errors="ignore").strip(), "")
    
    last_trigger = 0

    try:
        for raw_line in line_source:
            # raw_line may already be stripped
            if not raw_line:
                continue

            mic_value = parse_int_line(raw_line)
            if mic_value is None:
                continue

            buffer.append(mic_value)

            if len(buffer) >= window_size:
                window = list(buffer)[-window_size:]
                X = extract_features(window, num_features=num_features)

                # Run TFLite model
                interp.set_tensor(input_details[0]["index"], X)
                interp.invoke()
                output = interp.get_tensor(output_details[0]["index"])[0]
                prediction = int(np.argmax(output))

                # Clap detected -> send trigger
                if prediction == 1 and (time.time() - last_trigger) > COOLDOWN:
                    print("👏 CLAP detected!")
                    if ser:
                        ser.write(b'1')
                    last_trigger = time.time()
                else:
                    if ser:
                        ser.write(b'0')

    except KeyboardInterrupt:
        print("\n🛑 Exiting...")
    finally:
        if ser and ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()

