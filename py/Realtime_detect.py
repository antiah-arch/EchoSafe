from collections import deque
from time import time
import numpy as np
import joblib
from numpy.fft import rfft
from serial_helper import open_serial, close_serial
from config import COM_PORT, BAUD_RATE

# ==========================
WINDOW_SIZE = 64
FEATURE_COUNT = 16
COOLDOWN = 1.0  # seconds
MODEL_PATH = "clap_model.pkl"

# ==========================
def extract_features(signal):
    fft_vals = np.abs(rfft(signal))
    return np.array(
        [np.mean(fft_vals[i::FEATURE_COUNT]) for i in range(FEATURE_COUNT)]
    ).reshape(1, -1)

# ==========================
model = joblib.load(MODEL_PATH)
buffer = deque(maxlen=WINDOW_SIZE)
last_trigger = 0

ser = open_serial(COM_PORT, BAUD_RATE)

print("🎧 Listening for claps...")

try:
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line.isdigit():
            buffer.append(int(line))

        if len(buffer) == WINDOW_SIZE:
            features = extract_features(np.array(buffer))
            prediction = model.predict(features)[0]

            if prediction == 1 and (time() - last_trigger) > COOLDOWN:
                print("CLAP DETECTED")
                ser.write(b"1")
                last_trigger = time()
            else:
                ser.write(b"0")

finally:
    close_serial(ser)
