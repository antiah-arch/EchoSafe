import csv
from collections import deque
from time import time, sleep
import serial
from utils import error, warning, success, subtext

# ----------------------------
# CONFIG
# ----------------------------
SERIAL_PORT = "COM3"
BAUD_RATE = 9600

WINDOW_SIZE = 25
CALIBRATION_SECONDS = 3
CLAP_THRESHOLD_MULT = 2.8
LED_ON_TIME = 0.3

OUTPUT_CSV = "sounds_raw.csv"
USE_ARDUINO = True

# ----------------------------
# SERIAL SETUP
# ----------------------------
if USE_ARDUINO:
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        sleep(2)
        success(f"Connected to Arduino on {SERIAL_PORT}")
    except serial.SerialException as e:
        error(f"Failed to connect to Arduino: {e}")
        exit(1)

# ----------------------------
# AUTO CALIBRATION
# ----------------------------
subtext("Calibrating noise floor...")
calibration = []
start = time()

while time() - start < CALIBRATION_SECONDS:
    line = ser.readline()
    if line and line.strip().isdigit():
        calibration.append(int(line))

if not calibration:
    error("Calibration failed: no data")

baseline_noise = sum(calibration) / len(calibration)
CLAP_THRESHOLD = baseline_noise * CLAP_THRESHOLD_MULT
success(f"Noise floor: {baseline_noise:.1f} | Clap threshold: {CLAP_THRESHOLD:.1f}")

# ----------------------------
# WINDOW + TIMING
# ----------------------------
window = deque(maxlen=WINDOW_SIZE)
last_led = 0
start_time = time()

# ----------------------------
# CSV SETUP
# ----------------------------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "time",
        "peak",
        "duration",
        "energy",
        "sharpness",
        "label"
    ])
    subtext(f"Logging to {OUTPUT_CSV}")

    # ----------------------------
    # MAIN LOOP
    # ----------------------------
    while True:
        line = ser.readline()
        if not line:
            continue

        try:
            mic = int(line.decode("utf-8").strip())
        except ValueError:
            continue

        window.append(mic)

        if len(window) < WINDOW_SIZE:
            continue

        peak = max(window)
        energy = sum(abs(v - baseline_noise) for v in window)
        duration = WINDOW_SIZE * 0.01
        sharpness = peak / (baseline_noise + 1)

        now = time() - start_time

        label = "unknown"

        if peak > CLAP_THRESHOLD and duration < 0.25:
            label = "clap"
            success(f"CLAP 👏 peak={peak}")
            if time() - last_led > LED_ON_TIME:
                ser.write(b'1')
                sleep(LED_ON_TIME)
                ser.write(b'0')
                last_led = time()
        else:
            subtext(f"sound peak={peak}")

        writer.writerow([
            now,
            peak,
            duration,
            energy,
            sharpness,
            label
        ])
