import csv
import os
from collections import deque
from time import time, sleep
import serial
from utils import error, warning, success, subtext
from math import sqrt

# ----------------------------
# CONFIG
# ----------------------------
SERIAL_PORT = "COM3"
BAUD_RATE = 9600
WINDOW_SIZE = 25
CALIBRATION_SECONDS = 3
CLAP_THRESHOLD_MULT = 2.8
LED_ON_TIME = 0.3
TOGGLE_LED_MODE = False  # False = LED flashes, True = LED stays until next sound
USE_ARDUINO = True
SIMILARITY_THRESHOLD = 0.15  # relative difference (15%) for matching sounds
SOUNDS_DB_DIR = "sounds_db"  # folder for all sound CSVs

# Ensure DB folder exists
os.makedirs(SOUNDS_DB_DIR, exist_ok=True)

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
# LOAD EXISTING SOUNDS DB
# ----------------------------
sounds_db = {}  # key = CSV filename, value = list of representative features

def load_sounds_db():
    for file in os.listdir(SOUNDS_DB_DIR):
        if file.endswith(".csv"):
            path = os.path.join(SOUNDS_DB_DIR, file)
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                peaks = []
                energies = []
                durations = []
                sharpnesses = []
                for row in reader:
                    try:
                        peaks.append(float(row["peak"]))
                        energies.append(float(row["energy"]))
                        durations.append(float(row["duration"]))
                        sharpnesses.append(float(row["sharpness"]))
                    except KeyError:
                        continue
                if peaks:
                    # store average features
                    sounds_db[file] = {
                        "peak": sum(peaks)/len(peaks),
                        "energy": sum(energies)/len(energies),
                        "duration": sum(durations)/len(durations),
                        "sharpness": sum(sharpnesses)/len(sharpnesses)
                    }

load_sounds_db()
subtext(f"Loaded {len(sounds_db)} sounds from DB")

# ----------------------------
# SIMILARITY FUNCTION
# ----------------------------
def is_similar(f1, f2):
    # relative difference for each feature
    for key in ["peak", "energy", "duration", "sharpness"]:
        if f2[key] == 0:
            return False
        diff = abs(f1[key] - f2[key]) / f2[key]
        if diff > SIMILARITY_THRESHOLD:
            return False
    return True

# ----------------------------
# FIND MATCHING SOUND CSV
# ----------------------------
def find_matching_csv(features):
    for fname, fdb in sounds_db.items():
        if is_similar(features, fdb):
            return fname
    return None

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

    features = {"peak": peak, "energy": energy, "duration": duration, "sharpness": sharpness}
    label = "unknown"

    # ----------------------------
    # Clap detection
    # ----------------------------
    if peak > CLAP_THRESHOLD and duration < 0.25:
        label = "clap"
        success(f"CLAP 👏 peak={peak}")
        if TOGGLE_LED_MODE:
            ser.write(b'1')  # LED stays on
        else:
            if time() - last_led > LED_ON_TIME:
                ser.write(b'1')
                sleep(LED_ON_TIME)
                ser.write(b'0')
                last_led = time()
        # log to a dedicated CSV
        csv_file = os.path.join(SOUNDS_DB_DIR, "clap.csv")
    else:
        # ----------------------------
        # Try to match existing sound
        # ----------------------------
        match = find_matching_csv(features)
        if match:
            csv_file = os.path.join(SOUNDS_DB_DIR, match)
            label = os.path.splitext(match)[0]
            subtext(f"Matched sound: {label} peak={peak}")
        else:
            # Create new unknown_X CSV
            unknown_count = len([f for f in os.listdir(SOUNDS_DB_DIR) if f.startswith("unknown_")])
            csv_file = os.path.join(SOUNDS_DB_DIR, f"unknown_{unknown_count+1}.csv")
            label = f"unknown_{unknown_count+1}"
            subtext(f"New unknown sound detected: {label} peak={peak}")
            # also add to sounds_db so future sounds can match
            sounds_db[f"{label}.csv"] = features

    # ----------------------------
    # Append to CSV
    # ----------------------------
    write_header = not os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["time","peak","duration","energy","sharpness","label"])
        writer.writerow([now, peak, duration, energy, sharpness, label])
