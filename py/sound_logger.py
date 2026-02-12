import csv
import os
import serial
from time import time, sleep
from math import sqrt

# ---------------- CONFIG ----------------
SERIAL_PORT = "COM3"
BAUD_RATE = 9600
CALIBRATION_SECONDS = 3
SOUNDS_DB_DIR = "sounds_db"

DISTANCE_THRESHOLD = 1.2   # classification sensitivity
LEARNING_RATE = 0.2        # how fast centroids adapt (0.1–0.3 good)

MIN_SOUND_SAMPLES = 6

os.makedirs(SOUNDS_DB_DIR, exist_ok=True)

# ---------------- SERIAL ----------------
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
sleep(2)
print("Connected to Arduino")

# ---------------- CALIBRATION ----------------
print("Calibrating...")
calibration = []
start = time()

while time() - start < CALIBRATION_SECONDS:
    line = ser.readline()
    if line and line.strip().isdigit():
        calibration.append(int(line))

if not calibration:
    print("Calibration failed.")
    exit()

baseline_noise = sum(calibration) / len(calibration)
print("Baseline:", baseline_noise)

# dynamic thresholds
SOUND_START_THRESHOLD = baseline_noise * 1.8
SOUND_END_THRESHOLD = baseline_noise * 1.2

# ---------------- DATABASE ----------------
sounds_db = {}

def load_database():
    for file in os.listdir(SOUNDS_DB_DIR):
        if file.endswith(".csv"):
            path = os.path.join(SOUNDS_DB_DIR, file)
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    continue
                avg_features = {
                    "peak": sum(float(r["peak"]) for r in rows)/len(rows),
                    "duration": sum(float(r["duration"]) for r in rows)/len(rows),
                    "energy": sum(float(r["energy"]) for r in rows)/len(rows),
                    "sharpness": sum(float(r["sharpness"]) for r in rows)/len(rows),
                    "variance": sum(float(r["variance"]) for r in rows)/len(rows),
                }
                sounds_db[file] = avg_features

load_database()
print("Loaded", len(sounds_db), "known sounds")

# ---------------- FEATURE DISTANCE ----------------
def distance(f1, f2):
    return (
        abs(f1["peak"] - f2["peak"]) / (f2["peak"] + 1) +
        abs(f1["duration"] - f2["duration"]) / (f2["duration"] + 0.001) +
        abs(f1["energy"] - f2["energy"]) / (f2["energy"] + 1) +
        abs(f1["sharpness"] - f2["sharpness"]) / (f2["sharpness"] + 0.01) +
        abs(f1["variance"] - f2["variance"]) / (f2["variance"] + 1)
    )

# ---------------- CLASSIFICATION ----------------
def classify(features):
    best_match = None
    best_dist = 999

    for fname, centroid in sounds_db.items():
        d = distance(features, centroid)
        if d < best_dist:
            best_dist = d
            best_match = fname

    if best_match and best_dist < DISTANCE_THRESHOLD:
        label = os.path.splitext(best_match)[0]
        return label, os.path.join(SOUNDS_DB_DIR, best_match)

    # create new unknown class
    unknown_count = len([f for f in os.listdir(SOUNDS_DB_DIR) if f.startswith("unknown_")])
    label = f"unknown_{unknown_count+1}"
    filename = f"{label}.csv"
    sounds_db[filename] = features
    return label, os.path.join(SOUNDS_DB_DIR, filename)

# ---------------- LEARNING UPDATE ----------------
def update_centroid(filename, new_features):
    centroid = sounds_db.get(filename)
    if not centroid:
        sounds_db[filename] = new_features
        return

    for key in centroid:
        centroid[key] = (
            (1 - LEARNING_RATE) * centroid[key] +
            LEARNING_RATE * new_features[key]
        )

# ---------------- CSV LOGGING ----------------
def log_sound(path, features, label):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["time","peak","duration","energy","sharpness","variance","label"])
        writer.writerow([
            time(),
            features["peak"],
            features["duration"],
            features["energy"],
            features["sharpness"],
            features["variance"],
            label
        ])

# ---------------- MAIN LOOP ----------------
recording = False
current_sound = []

print("Listening...")

while True:
    line = ser.readline()
    if not line:
        continue

    try:
        mic = int(line.decode().strip())
    except:
        continue

    # ---- SOUND START ----
    if not recording and mic > SOUND_START_THRESHOLD:
        recording = True
        current_sound = [mic]
        continue

    # ---- RECORDING ----
    if recording:
        current_sound.append(mic)

        # ---- SOUND END ----
        if mic < SOUND_END_THRESHOLD and len(current_sound) > MIN_SOUND_SAMPLES:
            recording = False

            peak = max(current_sound)
            duration = len(current_sound) / 100
            avg = sum(current_sound)/len(current_sound)
            energy = sum(abs(v - baseline_noise) for v in current_sound)
            sharpness = peak / (avg + 1)
            variance = sum((v-avg)**2 for v in current_sound)/len(current_sound)

            features = {
                "peak": peak,
                "duration": duration,
                "energy": energy,
                "sharpness": sharpness,
                "variance": variance
            }

            # ---- ADVANCED CLAP DETECTION ----
            if (
                peak > baseline_noise * 2.5 and
                duration < 0.18 and
                sharpness > 1.4 and
                variance > 200
            ):
                label = "clap"
                filename = "clap.csv"
                path = os.path.join(SOUNDS_DB_DIR, filename)
                print("👏 CLAP DETECTED")
            else:
                label, path = classify(features)
                filename = os.path.basename(path)
                print("Detected:", label)

            log_sound(path, features, label)
            update_centroid(filename, features)
