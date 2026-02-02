# import time
# import csv
# import atexit
# from config import COM_PORT, BAUD_RATE
# from serial_helper import open_serial, close_serial

# # ==========================
# # CONFIG
# # ==========================
# DURATION = 10        # seconds to record
# LABEL = 1            # 1 = clap, 0 = noise
# OUTPUT_FILE = f"sound_data_label{LABEL}.csv"

# # ==========================
# # MAIN
# # ==========================
# def main():
#     # Open serial connection to Arduino
#     ser = open_serial(COM_PORT, BAUD_RATE)
#     atexit.register(close_serial, ser)  # auto-close on crash

#     print(f"Recording for {DURATION}s (label={LABEL})...")

#     with open(OUTPUT_FILE, "w", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["time", "mic_value", "label"])

#         start_time = time.time()

#         while time.time() - start_time < DURATION:
#             line = ser.readline().decode(errors="ignore").strip()

#             # Only keep valid numeric readings
#             if line.isdigit():
#                 timestamp = time.time() - start_time
#                 mic_value = int(line)
#                 writer.writerow([timestamp, mic_value, LABEL])

#     close_serial(ser)
#     print(f"Saved data to {OUTPUT_FILE}")

# # ==========================
# if __name__ == "__main__":
#     main()
# recording.py
import csv
from collections import deque
from time import time
import serial
from utils import error, warning, success, subtext  # use your fixed utils.py

# ----------------------------
# CONFIG
# ----------------------------
SERIAL_PORT = "COM3"       # change to your Arduino port
BAUD_RATE = 9600
WINDOW_SIZE = 20           # rolling baseline samples
CLAP_THRESHOLD = 200       # threshold for detecting claps
OUTPUT_CSV = "output.csv"

USE_ARDUINO = False  # Set True to read live from Arduino

# ----------------------------
# MIC VALUES SETUP
# ----------------------------
if USE_ARDUINO:
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        mic_values = (int(line.strip()) for line in ser)
        success("Connected to Arduino!")
    except Exception as e:
        error(f"Failed to connect to Arduino: {e}")
else:
    # Test stream of mic values
    mic_values = [100, 120, 130, 500, 110, 125, 600, 120]
    warning("Using test mic values instead of Arduino")

# ----------------------------
# ROLLING WINDOW & TIMER
# ----------------------------
window = deque(maxlen=WINDOW_SIZE)
start_time = time()

# ----------------------------
# CSV SETUP
# ----------------------------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "mic_value", "clap_factor"])  # CSV header
    subtext(f"Logging to {OUTPUT_CSV}")

    # ----------------------------
    # MAIN LOOP
    # ----------------------------
    for mic in mic_values:
        # Update rolling baseline
        window.append(mic)
        baseline = sum(window) / len(window)

        # Compute clap factor
        clap_factor = max(0.0, mic - baseline)
        # Optional normalization: clap_factor = min(1.0, clap_factor / 300)

        # Current time
        current_time = time() - start_time

        # Write to CSV
        writer.writerow([current_time, mic, clap_factor])

        # Real-time detection
        if clap_factor > CLAP_THRESHOLD:
            success(f"CLAP DETECTED 👏 t={current_time:.3f}s, mic={mic}, factor={clap_factor:.1f}")
        else:
            subtext(f"t={current_time:.3f}s, mic={mic}, factor={clap_factor:.1f}")
