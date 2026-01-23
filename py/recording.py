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
from collections.abc import Iterator
import csv
from source import DataEntry


def record(output: object,data:Iterator[DataEntry],seconds: int) -> None:
    with csv.writer(output) as writer:
        for entry in data:
            if entry.time < seconds:
                writer.writerow(entry.to_csv_entry())