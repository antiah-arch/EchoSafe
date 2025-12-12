import serial
import time
import csv

# 🔧 Adjust to your Arduino's COM port
port = 'COM3'       # Windows example
baud = 115200        # Must match Arduino Serial.begin(115200)
duration = 10        # seconds to record
label = 1            # 1 = clap, 0 = noise (you can change this)

ser = serial.Serial(port, baud, timeout=1)
time.sleep(2)  # wait for connection

filename = f"sound_data_label{label}.csv"

with open(filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["time", "mic_value", "label"])

    print(f"🎙️ Recording for {duration} seconds... (label={label})")
    start = time.time()
    while time.time() - start < duration:
        line = ser.readline().decode().strip()
        if line.isdigit():
            timestamp = time.time() - start
            mic_value = int(line)
            writer.writerow([timestamp, mic_value, label])

print(f"✅ Saved {filename}")
ser.close() 