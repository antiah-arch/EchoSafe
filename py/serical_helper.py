import serial
import time
import sys

def open_serial(port, baud):
    """Open serial safely; exit if port is busy."""
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # give Arduino time to reset
        return ser
    except serial.SerialException as e:
        print(f" Could not open {port}: {e}")
        sys.exit(1)

def close_serial(ser):
    if ser and ser.is_open:
        ser.close()
        print(f" Closed serial port")
