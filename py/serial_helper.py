"""
serial_helper.py
----------------
Shared serial port utilities for EchoSafe.
All files that talk to the Arduino should import from here
rather than duplicating connection logic.
"""

import time
import serial

from config import RECONNECT_ATTEMPTS, RECONNECT_DELAY


# ── Open / close ──────────────────────────────────────────────────────────────

def open_serial(
    port: str,
    baud: int,
    timeout: float = 1.0,   # FIX 5: configurable timeout instead of hardcoded 1
) -> serial.Serial:
    """
    Open a serial connection to the Arduino.

    FIX 7: raises serial.SerialException on failure instead of calling
    sys.exit() so callers can decide whether to retry, exit, or handle
    the error themselves.
    """
    try:
        ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2)   # give Arduino time to reset after connection
        print(f"Connected to Arduino on {port}")
        return ser
    except serial.SerialException as e:
        raise serial.SerialException(f"Could not open {port}: {e}") from e


def close_serial(ser: serial.Serial, verbose: bool = False) -> None:
    """
    Close a serial connection safely.

    FIX 8: only prints if verbose=True — avoids noisy output during normal shutdown.
    FIX 1 (original): catches SerialException so unplugging mid-session doesn't crash.
    """
    if ser and ser.is_open:
        try:
            ser.close()
            if verbose:
                print("Serial port closed.")
        except serial.SerialException as e:
            print(f"Warning: could not close serial port cleanly: {e}")


# ── Reconnection ──────────────────────────────────────────────────────────────

def reconnect_serial(
    port: str,
    baud: int,
    attempts: int = RECONNECT_ATTEMPTS,
    delay: float = RECONNECT_DELAY,
    timeout: float = 1.0,
) -> serial.Serial | None:
    """
    FIX 4: shared reconnection helper — replaces duplicate implementations
    in ai_classifier.py and listener.py.

    Tries up to `attempts` times with `delay` seconds between each attempt.
    Returns the new Serial object on success, or None if all attempts fail.
    """
    print(f"\n⚠️  Serial connection lost. Attempting to reconnect on {port}...")
    for attempt in range(1, attempts + 1):
        try:
            time.sleep(delay)
            ser = serial.Serial(port, baud, timeout=timeout)
            time.sleep(2)
            print(f"✅ Reconnected to {port} (attempt {attempt}/{attempts})")
            return ser
        except serial.SerialException:
            print(f"  Attempt {attempt}/{attempts} failed...")
    print(f"❌ Could not reconnect after {attempts} attempts.")
    return None


# ── Safe write ────────────────────────────────────────────────────────────────

def send(ser: serial.Serial, value: bytes, verbose: bool = False) -> bool:
    """
    FIX 6: safe wrapper around ser.write() that catches SerialException
    instead of letting a mid-write disconnect crash the caller.

    Args:
        ser:     open Serial object
        value:   bytes to send (e.g. b'1', b'0')
        verbose: print a warning on failure

    Returns True on success, False on failure.
    """
    try:
        ser.write(value)
        return True
    except serial.SerialException as e:
        if verbose:
            print(f"Warning: serial write failed: {e}")
        return False