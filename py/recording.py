"""
recording.py
------------
Records raw mic data from Arduino to CSV for use with trainer.py.

Usage:
    python recording.py
    python recording.py --port COM5 --duration 60 --output my_data.csv
    python recording.py --verbose
"""

import argparse
import csv
import os
import sys
from collections import deque
from time import time

import serial

from config import (
    COM_PORT, BAUD_RATE,
    SOUNDS_DB_DIR, TRAINING_DATA_DIR,
    RECORDING_WINDOW_SIZE as WINDOW_SIZE,
    CALIBRATION_SECONDS,
    CLAP_THRESHOLD_MULT,
    CLAP_DURATION_MAX,
    LED_ON_TIME,
    RAW_CSV,
)
from serial_helper import open_serial, close_serial, reconnect_serial, send, auto_detect_port
from utils import success


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    # FIX 5: argparse so all settings can be overridden at runtime
    parser = argparse.ArgumentParser(
        prog="recording",
        description="Record raw Arduino mic data to CSV for EchoSafe training",
    )
    parser.add_argument(
        "--port",
        default=COM_PORT,
        help=f"serial port (default: {COM_PORT})",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=BAUD_RATE,
        help=f"baud rate (default: {BAUD_RATE})",
    )
    parser.add_argument(
        "--label",
        default=None,
        metavar="NAME",
        help="sound label for this recording session e.g. clap, knock, whistle. "
             f"Saves to {TRAINING_DATA_DIR}/label_<NAME>.csv. "
             "If not given, saves to RAW_CSV with label=unknown.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="CSV_PATH",
        help="explicit CSV path (overrides --label derived path)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        metavar="SECONDS",
        help="stop after this many seconds (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        # FIX 8: per-window subtext gated behind --verbose
        help="print every window's peak, duration, and label",
    )
    return parser.parse_args()


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _read_int(ser: serial.Serial) -> int | None:
    """
    Read one line from serial and return it as an int, or None if invalid.
    FIX 1 & 12: always decode + strip before isdigit() and int() conversion
    so bytes like b'512\\r\\n' are handled correctly on all platforms.
    """
    raw = ser.readline()
    if not raw:
        return None
    decoded = raw.decode(errors="ignore").strip()
    if not decoded.isdigit():
        return None
    return int(decoded)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def record(
    label: str | None = None,
    output_csv: str | None = None,
    port: str = COM_PORT,
    baud: int = BAUD_RATE,
    duration_seconds: int | None = None,
    verbose: bool = False,
) -> None:
    """
    Record raw mic data from Arduino to CSV.

    Args:
        label:            sound label for this session (e.g. "clap", "knock").
                          Determines output CSV path as label_<name>.csv.
        output_csv:       explicit CSV path; overrides label-derived path.
        port:             serial port the Arduino is connected to
        baud:             baud rate
        duration_seconds: stop after this many seconds; None = run until Ctrl+C
        verbose:          print every window's features to console
    """

    # ── Derive output CSV from label if not given explicitly ─────────────────
    import re as _re
    if output_csv is None:
        if label:
            safe = _re.sub(r'[^\w-]', '_', label.strip()).strip('_') or 'unknown'
            output_csv = os.path.join(TRAINING_DATA_DIR, f"label_{safe}.csv")
            print(f"Label: {label!r}  →  {output_csv}")
        else:
            output_csv = RAW_CSV
            print(f"No --label given — saving to {RAW_CSV} with label=unknown")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    # ── Serial setup ──────────────────────────────────────────────────────────
    try:
        if port == COM_PORT:
            detected = auto_detect_port(baud)
            if detected and detected != port:
                print(f"Auto-detected Arduino on {detected}")
                port = detected
        ser = open_serial(port, baud)
    except serial.SerialException as e:
        print(f"Failed to connect to Arduino: {e}")
        sys.exit(1)

    # ── Calibration ───────────────────────────────────────────────────────────
    print("Calibrating noise floor...")
    calibration: list[int] = []
    cal_start = time()

    while time() - cal_start < CALIBRATION_SECONDS:
        # FIX 12: use _read_int() so decoding is correct
        val = _read_int(ser)
        if val is not None:
            calibration.append(val)

    if not calibration:
        print("Calibration failed: no data received from Arduino.")
        # FIX 2: use close_serial() consistently, not raw ser.close()
        close_serial(ser)
        sys.exit(1)

    baseline_noise = sum(calibration) / len(calibration)
    clap_threshold = baseline_noise * CLAP_THRESHOLD_MULT
    success(f"Noise floor: {baseline_noise:.1f} | Clap threshold: {clap_threshold:.1f}")

    # ── Window + timing ───────────────────────────────────────────────────────
    window: deque = deque(maxlen=WINDOW_SIZE)
    last_led    = 0.0
    sound_start: float | None = None
    prev_above  = False          # FIX 3: track edge so clap needs a rising+falling edge
    start_time  = time()
    deadline    = (start_time + duration_seconds) if duration_seconds else None

    # ── CSV setup ─────────────────────────────────────────────────────────────
    needs_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0

    try:
        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if needs_header:
                writer.writerow(["time", "peak", "duration", "energy", "sharpness", "label"])
            print(f"Logging to {output_csv}")

            # ── Main loop ─────────────────────────────────────────────────────
            while True:
                # FIX 11: check deadline before readline so we don't stall
                # for up to 1 s past the deadline on each iteration
                if deadline and time() >= deadline:
                    print(f"\nRecording complete ({duration_seconds}s).")
                    break

                # FIX 4: catch SerialException and attempt reconnection
                try:
                    mic = _read_int(ser)
                except (serial.SerialException, OSError):
                    ser = reconnect_serial(port, baud)
                    if ser is None:
                        print("Could not reconnect. Exiting.")
                        break
                    window.clear()
                    continue

                # FIX 11: check deadline again after readline unblocks
                if deadline and time() >= deadline:
                    print(f"\nRecording complete ({duration_seconds}s).")
                    break

                if mic is None:
                    continue

                # Non-blocking LED off
                if last_led and time() - last_led >= LED_ON_TIME:
                    send(ser, b"0")
                    last_led = 0.0

                window.append(mic)

                if len(window) < WINDOW_SIZE:
                    continue

                peak      = max(window)
                # FIX 7: mean energy (normalised by window size) so values
                # stay comparable if WINDOW_SIZE changes in config
                energy    = sum(abs(v - baseline_noise) for v in window) / len(window)
                sharpness = peak / (baseline_noise + 1)
                now       = time() - start_time
                above     = peak > clap_threshold

                # FIX 3: track sound duration using rising/falling edge,
                # not just "peak above threshold for < 0.25s"
                if above and not prev_above:
                    sound_start = time()           # rising edge
                if above and sound_start is not None:
                    duration = time() - sound_start
                elif not above and sound_start is not None:
                    duration = time() - sound_start  # capture at falling edge
                    sound_start = None
                else:
                    duration = 0.0

                prev_above = above

                # Use the session label if given; otherwise try to infer from signal
                row_label = label if label else "unknown"
                # Still detect and flag claps for reference even in labelled sessions
                if not above and duration > 0.0 and duration < CLAP_DURATION_MAX:
                    success(f"CLAP 👏 peak={peak} dur={duration:.3f}s")
                    if not last_led:
                        send(ser, b"1")
                        last_led = time()
                elif verbose:
                    print(f"  peak={peak:.0f}  dur={duration:.3f}s  "
                          f"energy={energy:.1f}  label={row_label}", end="\r")

                writer.writerow([
                    round(now, 3),
                    peak,
                    round(duration, 4),
                    round(energy, 4),
                    round(sharpness, 4),
                    row_label,
                ])
                f.flush()

    except KeyboardInterrupt:
        print("\nRecording stopped.")
    finally:
        if ser and ser.is_open:
            send(ser, b"0")        # ensure LED is off
            close_serial(ser)

    # FIX 9: return cleanly so sys.exit(main()) in __main__ gets a code
    return


def main() -> int:
    args = parse_args()
    record(
        label=args.label,
        output_csv=args.output,
        port=args.port,
        baud=args.baud,
        duration_seconds=args.duration,
        verbose=args.verbose,
    )
    return 0


# FIX 9: propagate exit code to shell
if __name__ == "__main__":
    sys.exit(main())