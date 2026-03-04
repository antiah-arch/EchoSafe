"""
recognizer.py
-------------
Records sounds from the PC microphone and saves them as .wav files
and rows in a single session CSV for use with trainer.py.

This is a data-collection tool — use it on machines without an Arduino
to build up a labelled sound dataset.

Usage:
    python recognizer.py
    python recognizer.py --output-dir my_sounds --verbose
    python recognizer.py --trigger 2.5 --device 1
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

# FIX 2: graceful ImportError with install instructions
try:
    import sounddevice as sd
    import soundfile as sf
except ImportError as e:
    print(
        f"Error: missing audio dependency — {e}\n"
        "Install with:  pip install sounddevice soundfile"
    )
    sys.exit(1)

from config import (
    SAMPLE_RATE,
    DEVICE_INDEX,
    PC_FRAME_SIZE as FRAME_SIZE,
    CALIBRATION_SECONDS,
    MIN_FRAMES,
    MAX_FRAMES,
    PC_TRIGGER_MULT as TRIGGER_MULTIPLIER,
    SOUNDS_DB_DIR,          # FIX 8: use directly, no alias
)

# Session CSV path — one file per run, all recordings appended to it
SESSION_CSV_FIELDS = ["time", "energy", "zcr", "centroid", "label", "filepath"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    # FIX 3: argparse so settings can be overridden at runtime
    parser = argparse.ArgumentParser(
        prog="recognizer",
        description="Record PC microphone sounds for EchoSafe training",
    )
    parser.add_argument(
        "--output-dir",
        default=SOUNDS_DB_DIR,
        help=f"directory to save .wav files and CSV (default: {SOUNDS_DB_DIR})",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=DEVICE_INDEX,
        metavar="INDEX",
        help="microphone device index (default: system default)",
    )
    parser.add_argument(
        "--trigger",
        type=float,
        default=TRIGGER_MULTIPLIER,
        metavar="MULTIPLIER",
        help=f"trigger = baseline * MULTIPLIER (default: {TRIGGER_MULTIPLIER})",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=MIN_FRAMES,
        help=f"minimum frames before saving a recording (default: {MIN_FRAMES})",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=MAX_FRAMES,
        help=f"maximum frames before force-saving (default: {MAX_FRAMES})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        # FIX 6: live level display is gated behind --verbose
        help="print live mic level on every frame",
    )
    return parser.parse_args()


# ── FEATURE EXTRACTION ────────────────────────────────────────────────────────

def extract_features(waveform: np.ndarray) -> dict:
    # FIX 4: guard against empty waveform so we never produce NaN/inf in CSV
    if len(waveform) == 0:
        return {"energy": 0.0, "zcr": 0.0, "centroid": 0.0}

    energy = float(np.mean(np.abs(waveform)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(waveform)))))
    denom = float(np.sum(np.abs(waveform))) + 1e-6
    centroid = float(np.sum(np.arange(len(waveform)) * np.abs(waveform)) / denom)
    return {"energy": energy, "zcr": zcr, "centroid": centroid}


# ── SAVE ──────────────────────────────────────────────────────────────────────

def save(
    frames: list[np.ndarray],
    output_dir: str,
    session_writer: csv.DictWriter,
) -> str | None:
    """
    Save a recording as a .wav file and append a row to the session CSV.
    FIX 5: appends to single session CSV instead of creating one CSV per recording.
    FIX 7: wrapped in try/except so a save failure during KeyboardInterrupt
           doesn't hide the original error or lose all context.
    Returns the wav path on success, None on failure.
    """
    try:
        audio = np.concatenate(frames)

        # FIX 4: guard against empty audio
        if len(audio) == 0:
            print("Warning: empty audio, skipping save.")
            return None

        ts = int(time.time())
        wav_path = os.path.join(output_dir, f"sound_{ts}.wav")
        sf.write(wav_path, audio, SAMPLE_RATE)

        feats = extract_features(audio)
        session_writer.writerow({
            "time":     round(time.time(), 3),
            "energy":   round(feats["energy"], 6),
            "zcr":      round(feats["zcr"], 6),
            "centroid": round(feats["centroid"], 3),
            "label":    "unknown",
            "filepath": wav_path,
        })
        print(f"\nSaved -> {wav_path}")
        return wav_path

    except Exception as e:
        print(f"\nWarning: could not save recording: {e}")
        return None


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    output_dir: str   = args.output_dir
    device: int | None = args.device
    trigger_mult: float = args.trigger
    min_frames: int   = args.min_frames
    max_frames: int   = args.max_frames
    verbose: bool     = args.verbose

    # FIX 11: makedirs inside main(), not at module level
    os.makedirs(output_dir, exist_ok=True)

    # FIX 5: single session CSV — open once, append all recordings
    session_csv = os.path.join(output_dir, f"session_{int(time.time())}.csv")
    session_file = open(session_csv, "w", newline="")
    session_writer = csv.DictWriter(session_file, fieldnames=SESSION_CSV_FIELDS)
    session_writer.writeheader()
    print(f"Session CSV: {session_csv}")

    # FIX 10: one shared InputStream for both calibration and recording
    # avoids device conflict from opening and closing the mic twice
    try:
        with sd.InputStream(
            device=device,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SIZE,
        ) as stream:

            # ── Calibration ───────────────────────────────────────────────────
            print(f"\nCalibrating for {CALIBRATION_SECONDS}s — stay quiet...")
            cal_values: list[float] = []
            total_cal_frames = CALIBRATION_SECONDS * SAMPLE_RATE // FRAME_SIZE

            for i in range(total_cal_frames):
                frame, _ = stream.read(FRAME_SIZE)
                level = float(np.max(np.abs(frame)))
                cal_values.append(level)
                print(f"  Calibrating {i+1}/{total_cal_frames}  level={level:.6f}", end="\r")

            # FIX 2 / original FIX 3: guard against empty calibration
            if not cal_values:
                print("\nCalibration failed: no audio frames received.")
                sys.exit(1)

            baseline = float(np.mean(cal_values))
            trigger_level = baseline * trigger_mult

            print(f"\nBaseline:      {baseline:.6f}")
            print(f"Trigger level: {trigger_level:.6f}  (baseline × {trigger_mult})")
            print("\n🎙  Listening...  (Ctrl+C to stop)\n")

            # ── Recording loop ────────────────────────────────────────────────
            recording = False
            frames: list[np.ndarray] = []

            try:
                while True:
                    frame, _ = stream.read(FRAME_SIZE)
                    frame = frame.flatten()
                    level = float(np.max(np.abs(frame)))

                    # FIX 6: live level only printed in verbose mode
                    if verbose:
                        print(f"[verbose] level={level:.6f}  "
                              f"trigger={trigger_level:.6f}  "
                              f"recording={recording}", end="\r")

                    if level > trigger_level:
                        if not recording:
                            recording = True
                            frames = [frame]
                            print("\n🔥 Sound detected — recording...")
                        else:
                            frames.append(frame)
                            if len(frames) >= max_frames:
                                recording = False
                                print("\n⚠️  Max length reached, saving...")
                                save(frames, output_dir, session_writer)
                                session_file.flush()
                                frames = []
                                print("Listening...")
                    else:
                        if recording and len(frames) >= min_frames:
                            recording = False
                            save(frames, output_dir, session_writer)
                            session_file.flush()
                            frames = []
                            print("Listening...")
                        elif recording:
                            recording = False
                            frames = []
                            print("\n(too short — discarded)")

            except KeyboardInterrupt:
                # FIX 7: save() is wrapped so a failure here won't crash cleanup
                if recording and len(frames) >= min_frames:
                    print("\nInterrupted — saving partial recording...")
                    save(frames, output_dir, session_writer)
                print("\nStopped.")

    finally:
        # Always close the session CSV cleanly
        session_file.flush()
        session_file.close()
        print(f"Session log saved: {session_csv}")


# FIX 9: __main__ guard so the script is safely importable
if __name__ == "__main__":
    main()