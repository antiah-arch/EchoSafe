"""
check_setup.py
--------------
EchoSafe environment diagnostic.

Run this before using EchoSafe for the first time, or when something
isn't working and you're not sure why.

    python check_setup.py
    python check_setup.py --port COM5   # test a specific port

Checks:
  1. Python version (3.10+ required)
  2. All required packages are importable
  3. sounds_db/ directory is writable
  4. Model files exist (warns if missing, not an error — train first)
  5. Serial port is accessible (if --port given or COM_PORT in config)
  6. Arduino is responding (optional — sends a few reads)
"""

import argparse
import importlib
import os
import sys


# ── Helpers ───────────────────────────────────────────────────────────────────

PASS  = "  ✅"
FAIL  = "  ❌"
WARN  = "  ⚠️ "
INFO  = "  ℹ️ "

_failures = 0
_warnings = 0

def ok(msg: str)   -> None: print(f"{PASS} {msg}")
def fail(msg: str) -> None:
    global _failures
    _failures += 1
    print(f"{FAIL} {msg}")
def warn(msg: str) -> None:
    global _warnings
    _warnings += 1
    print(f"{WARN} {msg}")
def info(msg: str) -> None: print(f"{INFO} {msg}")
def section(title: str) -> None: print(f"\n── {title} {'─' * max(0, 44 - len(title))}")


# ── Checks ────────────────────────────────────────────────────────────────────

def check_python() -> None:
    section("Python version")
    v = sys.version_info
    if v >= (3, 10):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        fail(f"Python {v.major}.{v.minor}.{v.micro} — EchoSafe requires 3.10 or newer")


def check_packages() -> None:
    section("Required packages")

    packages = [
        ("numpy",        "numpy",       True),
        ("pandas",       "pandas",      True),
        ("sklearn",      "scikit-learn",True),
        ("joblib",       "joblib",      True),
        ("tensorflow",   "tensorflow",  True),
        ("librosa",      "librosa",     True),
        ("soundfile",    "soundfile",   True),
        ("sounddevice",  "sounddevice", False),  # optional — only recognizer.py
        ("serial",       "pyserial",    True),
        ("colorama",     "colorama",    False),  # optional — plain text fallback
    ]

    for mod, pkg, required in packages:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            ok(f"{pkg}=={ver}")
        except ImportError:
            if required:
                fail(f"{pkg} not installed — run: pip install {pkg}")
            else:
                warn(f"{pkg} not installed (optional) — run: pip install {pkg}")
        except Exception as e:
            # Some packages (sounddevice) raise OSError on import if system libs missing
            if required:
                fail(f"{pkg} import error: {e}")
            else:
                warn(f"{pkg} import error (optional): {e}")


def check_directories() -> None:
    section("Directories and file permissions")

    try:
        from config import SOUNDS_DB_DIR, LOG_CSV, CLAP_CSV, NOISE_CSV
    except ImportError:
        fail("config.py not found — are you running from the EchoSafe directory?")
        return

    os.makedirs(SOUNDS_DB_DIR, exist_ok=True)

    # Test write permission
    test_file = os.path.join(SOUNDS_DB_DIR, ".write_test")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        ok(f"{SOUNDS_DB_DIR}/ is writable")
    except OSError as e:
        fail(f"{SOUNDS_DB_DIR}/ is not writable: {e}")

    # Test log file location
    log_dir = os.path.dirname(os.path.abspath("echosafe.log"))
    try:
        test = os.path.join(log_dir, ".log_test")
        open(test, "w").close()
        os.remove(test)
        ok(f"Log directory writable ({log_dir})")
    except OSError:
        warn(f"Log directory not writable — echosafe.log may fail")


def check_models() -> None:
    section("Model files")

    try:
        from config import CLAP_MODEL_PATH, CNN_MODEL_FILE, LISTENER_MODEL_PATH, LABEL_MAP_FILE
    except ImportError:
        fail("config.py not found")
        return

    models = [
        (CLAP_MODEL_PATH,    "Clap detector (logistic regression)",  "Run: python trainer.py"),
        (CNN_MODEL_FILE,     "CNN classifier",                       "Run: python ai_classifier.py (trains live)"),
        (LISTENER_MODEL_PATH,"TFLite listener model",                "Export from trained CNN or provide externally"),
        (LABEL_MAP_FILE,     "Label map (JSON)",                     "Created automatically by ai_classifier.py"),
    ]

    for path, desc, hint in models:
        if os.path.exists(path):
            size = os.path.getsize(path)
            ok(f"{path}  ({desc}, {size:,} bytes)")
        else:
            warn(f"{path} not found  ({desc})\n       Hint: {hint}")


def check_serial(port: str | None) -> None:
    section("Serial port")

    try:
        from config import COM_PORT, BAUD_RATE
    except ImportError:
        fail("config.py not found")
        return

    target_port = port or COM_PORT

    try:
        import serial
        import serial.tools.list_ports
    except ImportError:
        fail("pyserial not installed — run: pip install pyserial")
        return

    # List available ports
    available = [p.device for p in serial.tools.list_ports.comports()]
    if available:
        info(f"Available ports: {', '.join(available)}")
    else:
        warn("No serial ports detected — is the Arduino connected?")

    # Try to open the target port
    try:
        ser = serial.Serial(target_port, BAUD_RATE, timeout=2.0)
        ok(f"Opened {target_port} at {BAUD_RATE} baud")

        # Try reading a few lines to confirm Arduino is sending data
        print(f"       Reading 3 samples from Arduino (2s timeout)...")
        samples = []
        for _ in range(3):
            line = ser.readline().decode(errors="ignore").strip()
            if line.isdigit():
                samples.append(int(line))

        ser.close()

        if samples:
            ok(f"Arduino responding — sample values: {samples}")
        else:
            warn(f"Port opened but no integer data received — "
                 "check the Arduino sketch is uploaded and sending ADC values")

    except serial.SerialException as e:
        if available:
            fail(f"Could not open {target_port}: {e}\n"
                 f"       Try one of the available ports: {', '.join(available)}\n"
                 f"       Or set COM_PORT in config.py")
        else:
            fail(f"Could not open {target_port}: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="check_setup",
        description="Verify EchoSafe environment before first use",
    )
    parser.add_argument(
        "--port",
        default=None,
        metavar="PORT",
        help="serial port to test (default: COM_PORT from config.py)",
    )
    parser.add_argument(
        "--skip-serial",
        action="store_true",
        help="skip serial port check (useful when Arduino is not connected)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("\nEchoSafe — environment check")
    print("=" * 48)

    check_python()
    check_packages()
    check_directories()
    check_models()

    if not args.skip_serial:
        check_serial(args.port)
    else:
        section("Serial port")
        info("Skipped (--skip-serial)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 48}")
    if _failures == 0 and _warnings == 0:
        print("✅  All checks passed — EchoSafe is ready.")
    elif _failures == 0:
        print(f"⚠️   {_warnings} warning(s) — EchoSafe should work but review the above.")
    else:
        print(f"❌  {_failures} failure(s), {_warnings} warning(s) — fix the above before running.")
    print()

    return 1 if _failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())