# ── config.py ─────────────────────────────────────────────────────────────────
# Central configuration for EchoSafe.
# All tunable settings live here — no need to edit other files.

# ── Serial ────────────────────────────────────────────────────────────────────
COM_PORT  = "COM3"      # Arduino serial port  (e.g. COM5, /dev/ttyUSB0)
BAUD_RATE = 115200      # Arduino baud rate — must match sketch (raised for 8kHz sampling)
BAUDRATE  = 115200      # High-speed baud rate (source.py / cli.py pipeline)

# ── Audio — shared ────────────────────────────────────────────────────────────
ARDUINO_SAMPLE_RATE    = 8000   # Hz — Arduino ADC at ~8kHz (125µs timer, no delay())
                                # Nyquist = 4000 Hz → covers smoke alarm (3150 Hz),
                                # CO alarm (3100 Hz), and siren sweeps (500–1500 Hz)
ARDUINO_SAMPLE_INTERVAL = 1.0 / ARDUINO_SAMPLE_RATE  # seconds per sample (0.000125)
PC_SAMPLE_RATE      = 16000  # Hz — PC microphone rate (recognizer.py, librosa, ESC-50)

CALIBRATION_SECONDS = 3 # Seconds of silence used to measure the noise floor

# ── Audio — Arduino / serial pipeline ────────────────────────────────────────
RECORDING_WINDOW_SIZE = 200  # Samples per sliding window in recording.py (25ms @ 8kHz)
CLI_WINDOW_SIZE       = 256  # Default FFT window size for cli.py / trainer.py

# ── Audio — PC microphone (recognizer.py) ────────────────────────────────────
DEVICE_INDEX   = None   # Mic device index; None = system default
PC_FRAME_SIZE  = 1024   # Samples per PC audio frame
MIN_FRAMES     = 6      # Minimum frames before a PC recording is saved
MAX_FRAMES     = 500    # Maximum frames before a PC recording is force-saved (~32 s)
PC_TRIGGER_MULT = 3.0   # PC mic: level must exceed baseline * this to start recording

# ── Audio — CNN spectrogram ───────────────────────────────────────────────────
SPEC_H = 64             # Mel spectrogram height (frequency bins)
SPEC_W = 64             # Mel spectrogram width  (time frames)

# ── FFT features ──────────────────────────────────────────────────────────────
NUM_FEATURES  = 13      # Features for listener.py TFLite model
FEATURE_COUNT        = 16   # FFT frequency bins (must match training)
CLAP_WINDOW          = 256  # Samples per FFT window @ 8kHz = 32ms, freq resolution = 31Hz
FFT_STRIDE           = 32   # Samples to advance between windows (8x overlap → 4ms resolution)

# ── Frequency band features (distance-invariant) ──────────────────────────────
# At 8000 Hz with CLAP_WINDOW=256, each FFT bin = 8000/256 = 31.25 Hz
# Band edges are in Hz; converted to bin indices at runtime.
#
# Bands are chosen to bracket known alarm frequencies:
#   LOW  :    0–500 Hz  — ambient / HVAC rumble (not alarms)
#   MID  :  500–1500 Hz — siren sweep range
#   HIGH : 1500–4000 Hz — smoke/CO alarm tones (3100–3150 Hz)
#
# The FEATURE added to training is:
#   [low_ratio, mid_ratio, high_ratio] = band_energy / total_energy
# These ratios are nearly constant regardless of how far away the source is.
FREQ_BAND_LOW_HZ  = (0,    500)    # low band edge (Hz)
FREQ_BAND_MID_HZ  = (500,  1500)   # mid band edge (Hz)
FREQ_BAND_HIGH_HZ = (1500, 4000)   # high band edge (Hz)

# ── Alarm tone templates (rule-based, no training needed) ─────────────────────
# Fraction of total spectral energy that must sit within ±ALARM_FREQ_TOLERANCE Hz
# of the target frequency for the rule to fire.
ALARM_FREQ_TOLERANCE_HZ = 200      # ± Hz around target tone
ALARM_ENERGY_FRACTION   = 0.35     # 35% of energy must be in the target band

# Known alarm tones and their common names
ALARM_TONES = {
    "smoke_alarm":  3150,   # EN 54-3 / UL 217 mandated T3 tone
    "co_alarm":     3100,   # CO detector standard tone
    "siren_low":     500,   # lower edge of siren sweep
    "siren_high":   1400,   # upper edge of siren sweep
}

# ── Temporal alarm pattern detector ───────────────────────────────────────────
# Smoke alarms use the T3 pattern: 3 short beeps (~0.5s each), then silence (~1.5s).
# CO alarms use 4 beeps then silence.
# The detector tracks the on/off rhythm across consecutive sound events.
#
# A "pulse" = one detected sound event. The detector counts pulses and gaps.
PATTERN_WINDOW_SEC   = 8.0   # Time window to look for a repeating pattern
PATTERN_MIN_PULSES   = 3     # Minimum pulses in window to consider a pattern
PATTERN_MAX_PULSE_SEC = 1.2  # A single alarm beep is shorter than this
PATTERN_MIN_GAP_SEC   = 0.3  # Minimum silence between alarm beeps
PATTERN_REPEAT_THRESH = 2    # Pattern must repeat this many times to confirm alarm

# Derivative-based spike detector — runs per-sample in parallel with FFT
DERIVATIVE_WINDOW    = 4    # Samples for smoothed first-derivative
SPIKE_THRESHOLD_MULT = 8.0  # Spike if derivative > baseline * this value
SPIKE_COOLDOWN       = 0.3  # Min seconds between separate spike events
SPIKE_MIN_GAP        = 5    # Min samples between two spikes to be counted separately

# ── Detection thresholds ──────────────────────────────────────────────────────
COOLDOWN     = 1.0      # Minimum seconds between clap detections
LED_ON_TIME  = 0.3      # Seconds the LED + vibration motor stay on after detection
MOTOR_PIN    = 9        # Arduino pin driving the vibration motor transistor base

CLAP_THRESHOLD_MULT = 2.8  # recording.py: peak must exceed baseline * this to be a clap
CLAP_DURATION_MAX   = 0.25 # recording.py: clap spike must fall back below threshold within this many seconds
TRIGGER_MULT        = 1.8  # ai_classifier.py: mic must exceed baseline * this to start recording
SILENCE_MULT        = 1.2  # ai_classifier.py: mic must drop below baseline * this to stop

MIN_SOUND_SAMPLES = 64     # Minimum samples — at 8kHz this is 8ms (was 8 samples @ 100Hz)
MAX_SOUND_SAMPLES = 320000 # Maximum samples before force-stop (~40 s @ 8kHz)

CNN_CONFIDENCE_THRESHOLD    = 0.6  # CNN predictions below this are labelled "uncertain"
MIN_TRAINING_CONFIDENCE     = 0.4  # ai_classifier: warn if training on a sound the CNN
                                   #   was less than this confident about
CLAP_CONFIDENCE_THRESHOLD = 0.7  # listener.py: TFLite clap probability must exceed this
CLAP_CONFIDENCE_BORDER_LOW  = 0.4  # trainer.py: borderline confidence warning range (low)
CLAP_CONFIDENCE_BORDER_HIGH = 0.6  # trainer.py: borderline confidence warning range (high)

# ── Rolling noise ─────────────────────────────────────────────────────────────
ROLLING_NOISE_LEN    = 16000 # Deque length for dynamic baseline (2 s @ 8kHz)
BASELINE_UPDATE_EVERY = 800  # Recalculate dynamic baseline every N samples (every 100ms)

# ── Model / data files ────────────────────────────────────────────────────────
CLAP_MODEL_PATH     = "clap_model.pkl"       # kept for backward compat
SOUND_MODEL_PATH    = "sound_model.pkl"      # Multi-class logistic regression detector
LISTENER_MODEL_PATH = "sound_model.tflite"  # TFLite model used by listener.py
CNN_MODEL_FILE  = "cnn_sound_model.h5"   # CNN spectrogram classifier
LABEL_MAP_FILE  = "label_map.json"       # Class name ↔ index mapping

SOUNDS_DB_DIR      = "sounds_db"         # Directory for all wav files and CSVs
TRAINING_DATA_DIR  = SOUNDS_DB_DIR       # trainer.py looks here for label_*.csv files

# CSV naming convention: label_<name>.csv  e.g. label_clap.csv, label_knock.csv
# recording.py --label <name> creates these automatically.
# All CSV outputs go inside SOUNDS_DB_DIR
LOG_CSV         = f"{SOUNDS_DB_DIR}/detections.csv"   # ai_classifier.py detection log
RAW_CSV         = f"{SOUNDS_DB_DIR}/sounds_raw.csv"   # recording.py raw mic data
CLAP_CSV        = f"{SOUNDS_DB_DIR}/label_clap.csv"   # clap training examples
NOISE_CSV       = f"{SOUNDS_DB_DIR}/label_noise.csv"  # noise/background examples

SAVE_WAV        = True                   # Set False to skip wav saving in ai_classifier.py

# ── Training ──────────────────────────────────────────────────────────────────
HISTORY_LEN       = 3    # Predictions to smooth over per sound
TRAIN_EPOCHS      = 5    # Epochs per live training update in ai_classifier.py
AUTOSAVE_INTERVAL = 120  # Seconds between model autosaves

# ── Reconnection ──────────────────────────────────────────────────────────────
RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY    = 3.0  # Seconds between serial reconnection attempts

# ── Misc ──────────────────────────────────────────────────────────────────────
REPO_URL = "https://github.com/antiah-arch/EchoSafe"