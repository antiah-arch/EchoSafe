# ── config.py ─────────────────────────────────────────────────────────────────
# Central configuration for EchoSafe.
# All tunable settings live here — no need to edit other files.

# ── Serial ────────────────────────────────────────────────────────────────────
COM_PORT  = "COM3"      # Arduino serial port  (e.g. COM5, /dev/ttyUSB0)
BAUD_RATE = 9600        # Arduino baud rate    (recording.py, ai_classifier.py, listener.py)
BAUDRATE  = 115200      # High-speed baud rate (source.py / cli.py pipeline)

# ── Audio — shared ────────────────────────────────────────────────────────────
ARDUINO_SAMPLE_RATE    = 100    # Hz — Arduino ADC sample rate (10ms delay in sketch)
ARDUINO_SAMPLE_INTERVAL = 1.0 / ARDUINO_SAMPLE_RATE  # seconds per sample (0.01)
PC_SAMPLE_RATE      = 16000  # Hz — PC microphone rate (recognizer.py, librosa, ESC-50)

CALIBRATION_SECONDS = 3 # Seconds of silence used to measure the noise floor

# ── Audio — Arduino / serial pipeline ────────────────────────────────────────
RECORDING_WINDOW_SIZE = 25   # Samples per sliding window in recording.py
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
FEATURE_COUNT = 16      # Features for clap_model.pkl (ai_classifier.py / trainer.py)
CLAP_WINDOW   = 64      # Samples per FFT clap-detection window

# ── Detection thresholds ──────────────────────────────────────────────────────
COOLDOWN     = 1.0      # Minimum seconds between clap detections
LED_ON_TIME  = 0.3      # Seconds the Arduino LED stays on after a clap

CLAP_THRESHOLD_MULT = 2.8  # recording.py: peak must exceed baseline * this to be a clap
CLAP_DURATION_MAX   = 0.25 # recording.py: clap spike must fall back below threshold within this many seconds
TRIGGER_MULT        = 1.8  # ai_classifier.py: mic must exceed baseline * this to start recording
SILENCE_MULT        = 1.2  # ai_classifier.py: mic must drop below baseline * this to stop

MIN_SOUND_SAMPLES = 8      # Minimum samples before a recording is considered valid
MAX_SOUND_SAMPLES = 4000   # Maximum samples before force-stop (~40 s @ ARDUINO_SAMPLE_RATE)

CNN_CONFIDENCE_THRESHOLD  = 0.6  # CNN predictions below this are labelled "uncertain"
CLAP_CONFIDENCE_THRESHOLD = 0.7  # listener.py: TFLite clap probability must exceed this
CLAP_CONFIDENCE_BORDER_LOW  = 0.4  # trainer.py: borderline confidence warning range (low)
CLAP_CONFIDENCE_BORDER_HIGH = 0.6  # trainer.py: borderline confidence warning range (high)

# ── Rolling noise ─────────────────────────────────────────────────────────────
ROLLING_NOISE_LEN    = 200  # Deque length for dynamic baseline
BASELINE_UPDATE_EVERY = 10  # Recalculate dynamic baseline every N samples

# ── Model / data files ────────────────────────────────────────────────────────
CLAP_MODEL_PATH     = "clap_model.pkl"       # Logistic regression clap detector
LISTENER_MODEL_PATH = "sound_model.tflite"  # TFLite model used by listener.py
CNN_MODEL_FILE  = "cnn_sound_model.h5"   # CNN spectrogram classifier
LABEL_MAP_FILE  = "label_map.json"       # Class name ↔ index mapping

SOUNDS_DB_DIR   = "sounds_db"            # Directory for all wav files and CSVs

# All CSV outputs go inside SOUNDS_DB_DIR
LOG_CSV         = f"{SOUNDS_DB_DIR}/detections.csv"   # ai_classifier.py detection log
RAW_CSV         = f"{SOUNDS_DB_DIR}/sounds_raw.csv"   # recording.py raw mic data
CLAP_CSV        = f"{SOUNDS_DB_DIR}/sound_data_label1.csv"  # trainer.py clap examples
NOISE_CSV       = f"{SOUNDS_DB_DIR}/sound_data_label0.csv"  # trainer.py noise examples

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