# EchoSafe

Real-time clap detection and sound classification using an Arduino microphone and a Python ML pipeline.

EchoSafe listens to an Arduino over serial, detects clap events using a fast FFT logistic-regression model, and classifies other sounds using a CNN spectrogram classifier — all in real time. Every detected sound can be labelled on the fly to train the CNN without stopping the system.

---

## Requirements

- Python 3.10 or newer (uses `match/case` and `X | Y` type unions)
- Arduino Uno/Nano with a microphone module (see [Arduino sketch](#arduino-sketch))
- Windows, macOS, or Linux

---

## Installation

```bash
git clone https://github.com/antiah-arch/EchoSafe.git
cd EchoSafe

pip install -r requirements.txt

# Optional: install as commands so you can run `echosafe` from anywhere
pip install -e .
```

---

## Arduino sketch

The Arduino must send raw ADC microphone readings as plain integers over serial, one per line, at 9600 baud. A minimal sketch:

```cpp
const int MIC_PIN = A0;

void setup() {
  Serial.begin(9600);
}

void loop() {
  Serial.println(analogRead(MIC_PIN));  // sends e.g. "512\n"
  delay(10);                            // ~100 samples per second
}
```

Upload this to your Arduino before running any EchoSafe scripts. Set `COM_PORT` in `config.py` to match your board's port (e.g. `COM3` on Windows, `/dev/ttyUSB0` on Linux).

---

## Quick start

```bash
# 1. Download training data (ESC-50 clap + noise samples)
make download

# 2. Train the clap detector
make train

# 3. Run the full classifier
make run
```

Or using the CLI directly:

```bash
python main.py record   # record raw mic data
python main.py train    # train clap model
python main.py run      # run live inference
python main.py review   # review and relabel wav files
```

---

## File overview

| File | Purpose |
|------|---------|
| `config.py` | All tunable settings — edit this first |
| `main.py` | Unified CLI entry point (`record` / `train` / `run` / `review`) |
| `ai_classifier.py` | Full pipeline: FFT clap detector + CNN classifier + live training |
| `listener.py` | Lightweight TFLite-only clap detector — minimal dependencies |
| `recording.py` | Record raw Arduino mic data to CSV |
| `recognizer.py` | Record sounds from PC microphone to wav + CSV |
| `trainer.py` | Train logistic regression clap model from CSV data |
| `dataset_downloader.py` | Download and prepare the ESC-50 dataset |
| `sounds_review.py` | Review, relabel, and delete recorded wav files |
| `label_ui.py` | Threaded labelling UI — keeps `ai_classifier.py` listening while you type |
| `source.py` | Serial / file / microphone data stream abstraction |
| `serial_helper.py` | Shared serial open / close / reconnect / send helpers |
| `model_versioning.py` | Timestamped model saves with automatic pruning |
| `utils.py` | Coloured console output + rotating log file (`echosafe.log`) |

---

## Two inference modes

**`ai_classifier.py`** — the full pipeline. Runs both the FFT clap detector and a CNN spectrogram classifier simultaneously. After each detected sound it prompts you to label it and immediately fine-tunes the CNN. Best for training and building a labelled dataset.

**`listener.py`** — lightweight TFLite-only listener. Only detects claps, no CNN, no labelling. Designed for long unattended runs on low-power hardware.

---

## Configuration

All settings are in `config.py`. Key values:

| Setting | Default | Description |
|---------|---------|-------------|
| `COM_PORT` | `COM3` | Arduino serial port |
| `BAUD_RATE` | `9600` | Must match Arduino sketch |
| `CLAP_THRESHOLD_MULT` | `2.8` | How loud a spike must be to count as a clap |
| `CNN_CONFIDENCE_THRESHOLD` | `0.6` | Below this the CNN labels a sound "uncertain" |
| `SAVE_WAV` | `True` | Set `False` to skip saving wav files |
| `SOUNDS_DB_DIR` | `sounds_db` | Directory for all wav files and CSVs |

---

## Data pipeline

```
dataset_downloader.py ──► sounds_db/sound_data_label1.csv  (clap examples)
recording.py          ──► sounds_db/sound_data_label0.csv  (noise examples)
                                        │
                                  trainer.py
                                        │
                                  clap_model.pkl
                                        │
                            ai_classifier.py / listener.py
                                        │
                              sends b"1" to Arduino LED
```

---

## Model versioning

Every time a model is saved, a timestamped backup is created alongside it:

```
clap_model.pkl                  ← canonical file (always the latest)
clap_model_20240415_143022.pkl  ← backup
clap_model_20240415_150811.pkl  ← backup
clap_model_20240415_161234.pkl  ← backup (3 max, older ones pruned)
```

To restore a previous model, rename the desired backup to the canonical filename.

---

## Log file

All output is written to `echosafe.log` (rotates at 1 MB, keeps 3 backups). To change the path:

```bash
ECHOSAFE_LOG=/var/log/echosafe.log python ai_classifier.py
```

---

## Make shortcuts

```bash
make install       # install dependencies
make download      # fetch ESC-50 dataset
make record        # record from Arduino
make train         # train clap model
make run           # run full classifier
make listen        # run lightweight listener
make review        # review wav files
make clean         # delete all models and data
make help          # show all commands
```