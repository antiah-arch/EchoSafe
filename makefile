# EchoSafe Makefile
# Usage: make <target>
# Run `make help` to see all available commands.
#
# Requires GNU Make. On Windows use `nmake` or run commands directly.

.DEFAULT_GOAL := help
PYTHON        := python
PORT          ?= COM3        # override: make record PORT=COM5
DURATION      ?= 60          # override: make record DURATION=120

# ── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo ""
	@echo "  EchoSafe — available commands"
	@echo "  ─────────────────────────────────────────────"
	@echo "  Setup"
	@echo "    make install        Install all Python dependencies"
	@echo "    make install-dev    Install + dev tools (pytest, ruff)"
	@echo "    make check          Verify setup (imports, port, model files)"
	@echo ""
	@echo "  Data collection"
	@echo "    make record         Record mic data from Arduino to CSV"
	@echo "    make recognise      Record sounds from PC microphone"
	@echo "    make download       Download and prepare ESC-50 dataset"
	@echo ""
	@echo "  Training"
	@echo "    make train          Train clap detector from sounds_db CSVs"
	@echo "    make train-verbose  Train with full classification report"
	@echo ""
	@echo "  Inference"
	@echo "    make run            Run full CNN classifier (ai_classifier.py)"
	@echo "    make listen         Run lightweight TFLite listener"
	@echo "    make review         Review and relabel recorded wav files"
	@echo ""
	@echo "  Maintenance"
	@echo "    make clean-models   Delete all saved model files"
	@echo "    make clean-data     Delete all CSVs and wav files in sounds_db/"
	@echo "    make clean          clean-models + clean-data"
	@echo "    make test           Run unit tests"
	@echo "    make lint           Run ruff linter"
	@echo ""
	@echo "  Override defaults:  make record PORT=COM5 DURATION=120"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────

.PHONY: install
install:
	$(PYTHON) -m pip install -r requirements.txt

.PHONY: install-dev
install-dev:
	$(PYTHON) -m pip install -r requirements.txt pytest ruff

.PHONY: check
check:
	$(PYTHON) check_setup.py

# ── Data collection ───────────────────────────────────────────────────────────

.PHONY: record
record:
	$(PYTHON) recording.py --port $(PORT) --duration $(DURATION)

.PHONY: recognise
recognise:
	$(PYTHON) recognizer.py

.PHONY: download
download:
	$(PYTHON) dataset_downloader.py

# ── Training ──────────────────────────────────────────────────────────────────

.PHONY: train
train:
	$(PYTHON) trainer.py

.PHONY: train-verbose
train-verbose:
	$(PYTHON) trainer.py --verbose

# ── Inference ─────────────────────────────────────────────────────────────────

.PHONY: run
run:
	$(PYTHON) ai_classifier.py --port $(PORT)

.PHONY: listen
listen:
	$(PYTHON) listener.py --port $(PORT)

.PHONY: review
review:
	$(PYTHON) sounds_review.py

# ── Maintenance ───────────────────────────────────────────────────────────────

.PHONY: clean-models
clean-models:
	@echo "Deleting model files..."
	-rm -f clap_model.pkl clap_model_*.pkl
	-rm -f cnn_sound_model.h5 cnn_sound_model_*.h5
	-rm -f sound_model.tflite
	-rm -f label_map.json
	@echo "Done."

.PHONY: clean-data
clean-data:
	@echo "Deleting sounds_db contents..."
	-rm -rf sounds_db/
	@echo "Done."

.PHONY: clean
clean: clean-models clean-data

.PHONY: test
test:
	$(PYTHON) -m pytest tests/ -v

.PHONY: lint
lint:
	$(PYTHON) -m ruff check .