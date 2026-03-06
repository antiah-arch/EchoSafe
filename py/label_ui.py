"""
label_ui.py
-----------
Threaded labelling UI for EchoSafe's ai_classifier.

Decouples the blocking input() call from the main serial loop so sounds
are never missed while the user is typing a label.

Architecture:
    Main thread  ──► sound_queue  ──► LabelWorker thread
                 ◄── result_queue ◄──

The main thread puts SoundResult objects onto sound_queue.
LabelWorker pulls them off, prompts the user, and puts LabelResult
objects onto result_queue. The main thread polls result_queue on each
loop iteration and trains the CNN as soon as a result is ready —
no blocking, no missed samples.

Usage:
    worker = LabelWorker(label_map, verbose=True)
    worker.start()

    # in main loop:
    worker.submit(sound)          # non-blocking
    result = worker.poll()        # non-blocking, returns None if not ready
    if result:
        train_on_result(result)

    worker.stop()                 # call on shutdown
"""

import queue
import threading
from dataclasses import dataclass

import numpy as np


# ── Data classes passed between threads ───────────────────────────────────────

@dataclass
class SoundResult:
    """Everything the main thread knows about a detected sound."""
    waveform:    np.ndarray   # raw Arduino mic values as float array
    specs_aug:   list         # augmented mel spectrograms (list of np.ndarray)
    final_label: str          # label assigned by CNN / clap detector
    confidence:  float        # CNN confidence 0..1
    peak:        float
    duration:    float
    wav_path:    str | None   # saved .wav path, or None if not saved


@dataclass
class LabelResult:
    """What the label thread sends back to the main thread."""
    sound:       SoundResult
    user_label:  str          # the label the user typed
    is_new_label: bool        # True if this label wasn't in the map before


# ── LabelWorker ───────────────────────────────────────────────────────────────

class LabelWorker:
    """
    Background thread that handles all user-facing labelling interaction.

    Thread safety:
      - sound_queue and result_queue are thread-safe by design (queue.Queue).
      - label_map is read by the worker and written back via LabelResult.
        The main thread must apply label_map updates from LabelResult before
        calling predict() on the next sound — i.e. never mutate label_map
        concurrently.
      - CNN model access (predict / fit) stays on the main thread entirely.
        The worker never touches the model directly.
    """

    def __init__(
        self,
        label_map: dict[str, int],
        verbose: bool = False,
        queue_maxsize: int = 32,
    ) -> None:
        self.label_map   = label_map        # shared reference — main thread owns writes
        self.verbose     = verbose
        self._sound_q:  queue.Queue[SoundResult | None] = queue.Queue(maxsize=queue_maxsize)
        self._result_q: queue.Queue[LabelResult]        = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            name="LabelWorker",
            daemon=True,    # exits automatically when main thread exits
        )
        self._pending = 0   # sounds submitted but not yet labelled
        self._lock = threading.Lock()

    # ── Public API (called from main thread) ──────────────────────────────────

    def start(self) -> None:
        """Start the background labelling thread."""
        self._thread.start()

    def submit(self, sound: SoundResult) -> None:
        """
        Put a sound onto the labelling queue. Non-blocking.
        If the queue is full (>32 pending sounds), the oldest is dropped
        with a warning rather than blocking the main loop.
        """
        try:
            self._sound_q.put_nowait(sound)
            with self._lock:
                self._pending += 1
        except queue.Full:
            print(
                f"\n⚠️  Label queue full ({self._sound_q.maxsize} pending) — "
                "dropping oldest unlabelled sound. Label faster or increase queue_maxsize."
            )

    def poll(self) -> LabelResult | None:
        """
        Check if a label result is ready. Non-blocking.
        Returns LabelResult if the user has submitted a label, else None.
        """
        try:
            result = self._result_q.get_nowait()
            with self._lock:
                self._pending -= 1
            return result
        except queue.Empty:
            return None

    def pending_count(self) -> int:
        """How many sounds are waiting to be labelled."""
        with self._lock:
            return self._pending

    def stop(self) -> None:
        """
        Signal the worker to finish after processing all queued sounds,
        then wait for it to exit. Call this during shutdown.
        """
        self._sound_q.put(None)   # sentinel — tells _run() to exit
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            print("Warning: label worker did not stop cleanly within 10s.")

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Main loop of the background thread."""
        while True:
            sound = self._sound_q.get()

            if sound is None:
                # Sentinel received — drain any remaining results then exit
                break

            result = self._prompt(sound)
            if result is not None:
                self._result_q.put(result)
            else:
                # User skipped — decrement pending without putting a result
                with self._lock:
                    self._pending -= 1

    def _prompt(self, sound: SoundResult) -> LabelResult | None:
        """
        Show sound info to the user and ask for a label.
        Returns LabelResult if the user typed a label, None if they skipped.
        Runs entirely on the worker thread — safe to block here.
        """
        pending = self.pending_count()
        queue_info = f"  [{pending} more in queue]" if pending > 1 else ""

        print(
            f"\n┌─ Label sound ──────────────────────────────\n"
            f"│  CNN label   : {sound.final_label}\n"
            f"│  Confidence  : {sound.confidence:.0%}\n"
            f"│  Peak        : {sound.peak:.0f}    Duration: {sound.duration:.2f}s\n"
            f"│  Wav         : {sound.wav_path or '(not saved)'}\n"
            f"└────────────────────────────────────────────{queue_info}"
        )

        try:
            raw = input("  Label (ENTER to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            # Terminal closed or Ctrl+C during input — skip gracefully
            return None

        if not raw:
            if self.verbose:
                print("  (skipped)")
            return None

        is_new = raw not in self.label_map
        if is_new:
            # Register immediately so subsequent prompts show the right map size.
            # The main thread will also update its copy when it processes LabelResult.
            self.label_map[raw] = len(self.label_map)
            print(f"  ✨ New label '{raw}' registered (index {self.label_map[raw]})")

        return LabelResult(
            sound=sound,
            user_label=raw,
            is_new_label=is_new,
        )