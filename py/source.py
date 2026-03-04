"""
source.py
---------
Data source abstractions for EchoSafe's CLI pipeline.

Defines the Source ADT (SerialSource, MicrophoneSource, FileSource),
DataEntry (one timestamped mic reading), and DataStream (iterator + closer).
"""

import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from io import TextIOWrapper
from time import time

import serial  # FIX 9: single serial import — use serial.Serial throughout

from config import BAUDRATE
from serial_helper import open_serial as _open_serial
from utils import warning


# ── Source ADT ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SerialSource:
    port: str


@dataclass(frozen=True)
class MicrophoneSource:
    default: bool = False
    index: int | None = None
    substring: str | None = None


@dataclass(frozen=True)
class FileSource:
    path: str


Source = SerialSource | MicrophoneSource | FileSource


# ── DataEntry ──────────────────────────────────────────────────────────────────

@dataclass
class DataEntry:
    time: float
    microphone: int
    clap_confidence: float  # score in 0..1 of how much of a clap it is

    @staticmethod
    def from_csv_entry(s: str) -> "DataEntry | None":
        parts = s.strip().split(",")

        if len(parts) < 3:
            warning(
                f"could not parse CSV entry {parts!r}: "
                f"expected 3 columns, got {len(parts)}."
            )
            return None

        raw_time, raw_microphone, raw_clap_confidence = parts[:3]

        try:
            entry_time      = float(raw_time)
            microphone      = int(raw_microphone)
            clap_confidence = float(raw_clap_confidence)
        except ValueError:
            warning(
                f"could not parse mangled CSV entry {parts!r}: "
                "invalid numeric value."
            )
            return None

        return DataEntry(entry_time, microphone, clap_confidence)

    def to_csv_entry(self) -> str:
        # FIX 7: kept as a utility — useful for writing DataEntry back to CSV
        # (e.g. in a future export or replay feature)
        return f"{self.time},{self.microphone},{self.clap_confidence}"

    @staticmethod
    def from_mic_iterable(microphone_values: Iterable[int]) -> "Iterator[DataEntry]":
        # FIX 8: capture start inside the generator so timing begins when the
        # first value is consumed, not when from_mic_iterable() is called
        def _gen() -> Iterator["DataEntry"]:
            start = time()
            for mic in microphone_values:
                yield DataEntry(time() - start, mic, 0.0)
        return _gen()


# ── Serial ─────────────────────────────────────────────────────────────────────

def initiate_serial_connection(com_port: str) -> serial.Serial:
    """
    Open a serial connection via shared serial_helper.
    FIX 5: error() from utils raises RuntimeError making sys.exit() after it
    unreachable. Replaced with print() + sys.exit(1) for clarity.
    FIX 9: returns serial.Serial (not the redundant Serial alias).
    """
    try:
        return _open_serial(com_port, BAUDRATE)
    except serial.SerialException as e:
        print(f"Could not open serial port {com_port!r}: {e}")
        sys.exit(1)


def open_serial_data(serial_connection: serial.Serial) -> Iterator[DataEntry]:
    """
    FIX 1: replaced iter(callable, sentinel) pattern with an explicit generator.
    The old iter(lambda: readline(), '') stopped permanently the moment Arduino
    sent no data (timeout), silently terminating the stream.

    FIX 2: replaced isdigit() filter with try/except int() so negative values
    or unexpected formats produce a warning rather than silent drops.

    FIX 3: removed redundant int() in from_mic_iterable — values are already
    ints after the try/except parse below.
    """
    def _gen() -> Iterator[DataEntry]:
        start = time()
        while True:
            raw = serial_connection.readline()
            if not raw:
                # Timeout with no data — keep waiting, don't terminate stream
                continue
            decoded = raw.decode(errors="ignore").strip()
            if not decoded:
                continue
            try:
                mic = int(decoded)
            except ValueError:
                # FIX 2: warn on unexpected format instead of silently dropping
                warning(f"could not parse serial value {decoded!r}: not an integer")
                continue
            yield DataEntry(time() - start, mic, 0.0)

    return _gen()


# ── File ───────────────────────────────────────────────────────────────────────

def open_file_data(lines: TextIOWrapper) -> Iterator[DataEntry]:
    return (
        entry
        for entry in (DataEntry.from_csv_entry(line) for line in lines)
        if entry is not None
    )


# ── Microphone ─────────────────────────────────────────────────────────────────

def open_microphone_data(source: MicrophoneSource) -> Iterator[DataEntry]:
    # FIX 6: kept as a named stub so cli.py can reference it and future
    # implementation has a clear home. NotImplementedError is intentional.
    raise NotImplementedError(
        "Microphone source is not yet implemented. "
        "Use serial:COMPORT or file:PATH instead."
    )


# ── DataStream ─────────────────────────────────────────────────────────────────

@dataclass
class DataStream:
    iterator: Iterator[DataEntry]
    backer: TextIOWrapper | serial.Serial  # FIX 9: serial.Serial not the alias

    def close(self) -> None:
        # FIX 4: catch only the exceptions that can actually occur on close,
        # not bare Exception which hides real bugs like AttributeError
        try:
            self.backer.close()
        except (serial.SerialException, OSError) as e:
            warning(f"Could not close data stream cleanly: {e}")