import argparse
import os
import sys
from argparse import Namespace
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

from source import (
    DataStream,
    FileSource,
    MicrophoneSource,
    SerialSource,
    Source,
    initiate_serial_connection,
    open_file_data,
    open_serial_data,
)
from utils import error
from config import CLI_WINDOW_SIZE as WINDOW_SIZE, FEATURE_COUNT, REPO_URL, COM_PORT, BAUD_RATE


# ── Command dataclasses ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Record:
    seconds: int        # FIX 7: no longer optional — always an int, defaults to 60


@dataclass(frozen=True)
class Train:
    window_size: int
    feature_count: int


@dataclass(frozen=True)
class Run:
    pass


@dataclass(frozen=True)
class Review:
    directory: str
    export: str | None  # FIX 6: new command for sounds_review


Command = Train | Record | Run | Review


# ── Source parsers ─────────────────────────────────────────────────────────────

def parse_serial_path(stream: Iterator[str]) -> SerialSource:
    port = next(stream, None)
    if port is None:
        error("serial: requires a COMPORT, eg. serial:COM0")
    return SerialSource(port)          # type: ignore[arg-type]


def parse_file_path(stream: Iterator[str]) -> FileSource:
    path = next(stream, None)
    if path is None:
        error("file: requires a PATH, eg. file:./data.csv")
    return FileSource(path)            # type: ignore[arg-type]


def validate_serial_port(port: str) -> str:
    """
    FIX 8: validate port format at parse time so bad values give a clear
    error immediately rather than crashing deep inside open_source().
    Accepts Windows COM ports (COM1–COM256) and Unix /dev/ paths.
    """
    import re
    if re.match(r"^COM\d{1,3}$", port, re.IGNORECASE):
        return port.upper()
    if port.startswith("/dev/"):
        return port
    error(
        f"Invalid serial port {port!r}. "
        "Expected a Windows COM port (e.g. COM3) or a Unix device path (e.g. /dev/ttyUSB0)."
    )


# ── Args dataclass ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Args:
    source: Source
    output: str
    verbose: bool
    command: Command
    model: str
    port: str       # FIX 4: explicit port field so all subcommands can override it
    baud: int       # FIX 5: explicit baud field

    @staticmethod
    def source_parser(source: str) -> Source:
        stream = iter(source.split(":"))
        first = next(stream, None)

        if first is None:
            error("empty source")

        if first == "serial":
            return parse_serial_path(stream)

        if first == "file":
            return parse_file_path(stream)

        if first == "microphone":
            submethod = next(stream, None)
            if submethod is None:
                error("microphone: requires a submethod eg. --source microphone:default")

            if submethod == "default":
                return MicrophoneSource(default=True)

            if submethod == "index":
                i = next(stream, None)
                if i is None:
                    error("microphone:index requires a number, eg. --source microphone:index:0")
                if not i.isdigit():                        # type: ignore[union-attr]
                    error(f'{i} is not a digit in "microphone:index:{i}"')
                return MicrophoneSource(index=int(i))      # type: ignore[arg-type]

            if submethod == "name":
                name = next(stream, None)
                if name is None:
                    error(
                        "microphone:name requires a name which may be a substring "
                        "of the full system name, eg. --source microphone:name:built-in"
                    )
                return MicrophoneSource(substring=name)

            error(f"Unknown microphone submethod: {submethod}")

        error(f"Unknown source method: {first}")

    @staticmethod
    def from_parsed_args(raw: Namespace) -> "Args":
        source: Source = Args.source_parser(raw.source)
        output: str = raw.output
        verbose: bool = raw.verbose
        model: str = getattr(raw, "model", "")
        port: str = validate_serial_port(raw.port)   # FIX 8: validate at parse time
        baud: int = raw.baud
        command: Command

        match raw.command:
            case "train":
                command = Train(raw.window_size, raw.feature_count)
            case "record":
                command = Record(raw.time)             # FIX 7: always int
            case "run":
                command = Run()
            case "review":                             # FIX 6
                command = Review(raw.dir, getattr(raw, "export", None))
            case _:
                error(f"unknown sub-command {raw.command}")

        return Args(source, output, verbose, command, model, port, baud)

    def open_output(self) -> tuple[object, bool]:
        stream = iter(self.output.strip().split(":"))
        first = next(stream, None)

        if first == "stdout":
            return sys.stdout.buffer, False

        if first == "file":
            path = parse_file_path(stream).path
            # FIX 3: ensure parent directory exists before opening file
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            return open(path, "wb"), True

        if first == "serial":
            port = parse_serial_path(stream).port
            return initiate_serial_connection(port), True

        error(f"invalid output: {self.output!r}")

    def open_source(self) -> DataStream:
        if isinstance(self.source, SerialSource):
            serial_connection = initiate_serial_connection(
                cast(SerialSource, self.source).port
            )
            iterator = open_serial_data(serial_connection)
            return DataStream(iterator, serial_connection)

        if isinstance(self.source, FileSource):
            path = cast(FileSource, self.source).path
            if not os.path.exists(path):
                error(f"file {path} does not exist")
            file = open(path, "r", encoding="utf-8", errors="ignore")
            iterator = open_file_data(file)
            return DataStream(iterator, file)

        if isinstance(self.source, MicrophoneSource):
            error("microphone source is not yet implemented")

        error("unknown source type")


# ── CLI definition ─────────────────────────────────────────────────────────────

def parse_command_line() -> Args:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("-v", "--verbose", action="store_true")
    shared.add_argument(
        "-s", "--source",
        help=(
            "source of sound data: "
            "serial:COMPORT | microphone:[default|index:N|name:STR] | file:PATH"
        ),
        metavar="SOURCE",
        default="microphone:default",
    )
    # FIX 4 & 5: --port and --baud in shared parser so every subcommand inherits them
    shared.add_argument(
        "--port",
        default=COM_PORT,
        metavar="PORT",
        help=f"serial port to use (default: {COM_PORT})",
    )
    shared.add_argument(
        "--baud",
        type=int,
        default=BAUD_RATE,
        metavar="BAUD",
        help=f"serial baud rate (default: {BAUD_RATE})",
    )

    parser = argparse.ArgumentParser(
        prog="echosafe",
        description="EchoSafe — Arduino sound classification pipeline",
        epilog=f"for more information see {REPO_URL}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── record ────────────────────────────────────────────────────────────────
    record = subparsers.add_parser(
        "record",
        parents=[shared],
        help="record raw mic data from Arduino to CSV",
    )
    record.add_argument(
        "-t", "--time",
        metavar="SECONDS",
        type=int,
        default=60,                    # FIX 7: sensible default instead of None
        help="recording duration in seconds (default: 60)",
    )
    record.add_argument(
        "-o", "--output",
        metavar="FILE",
        default="recordings/recording.csv",
        help="CSV file to write data to (default: recordings/recording.csv)",
    )

    # ── train ─────────────────────────────────────────────────────────────────
    train = subparsers.add_parser(
        "train",
        parents=[shared],
        help="train the clap detection model from labelled CSV data",
    )
    train.add_argument("-f", "--feature-count", type=int, default=FEATURE_COUNT,
                       help=f"number of FFT features (default: {FEATURE_COUNT})")
    train.add_argument("-w", "--window-size", type=int, default=WINDOW_SIZE,
                       help=f"FFT window size in samples (default: {WINDOW_SIZE})")
    train.add_argument(
        "-m", "--model",
        default="./models/model.tflite",
        metavar="MODEL_PATH",
        help="tflite model output path (default: ./models/model.tflite)",
    )
    train.add_argument(
        "-o", "--output",
        default="stdout",
        metavar="OUTPUT",
        help="where to write output: stdout | serial:COMPORT | file:PATH",
    )

    # ── run ───────────────────────────────────────────────────────────────────
    # FIX 1: variable renamed to run_cmd so it doesn't shadow the run_command()
    # function defined below, which would cause a confusing crash at call time
    run_cmd = subparsers.add_parser(
        "run",
        parents=[shared],
        help="run live inference using a trained model",
    )
    run_cmd.add_argument(
        "-m", "--model",
        default="./models/model.tflite",
        metavar="MODEL_PATH",
        help="tflite model file to run (default: ./models/model.tflite)",
    )
    run_cmd.add_argument(
        "-o", "--output",
        default="stdout",
        metavar="OUTPUT",
        help="where to write results: stdout | serial:COMPORT | file:PATH",
    )

    # ── review ────────────────────────────────────────────────────────────────
    # FIX 6: wire sounds_review.py into the main echosafe CLI
    review = subparsers.add_parser(
        "review",
        parents=[shared],
        help="interactively review and relabel recorded .wav files",
    )
    review.add_argument(
        "--dir",
        default="sounds_db",
        metavar="DIRECTORY",
        help="directory of .wav files to review (default: sounds_db)",
    )
    review.add_argument(
        "--export",
        default=None,
        metavar="CSV_PATH",
        help="export reviewed labels to this CSV path after finishing",
    )

    parsed = parser.parse_args()
    return Args.from_parsed_args(parsed)


# FIX 1: renamed from run() to run_command() to avoid clashing with the
# run_cmd subparser variable above
def run_command(source_iterator, model_path: str) -> None:
    """Placeholder — wired up via listener.main() in main.py."""
    raise NotImplementedError("run command is not yet implemented")