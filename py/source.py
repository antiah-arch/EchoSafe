import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from io import TextIOWrapper
from time import time
from typing import cast

from serial import Serial
from utils import error, success, warning

BAUDRATE = 115200  # speed of communication over connection in baud
QUANTITY: int = 0
TIME_LABEL = "time"
MIC_VALUE_LABEL = "mic_value"
QUANTITY_LABEL = "label"


# Source ADT
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


def source_parser(source: str) -> Source:
    stream = iter(source.split(":"))
    first = next(stream, None)
    match first:
        case None:
            error("empty source")
        case "serial":
            port = next(stream, None)
            match port:
                case None:
                    error("serial: requires a COMPORT, eg. --source serial:COM0")
                case _:
                    return SerialSource(port)
        case "file":
            path = next(stream, None)
            match path:
                case None:
                    error("file: requires a PATH, eg. --source file:./data.csv")
                case _:
                    return FileSource(path)
        case "microphone":
            submethod = next(stream, None)
            match submethod:
                case None:
                    error(
                        "microphone: requires a submethod eg. --source microphone:default"
                    )
                case "default":
                    return MicrophoneSource(default=True)
                case "index":
                    i = next(stream, None)
                    match i:
                        case None:
                            error(
                                "microphone:index requires a number, eg. --source microphone:index:0"
                            )
                        case _:
                            if not i.isdigit():
                                error(f'{i} is not a digit in "microphone:index:{i}"')
                            return MicrophoneSource(index=int(i))
                case "name":
                    name = next(stream, None)
                    match name:
                        case None:
                            error(
                                "microphone:index requires a name which may be a substring of the full system name, eg. --source microphone:name:built-in"
                            )
                        case _:
                            return MicrophoneSource(substring=name)
                case _:
                    error(f"Unknown method {submethod}")
        case _:
            error(f"Unknown method {first}")


@dataclass
class DataEntry:
    time: float
    microphone: int
    clap_confidence: float # score of how much of a clap it is.

    @staticmethod
    def from_csv_entry(s: str) -> "DataEntry|None":
        parsed = s.strip().split(",")[:3]
        raw_time, raw_microphone, raw_clap_confidence = parsed
        # if any arent digits
        if any(map(lambda a: not a.isdigit(), parsed)):
            warning(f"could not parse mangled CSV entry {parsed} non-digit present.")
            return None
        else:
            time = float(raw_time)
            microphone = int(raw_microphone)
            clap_confidence = float(raw_clap_confidence)
            return DataEntry(time, microphone, clap_confidence)

    def to_csv_entry(self) -> str:
        return f"{self.time},{self.microphone},{self.clap_confidence}"

    @staticmethod
    def from_mic_iterable(microphone_values: Iterable[int]) -> "Iterator[DataEntry]":
        start = time()
        return map(
            lambda mic: DataEntry(time() - start, int(mic), CLAP_FACTOR), microphone_values
        )
        # filter(lambda mic: not mic.isdigit(), microphone_values),


def initiate_serial_connection(com_port: str) -> Serial:
    serial_connection = Serial(com_port, BAUDRATE, timeout=1)
    success(f"connected to device on port {com_port}")
    return serial_connection


def open_microphone_data(microphone_raw: str) -> Iterator[DataEntry]:
    error("microphone not implemented")


def open_serial_data(serial_connection: Serial) -> Iterator[DataEntry]:
    microphone_values: Iterator[str] = iter(
        lambda: serial_connection.readline().decode(errors="ignore").strip(), ""
    )
    
    return DataEntry.from_mic_iterable(
        map(
            lambda mic: int(mic),
            filter(lambda mic: mic.isdigit(), microphone_values),
        )
    )


def open_file_data(lines: TextIOWrapper) -> Iterator[DataEntry]:
    return (
        cast(  # needed because pyright is unaware of the type narrowing in the filter
            Iterator[DataEntry],
            filter(
                lambda x: isinstance(x, DataEntry),
                map(lambda entry: DataEntry.from_csv_entry(entry), lines),
            ),
        )
    )


@dataclass
class DataStream:
    iterator: Iterator[DataEntry]
    backer: TextIOWrapper | Serial

    def close(self):
        self.backer.close()


