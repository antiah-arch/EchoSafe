import argparse
from argparse import Namespace
from collections.abc import Iterator
from dataclasses import dataclass
import os
import sys
from typing import cast
from serial import Serial
from py.source import (
    DataEntry,
    DataStream,
    FileSource,
    MicrophoneSource,
    SerialSource,
    Source,
    initiate_serial_connection,
    open_file_data,
    open_serial_data,
)
from py.utils import Writeable, WriteableAndCloseable, error


WINDOW_SIZE = 256
FEATURE_COUNT: int = 20
REPO_URL = "https://github.com/antiah-arch/EchoSafe"




@dataclass(frozen=True)
class Record:
    seconds: int


@dataclass(frozen=True)
class Run:
    window_size: int
    feature_count: int


Command = Run | Record

def parse_serial_path(stream:Iterator[str]) -> SerialSource:
    port = next(stream, None)
    match port:
        case None:
            error("serial: requires a COMPORT, eg. serial:COM0")
        case _:
            return SerialSource(port)
def parse_file_path(stream:Iterator[str]) -> FileSource:
    path = next(stream, None)
    match path:
        case None:
            error("file: requires a PATH, eg. file:./data.csv")
        case _:
            return FileSource(path)

@dataclass(frozen=True)
class Args:
    source: Source
    output: str
    verbose:bool
    command: Command
    model:str
    @staticmethod
    def source_parser(source: str) -> Source:
        stream = iter(source.split(":"))
        first = next(stream, None)
        match first:
            case None:
                error("empty source")
            case "serial":
                return parse_serial_path(stream)
            case "file":
                return parse_file_path(stream)
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
                                    error(
                                        f'{i} is not a digit in "microphone:index:{i}"'
                                    )
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

    @staticmethod
    def from_parsed_args(raw: Namespace) -> "Args":
        print(raw)
        source : Source = Args.source_parser(raw.source)
        output : str = raw.output
        verbose : bool = raw.verbose
        model : str = raw.model
        command: Command
        match raw.command:
            case "run":
                command = Run(raw.window_size, raw.feature_count)
            case "record":
                command = Record(raw.seconds)
            case _:
                error(f"unknown sub-option {raw.command}")
        return Args(source, output, verbose, command, model)
    # second value tells it if it SHOULD be closed. stdout should not be.
    # could wrap in another ADT but simple generics should be sufficent here.
    def open_output(self) -> tuple[WriteableAndCloseable, bool]:
        stream = iter(self.output.strip().split(":"))
        first = next(stream)
        match first:
            case "stdout":
                return sys.stdout.buffer, False  # BinaryIO | Any
            case "file":
                path = parse_file_path(stream).path
                return open(path, "wb", errors="ignore"), True  # IO[Any]
            case "serial":
                port = parse_serial_path(stream).port
                return initiate_serial_connection(port), True  # pyserial.Serial
            case _:
                error("invalid output")

    def open_source(self) -> DataStream:  # type: ignore
        match self.source:
            case x if isinstance(x, SerialSource):
                serial_connection = initiate_serial_connection(
                    cast(SerialSource, self.source).port
                )
                iterator = open_serial_data(serial_connection)
                return DataStream(iterator, serial_connection)

            case x if isinstance(x, FileSource):
                path = cast(FileSource, self.source).path
                if not os.path.exists(path):
                    error(f"file {path} does not exist")
                file = open(path, "r", encoding="utf-8", errors="ignore")
                iterator = open_file_data(file)
                return DataStream(iterator, file)

            case x if isinstance(x, MicrophoneSource):
                error("microphone is not implemented")

            case _:
                error("unknown source type")
 # TODO
        # pass  # TODO

# @dataclass
# class Output:
#     inner:Writeable|WriteableAndCloseable
#     def write(self,data:ReadableBuffer) -> int:
#         return self.inner.write(data)
#     def close(self) -> None:
#         if isinstance(self.inner,):





def parse_command_line() -> Args:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("-v", "--verbose", action="store_true")
    shared.add_argument(
        "-s",
        "--source",
        help="choose source of sound data, either serial:COMPORT, microphone:[default | index:N | name:STR] | file:PATH",
        metavar="SOURCE",
        default="microphone:default",
        # required=True,
    )

    parser = argparse.ArgumentParser(
        prog="echosafe",
        description="Arduino to TensorFlowLite Interface bridge",  # whatever that means
        epilog=f"for more information see {REPO_URL}",
    )
    subparsers = parser.add_subparsers(dest="command")
    record = subparsers.add_parser("record", parents=[shared])
    record.add_argument(
        "-t", "--time", metavar="SECONDS", help="time in seconds to record"
    )
    record.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="file to write data to",
        default="recordings/recording.csv",
    )

    run = subparsers.add_parser("run", parents=[shared])
    run.add_argument("-f", "--feature-count", type=int, default=FEATURE_COUNT)
    run.add_argument("-w", "--window-size", type=int, default=WINDOW_SIZE)
    run.add_argument(
        "-m",
        "--model",
        default="./models/model.tflite",
        metavar="MODEL_PATH",
        help="tflite model file to use",
    )
    run.add_argument("-o","--output",
        default="stdout",
        metavar="OUTPUT",
        help="where to output data, can be stdout | serial:COMPORT | file:PATH"
    )

    parsed = parser.parse_args()
    return Args.from_parsed_args(parsed)
