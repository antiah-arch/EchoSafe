#!/usr/bin/env python
import argparse
import csv
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum, auto
from sys import argv
from time import time
from typing import TYPE_CHECKING, NoReturn, cast

import colored
import numpy as np
import pandas
import scipy
import serial
from numpy.fft import rfft
from numpy.typing import NDArray
from pandas.api.extensions import ExtensionArray
from serial import Serial
from argparse import Namespace
# from numpy.typing import ExtensionArray

# allows type checker to see tf while still lazy loading per module
if TYPE_CHECKING:
    import tensorflow as tf

# inputs
# training_data_directory = "model_input_data"
# defaults
BAUDRATE = 9600  # speed of communication over connection in baud
WINDOW_SIZE = 256
FEATURE_COUNT: int = 20
REPO_URL = "https://github.com/antiah-arch/EchoSafe"
COM_PORT: str = "COM3"
QUANTITY = 0
ERROR_RETURN = 2
TIME_LABEL = "time"
MIC_VALUE_LABEL = "mic_value"
QUANTITY_LABEL = "label"


def error(msg: str) -> NoReturn:
    print(colored.stylize(msg, colored.fore("red")))
    exit(ERROR_RETURN)


def warning(msg: str) -> None:
    print(colored.stylize(msg, colored.fore("yellow")))


def success(msg: str) -> None:
    print(colored.stylize(msg, colored.fore("green")))


def subtext(msg: str) -> None:
    print(colored.stylize(msg, colored.fore("dark_gray")))


def extract_features(signal: list[int], feature_count: int) -> NDArray:
    fft_vals = np.abs(rfft(signal))
    features = np.array(
        [np.mean(fft_vals[i::feature_count]) for i in range(feature_count)]
    )
    return np.array([features], dtype=np.float32)


# def load_data():
#     features = []
#     labels = []
#     data: list[str] = os.listdir("model_input_data")
#     for data_file in data:
#         # assert os.path.exists(data_file)
#         csv_data: pandas.DataFrame = pandas.read_csv(data_file)
#         # label = int(csv_data["label"].iloc[0])
#         features.append(extract_features(csv_data))
#         labels.append(int(csv_data["label"].iloc[0]))


def initialize_model(model: str) -> "tf.lite.Interpreter":
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()
    return interpreter


# features of this program, it supports running on a standard environment as well as on an arduino.
# decided by the test option, which specifies a replacement to the serial datastream
@dataclass
class Config:
    window_size: int
    com_port: str
    verbose: bool
    model: str
    feature_count: int
    test: str | None


# /Config


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

@dataclass(frozen=True)
class Record:
    seconds:int

@dataclass(frozen=True)
class Run:
    window_size:int
    feature_count:int
Command = Run|Record

@dataclass(frozen=True)
class Args:
    source: Source
    output:str
    command:Command
    @staticmethod
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
    def from_parsed_args(raw:Namespace) -> "Args":
        # subcommand = raw.run if raw.run is not None else raw.record if raw.record else None
        # if raw.source is None:
        #     # error("missing data source")
        print(raw)
        source = Args.source_parser(raw.source)
        output = raw.output if raw.output is not None else raw.model
        command : Command
        match raw.command:
            case "run":
                command = Run(raw.window_size,raw.feature_count)
            case "record":
                command = Record(raw.seconds)
            case _:
                error(f"unknown sub-option {raw.command}")
        return Args(source,output,command)
    def open_source(self) -> Iterator[DataEntry]:
        pass # TODO

def parse_command_line() -> Args:
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

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("-v", "--verbose", action="store_true")
    shared.add_argument(
        "-s",
        "--source",
        help="choose source of sound data, either serial:COMPORT, microphone:[default | index:N | name:STR] | file:PATH",
        metavar="SOURCE",
        default="microphone:default"
        # required=True,
    )

    parser = argparse.ArgumentParser(
        prog="echosafe",
        description="Arduino to TensorFlowLite Interface bridge",  # whatever that means
        epilog=f"for more information see {REPO_URL}",
    )
    subparsers = parser.add_subparsers(dest="command")
    record = subparsers.add_parser("record",parents=[shared])
    record.add_argument(
        "-t", "--time", metavar="SECONDS", help="time in seconds to record"
    )
    record.add_argument("-o", "--output", metavar="FILE", help="file to write data to",default="recordings/recording.csv")

    run = subparsers.add_parser("run",parents=[shared])
    run.add_argument("-f","--feature-count", type=int, default=FEATURE_COUNT)
    run.add_argument("-w", "--window-size", type=int, default=WINDOW_SIZE)
    run.add_argument(
        "-o",
        "--output",
        default="./models/model.tflite",
        metavar="FILE",
        help="file path to write model",
    )

    parsed = parser.parse_args()
    return Args.from_parsed_args(parsed)



@dataclass
class DataEntry:
    time: float
    microphone: int
    quantity: int

    @staticmethod
    def from_csv_entry(s: str) -> "DataEntry|None":
        parsed = s.strip().split(",")[:3]
        raw_time, raw_microphone, raw_quantity = parsed
        # if any arent digits
        if any(map(lambda a: not a.isdigit(), parsed)):
            warning(f"could not parse mangled CSV entry {parsed} non-digit present.")
            return None
        else:
            time = float(raw_time)
            microphone = int(raw_microphone)
            quantity = int(raw_quantity)
            return DataEntry(time, microphone, quantity)

    def to_csv_entry(self) -> str:
        return f"{self.time},{self.microphone},{self.quantity}"

    @staticmethod
    def from_mic_iterable(microphone_values: Iterable[int]) -> "Iterator[DataEntry]":
        start = time()
        return map(
            lambda mic: DataEntry(time() - start, int(mic), QUANTITY), microphone_values
        )
        # filter(lambda mic: not mic.isdigit(), microphone_values),

    # @staticmethod
    # def from_iterable(xs: Iterable) -> "Iterator[DataEntry]":
    #     return cast(  # needed because pyright is unaware of the type narrowing in the filter
    #         Iterator[DataEntry],
    #         filter(
    #             lambda x: isinstance(x, DataEntry),
    #             map(lambda entry: DataEntry.from_csv_entry(entry), xs),
    #         ),
    #     )

    # @staticmethod


# returns an iterator over the serial connection
#

def override_file(path:str) -> bool:
    print(f"Override file \"{path}\"?")
    subtext(f"this file already exists on disk")
    resp = input("Y\n").strip().lower()
    return resp == "y";

def initiate_serial_connection(com_port: str) -> Serial:
    serial_connection = Serial(com_port, BAUDRATE, timeout=1)
    success(f"connected to device on port {com_port}")
    return serial_connection


def open_serial_data(serial_connection: serial.Serial) -> Iterator[DataEntry]:
    microphone_values: Iterator[str] = iter(
        lambda: serial_connection.readline().decode(errors="ignore").strip(), ""
    )
    return DataEntry.from_mic_iterable(
        map(
            lambda mic: int(mic),
            filter(lambda mic: mic.isdigit(), microphone_values),
        )
    )


def open_simulated_data(path: str) -> Iterator[DataEntry]:
    # assert isinstance(config.test, str)
    if not os.path.exists(path):
        error(f"test data file {path} not found.")
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as csv_lines:
            return cast(  # needed because pyright is unaware of the type narrowing in the filter
                Iterator[DataEntry],
                filter(
                    lambda x: isinstance(x, DataEntry),
                    map(lambda entry: DataEntry.from_csv_entry(entry), csv_lines),
                ),
            )
            # return DataEntry.from_iterable(csv_lines)

            # map(lambda entry:,f )
            # for line in f:
            #     token = line.strip().split(",")[0]
            #     if token.isdigit():
            #         yield token
            #     else:
            #         continue
            #     # val = int(token) if token.isdigit() else None
            # if val is not None:
            #     yield str(val)
        # raise SystemExit(f"simulated data file {config.test} not found.")


def record(path: str, data: Iterator[DataEntry], seconds: int) -> None:
    with open(path, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([TIME_LABEL, MIC_VALUE_LABEL, QUANTITY_LABEL])
        print(f"recording for {seconds} seconds")
        for entry in data:
            writer.writerow([ entry.time, entry.microphone, entry.quantity])
        success(f"finished recording to {path}")


def record_serial(path: str, com_port: str, seconds: int) -> None:
    with initiate_serial_connection(com_port) as serial_connection:
        record(path, open_serial_data(serial_connection), seconds)


# def detect_claps():


def main():
    args: Args = parse_command_line()
    match args.command:
        case command if isinstance(command,Run):
            # open source Iterator[DataEntry]
            # pass to training function
            # finish
            pass
        case command if isinstance(command,Record):
            # match args.source:
            # open source Iterator[DataEntry]
            # pass to recorder function
            # finish
    # interpretor = initialize_model(args.model)



if __name__ == "__main__":
    print(argv)
    main()
