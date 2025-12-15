import argparse
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import colored
import numpy as np
import pandas
import scipy
from numpy.typing import ArrayLike, NDArray
from pandas.api.extensions import ExtensionArray

# from numpy.typing import ExtensionArray

# allows type checker to see tf while still lazy loading per module
if TYPE_CHECKING:
    import tensorflow as tf

# inputs
# training_data_directory = "model_input_data"
# defaults
WINDOW_SIZE = 256
FEATURE_COUNT: int = 20
REPO_URL = "https://github.com/antiah-arch/EchoSafe"
COM_PORT: str = "COM3"


def extract_features(data: pandas.DataFrame) -> np.ndarray:
    signal: ExtensionArray | NDArray = data["mic_value"].values
    window_size = 256
    features: list = []  # x
    labels: list[str] = []  # y
    # loops through a range with a step of window_size (256) 0,256,512.. and create a new slice of strings
    for i in range(0, len(signal) - window_size, window_size):
        frame: list[str] = signal[i : i + window_size]
        fft_vals = np.abs(scipy.fft.rfft(frame))
        features.append(np.mean(fft_vals))
    features_slice = features[:FEATURE_COUNT]
    if len(features_slice) < FEATURE_COUNT:
        features_slice += [0] * (
            FEATURE_COUNT - len(features_slice)
        )  # this can probably be rewritten its gross
    return np.array(features_slice)


def load_data():
    features = []
    labels = []
    data: list[str] = os.listdir("model_input_data")
    for data_file in data:
        # assert os.path.exists(data_file)
        csv_data: pandas.DataFrame = pandas.read_csv(data_file)
        # label = int(csv_data["label"].iloc[0])
        features.append(extract_features(csv_data))
        labels.append(int(csv_data["label"].iloc[0]))


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


def load_command_line(args: list[str]) -> Config:
    parser = argparse.ArgumentParser(
        prog="echosafe",
        description="Arduino to TensorFlowLite Interface bridge",  # whatever that means
        epilog=f"for more information see {REPO_URL}",
    )
    parser.add_argument("-c", "--com", default=COM_PORT)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-m", "--model", default="./sound_model.tflite", help="Path to TFLite model"
    )
    parser.add_argument("-s", "--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("-f", "--feature-count", type=int, default=FEATURE_COUNT)
    parser.add_argument(
        "-t",
        "--test",
        help="run in test mode using a simulated data stream CSV file.",
    )
    parsed = parser.parse_args(args=args)
    return Config(
        window_size=parsed.window_size,
        feature_count=parsed.feature_count,
        com_port=parsed.com,
        verbose=parsed.verbose,
        model=parsed.model,
        test=parsed.test,
    )


def error(msg: str) -> None:
    print(colored.stylize(msg, colored.fore("red")))


def warning(msg: str) -> None:
    print(colored.stylize(msg, colored.fore("yellow")))


def success(msg: str) -> None:
    print(colored.stylize(msg, colored.fore("green")))


def subtext(msg: str) -> None:
    print(colored.stylize(msg, colored.fore("dark_gray")))


# /load_command_line
# returns an iterator over the serial connection
def open_serial(config: Config) -> Iterator[str]:
    # setup arduino connection
    import serial

    DATA_COMM_SPEED = 9600  # speed of communication over connection in baud
    serial_connection = serial.Serial(
        config.com_port, DATA_COMM_SPEED, timeout=1
    )  # what is 9600
    print(f"connected to device on port {config.com_port}")
    return iter(
        lambda: serial_connection.readline().decode(errors="ignore").strip(), ""
    )


ERROR_RETURN = 2


def open_simulated_data(config: Config) -> Iterator[str]:
    assert isinstance(config.test, str)
    if not os.path.exists(config.test):
        error(f"test data file {config.test} not found.")
        exit(ERROR_RETURN)

        # raise SystemExit(f"simulated data file {config.test} not found.")


def main(args: list[str]):
    config: Config = load_command_line(args)
    interpreter = initialize_model(config.model)
    if isinstance(config.test, str):
        pass
    else:
        pass
