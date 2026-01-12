from collections import deque
from collections.abc import Iterator
from time import time
from typing import TYPE_CHECKING

import numpy as np

from numpy.fft import rfft
from numpy.typing import NDArray
from source import DataEntry
from utils import subtext

if TYPE_CHECKING:
    import tensorflow as tf


def initialize_model(model: str) -> "tf.lite.Interpreter":
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()
    return interpreter


def extract_features(signal: list[int], feature_count: int) -> NDArray:
    fft_vals: NDArray = np.abs(rfft(signal))
    features = np.array(
        [np.mean(fft_vals[i::feature_count]) for i in range(feature_count)]
    )
    return np.array([features], dtype=np.float32)


COOLDOWN: int = 10


def train(
    interpreter: "tf.lite.Interpreter",
    output: SupportsWrite,
    window_size: int,
    feature_count: int,
    data_stream: Iterator[DataEntry],
):
    buffer: deque[int] = deque(maxlen=window_size)
    last_trigger: int = 0
    for entry in data_stream:
        if (
            len(buffer) >= window_size
        ):  # if buffer length is greater than or equal to window size,
            # but it can only ever be equal to...

            # all of this being each iteration feels wrong.
            window: list[int] = list(buffer)[
                -window_size:
            ]  # cast to list might degrade benefits of deque
            features = extract_features(window, feature_count)
            index = interpreter.get_input_details()[0][
                "index"
            ]  # magic weird data access

            interpreter.set_tensor(index, features)
            interpreter.invoke()
            model_output = interpreter.get_tensor(
                interpreter.get_output_details()[0]["index"]
            )[0]
            prediction = int(np.argmax(model_output))  # what does this do
            # clap detection

            if prediction == 1 and (time() - last_trigger) > COOLDOWN:
                subtext("clap detected")
                output.write(b"1")  # send signal down to output line
            else:
                output.write(b"0")  # seems redundant?
                # couldnt we just send a blip when something happens?
