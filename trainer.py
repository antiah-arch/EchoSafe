import os
import pandas
import numpy as np
from numpy.typing import ExtensionArray
import scipy
# inputs
# training_data_directory = "model_input_data"
NUM_FEATURES : int = 20
def extract_features(data: pandas.DataFrame) -> np.ndarray:
    signal : ExtensionArray = data["mic_value"].values
    window_size = 256
    features : list = [] # x
    labels : list[str] = [] # y
    for i in range(0,len(signal) - window_size, window_size):
        frame: list[str] = signal[i : i + window_size]
        fft_vals = np.abs(scipy.rfft(frame))
        features.append(np.mean(fft_vals))
    features_slice = features[:NUM_FEATURES]
    if len(features_slice) < NUM_FEATURES:
        features_slice += [0] * (NUM_FEATURES - len(features_slice)) # this can probably be rewritten its gross
    return np.array(features_slice)

def load_data():
    features = [];
    data : list[str] = os.listdir("model_input_data")
    for data_file in data:
        # assert os.path.exists(data_file)
        csv_data : pandas.DataFrame = pandas.read_csv(data_file)
        # label = int(csv_data["label"].iloc[0])
        features.append(extract_features(csv_data))
        labels.append(int(csv_data["label"].iloc[0]))
