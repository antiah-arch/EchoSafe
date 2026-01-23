import pandas as pd
import numpy as np
from numpy.fft import rfft
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


FEATURE_COUNT = 16
WINDOW_SIZE = 64
MODEL_OUT = "clap_model.pkl"


def extract_features(signal: np.ndarray) -> np.ndarray:
    fft_vals = np.abs(rfft(signal))
    features = np.array(
        [np.mean(fft_vals[i::FEATURE_COUNT]) for i in range(FEATURE_COUNT)]
    )
    return features


clap = pd.read_csv("sound_data_label1.csv")
noise = pd.read_csv("sound_data_label0.csv")
data = pd.concat([clap, noise]).reset_index(drop=True)


X, y = [], []

for label in [0, 1]:
    subset = data[data["label"] == label]["mic_value"].values

    for i in range(len(subset) - WINDOW_SIZE):
        window = subset[i : i + WINDOW_SIZE]
        features = extract_features(window)
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(f"✅ Model accuracy: {accuracy * 100:.2f}%")


joblib.dump(model, MODEL_OUT)
print(f"💾 Saved model to {MODEL_OUT}")
