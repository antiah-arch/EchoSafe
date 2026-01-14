import atexit
import time
from collections import deque
import numpy as np
import tensorflow as tf
from numpy.fft import rfft

from config import COM_PORT, BAUD_RATE, WINDOW_SIZE, NUM_FEATURES, COOLDOWN
from serial_helper import open_serial, close_serial

def extract_features(signal, num_features=NUM_FEATURES):
    fft_vals = np.abs(rfft(signal))
    features = np.array([np.mean(fft_vals[i::num_features]) for i in range(num_features)])
    return np.array([features], dtype=np.float32)

def load_interpreter(model_path):
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()

def main():
    interp, input_details, output_details = load_interpreter("sound_model.tflite")
    buffer = deque(maxlen=WINDOW_SIZE)

    # Safe COM access
    ser = open_serial(COM_PORT, BAUD_RATE)
    import atexit; atexit.register(close_serial, ser)

    last_trigger = 0
    print(f" Listening on {COM_PORT}...")

    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line.isdigit():
                continue
            mic_value = int(line)
            buffer.append(mic_value)

            if len(buffer) >= WINDOW_SIZE:
                window = list(buffer)[-WINDOW_SIZE:]
                X = extract_features(window)
                interp.set_tensor(input_details[0]["index"], X)
                interp.invoke()
                output = interp.get_tensor(output_details[0]["index"])[0]
                prediction = int(np.argmax(output))

                if prediction == 1 and (time.time() - last_trigger) > COOLDOWN:
                    print(" CLAP detected!")
                    ser.write(b'1')
                    last_trigger = time.time()
                else:
                    ser.write(b'0')
    except KeyboardInterrupt:
        print("\n Exiting...")
        close_serial(ser)

if __name__ == "__main__":
    main()
