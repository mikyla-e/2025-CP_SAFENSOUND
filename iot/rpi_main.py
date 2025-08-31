import time
import socket
import numpy as np
import librosa
import sounddevice as sd
import RPi.GPIO as GPIO
import joblib

# Load the pre-trained model
model = joblib.load("model.joblib")

def listen_and_classify():
    # # Record audio
    # duration = 5  # seconds
    # fs = 16000  # sample rate
    # print("[INFO] Listening...")
    # audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    # sd.wait()  # Wait until recording is finished
    # audio = audio.flatten()

    # # Preprocess the audio
    # mfccs = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=13)
    # mfccs = np.mean(mfccs.T, axis=0)

    # # Classify the audio
    # prediction = model.predict([mfccs])
    # print(f"[INFO] Prediction: {prediction}")
    # if prediction == "target_sound":
    #     trigger_action()


def trigger_action():
    print("[ACTION] Target sound detected! Triggering action...")
    # Example action: Blink an LED connected to GPIO pin 18


try:
    while True:
        listen_and_classify()
        time.sleep(1)  # Small delay before next recording
except KeyboardInterrupt:
    print("\n[INFO] Exiting gracefully...")
    GPIO.cleanup()