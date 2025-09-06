# --- general ---
import os
import time
from time import sleep
import datetime

# --- networking ---
import socket
import paho.mqtt.client as mqtt     
# import bluetooth

# --- audio processing ---
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

import librosa as lb
import librosa.display as ld
import sounddevice as sd
import soundfile as sf

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- hardware control ---
import RPi.GPIO as GPIO

# GPIO setup ---------------------------------------
# GPIO.setmode(GPIO.BOARD)
# GPIO.setmode(GPIO.BCM)

# light_pin_1 =
# light_pin_2 = 
# light_pin_3 =

# buzzer_pin =

# reset_pin =

# GPIO.setup([light_pin_1, light_pin_2, light_pin_3, buzzer_pin], GPIO.OUT)
# GPIO.setup(reset_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)


# functions -----------------------------------------

model = joblib.load("mfcc_rf_model.joblib")

audio_data = None
audio_wav = None
audio_duration = 5 #seconds
sample_rate = 16000

emergency_count = 0

def connection():
    
    return


def get_audio():
    # get audio from microphone
    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        print(f"Recording audio... at {date_time}")
        audio = sd.rec(int(audio_duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()

        audio_wav = f"recording_{date_time}.wav"
        sf.write(audio_wav, audio, sample_rate)
        print(f"Audio recorded and saved as {audio_wav}")

    except Exception as e:
        print("Audio recording failed:", e)
        return None, None


    return audio.flatten(), audio_wav


# def extract_features():
#     # extract features from audio
#     return features
def extract_mfcc(audio, sample_rate, n_mfcc=40, hop_length=512, max_len=160):
    mfcc = lb.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length)
    
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0,0),(0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    
    flat_mfcc = mfcc.flatten()
    
    return flat_mfcc


def inference(audio):
    emergency_count = 0
    emergency_detected = False

    # extracting
    audio_features = extract_mfcc(audio, sample_rate)
    audio_reshaped = audio_features.reshape(1, -1)

    # predicting
    prediction = model.predict_proba(audio_reshaped)
    threshold = 0.3
    prediction = (prediction[:, 1] >= threshold).astype(int)

    #add alarm and emergency logic

    print(f"Prediction for {audio_wav}: {'emergency' if prediction[0] == 1 else 'non-emergency'}")
    
    # alarming sound = 3 times before emergency is confirmed
    # emergency sound = 2 times after emergency is confirmed


    # if prediction[0] == 1:
    #     emergency_detected = True
    # else:
    #     emergency_detected = False
    
    # if emergency_detected:
    #     emergency_count += 1
    #     trigger_alarm()
    # else:
    #     emergency_count = 0

    return

def trigger_alarm():
    # trigger alarm if emergency was detected
    if emergency_count > 0:
        print(f"Emergency detected {emergency_count} times.")

    return


# main loop -----------------------------------------
try:
    while True:
        # audio_data, audio_wav = get_audio()
        audio_data = "../ml/datasets/alarming.wav"
        if audio_data is not None:
            inference(audio_data)
        time.sleep(1)
except KeyboardInterrupt:
    print("\n[INFO] Exiting gracefully...")
    GPIO.cleanup()