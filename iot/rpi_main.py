# --- general ---
import os
import time
from time import sleep
import datetime

# --- networking ---
import socket
import paho.mqtt.client as mqtt     
import bluetooth

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
from sklearn.model_selection import train_test_split

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

# load model vvv
# model = joblib.load("model.joblib")


# functions -----------------------------------------
ml_model_path = ""

audio_wav = None
audio_duration = 5 #seconds
sample_rate = 16000


def connection():
    



def get_audio():
    # get audio from microphone
    audio_wav = None
    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        print(f"Recording audio... at {date_time}")
        audio = sd.rec(int(audio_duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()

        audio_wav = f"recording_{date_time}.wav"
        lb.output.write_wav(audio_wav, audio, sample_rate)
        print(f"Audio recorded and saved as {audio_wav}")

    except Exception as e:
        print("Audio recording failed:", e)


    return audio_wav


def inference():

    emergency_count = 0
    emergency_detected = False

    # proceeding 2-4 audios after the first emergency audio detected must also be detected as emergency to trigger alarm (?)
    if emergency_detected:
        emergency_count += 1
        trigger_alarm()
    else:
        emergency_count = 0

def extract_features():
    # extract features from audio
    features =

    
    return features

def trigger_alarm():
    # trigger alarm if emergency was detected


    

try:
    while True:
        get_audio()
        time.sleep(1)
except KeyboardInterrupt:
    print("\n[INFO] Exiting gracefully...")
    GPIO.cleanup()