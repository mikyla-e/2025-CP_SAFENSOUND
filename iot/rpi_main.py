# --- general ---
import sys
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

# hardware control ---------------------------------
# import RPi.GPIO as GPIO

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

# database
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database.db_connection import Database

db = Database()
print("Database connected successfully.")

# ml model
model = joblib.load("ml/mfcc/mfcc_rf_model.joblib")
print("Model loaded successfully.")

audio_data = None
audio_wav = None
audio_duration = 5 #seconds
sample_rate = 16000

alarming_count = 0
emergency_count = 0
nonemergency_count = 0
emergency_detected = False

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


def inference(audio, wav_name):
    global emergency_count, alarming_count, nonemergency_count, emergency_detected

    # extracting
    audio_features = extract_mfcc(audio, sample_rate)

    # predicting
    prediction = model.predict(audio_features.reshape(1, -1))
    # prediction = model.predict_proba(audio_reshaped)
    # threshold = 0.3
    # prediction = (prediction[:, 1] >= threshold).astype(int)

    #add alarm and emergency logic

    predicted_class = prediction[0]

    print(f"\nPrediction for {wav_name}: {'emergency' if predicted_class == 2 else 'alarming' if predicted_class == 1 else 'non-emergency'}")

    if predicted_class == 1:
        alarming_count += 1
        print("ALARMING sound detected. \nAlarm count:", alarming_count, "\nEmergency count:", emergency_count)

        if alarming_count >= 4:
            emergency_detected = True
            trigger_alarm()
        
        elif alarming_count == 2 and emergency_count >= 1:
            emergency_detected = True
            trigger_alarm()

    elif predicted_class == 2:
        emergency_count += 1
        print("EMERGENCY sound detected. \nAlarm count:", alarming_count, "\nEmergency count:", emergency_count)
        if emergency_count >= 2:
            emergency_detected = True
            trigger_alarm()

        elif emergency_count == 1 and alarming_count >= 2:
            emergency_detected = True
            trigger_alarm()

    elif predicted_class == 0:
        print("No emergency detected.")
        nonemergency_count += 1
        if nonemergency_count >= 6:
            alarming_count = 0
            emergency_count = 0
            nonemergency_count = 0
            print("System reset due to consecutive non-emergency sounds.")

    
    # alarming sound = 4 times before emergency is confirmed
    # emergency sound = 2 times after emergency is confirmed

    return

def trigger_alarm():
    # trigger alarm if emergency was detected
    global emergency_detected, emergency_count, alarming_count, nonemergency_count
    buzzer_trigger = 0
    nonemergency_count = 0

    while emergency_detected == True:
        print(f"\nALARM TRIGGERED")
        time.sleep(3)
        buzzer_trigger += 1
        if buzzer_trigger == 3:
            emergency_detected = False
            print(f"\nReset alarm.")
    else:
        emergency_count = 0
        alarming_count = 0



# main loop -----------------------------------------

try:
    while True:
        # audio_data, audio_wav = get_audio()
        example_wav = "ml/datasets/alarming/babycry-3.wav"

        if example_wav is not None:
            # Load the audio file
            audio_data, sr = lb.load(example_wav, sr=sample_rate, mono=True)
            inference(audio_data, example_wav)
        
        time.sleep(5) 
            
        example_wav = "ml/datasets/emergency/gunshot_02.wav"

        if example_wav is not None:
            # Load the audio file
            audio_data, sr = lb.load(example_wav, sr=sample_rate, mono=True)
            inference(audio_data, example_wav)

        time.sleep(5) 

        example_wav = "ml/datasets/alarming/babycry-3.wav"

        if example_wav is not None:
            # Load the audio file
            audio_data, sr = lb.load(example_wav, sr=sample_rate, mono=True)
            inference(audio_data, example_wav)

        time.sleep(5)   

        example_wav = "ml/datasets/alarming/babycry-3.wav"

        if example_wav is not None:
            # Load the audio file
            audio_data, sr = lb.load(example_wav, sr=sample_rate, mono=True)
            inference(audio_data, example_wav)

        time.sleep(5) 

        example_wav = "ml/datasets/alarming/babycry-3.wav"

        if example_wav is not None:
            # Load the audio file
            audio_data, sr = lb.load(example_wav, sr=sample_rate, mono=True)
            inference(audio_data, example_wav)

        time.sleep(5)  
        
        example_wav = "ml/datasets/emergency/gunshot_02.wav"

        if example_wav is not None:
            # Load the audio file
            audio_data, sr = lb.load(example_wav, sr=sample_rate, mono=True)
            inference(audio_data, example_wav)

        time.sleep(5) 
except KeyboardInterrupt:
    print("\n[INFO] Exiting gracefully...")
    # GPIO.cleanup()