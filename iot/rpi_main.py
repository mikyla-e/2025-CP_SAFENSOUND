# --- general ---
import os
import time
import datetime

# --- networking ---
import socket
import paho.mqtt.client as mqtt

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
# Load the pre-trained model
# model = joblib.load("model.joblib")


# functions -----------------------------------------
def listen_and_classify():
    # Record audio

def extract_features():
    # extract features from audio

def trigger_alarm():
    # trigger alarm if emergency was detected



try:
    while True:
        listen_and_classify()
        time.sleep(1)  # Small delay before next recording
except KeyboardInterrupt:
    print("\n[INFO] Exiting gracefully...")
    GPIO.cleanup()