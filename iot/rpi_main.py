# --- general ---
import sys
import os
import time
from time import sleep
import datetime
import json

# --- networking ---
import socket
import paho.mqtt.client as mqtt
import threading
import requests
import serial
import aiohttp

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

# esp32_receiver_ip = None
esp32_serial = None
esp32_port = "COM5"

class LaptopDiscoverServer:
    def __init__(self):
        self.running = True
        self.laptop_ip = self.get_laptop_ip()

    def get_laptop_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception as e:
            print("Error getting laptop IP:", e)
            return "localhost"
        
    def discovery_listener(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', 8081))

        print(f"🔍 Discovery server listening on {self.laptop_ip}:8081")

        while self.running:
            try:
                data, addr = sock.recvfrom(1024)
                message = data.decode('utf-8').strip()
                print(f"Received discovery message from {addr[0]}: {message}")

                if message == "DISCOVER_MAIN_DEVICE":
                    response = f"MAIN_DEVICE_HERE:{self.laptop_ip}"
                    sock.sendto(response.encode('utf-8'), addr)
                    print(f"Sent response to {addr[0]}: {self.laptop_ip}")
            
            except Exception as e:
                if self.running:
                    print(f"Error in discovery listener: {e}")


        sock.close()

    def start(self):
        discovery_thread = threading.Thread(target=self.discovery_listener, daemon=True)
        discovery_thread.start()
        print(f"Discovery server started on {self.laptop_ip}:8081.")

        return discovery_thread
    
    def stop(self):
        self.running = False

# audio recording and receiving --------------------
def get_audio_local():
    # get audio from microphone
    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        print("\n//////////////////////////////////")
        print(f"Recording audio... at {date_time}")
        audio = sd.rec(int(audio_duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()

        audio_folder = "recorded_audio"
        if not os.path.exists(audio_folder):
            os.makedirs(audio_folder)
        
        audio_wav = os.path.join(audio_folder, f"recording_{date_time}.wav")
        sf.write(audio_wav, audio, sample_rate)
        print(f"Audio recorded and saved as {audio_wav}")

    except Exception as e:
        print("Audio recording failed:", e)
        return None, None

    return audio.flatten(), audio_wav

def receive_audio_esp32():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 8080))

    while True:
        try:
            data, addr = sock.recvfrom(4096)

            audio_data = json.loads(data.decode('utf-8'))

            room_id = audio_data.get('roomID')
            timestamp = audio_data.get('timestamp')
            audio_samples = audio_data.get('audioData')

            print(f"Received audio data from {addr}: Room {room_id}")

            thread = threading.Thread(
                target=process_audio,
                args=(audio_samples, room_id, timestamp)
            )
            thread.start()

        except Exception as e:
            print(f"UDP error: {e}")

def init_esp32_serial():
    global esp32_serial
    try:
        esp32_serial = serial.Serial(esp32_port, 115200, timeout=1)
        time.sleep(2)
        print(f"Connected to ESP32 on {esp32_port}")

        esp32_serial.write(b'STATUS\n')
        response = esp32_serial.readline().decode('utf-8').strip()
        print(f"Received from ESP32: {response}")
        return True

    except Exception as e:
        print(f"Failed to connect to ESP32 on {esp32_port}: {e}")
        esp32_serial = None


# audio processing and inference --------------------
def process_audio(audio_data_int16, room_id=None, timestamp=None):
    try:
        audio_data = np.array(audio_data_int16, dtype=np.float32) / 32768.0

        inference(audio_data, f"Room{room_id}_{timestamp}", room_id)
        return True
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return False


def extract_mfcc(audio, sample_rate, n_mfcc=40, hop_length=512, max_len=160):
    mfcc = lb.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length)
    
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0,0),(0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    
    flat_mfcc = mfcc.flatten()
    
    return flat_mfcc

def inference(audio, wav_name, room_id=None):
    global emergency_count, alarming_count, nonemergency_count, emergency_detected

    # extracting
    audio_features = extract_mfcc(audio, sample_rate)

    # predicting
    prediction = model.predict(audio_features.reshape(1, -1))

    # prediction with threshold
    # prediction = model.predict_proba(audio_reshaped)
    # threshold = 0.3
    # prediction = (prediction[:, 1] >= threshold).astype(int)


    # alarm and emergency logic
    predicted_class = prediction[0]

    print(f"\nPrediction for {wav_name}: {'emergency' if predicted_class == 2 else 'alarming' if predicted_class == 1 else 'non-emergency'}")

    # alarming sound = 4 times before emergency is confirmed
    # emergency sound = 2 times after emergency is confirmed
    
    if predicted_class == 1:
        alarming_count += 1
        print("ALARMING sound detected. \nAlarm count:", alarming_count, "\nEmergency count:", emergency_count)

        if alarming_count >= 4:
            emergency_detected = True
            trigger_alarm(room_id)
        
        elif alarming_count == 2 and emergency_count >= 1:
            emergency_detected = True
            trigger_alarm(room_id)

    elif predicted_class == 2:
        emergency_count += 1
        print("EMERGENCY sound detected. \nAlarm count:", alarming_count, "\nEmergency count:", emergency_count)
        if emergency_count >= 2:
            emergency_detected = True
            trigger_alarm(room_id)

        elif emergency_count == 1 and alarming_count >= 2:
            emergency_detected = True
            trigger_alarm(room_id)

    elif predicted_class == 0:
        print("No emergency detected.")
        nonemergency_count += 1
        if nonemergency_count >= 6:
            alarming_count = 0
            emergency_count = 0
            nonemergency_count = 0
            print("System reset due to consecutive non-emergency sounds.")


# alarm triggering -----------------------------------
def trigger_alarm(room_id=None):
    # trigger alarm if emergency was detected
    global emergency_detected, emergency_count, alarming_count, nonemergency_count

    nonemergency_count = 0

    if (emergency_detected == True):
        print(f"\nALARM TRIGGERED")

        send_alert(room_id, action="Emergency")

    emergency_count = 0
    alarming_count = 0
    emergency_detected = False

async def send_alert(room_id, action=None):
    success_web = False
    success_esp32 = False

    # esp32
    try:
        if esp32_serial and esp32_serial.is_open:
            if "Emergency" in action:
                command = f"ALERT: {room_id}\n"
            esp32_serial.write(command.encode())
            
            response = esp32_serial.readline().decode().strip()
            if response:
                print(f"Received from ESP32: {response}")
            
            success_esp32 = True
        else:
            print("Serial port not open.")

    except Exception as e:
        print("Failed to communicate with ESP32:", e)

    # web
    try:
        payload_web = {
            "room_id": room_id,
            "action": action
        }

        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:5000/api/alert", json=payload_web, timeout=2) as response:
                if response.status == 200:
                    print(f"Alert sent to Web Dashboard for Room {room_id}")
                    success_web = True
                else:
                    print(f"Failed to send alert to Web Dashboard for Room {room_id}")

    except Exception as e:
        print("Failed to send alert to Web Dashboard:", e)

    return success_web, success_esp32
    

# main loop -----------------------------------------

if __name__ == "__main__":
    try:
        if not init_esp32_serial():
            print("Exiting due to Receiver connection failure.")
            exit(1)

        discovery_server = LaptopDiscoverServer()
        discovery_server.start()
        
        audio_thread = threading.Thread(target=receive_audio_esp32, daemon=True)
        audio_thread.start()
        
        print("=" * 60)
        print(f"💻 LAPTOP IP: {discovery_server.laptop_ip}")
        print(f"🔍 Discovery server: Port 8081")
        print(f"🎵 Audio receiver: Port 8080")
        print("=" * 60)
        
        # Main loop for local recording
        while True:
            audio_data, audio_wav = get_audio_local()
            if audio_data is not None and audio_wav is not None:
                inference(audio_data, audio_wav)
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[INFO] Exiting gracefully...")
        if esp32_serial and esp32_serial.is_open:
            esp32_serial.close()
        discovery_server.stop()
