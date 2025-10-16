# --- general ---
import sys
import os
import time
from time import sleep
from datetime import datetime
import json
import struct
import wave
import traceback

# --- networking ---
import socket
import paho.mqtt.client as mqtt
import threading
import requests
import serial
import aiohttp
import asyncio

# --- audio processing ---
import joblib

import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, models
# from tensorflow.keras.utils import to_categorical

import librosa as lb
import librosa.display as ld
import sounddevice as sd
import soundfile as sf


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

loudness_threshold = 6000 
loud_threshold_ms = 1000 #1 second
loud_duration_ms = 0

alarming_count = 0
emergency_count = 0
nonemergency_count = 0
emergency_detected = False

# esp32_receiver_ip = None
esp32_serial = None
esp32_port = "COM9"

stop_event = threading.Event()

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
        sock.bind(('0.0.0.0', 8080))

        print(f"Discovery server listening on {self.laptop_ip}:8080")

        while self.running and not stop_event.is_set():
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
        print(f"Discovery server started on {self.laptop_ip}:8080.")

        return discovery_thread
    
    def stop(self):
        self.running = False
        print("Discovery listener stopped.")

def init_esp32_serial():
    global esp32_serial

    try:
        esp32_serial = serial.Serial(esp32_port, 115200, timeout=1)
        time.sleep(2)
        print(f"Connected to ESP32 on {esp32_port}")

        ready = False
        for _ in range(11):
            line = esp32_serial.readline().decode('utf-8').strip()
            if "Receiver ready!" in line:
                print(f"Received from ESP32: {line}")
                ready = True
                break
            time.sleep(0.5)
        if not ready:
            print("ESP32 did not send 'Receiver ready!' message.")
            return False
        
        return True

    except Exception as e:
        print(f"Failed to connect to ESP32 on {esp32_port}: {e}")
        esp32_serial = None
        return False


# audio recording and receiving --------------------
def get_audio_local():
    # get audio from microphone
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        print("/" * 60 + "\n")
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


def receive_audio_data():
    from database.db_connection import Database
    db = Database()
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 8081))
    sock.settimeout(1.0)

    audio_chunks = {}
    chunk_timestamps = {}
    EXPECTED_TOTAL_SAMPLES = 80000

    last_packet_time = time.time()

    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(65536)

            if len(data) < 12:
                print(f"Received packet too small from {addr}: {len(data)} bytes")
                continue

            room_id = int.from_bytes(data[0:4], 'little')
            timestamp = int.from_bytes(data[4:8], 'little')
            chunk_samples = int.from_bytes(data[8:12], 'little')

            audio_chunk = np.frombuffer(data[12:], dtype=np.int16)
            print(f"Room {room_id}: Received {len(audio_chunk)} samples (Expected: {chunk_samples})")

            if room_id not in audio_chunks:
                audio_chunks[room_id] = []
                chunk_timestamps[room_id] = time.time()

            audio_chunks[room_id].append(audio_chunk)
            total_samples = sum(len(chunk) for chunk in audio_chunks[room_id])

            print(f"Room {room_id}: Total samples received so far: {total_samples}/{EXPECTED_TOTAL_SAMPLES}")

            if total_samples >= EXPECTED_TOTAL_SAMPLES:
                print(f"✅ Received complete audio data: {total_samples} samples from Room {room_id}")
                full_audio = np.concatenate(audio_chunks[room_id])[:EXPECTED_TOTAL_SAMPLES]
                
                # Save the audio as a WAV file
                datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                wav_filename = f"recorded_audio/Room{room_id}_{datetime_str}.wav"
                save_wav(wav_filename, full_audio, sample_rate)

                # Process the audio
                thread = threading.Thread(
                    target=process_audio,
                    args=(full_audio, room_id, timestamp)
                )
                thread.start()

                # Reset buffer for the next transmission
                del audio_chunks[rid]
                del chunk_timestamps[rid]

            last_packet_time = time.time()
            for rid in list(chunk_timestamps.keys()):
                if last_packet_time - chunk_timestamps[rid] > 40:
                    print(f"⏱️ Timeout: Discarding incomplete recording from room {rid}")
                    del audio_chunks[rid]
                    del chunk_timestamps[rid]
                    

            # print("Audio data receiver is running...")

            # data, addr = sock.recvfrom(4096)
            # if not data:
            #     continue
            # try:
            #     audio_data = json.loads(data.decode('utf-8'))
            # except json.JSONDecodeError as e:
            #     print(f"Invalid JSON received from {addr}: {data[:100]}... Error: {e}")
            #     continue

            # room_id = audio_data.get('roomID')
            # timestamp = audio_data.get('timestamp')
            # audio_samples = audio_data.get('audioData')

            # if not room_id or not timestamp or not audio_samples:
            #     print(f"Incomplete data received from {addr}: {audio_data}")
            #     continue
            # if len(audio_samples) == 0:
            #     print(f"No audio samples received from {addr}")
            #     continue

            # print(f"Received audio data {len(audio_samples)} from {addr}: Room {room_id}")

            # thread = threading.Thread(
            #     target=process_audio,
            #     args=(audio_samples, room_id, timestamp)
            # )
            # thread.start()
        except socket.timeout:
            continue
        except Exception as e:
            print(f"Audio Data - UDP error: {e}")
            traceback.print_exc()

    sock.close()
    print("Audio data receiver stopped.")

def receive_reset_signals():
    from database.db_connection import Database
    db = Database()
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 8082))
    sock.settimeout(1.0)
    print("Reset signal receiver is running...")

    while not stop_event.is_set():
        
        try:
            data, addr = sock.recvfrom(1024)
            print(f"Raw reset packet from {addr}: {data}")
            try:
                reset_data = json.loads(data.decode('utf-8'))
            except json.JSONDecodeError:
                print(f"Invalid JSON received from {addr}: {data}")
                continue

            room_id = reset_data.get('roomID')
            action = reset_data.get('action')
            print(f"Received reset data from {addr}: Room {room_id}, Action: {action}")
            if not room_id or not action:
                print(f"Incomplete reset data received from {addr}: {reset_data}")
                continue

            current_time = datetime.now().strftime("%H:%M %p")
            current_date = datetime.now().strftime("%m/%d/%y")

            if action == "reset":
                operation = "Alert Acknowledged"

                print(f"Received audio data from {addr}: Room {room_id}")
                db.insert_history(action=operation, date=current_date, time=current_time, room_id=room_id)
                
                retry = 0
                try:
                    while retry < 3:
                        success_web, success_esp32 = asyncio.run(send_reset(room_id, operation))
                        if success_web and success_esp32:
                            print(f"Sent reset command from {room_id}")
                            break
                        else:
                            print("Failed to send reset command. Retrying...")
                            sleep(2)
                            retry += 1
                            success_web, success_esp32 = asyncio.run(send_reset(room_id, operation))
                            if retry == 3 and not success_web and not success_esp32:
                                print("Failed to send reset command after 3 attempts.")
                except Exception as e:
                    print(f"Error sending reset command: {e}")
        except socket.timeout:
            continue
        except Exception as e:
            print(f"Reset Signal - UDP error: {e}")
            
    sock.close()
    print("Reset signal receiver stopped.")

def save_wav(filename, audio_data, sample_rate):
    """Save int16 audio data as WAV file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data, dtype=np.int16)

    try:
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes = 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        print(f"✅ Audio saved: {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to save WAV: {e}")
        return False
    

# audio processing and inference --------------------
def process_audio(audio_data_int16, room_id=None, timestamp=None):
    global loud_duration_ms
    
    num_samples = len(audio_data_int16)
    duration_ms = (num_samples / sample_rate) * 1000

    try:
        audio_data = np.array(audio_data_int16, dtype=np.int16)
        avg_amplitude = np.mean(np.abs(audio_data))
        
        print(f"Loudness: {avg_amplitude}, Duration: {duration_ms}ms\n")

        if avg_amplitude > loudness_threshold:
            loud_duration_ms += duration_ms
        else:
            loud_duration_ms = 0
            
        if loud_duration_ms >= loud_threshold_ms:
            loud_duration_ms = 0
            inference(audio_data, f"Room{room_id}_{timestamp}", room_id)
            
        return True
        
    except Exception as e:
        print(f"Error processing audio: {e}")

        return False

def extract_features(audio, sample_rate, max_len=160):
    mfcc = lb.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20, hop_length=512)
    
    zcr = lb.feature.zero_crossing_rate(audio)
    spectral_centroid = lb.feature.spectral_centroid(y=audio, sr=sample_rate)
    spectral_rolloff = lb.feature.spectral_rolloff(y=audio, sr=sample_rate)
    rms = lb.feature.rms(y=audio)
    
    features = [mfcc, spectral_centroid, spectral_rolloff, zcr, rms]
    extracted_features = []

    for feature in features:
        if feature.shape[1] < max_len:
            feature = np.pad(feature, ((0,0),(0, max_len - feature.shape[1])), mode='constant')
        else:
            feature = feature[:, :max_len]
        
        feature_stat = [
            np.mean(feature, axis=1),
            np.std(feature, axis=1),
            np.min(feature, axis=1),
            np.max(feature, axis=1)
        ]
        
        for stat in feature_stat:
            extracted_features.append(stat.flatten())

    return np.concatenate(extracted_features)

def inference(audio, wav_name, room_id=None):
    global emergency_count, alarming_count, nonemergency_count, emergency_detected

    # extracting
    audio_features = extract_features(audio, sample_rate)

    # predicting
    prediction = model.predict(audio_features.reshape(1, -1))


    # alarm and emergency logic
    predicted_class = prediction[0]

    # alarming sound = 4 times before emergency is confirmed
    # emergency sound = 2 times after emergency is confirmed

    print(f"\nPrediction for {wav_name}: ")

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
    global emergency_detected, emergency_count, alarming_count, nonemergency_count, success_web, success_esp32

    alarming_count = 0
    emergency_count = 0
    nonemergency_count = 0

    if (emergency_detected == True):
        print(f"\nALARM TRIGGERED")
        action = "Emergency Detected"
        current_time = datetime.now().strftime("%H:%M %p")
        current_date = datetime.now().strftime("%m/%d/%y")

        db.insert_history(action=action, date=current_date, time=current_time, room_id=room_id)
        try:
            retry = 0
            while retry < 3:
                success_web = asyncio.run(send_alert_web(room_id, action))
                success_esp32 = asyncio.run(send_alert_esp(room_id, action))
                if success_esp32 and success_web:
                    print(f"Sent emergency alert from {room_id}")
                    break
                else:
                    print("Failed to send alert. Retrying...")
                    sleep(2)

                    retry += 1
                    success_web = asyncio.run(send_alert_web(room_id, action))
                    success_esp32 = asyncio.run(send_alert_esp(room_id, action))
                    if retry == 3 and not success_esp32 and not success_web:
                        print("Failed to send alert after 3 attempts.")

        except Exception as e:
            print(f"Error sending alert: {e}")

async def send_alert_web(room_id, action=None):
    success_web = False

    if "Emergency Detected" in action:
        # web
        try:
            payload_web = {
                "room_id": room_id,
                "action": action
            }

            async with aiohttp.ClientSession() as session:
                async with session.post("http://localhost:8000/api/alert", json=payload_web, timeout=10) as response:
                    if response.status == 200:
                        print(f"Alert sent to Web Dashboard for Room {room_id}")
                        success_web = True

        except Exception as e:
            print("Failed to send alert to Web Dashboard:", e)
            traceback.print_exc()

    return success_web

async def send_alert_esp(room_id, action=None):
    success_esp32 = False

    if "Emergency Detected" in action:
        # esp32
        try:
            if esp32_serial and esp32_serial.is_open:
                command = f"ALERT: {room_id}\n"
                print(command)
                esp32_serial.write(command.encode())

                responses = []
                responses.append(esp32_serial.readline().decode().strip())
                if responses:
                    for response in responses:
                        print(f"Received from ESP32: {response}")

                success_esp32 = True
            else:
                print("Serial port not open.")

        except Exception as e:
            print("Failed to send alert to ESP32 receiver:", e)

    return success_esp32


async def send_reset(room_id, action=None):
    success_web = False
    success_esp32 = False

    if "Alert Acknowledged" in action:
        try:
            if esp32_serial and esp32_serial.is_open:
                command = f"RESET: {room_id}\n"
                esp32_serial.write(command.encode())
                
                response = esp32_serial.readline().decode().strip()
                if response:
                    print(f"Received from ESP32: {response}")

                success_esp32 = True
            else:
                print("Serial port not open.")
        except Exception as e:
            print("Failed to send reset command to ESP32 receiver:", e)

        try:
            payload_web = {
                "room_id": room_id,
                "action": action
            }

            async with aiohttp.ClientSession() as session:
                async with session.post("http://localhost:8000/api/alert", json=payload_web, timeout=10) as response:
                    if response.status == 200:
                        print(f"Reset command sent to Web Dashboard for Room {room_id}")
                        success_web = True

        except Exception as e:
            print("Failed to send reset command to Web Dashboard:", e)
    
    return success_web, success_esp32
    

# main loop -----------------------------------------

async def main_loop():
    while not stop_event.is_set():
        await asyncio.sleep(1)

if __name__ == "__main__":
    if not init_esp32_serial():
        print("Exiting due to Receiver connection failure.")
        exit(1)

    print("Receiver connected successfully.")

    discovery_server = LaptopDiscoverServer()
    discovery_server.start()
    
    audio_thread = threading.Thread(target=receive_audio_data, daemon=True)
    audio_thread.start()

    reset_thread = threading.Thread(target=receive_reset_signals, daemon=True)
    reset_thread.start()
    
    print("=" * 60)
    print(f"LAPTOP IP: {discovery_server.laptop_ip}")
    print(f"Discovery server: Port 8080")
    print(f"Audio receiver: Port 8081")
    print(f"Reset signal: Port 8082")
    print("=" * 60)

    try:
        trigger = 0
        while True:
            # Main loop for laptop recording
            # audio_data, audio_wav = get_audio_local()
            # if audio_data is not None and audio_wav is not None:
            #     inference(audio_data, audio_wav)

            # Main loop for dataset
            while trigger < 4:
                audio_file_path = "ml/datasets/alarming/doorsmash_01.wav"
                audio_wav = "wav name"
                room_no = 1

                audio_data, _ = lb.load(audio_file_path, sr=sample_rate)
                inference(audio_data, audio_wav, room_no)

                time.sleep(1)
                print(f"Next audio... {trigger}")

                trigger += 1

            else:
                audio_file_path = "ml/datasets/non-emergency/bg-11.wav"

                audio_data, _ = lb.load(audio_file_path, sr=sample_rate)
                inference(audio_data, audio_wav, room_no)
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
        stop_event.set()
        discovery_server.stop()
        audio_thread.join()
        reset_thread.join()
        if esp32_serial and esp32_serial.is_open:
            esp32_serial.close()
        print("\nPorts closed successfully.")

