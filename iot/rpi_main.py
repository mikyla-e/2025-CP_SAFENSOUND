# --- general ---
import sys
import os
import time
from time import sleep
from datetime import datetime
import json
import wave
# import struct
import signal

# --- networking ---
import socket
# import paho.mqtt.client as mqtt
import threading
import requests
import serial
import aiohttp
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# --- audio processing ---
# import joblib
from tensorflow import keras

import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

import librosa as lb
# import sounddevice as sd
import soundfile as sf

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")


# hardware control ---------------------------------
from gpiozero import LED, Buzzer, Device
from gpiozero.pins.rpigpio import RPiGPIOFactory

try:
    # Close any existing pin factory
    if Device.pin_factory is not None:
        Device.pin_factory.close()
    Device.pin_factory = RPiGPIOFactory()
except Exception as e:
    print(f"Warning: Could not reset pin factory: {e}")

try:
    led_pin_1 = LED(17)
    led_pin_2 = LED(23)
    led_pin_3 = LED(25)

    buzzer_pin = Buzzer(16)
except Exception as e:
    print("Attempting cleanup and retry...")
    import subprocess
    subprocess.run(['sudo', 'killall', 'python3'], capture_output=True)
    import time
    time.sleep(1)
    led_pin_1 = LED(17)
    led_pin_2 = LED(23)
    led_pin_3 = LED(25)
    buzzer_pin = Buzzer(16)


# functions -----------------------------------------

# database
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database.db_connection import Database

db = Database()
print("Database connected successfully.")

# ml model
# model = joblib.load("ml/ml models/mfcc_rf_model.joblib") # mfcc + random forest

# model = keras.models.load_model("ml/ml models/lsms_cnn_model.keras") # lsms + cnn
# model = keras.models.load_model("ml/ml models/lsms_cnn_model_2.keras") # lsms + cnn
# model = keras.models.load_model("ml/ml models/emergency_classification/lsms_cnn_model_3.keras") # lsms + cnn
# noise_classifier = keras.models.load_model("ml/ml models/noise_classification/lsms_cnn_model.keras") # noise cnn


import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path="ml/ml models/emergency_classification/lsms_cnn_model_3.tflite")
interpreter.allocate_tensors()
print("Model loaded successfully.")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


noise_classifier = tflite.Interpreter(model_path="ml/ml models/noise_classification/lsms_cnn_model.tflite")
noise_classifier.allocate_tensors()
print("Model loaded successfully.")

noise_input_details = noise_classifier.get_input_details()
noise_output_details = noise_classifier.get_output_details()


audio_data = None
audio_wav = None
audio_duration = 5 #seconds
sample_rate = 16000

alarming_count = 0
emergency_count = 0
nonemergency_count = 0
emergency_detected = False
alarming_detected = False
sound_type = None
alerted_rpi = False

esp32_serial = None
esp32_port = "COM9"

sns_port = 47845
disc_port = 60123
audio_port = 54321
reset_port = 58080
shutdown_port = 52346


stop_event = threading.Event()

# def init_esp32_serial():
#     global esp32_serial
#     try:
#         esp32_serial = serial.Serial(esp32_port, 115200, timeout=1)
#         print(f"ESP32 serial connected on {esp32_port}")
#         return True
#     except Exception as e:
#         print(f"Could not open ESP32 serial port: {e}")
#         esp32_serial = None
#         return False

class RPIDiscoverServer:
    def __init__(self):
        self.running = True
        self.RPI_ip = self.get_RPI_ip()

    def get_RPI_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception as e:
            print("Error getting RPI IP:", e)
            return "localhost"
        
    def discovery_listener(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', disc_port))

        print(f"Discovery server listening on {self.RPI_ip}:{disc_port}")

        while self.running and not stop_event.is_set():
            try:
                data, addr = sock.recvfrom(1024)
                message = data.decode('utf-8').strip()
                print(f"Received discovery message from {addr[0]}: {message}")

                if message == "SENDER_HERE":
                    response = f"RPI_HERE:{self.RPI_ip}"

                    sock.sendto(response.encode('utf-8'), addr)
                    print(f"Sent response to {addr[0]}: {response}")
            
            except Exception as e:
                if self.running:
                    print(f"Error in discovery listener: {e}")

        sock.close()

    def start(self):
        discovery_thread = threading.Thread(target=self.discovery_listener, daemon=True)
        discovery_thread.start()
        print(f"RPI Discovery server started on {self.RPI_ip}:{disc_port}.")

        return discovery_thread
    
    def stop(self):
        self.running = False
        print("RPI Discovery listener stopped.")

class ReusableTCPServer(HTTPServer):
    allow_reuse_address = True

    def server_bind(self):
        HTTPServer.server_bind(self)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        

class ShutdownHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/shutdown":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                if data.get("confirm"):
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": True, "message": "Shutting down..."}).encode())
                    self.flush
                    
                    print("\n*** REMOTE SHUTDOWN REQUESTED ***")
                    
                    stop_event.set()
                    time.sleep(1)
                    cleanup() 

                    def _poweroff():
                        try:
                            time.sleep(2)
                            import subprocess
                            subprocess.run(['sudo', 'shutdown', '-h', 'now'], check=True)
                        except Exception as e:
                            print(f"Failed to shutdown system: {e}")

                    threading.Thread(target=_poweroff, daemon=True).start()
                else:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Confirmation required"}).encode())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass

def run_shutdown_server():
    try: 
        server = ReusableTCPServer(('0.0.0.0', shutdown_port), ShutdownHandler)    
        server.timeout = 1.0
        
        print(f"Shutdown listener started on port {shutdown_port}")
        
        while not stop_event.is_set():
            server.handle_request()
        
        server.server_close()
        print("Shutdown server stopped.")
    except OSError as e:
        print(f"Failed to start shutdown server: {e}")
        import subprocess
        subprocess.run(['sudo', 'killall', 'python3'], capture_output=True)
        time.sleep(1)

        server = ReusableTCPServer(('0.0.0.0', shutdown_port), ShutdownHandler)
        server.timeout = 1.0
        print(f"Shutdown listener started on port {shutdown_port} (retry)")
        
        while not stop_event.is_set():
            server.handle_request()
        
        server.server_close()

def cleanup():
    """Turn off all GPIO devices"""
    print("\nCleaning up GPIO...")
    try:
        led_pin_1.off()
        led_pin_2.off()
        led_pin_3.off()
        buzzer_pin.off()
        led_pin_1.close()
        led_pin_2.close()
        led_pin_3.close()
        buzzer_pin.close()
    except Exception as e:
        print(f"Error during cleanup: {e}")
    print("GPIO cleanup complete.")

def signal_handler(signum, frame):
    """Handle termination signals (SIGTERM, SIGINT)"""
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    stop_event.set()
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# # audio recording and receiving --------------------
def bytes_to_mac_string(mac_bytes: bytes) -> str:
    return ':'.join(f'{b:02X}' for b in mac_bytes)

def receive_audio_data():
    # from database.db_connection import Database
    # get = Database()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', audio_port))
    sock.settimeout(1.0)

    audio_chunks = {}
    chunk_timestamps = {}
    EXPECTED_TOTAL_SAMPLES = 80000

    last_packet_time = time.time()

    device_room_cache = {}
    cache_timeout = 60  # seconds
    cache_timestamps = {}

    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(65536)

            if len(data) < 18:
                print(f"Received packet too small from {addr}: {len(data)} bytes")
                continue

            mac_add = data[0:6]                                    # bytes 0-5: MAC address
            device_add = bytes_to_mac_string(mac_add)   
            room_id = int.from_bytes(data[6:10], 'little')
            timestamp = int.from_bytes(data[10:14], 'little')
            chunk_samples = int.from_bytes(data[14:18], 'little')

            if not device_add or device_add == "00:00:00:00:00:00":
                print(f"Invalid MAC address received: {device_add}")
                continue

            current_time = time.time()

            if device_add is None or room_id is None or timestamp is None or chunk_samples is None:
                print(f"Incomplete data received from {addr}: Room ID{room_id}")
                continue

            if chunk_samples <= 0 or chunk_samples > 16000:
                print(f"Invalid chunk size from Room ID {room_id}: {chunk_samples} samples")
                continue

            audio_chunk = np.frombuffer(data[18:], dtype=np.int16)

            if room_id not in audio_chunks:
                audio_chunks[room_id] = []
                chunk_timestamps[room_id] = time.time()

            audio_chunks[room_id].append(audio_chunk)
            total_samples = sum(len(chunk) for chunk in audio_chunks[room_id])

            if total_samples >= EXPECTED_TOTAL_SAMPLES:
                full_audio = np.concatenate(audio_chunks[room_id])[:EXPECTED_TOTAL_SAMPLES]
                
                thread = threading.Thread(
                    target=process_audio,
                    args=(full_audio, device_add, room_id, timestamp)
                )
                thread.start()

                del audio_chunks[room_id]
                del chunk_timestamps[room_id]

            last_packet_time = time.time()
            for room_id in list(chunk_timestamps.keys()):
                if last_packet_time - chunk_timestamps[room_id] > 10:
                    print(f"Timeout: Discarding incomplete recording from room id {room_id}")
                    del audio_chunks[room_id]
                    del chunk_timestamps[room_id]

        except socket.timeout:
            continue
        except Exception as e:
            print(f"Audio Data - UDP error: {e}")

    sock.close()
    print("Audio data receiver stopped.")

def receive_reset_signals():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', reset_port))
    sock.settimeout(1.0)

    

    while not stop_event.is_set():
        
        try:
            data, addr = sock.recvfrom(1024)
            try:
                reset_data = json.loads(data.decode('utf-8'))
            except json.JSONDecodeError:
                print(f"Invalid JSON received from {addr}: {data}")
                continue

            room_id = reset_data.get('roomID')
            action = reset_data.get('action')
            device_add = reset_data.get('deviceAdd')
            if not room_id or not action:
                print(f"Incomplete reset data received from Room {room_id}: {reset_data}")
                continue

            if action == "reset":
                operation = "Alert Acknowledged"
                
                retry = 0
                try:
                    while retry < 3:
                        success_rpi = asyncio.run(send_reset_rpi(device_add, operation))
                        success_web = asyncio.run(send_reset_web(room_id, operation))
                        # success_esp32 = asyncio.run(send_reset_esp(room_id, operation))
                        if success_rpi and success_web:
                            print(f"Sent reset command from Room {room_id}")
                            break
                        else:
                            print("Failed to send reset command. Retrying...")
                            sleep(2)
                            retry += 1
                            if retry == 3 and not success_web and not success_rpi:
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
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data, dtype=np.int16)

    try:
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        print(f"\nAudio saved: {filename}")
        return True
    except Exception as e:
        print(f"Failed to save WAV: {e}")
        return False
    

# audio processing and inference --------------------

loudness_threshold = 6000 
loud_threshold_ms = 5000 #5 seconds
loud_duration_ms = 0

def process_audio(audio_data_int16, device_add=None, room_id=None, timestamp=None):
    global loud_duration_ms, emergency_detected
    frame_length = 1024
    hop_length = 200

    # print(f"process_audio DEBUG: device add = {device_add}")

    y_i16 = np.asarray(audio_data_int16, dtype=np.int16)
    y = y_i16.astype(np.float32) / 32768.0
    y = y - np.mean(y)

    rms = lb.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=False)[0]
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-6))

    noise_floor = np.percentile(rms_db, 30)
    margin_db = 8.0
    active = rms_db > (noise_floor + margin_db)

    active_ms = active.sum() * (hop_length / sample_rate) * 1000.0
    print(f"Activity={active_ms:.0f} ms")

    try:
        if active_ms >= 3100:
            emergency_detected = True
            trigger_alert(y_i16, device_add, room_id)
            return True
        if active_ms > 100 and active_ms < 3100:
            inference(y_i16, f"Room{room_id}_{timestamp}", device_add, room_id)
            return True
        else:
            print("Audio too quiet, skipping inference.")
            return False
    except Exception as e:
        print(f"Error processing audio: {e}")

        return False


def extract_features(audio, sample_rate, hop_length=200, win_length=400,frame_ms=25, n_mels=64, max_len=320):
    audio = np.asarray(audio).squeeze()

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)
    
    frame_length = int(sample_rate * frame_ms / 1000)
    n_fft = 1 << (frame_length - 1).bit_length()

    # mel-spectrogram + delta + delta-delta approach (new)

    melspectrogram = lb.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    melS_dB = lb.power_to_db(melspectrogram, ref=np.max)

    if melS_dB.shape[1] < max_len:
        melS_dB = np.pad(melS_dB, ((0,0),(0, max_len - melS_dB.shape[1])), mode='constant')
    else:
        melS_dB = melS_dB[:, :max_len]

    S_norm = (melS_dB - melS_dB.mean()) / (melS_dB.std() + 1e-6)
    # melS_dB = melS_dB[..., np.newaxis]
    delta = lb.feature.delta(S_norm)
    delta2 = lb.feature.delta(S_norm, order=2)

    features = np.stack([S_norm, delta, delta2], axis=-1)

    return features

    # #  mfcc+random forest approach (old)
    # n_fft=512

    # def float32(y):
    #     arr = np.asarray(y)
    #     if arr.dtype == np.int16:
    #         return arr.astype(np.float32) / 32768.0
    #     if arr.dtype == np.int32:
    #         return arr.astype(np.float32) / 2147483648.0
    #     return arr.astype(np.float32, copy=False)

    # y = float32(audio).squeeze()

    # mfcc = lb.feature.mfcc(y=y, sr=sample_rate, n_mfcc=20, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    # mfcc_delta = lb.feature.delta(mfcc)
    # mfcc_delta2 = lb.feature.delta(mfcc, order=2)

    # rms = lb.feature.rms(y=y, frame_length=win_length, hop_length=hop_length)
    # zcr = lb.feature.zero_crossing_rate(y=y, frame_length=win_length, hop_length=hop_length)
    # spectral_centroid = lb.feature.spectral_centroid(y=y, sr=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    # spectral_rolloff = lb.feature.spectral_rolloff(y=y, sr=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    # # chroma = lb.feature.chroma_stft(y=y, sr=sample_rate, hop_length=hop_length)

    # features = [mfcc, mfcc_delta, mfcc_delta2, spectral_centroid, spectral_rolloff, zcr, rms]
    # extracted_features = []

    # for feature in features:
    #     if feature.shape[1] < max_len:
    #         feature = np.pad(feature, ((0,0),(0, max_len - feature.shape[1])), mode='constant')
    #     else:
    #         feature = feature[:, :max_len]
        
    #     feature_stat = [
    #         np.mean(feature, axis=1),
    #         np.std(feature, axis=1),
    #         np.min(feature, axis=1),
    #         np.max(feature, axis=1)
    #     ]
        
    #     for stat in feature_stat:
    #         extracted_features.append(stat.flatten())

    # return np.concatenate(extracted_features)

def noise_classification(audio, sample_rate):
    noise_features = extract_features(audio, sample_rate).astype(np.float32)
    noise_features = np.expand_dims(noise_features, axis=0)

    # noise_prediction = noise_classifier.predict(noise_features)
    # noise_class = np.argmax(noise_prediction[0]) if noise_prediction.ndim == 2 else int(noise_prediction[0])

    noise_classifier.set_tensor(noise_input_details[0]['index'], noise_features)
    noise_classifier.invoke()
    noise_class = noise_classifier.get_tensor(noise_output_details[0]['index']) #tflite
    noise_class = np.argmax(noise_class[0]) if noise_class.ndim == 2 else int(noise_class[0])

    if noise_class == 0:
        sound_class = "Environment Sounds"
        print("Environment sound detected.")

    elif noise_class == 1:
        sound_class = "Defensive Speech"
        print("Defensive speech detected.")
        
    elif noise_class == 2:
        sound_class = "Hostile Speech"
        print("Hostile speech detected.")

    else:
        sound_class = "Unknown"

    return sound_class


def inference(audio, wav_name, device_add=None, room_id=None):
    global emergency_count, alarming_count, nonemergency_count, emergency_detected, alarming_detected, sound_type

    # print(f"\nProcessing audio for inference: {wav_name}")

    audio_features = extract_features(audio, sample_rate).astype(np.float32)
    features = np.expand_dims(audio_features, axis=0)

    # prediction = model.predict(features) # cnn
    # predicted_class = prediction[0] #random forest

    interpreter.set_tensor(input_details[0]['index'], features)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index']) #tflite

    predicted_class = np.argmax(prediction[0]) if prediction.ndim == 2 else int(prediction[0]) #cnn

    # to be changed
    # alarming sound = 2 times before alarming is confirmed
    # emergency sound = 1 times after emergency is confirmed

    print(f"\nPrediction for {wav_name}: {predicted_class}")

    if predicted_class == 1:
        alarming_count += 1
        print("ALARMING sound detected. \nAlarming count:", alarming_count, "\nEmergency count:", emergency_count)

        if alarming_count >= 2:
            alarming_detected = True
            sound_type = noise_classification(audio, sample_rate)
            print(f"Determined sound type: {sound_type}")
            trigger_alert(audio, sound_type, device_add, room_id)

    elif predicted_class == 2:
        emergency_count += 1
        print("EMERGENCY sound detected. \nAlarming count:", alarming_count, "\nEmergency count:", emergency_count)
        if emergency_count >= 1:
            emergency_detected = True
            sound_type = noise_classification(audio, sample_rate)
            print(f"Determined sound type: {sound_type}")
            trigger_alert(audio, sound_type, device_add, room_id)

    elif predicted_class == 0:
        print("No emergency detected.")
        nonemergency_count += 1
        if nonemergency_count >= 10:
            alarming_count = 0
            emergency_count = 0
            nonemergency_count = 0
            print("System reset due to consecutive non-emergency sounds.")


# alert triggering -----------------------------------
def trigger_alert(audio, sound_type=None, device_add=None, room_id=None):
    global alarming_detected, emergency_detected, emergency_count, alarming_count, nonemergency_count, success_web, success_esp32

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_filename = f"recorded_audio/ID{room_id}_{datetime_str}.wav"
    wav_path = os.path.abspath(wav_filename)
    
    save_wav(wav_filename, audio, sample_rate)

    alarming_count = 0
    emergency_count = 0
    nonemergency_count = 0

    if (emergency_detected == True):
        print(f"\nALERT TRIGGERED")
        action = "Emergency Alert Detected"
        print(action)

        try:
            retry = 0
            while retry < 3:
                # success_esp32 = asyncio.run(send_alert_esp(room_id, action))
                success_web = asyncio.run(send_alert_web(room_id, action, sound_type, wav_path))
                success_rpi = asyncio.run(send_alert_rpi(device_add, room_id, action))
                
                if success_rpi and success_web:
                    print(f"Sent emergency alert from Room {room_id}")
                    break
                else:
                    print("Failed to send alert. Retrying...")
                    sleep(2)

                    retry += 1
                    success_rpi = asyncio.run(send_alert_rpi(device_add, room_id, action))
                    success_web = asyncio.run(send_alert_web(room_id, action, sound_type, wav_path))
                    # success_esp32 = asyncio.run(send_alert_esp(room_id, action))
                    
                    if retry > 3:
                        print("Failed to send alert after 3 attempts.")

            emergency_detected = False

        except Exception as e:
            print(f"Error sending alert: {e}")

    if (alarming_detected == True):
        print(f"\nALERT TRIGGERED")
        action = "Alarming Alert Detected"

        try:
            retry = 0
            while retry < 3:
                # success_esp32 = asyncio.run(send_alert_esp(room_id, action))
                success_web = asyncio.run(send_alert_web(room_id, action, sound_type, wav_path))
                success_rpi = asyncio.run(send_alert_rpi(device_add, room_id, action))
                
                if success_rpi and success_web:
                    print(f"Sent alarming alert from Room {room_id}")
                    break
                else:
                    print("Failed to send alert. Retrying...")
                    sleep(2)

                    retry += 1
                    success_rpi = asyncio.run(send_alert_rpi(device_add, room_id, action))
                    success_web = asyncio.run(send_alert_web(room_id, action, sound_type, wav_path))
                    # success_esp32 = asyncio.run(send_alert_esp(room_id, action))
                    
                    if retry > 3:
                        print("Failed to send alert after 3 attempts.")

            alarming_detected = False

        except Exception as e:
            print(f"Error sending alert: {e}")

async def send_alert_web(room_id, action=None, sound_type=None, recording_path=None):

    success_web = False
    print(sound_type)

    if "Emergency Alert Detected" in action or "Alarming Alert Detected" in action:
        # web
        try:
            payload_web = {
                "room_id": room_id,
                "action": action,
                "sound_type": sound_type,
                "recording_path": recording_path
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(f"http://localhost:{sns_port}/api/alert", json=payload_web, timeout=10) as response:
                    if response.status == 200:
                        print(f"Alert sent to Web Dashboard for Room {room_id}")
                        success_web = True

        except Exception as e:
            print("Failed to send alert to Web Dashboard:", e)

    return success_web


async def send_reset_web(room_id, action=None):
    success_web = False

    if "Alert Acknowledged" in action:
        try:
            payload_web = {
                "room_id": room_id,
                "action": action,
                "sound_type": "",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(f"http://localhost:{sns_port}/api/alert", json=payload_web, timeout=10) as response:
                    if response.status == 200:
                        print(f"Reset command sent to Web Dashboard for Room {room_id}")
                        success_web = True

        except Exception as e:
            print("Failed to send reset command to Web Dashboard:", e)
    
    return success_web

# async def send_alert_esp(room_id, action=None):
#     success_esp32 = False

#     if "Emergency Alert Detected" in action or "Alarming Alert Detected" in action:
#         try:
#             if esp32_serial and esp32_serial.is_open:
#                 command = f"ALERT: {room_id}\n"
#                 esp32_serial.write(command.encode())

#                 responses = []
#                 responses.append(esp32_serial.readline().decode().strip())
#                 if responses:
#                     for response in responses:
#                         print(f"Received from ESP32: {response}")

#                 success_esp32 = True
#             else:
#                 print("Serial port not open.")

#         except Exception as e:
#             print("Failed to send alert to ESP32 receiver:", e)

#     return success_esp32


async def send_reset_esp(room_id, action=None):
    success_esp32 = False

    if "Alert Acknowledged" in action:
        try:
            if esp32_serial and esp32_serial.is_open:
                command = f"RESET: {room_id}\n"
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
            print("Failed to send reset command to ESP32 receiver:", e)
    
    return success_esp32

led1_active = False
led2_active = False
led3_active = False

async def send_alert_rpi(device_add, room_id, action=None):
    global alerted_rpi, led1_active, led2_active, led3_active
    from database.db_connection import Database
    get = Database()
    success_rpi = False

    device_id = get.fetch_device_id(device_add)
    print(f"Device ID for alert: {device_id}")

    if action is None:
        print("No action specified for RPI alert.")
        return success_rpi

    if "Emergency Alert Detected" in action or "Alarming Alert Detected" in action:

        try:
            await asyncio.sleep(1)
            match device_id:
                case 1:
                    led1_active = True
                    led_pin_1.blink(on_time=0.5, off_time=0.5)
                    print("LED 1 activated.")
                case 3:
                    led2_active = True
                    led_pin_2.blink(on_time=0.5, off_time=0.5)
                    print("LED 2 activated.")
                case 5:
                    led3_active = True
                    led_pin_3.blink(on_time=0.5, off_time=0.5)
                    print("LED 3 activated.")
                case _:
                    print(f"WARNING: Unknown device_id {device_id}, no LED activated!")
                    return success_rpi
                
            print(f"DEBUG: led1_active={led1_active}, led2_active={led2_active}, led3_active={led3_active}")
            print(f"DEBUG: alerted_rpi before check = {alerted_rpi}")

            if alerted_rpi is False and (led1_active or led2_active or led3_active):
                alerted_rpi = True
                buzzer_pin.beep(on_time=0.5, off_time=0.5)
                print("DEBUG: Buzzer should now be beeping!")
            else:
                print(f"DEBUG: Buzzer NOT activated. alerted_rpi={alerted_rpi}")
            
            success_rpi = True
        except Exception as e:
            print("Failed to send alert to Raspberry Pi:", e)

    return success_rpi


async def send_reset_rpi(device_add, action=None):
    global alerted_rpi, led1_active, led2_active, led3_active
    from database.db_connection import Database
    get = Database()
    success_rpi = False

    device_id = get.fetch_device_id(device_add)
    print(f"Device ID for reset: {device_id}")
    # print(f"send_reset_rpi DEBUG: device add = {device_add}")

    if action is None:
        print("No action specified for RPI alert.")
        return success_rpi
    
    if "Alert Acknowledged" in action:
        try:
            match device_id:
                case 1:
                    led1_active = False
                    led_pin_1.off()
                    print("LED 1 deactivated.")
                case 3:
                    led2_active = False
                    led_pin_2.off()
                    print("LED 2 deactivated.")
                case 5:
                    led3_active = False
                    led_pin_3.off()
                    print("LED 3 deactivated.")

            if (not led1_active and not led2_active and not led3_active):
                alerted_rpi = False
                buzzer_pin.off()

            success_rpi = True
        except Exception as e:
            print("Failed to send reset command to ESP32 receiver:", e)
    
    return success_rpi
    

# main loop -----------------------------------------

async def main_loop():
    while not stop_event.is_set():
        await asyncio.sleep(1)

if __name__ == "__main__":
    print("Starting SafeNSound Raspberry Pi Application...\n")

    discovery_server = RPIDiscoverServer()
    discovery_server.start()

    # init_esp32_serial()
    
    audio_thread = threading.Thread(target=receive_audio_data, daemon=True)
    audio_thread.start()

    reset_thread = threading.Thread(target=receive_reset_signals, daemon=True)
    reset_thread.start()

    shutdown_thread = threading.Thread(target=run_shutdown_server, daemon=True)
    shutdown_thread.start()
    
    print("=" * 60)
    print(f"Raspberry Pi IP: {discovery_server.RPI_ip}")
    print(f"Receiver Port: {audio_port}")
    print("=" * 60)

    print("System is running.\n")

    try:
        asyncio.run(main_loop())
        
    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        stop_event.set()
        time.sleep(0.5)

        audio_thread.join(timeout=2)
        reset_thread.join(timeout=2)
        shutdown_thread.join(timeout=2)

        discovery_server.stop()
        cleanup()

        print("\nPorts closed successfully.")
