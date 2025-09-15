#include <Arduino.h>
#include "audio.h"
#include "reset.h"
#include <WiFi.h>

//esp to esp connection
#include <esp_now.h>

uint8_t senderMAC[] = {0x00, 0x4B, 0x12, 0x3A, 0xC0, 0xC0}; //this esp32 (sender)
uint8_t receiverMAC[] = {0x00, 0x4B, 0x12, 0x3A, 0x4D, 0x78}; //receiver

typedef struct AudioRecording {
  int16_t audioData[160];
  size_t sampleCount;
  uint32_t timestamp;
  int roomID;
};

AudioRecording audioRecording;
bool audioReady = false;
const int ROOM_ID = 1;

void dataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  Serial.print("Send Status: ");
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Delivery Success" : "Delivery Fail");
}

// void dataReceived(const uint8_t *mac, const uint8_t *incomingData, int len) {
//   Serial.print("Audio received!");
// }

void setup() { // esp setup
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  // findMACAddress();

  if(esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  esp_now_register_send_cb(dataSent);
  // esp_now_register_recv_cb();

  esp_now_peer_info_t peerInfo;
  memcpy(peerInfo.peer_addr, receiverMAC, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer");
    return;
  }

  Serial.println("ESP-NOW Initialized");

  setupAudio();
  setupResetButton();
}


void findMACAddress() { //find mac address of esp32
  delay(1000);
  
  WiFi.mode(WIFI_STA);
  
  Serial.println("====================");
  Serial.println("ESP32 MAC Address:");
  Serial.println(WiFi.macAddress());
  Serial.println("====================");
  
  String mac = WiFi.macAddress();
  Serial.println("Copy this for your code:");
  Serial.print("uint8_t macAddress[] = {");
  
  for (int i = 0; i < 17; i += 3) {
    String hexByte = mac.substring(i, i + 2);
    Serial.print("0x" + hexByte);
    if (i < 15) Serial.print(", ");
  }
  Serial.println("};");
  Serial.println("====================");
}

//////////////////////////////////////////////////////////////////////////

void sendData() { //send data from audio
  if (audioReady) {
    esp_now_send(receiverMAC, (uint8_t *)&audioRecording, sizeof(audioRecording));
  }

  audioReady = false;
}

void getData() { //get data for reset button
  
}

void prepareAudio(int16_t* audio, size_t sampleCount) { //prepare audio data to be sent
  Serial.println("Getting audio recording...");

  size_t copyCount = min(sampleCount, (size_t)(sizeof(audioRecording.audioData) / sizeof(int16_t)));
  memcpy(audioRecording.audioData, audio, copyCount * sizeof(int16_t));

  audioRecording.sampleCount = copyCount;
  audioRecording.timestamp = millis();
  audioRecording.roomID = ROOM_ID;

  audioReady = true;

  Serial.printf("Audio ready: %d samples, %d bytes\n", copyCount, copyCount * sizeof(int16_t));

}

void loop() { //loops
  processAudioRecording();
  sendData();

  delay(1000);

}