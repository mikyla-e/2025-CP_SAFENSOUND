#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>

uint8_t receiverMAC[] = {0x00, 0x4B, 0x12, 0x3A, 0x4D, 0x78}; //this esp32 (receiver)
uint8_t senderMAC1[] = {0x00, 0x4B, 0x12, 0x3A, 0xC0, 0xC0}; //sender

typedef struct AudioRecording {
  uint16_t audioData[160];
  size_t sampleCount;
  uint32_t timestamp;
  int roomID;
};

struct RoomStatus {
  int emergencyCount;
  String roomName;

};

RoomStatus room1 = {0, "Room 1"};
RoomStatus room2 = {0, "Room 2"};
RoomStatus room3 = {0, "Room 3"};

void addMicrophone(uint8_t* macAddress);
void processAudioML();

void dataReceived(const uint8_t *mac, const uint8_t *incomingData, int len) {
  Serial.print("Audio received!");
}

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);

  if(esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  esp_now_register_recv_cb(dataReceived);

  addMicrophone(senderMAC1);

}

void addMicrophone(uint8_t* macAddress) {
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, macAddress, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add Microphone.");
  } else {
    Serial.println("Added Microphone from... ");
    Serial.printf("Added Microphone from... %02X:%02X:%02X:%02X:%02X:%02X\n", 
                  macAddress[0], macAddress[1], macAddress[2], 
                  macAddress[3], macAddress[4], macAddress[5]);
    
  }
}

void processAudioML() {

}


void loop() {
  // put your main code here, to run repeatedly:
}