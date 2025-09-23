#include <Arduino.h>
#include "audio.h"
#include "reset.h"
#include <WiFi.h>

//esp to esp connection
#include <esp_now.h>
uint8_t senderMAC[] = {0x00, 0x4B, 0x12, 0x3A, 0xC0, 0xC0}; //sender

uint8_t receiverMAC[] = {0x00, 0x4B, 0x12, 0x3A, 0x4D, 0x78}; //receiver

void setup() {
  Serial.begin(115200);

  delay(1000);
  
  // Initialize WiFi in station mode
  WiFi.mode(WIFI_STA);
  
  Serial.println("====================");
  Serial.println("ESP32 MAC Address:");
  Serial.println(WiFi.macAddress());
  Serial.println("====================");
  
  // Also print in array format for easy copying
  String mac = WiFi.macAddress();
  Serial.println("Copy this for your code:");
  Serial.print("uint8_t macAddress[] = {");
  
  // Convert MAC string to hex array format
  for (int i = 0; i < 17; i += 3) {
    String hexByte = mac.substring(i, i + 2);
    Serial.print("0x" + hexByte);
    if (i < 15) Serial.print(", ");
  }
  Serial.println("};");
  Serial.println("====================");

  // setupAudio();
  // setupResetButton();

}

void loop() {
  delay(5000);
  Serial.println("MAC: " + WiFi.macAddress());
  // processAudioRecording();
  // pressResetButton();

  //delay(10);

}