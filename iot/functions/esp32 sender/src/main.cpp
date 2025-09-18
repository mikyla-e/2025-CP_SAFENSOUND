#include <Arduino.h>
#include "audio.h"
#include "reset.h"
#include <WiFi.h>
#include <WebServer.h>
#include <DNSServer.h>
#include <EEPROM.h>
#include <ArduinoJson.h>
#include <WiFiUdp.h>

const char* ssid_ap = "ssid"; //ssid
const char* password_ap = ""; //password
// const int port = 8080; //port

WebServer server(80);
DNSServer dns;
WiFiUDP udp;

String stored_ssid = "";
String stored_password = "";
String stored_ip = ""; //lappy ip
bool wifi_configured = false;

#define SSID_ADDR 0
#define PASS_ADDR 32
#define IP_ADDR 64
#define CONFIG_FLAG_ADDR 96

typedef struct AudioRecording {
  int16_t audioData[160];
  size_t sampleCount;
  uint32_t timestamp;
  int roomID;
};

AudioRecording audioRecording;
bool audioReady = false;
const int ROOM_ID = 1;

/////////////////////////////////////////////////////////

void setup() { // esp setup
  Serial.begin(115200);
  EEPROM.begin(512);

  loadWiFiCredentials();

  if (wifi_configured && connectToWiFi()) {
    Serial.println("Connected to saved WiFi.");
    setupAudio();
    setupResetButton();
  } else {
    Serial.println("Failed to connect to saved WiFi. Starting captive portal.");
    startCaptivePortal();
  }
}

/////////////////////////////////////////////////////////

void saveWiFiCredentials() {
  EEPROM.writeString(SSID_ADDR, stored_ssid);
  EEPROM.writeString(PASS_ADDR, stored_password);
  EEPROM.writeString(IP_ADDR, stored_ip);
  EEPROM.writeBool(CONFIG_FLAG_ADDR, true);
  EPPROM.commit();

  wifi_configured = true;

  Serial.println(" Wifi Credentials saved to EPPROM");
}

void loadWiFiCredentials() {
  wifi_configured = EPPROM.readbool(CONFIG_FLAG_ADDR);
  if (wifi_configured) {
    stored_ssid = EEPROM.readString(SSID_ADDR);
    stored_password = EEPROM.readString(PASS_ADDR);
    stored_ip = EEPROM.readString(IP_ADDR);
    
    Serial.println("📖 Loaded saved credentials");
  }
}

void connectToWiFi() {
  if (stored_ssid.length() == 0) return false;

  WiFi.mode(WIFI_STA);
  WiFi.begin(stored_ssid.c_str(), stored_password.c_str());

  Serial.print("Connecting to " + stored_ssid + "...");

  int attempts = 0;

  while (WiFi.status() != WL_CONNECTED && attempt <20) {
    delay(500);
    Serial.print(".");
    attempts++
  }

  if (WiFi.status() == WL_CONNECTED){
    Serial.println("\nWiFi connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    return true;
  }

  Serial.println("Wifi connection failed");
  return false;
}

void startCaptivePortal() {
  Serial.println("Starting Captive Portal...")

  WiFi.mode(WIFI_AP);
  WiFi.softAP(ssid_ap, password_ap);

  dns.start(53, "*", WiFi.softAPIP());

  server.on("/", handleRoot);
  server.on("/configure", handleConfigure);
  server.on("/status", handleStatus);
  server.onNotFound(handleNotFound);

  server.begin();
  Serial.println("Captive Portal started");
  Serial.print("AP IP: ");
  Serial.println(WiFi.softAPIP());

}

void handleRoot() {
  String html_root = R"()"; // <--- captive portal ui @Danisa hehe 

  server.send(200, "text/html", html_root);
}

void handleConfigure() {
  if (server.hasArg("ssid") && server.harArg("laptop_ip")) {
    stored_ssid = server.arg("ssid");
    stored_password = server.arg("password");
    laptop_ip = server.arg("laptop_ip");

    saveWiFiCredentials();

    if (connectToWifi()) {
      String html_connected = R"()"; // <--- connected to wifi ui @Danisa hehe 
      server.send(200, "text/html", html_connected);
      delay(2000);
      ESP.restart();
    } else {
      String html_failed = R"()"; // <--- NOT connected to wifi ui @Danisa hehe 
      server.send(400, "text/html", html_failed);
    }
  }
}

void handleStatus() {
  DynamicJsonDocument doc(300);
  doc["connected"] = (WiFi.status() == WL_CONNECTED);
  doc["wifi_ip"] = WiFi.localIP().toString();
  doc["ap_ip"] = WiFi.softAPIP().toString();
  doc["room_id"] = ROOM_ID;
  doc["laptop_ip"] = laptop_ip;

  String jsonResponse;
  serializeJson(doc, jsonResponse);
  server.send(200, "application/json", jsonResponse);
}

void handleNotFound() {
  server.sendHeader("Location", "http://" + WiFi.softAPIP().toString(), true);
  server.send(302, "text/plain", "");
}

/////////////////////////////////////////////////////////

void sendData() {
  if (audioReady && WiFi.status() == WL_CONNECTED) {
    udp.beginPacket(laptop_ip.c_str(), 8080);

    DynamicJsonDocument jsonDoc(1024);
    jsonDoc["roomID"] = ROOM_ID;
    jsonDoc["timestamp"] = audioRecording.timestamp;
    jsonDoc["audioData"] = audioRecording.audioData;

    String sendAudioData;
    serializeJson(jsonDoc, sendAudioData);

    udp.print(sendAudioData);
    udp.endPacket();

    audioReady = false;
  }
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
  if(WiFi.getMode() == WIFI_AP) {
    dns.processNextRequest();
    server.handleClient();
  } else {
    processAudioRecording();
    sendData();
    delay(1000);
  }
}