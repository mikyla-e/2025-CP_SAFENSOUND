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
String laptop_ip = ""; //lappy ip
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
const int room_id = 1;

/////////////////////////////////////////////////////////

void setup() { // esp setup
  Serial.begin(115200);
  EEPROM.begin(512);

  loadWiFiCredentials();

  if (wifi_configured && connectToWiFi()) {
    Serial.println("Connected to saved WiFi.");

    if (laptop_ip.length() == 0 || discoverIP()) {
      Serial.println("Discovered Main Device IP: " + laptop_ip + ". Starting captive portal...");
      setupAudio();
      setupResetButton();
    } else {
      Serial.println("Could not discover main device. Starting captive portal...");
      startCaptivePortal();
      return;
    }
  } else {
    Serial.println("Failed to connect to saved WiFi. Starting captive portal.");
    startCaptivePortal();
  }
}

/////////////////////////////////////////////////////////

bool discoverIP() {
  WiFiUDP udp;
  udp.begin(8081);

  IPAddress broadcastIP = WiFi.localIP();
  broadcastIP[3] = 255;

  udp.beginPacket(broadcastIP, 8081);
  udp.print("DISCOVER_MAIN_DEVICE");
  udp.endPacket();

  unsigned long startTime = millis();
  while (millis() - startTime < 5000) {
    int packetSize = udp.parsePacket();
    if (packetSize) {
      String response = udp.readString();
      response.trim();

      if(response.startsWith("MAIN_DEVICE_HERE:")) {
        laptop_ip = response.substring(17);
        laptop_ip.trim();
        Serial.println("Discovered Main Device IP: " + laptop_ip);

        EEPROM.writeString(IP_ADDR, laptop_ip);
        EEPROM.commit();

        udp.stop();
        return true;
      }
    }
    delay(100);
  }

  udp.stop();
  Serial.println("Failed to discover laptop IP.");
  return false;
}

void saveWiFiCredentials() {
  EEPROM.writeString(SSID_ADDR, stored_ssid);
  EEPROM.writeString(PASS_ADDR, stored_password);
  EEPROM.writeString(IP_ADDR, laptop_ip);
  EEPROM.writeBool(CONFIG_FLAG_ADDR, true);
  EEPROM.commit();

  wifi_configured = true;

  Serial.println(" Wifi Credentials saved to EEPROM");
}

void loadWiFiCredentials() {
  wifi_configured = EEPROM.readBool(CONFIG_FLAG_ADDR);
  if (wifi_configured) {
    stored_ssid = EEPROM.readString(SSID_ADDR);
    stored_password = EEPROM.readString(PASS_ADDR);
    laptop_ip = EEPROM.readString(IP_ADDR);
    
    Serial.println("📖 Loaded saved credentials");
  }
}

bool connectToWiFi() {
  if (stored_ssid.length() == 0) return false;

  WiFi.mode(WIFI_STA);
  WiFi.begin(stored_ssid.c_str(), stored_password.c_str());

  Serial.print("Connecting to " + stored_ssid + "...");

  int attempts = 0;

  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
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
  Serial.println("Starting Captive Portal...");

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
  if (server.hasArg("ssid") && server.hasArg("laptop_ip")) {
    stored_ssid = server.arg("ssid");
    stored_password = server.arg("password");
    laptop_ip = server.arg("laptop_ip");

    saveWiFiCredentials();

    if (connectToWiFi()) {
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
  doc["room_id"] = room_id;
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

    DynamicJsonDocument jsonDoc(2048);
    jsonDoc["roomID"] = room_id;
    jsonDoc["timestamp"] = audioRecording.timestamp;
    jsonDoc["sampleCount"] = audioRecording.sampleCount;

    JsonArray audioArray = jsonDoc.createNestedArray("audioData");
    for (size_t i = 0; i < audioRecording.sampleCount; i++) {
      audioArray.add(audioRecording.audioData[i]);
    }

    String sendAudioData;
    serializeJson(jsonDoc, sendAudioData);

    udp.print(sendAudioData);

    int result = udp.endPacket();
        
    if (result) {
        Serial.println("✅ Sent");
    } else {
        Serial.println("❌ Failed");
    }

    audioReady = false;
  }
}

void prepareAudio(int16_t* audio, size_t sampleCount) { //prepare audio data to be sent
  Serial.println("Getting audio recording...");

  size_t copyCount = min(sampleCount, (size_t)(sizeof(audioRecording.audioData) / sizeof(int16_t)));
  memcpy(audioRecording.audioData, audio, copyCount * sizeof(int16_t));

  audioRecording.sampleCount = copyCount;
  audioRecording.timestamp = millis();
  audioRecording.roomID = room_id;

  audioReady = true;

  Serial.printf("Audio ready: %d samples, %d bytes\n", copyCount, copyCount * sizeof(int16_t));

}

void sendResetSignal() {
  udp.beginPacket(laptop_ip.c_str(), 8082);
  
  DynamicJsonDocument jsonDoc(1024);
    jsonDoc["roomID"] = room_id;
    jsonDoc["action"] = "reset";

    String sendResetData;
    serializeJson(jsonDoc, sendResetData);

  udp.endPacket();
}

void loop() { //loops
  if(WiFi.getMode() == WIFI_AP) {
    dns.processNextRequest();
    server.handleClient();
  } else {
    processAudioRecording();
    sendData();

    if (processResetButton()) {
      Serial.println("Reset button pressed.");
      sendResetSignal();
    }

    delay(1000); 
  }
}