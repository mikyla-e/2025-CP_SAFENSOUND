#include <Arduino.h>
#include "audio.h"
#include "reset.h"
#include <WiFi.h>
#include <WebServer.h>
#include <DNSServer.h>
#include <EEPROM.h>
#include <ArduinoJson.h>
#include <WiFiUdp.h>

const char* ssid_ap = "SafeNSound_ESP32"; //ssid
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

typedef struct {
  int16_t audioData[1024];
  size_t sampleCount;
  uint32_t timestamp;
  int roomID;
} AudioRecording;

AudioRecording audioRecording;
bool audioReady = false;
const int room_id = 1;

/////////////////////////////////////////////////////////

bool discoverIP() {
  WiFiUDP udp;
  udp.begin(8080);

  IPAddress broadcastIP = WiFi.localIP();
  broadcastIP[3] = 255;

  udp.beginPacket(broadcastIP, 8080);
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

  Serial.println("Wifi Credentials saved to EEPROM");
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

  while (WiFi.status() != WL_CONNECTED && attempts < 50) {
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

  if (WiFi.status() == WL_NO_SSID_AVAIL) {
    Serial.println("SSID not available.");
  } else if (WiFi.status() == WL_CONNECT_FAILED) {
    Serial.println("Connection failed. Check password.");
  } else if (WiFi.status() == WL_DISCONNECTED) {
    Serial.println("Disconnected from WiFi.");
  } else {
    Serial.println("Unknown error.");
  }

  return false;
}

////////////////////////////////////////////////////////////

void handleDiscoverIP() {
  String discovered_ip = "";
  if (discoverIP()) {
    discovered_ip = laptop_ip;
    server.send(200, "text/plain", discovered_ip);
  } else {
    server.send(500, "text/plain", "Discovery failed");
  }
}

void handleRoot() {
  String html_root = R"(
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <link rel="stylesheet" href="/captive_portal.css">
    </head>
    <body>
      <div class="container" id="configForm">
        <div class="header-icon">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
                <path d="M256 32c-17.7 0-32 14.3-32 32V80C153.3 92.7 96 158.3 96 240v112l-32 32v16H448V384l-32-32V240c0-81.7-57.3-147.3-128-160V64c0-17.7-14.3-32-32-32zm0 448c35.3 0 64-28.7 64-64H192c0 35.3 28.7 64 64 64z"/>
            </svg>
        </div>
        
        <h1>SafeNSound Configuration</h1>
        <p>Configure your device to connect to WiFi and communicate with your system.</p>
        
        <form action="/configure" method="POST">
          <h3>Settings</h3>
          <input type="text" name="ssid" placeholder="WiFi Network Name" required>
          <input type="password" name="password" placeholder="WiFi Password">
            
          <input type="text" name="laptop_ip" placeholder="Laptop IP Address">
            
          <button type="submit">Save</button>
        </form>
        
        <div id="status"></div>
      </div>
    </body>
    </html>
  )"; // <--- Configuration Form

  server.send(200, "text/html", html_root);
}

void handleConfigure() {
  if (server.hasArg("ssid") && server.hasArg("laptop_ip")) {
    stored_ssid = server.arg("ssid");
    stored_password = server.arg("password");
    laptop_ip = server.arg("laptop_ip");

    saveWiFiCredentials();

    if (connectToWiFi()) {
      String html_connected = R"(
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <link rel="stylesheet" href="/captive_portal.css">
        </head>
        <body>
            <div class="container" id="successMessage">
                <div class="header-icon" style="background-color: #4CAF50;">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
                        <path d="M256 48a208 208 0 1 1 0 416 208 208 0 1 1 0-416zm0 464A256 256 0 1 0 256 0a256 256 0 1 0 0 512zM369 209c9.4-9.4 9.4-24.6 0-33.9s-24.6-9.4-33.9 0l-111 111-47-47c-9.4-9.4-24.6-9.4-33.9 0s-9.4 24.6 0 33.9l64 64c9.4 9.4 24.6 9.4 33.9 0L369 209z"/>
                    </svg>
                </div>
                <h1>Configuration Saved!</h1>
                <p>ESP32 is now connected to WiFi. You can close this page.</p>
            </div>
        </body>
        </html>
      )"; // <--- Success Message
      server.send(200, "text/html", html_connected);
      delay(2000);
      ESP.restart();
    } else {
      String html_failed = R"(
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <link rel="stylesheet" href="/captive_portal.css">    
        </head>
        <body>
            <div class="container " id="errorMessage">
                <div class="header-icon" style="background-color: #f44336;">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
                        <path d="M256 48a208 208 0 1 1 0 416 208 208 0 1 1 0-416zm0 464A256 256 0 1 0 256 0a256 256 0 1 0 0 512zm0-384c-13.3 0-24 10.7-24 24V264c0 13.3 10.7 24 24 24s24-10.7 24-24V152c0-13.3-10.7-24-24-24zm32 224a32 32 0 1 0 -64 0 32 32 0 1 0 64 0z"/>
                    </svg>
                </div>
                <h1>Connection Failed</h1>
                <p>Could not connect to WiFi. Please check your credentials and try again.</p>
                <center><a href="/" class="link">Go Back</a></center>
            </div>
        </body>
        </html>
      )"; // <--- NOT connected to wifi ui @Danisa hehe 
      server.send(400, "text/html", html_failed);
    }
  }
}

void handleStatus() {
  JsonDocument doc;
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

void handleCSS() {
  String css = R"(
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&display=swap');

      :root {
          --primary: white;
          --second: #2D336B;
          --background: linear-gradient(133deg, #F7F7F7 13.25%, #A9B5DF 205.83%);
      }

      * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
      }

      body {
          font-family: "DM Sans", sans-serif;
          background: var(--background);
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 20px;
      }

      /* ALL BOXES */
      .container {
          background: var(--primary);
          padding: 30px;
          border-radius: 13px;
          box-shadow: 0 20.547px 20.547px 0 rgba(0, 0, 0, 0.10);
          width: 100%;
          max-width: 600px;
          margin-bottom: 20px;
      }

      /* BELL ICON CIRCLE */
      .header-icon {
          width: 60px;
          height: 60px;
          background-color: var(--second);
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          margin: 0 auto 20px;
          position: relative;
      }

      /* BELL ICON */
      .header-icon svg {
          width: 35px;
          height: 35px;
          fill: white;
      }

      /* SafeNSound Configuration */
      h1 {
          text-align: center;
          color: var(--second);
          font-size: clamp(1.3rem, 3vw, 1.8rem);
          font-weight: 600;
          margin-bottom: 10px;
      }

      /* Configure your device... */
      p {
          text-align: center;
          color: #666;
          margin-bottom: 25px;
          font-size: clamp(0.9rem, 2vw, 1rem);
      }

      /* ... Settings */
      h3 {
          color: var(--second);
          font-size: clamp(1rem, 2.5vw, 1.2rem);
          margin: 20px 0 10px;
          font-weight: 600;
          opacity: 0.8;
      }

      /* INPUT FIELDS */
      input {
          width: 100%;
          padding: 12px 15px;
          margin: 8px 0;
          border: 2px solid rgba(45, 51, 107, 0.2);
          border-radius: 10px;
          font-family: "DM Sans", sans-serif;
          font-size: clamp(0.9rem, 2vw, 1rem);
          transition: all 0.3s ease;
      }

      input:focus {
          outline: none;
          /* border-color: var(--second); */
          border-color: #2d336baf;
          box-shadow: 0 0 0 3px rgba(45, 51, 107, 0.1);
      }

      input::placeholder {
          color: #999;
      }

      button {
          width: 100%;
          background: var(--second);
          color: white;
          padding: 14px 20px;
          border: none;
          border-radius: 10px;
          cursor: pointer;
          font-family: "DM Sans", sans-serif;
          font-size: clamp(1rem, 2.5vw, 1.1rem);
          font-weight: 600;
          margin-top: 20px;
          transition: all 0.3s ease;
          box-shadow: 0 4px 15px rgba(45, 51, 107, 0.3);
      }

      button:hover {
          background: #1f2449;
          transform: translateY(-2px);
          box-shadow: 0 6px 20px rgba(45, 51, 107, 0.4);
      }

      button:active {
          transform: translateY(0);
      }

      .status {
          padding: 15px;
          margin: 15px 0;
          border-radius: 10px;
          font-size: clamp(0.9rem, 2vw, 1rem);
          text-align: center;
      }

      .success {
          background: rgba(76, 175, 80, 0.1);
          color: #2e7d32;
          border: 2px solid rgba(76, 175, 80, 0.3);
      }

      .error {
          background: rgba(244, 67, 54, 0.1);
          color: #c62828;
          border: 2px solid rgba(244, 67, 54, 0.3);
      }

      .link {
          display: inline-block;
          margin-top: 15px;
          color: var(--second);
          text-decoration: none;
          font-weight: 600;
          padding: 10px 20px;
          border: 2px solid var(--second);
          border-radius: 10px;
          transition: all 0.3s ease;
      }

      .link:hover {
          background: var(--second);
          color: white;
      }

      .hidden {
          display: none;
      }

      /* Responsive design */
      @media screen and (max-width: 768px) {
          body {
              padding: 15px;
          }

          .container {
              padding: 20px;
          }

          nav {
              height: 50px;
              margin-bottom: 15px;
          }
      }

      @media screen and (max-width: 375px) {
          .container {
              padding: 15px;
          }

          .header-icon {
              width: 50px;
              height: 50px;
          }

          .header-icon svg {
              width: 28px;
              height: 28px;
          }
      }
    )";
    server.send(200, "text/css", css);
  }

/////////////////////////////////////////////////////////

void startCaptivePortal() {
  Serial.println("Starting Captive Portal...");

  WiFi.mode(WIFI_AP);
  WiFi.softAP(ssid_ap, password_ap);

  dns.start(53, "*", WiFi.softAPIP());

  server.on("/", handleRoot);
  server.on("/configure", handleConfigure);
  server.on("/status", handleStatus);
  server.on("/captive_portal.css", handleCSS);
  server.on("/discover_ip", handleDiscoverIP);
  server.onNotFound(handleNotFound);

  server.begin();
  Serial.println("Captive Portal started");
  Serial.print("AP IP: ");
  Serial.println(WiFi.softAPIP());

}

/////////////////////////////////////////////////////////

void setup() { // esp setup
  Serial.begin(115200);
  EEPROM.begin(512);

  loadWiFiCredentials();

  if (wifi_configured && connectToWiFi()) {
    Serial.println("Connected to saved WiFi.");
    Serial.println("Discovered Main Device IP: " + laptop_ip);
    setupAudio();
    setupResetButton();
  } else {
    Serial.println("Failed to connect to saved WiFi. Starting captive portal.");
    startCaptivePortal();
    return;
  }
}

///////////////////////////////////////////////////////////

void sendData() {
  if (audioReady && WiFi.status() == WL_CONNECTED) {
    udp.beginPacket(laptop_ip.c_str(), 8081);

    JsonDocument jsonDoc;
    jsonDoc["roomID"] = room_id;
    jsonDoc["timestamp"] = audioRecording.timestamp;
    jsonDoc["sampleCount"] = audioRecording.sampleCount;

    JsonArray audioArray = jsonDoc["audioData"].to<JsonArray>();
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
  
  JsonDocument jsonDoc;
  jsonDoc["roomID"] = room_id;
  jsonDoc["action"] = "reset";

  String sendResetData;
  serializeJson(jsonDoc, sendResetData);

  int result = udp.endPacket();

  if (result) {
      Serial.println("Sent");
  } else {
      Serial.println("Failed");
  }
}

void loop() { //loops
  if(WiFi.getMode() == WIFI_AP) {
    dns.processNextRequest();
    server.handleClient();
  } else {
    Serial.println("Processing audio...");
    processAudioRecording();
    sendData();

    if (processResetButton()) {
      Serial.println("Reset button pressed.");
      sendResetSignal();
    }

    delay(5); 
  }
}