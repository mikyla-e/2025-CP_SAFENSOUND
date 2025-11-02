#include <Arduino.h>
#include "audio.h"
#include "reset.h"
#include <WiFi.h>
#include <WebServer.h>
#include <DNSServer.h>
#include <EEPROM.h>
#include <ArduinoJson.h>
#include <WiFiUdp.h>
#include "driver/gpio.h"

const int disc_port = 60123;
const int audio_port = 54321;
const int reset_port = 58080;

const char* ssid_ap = "SafeNSound_S"; //ssid
const char* password_ap = "SafeCapstone25"; //password

WebServer server(80);
DNSServer dns;
WiFiUDP udp;

const uint16_t timeoutMs = 4000;

String stored_ssid = "";
String stored_password = "";
String laptop_ip = "";
String auth_token = "";
bool wifi_configured = false;

#define SSID_ADDR 0
#define PASS_ADDR 32
#define IP_ADDR 64
#define CONFIG_FLAG_ADDR 96
#define TOKEN_ADDR 128

typedef struct {
  int16_t audioData[16000];
  size_t sampleCount;
  uint32_t timestamp;
  int roomID;
} AudioRecording;

AudioRecording audioRecording;
bool audioReady = false;
const int room_id = 2;

/////////////////////////////////////////////////////////

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

void handleRoot() {
  String html_root = R"(
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SafeNSound ESP32 Setup</title>

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
                <h3>WiFi Settings</h3>
                <input type="text" name="ssid" placeholder="WiFi Network Name '(SSID)'" required>
                
                <input type="password" name="password" placeholder="WiFi Password">
                
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
  if (server.hasArg("ssid")) {
    stored_ssid = server.arg("ssid");
    stored_password = server.arg("password");

    saveWiFiCredentials();

    if (connectToWiFi()) {
      String html_connected = R"(
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Setup Successful</title>

            <link rel="stylesheet" href="/captive_portal.css">
        </head>
        <body>
            <!-- Success Message -->
            <div class="container" id="successMessage">
                <a class="close-btn" href="/" aria-label="Close and return">x</a>

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
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Setup Error</title>

            <link rel="stylesheet" href="/captive_portal.css">
        </head>
        <body>
            <!-- Error Message -->
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
      )";
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
    @font-face {
      font-family: 'DM Sans';
      src: url('/static/fonts/DM_Sans/DMSans_18pt-Regular.ttf') format('truetype');
      font-weight: normal;
      font-style: normal;
    }

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
      justify-content: center;
      padding: 20px;
    }

    /* ALL BOXES */
    .container {
      background: var(--primary);
      padding: 40px;
      border-radius: 13px;
      box-shadow: 0 20.547px 20.547px 0 rgba(0, 0, 0, 0.10);
      width: 100%;
      max-width: 600px;
      margin-bottom: 20px;

      position: relative;
    }

    /* Close / Exit X button */
    .close-btn {
      position: absolute;
      top: 16px;
      right: 16px;
      width: 36px;
      height: 36px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 50%;
      background: transparent;
      border: 2px solid rgba(45, 51, 107, 0.12);
      color: var(--second);
      text-decoration: none;
      font-weight: 700;
      font-size: 1.1rem;
      line-height: 1;
      transition: all 0.18s ease;
      z-index: 5;
    }

    .close-btn:hover {
      background: var(--second);
      color: white;
      transform: translateY(-2px);
      border-color: var(--second);
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
      font-size: clamp(1.5rem, 4vw, 2rem);
      font-weight: 600;
      margin-bottom: 10px;
    }

    /* Configure your device... */
    p {
      text-align: center;
      color: #666;
      margin-bottom: 25px;
      font-size: clamp(0.95rem, 2.5vw, 1.1rem);
      line-height: 1.5;
    }

    /* ... Settings */
    h3 {
      color: var(--second);
      font-size: clamp(1.1rem, 3vw, 1.3rem);
      margin: 20px 0 10px;
      font-weight: 600;
      opacity: 0.8;
    }

    /* INPUT FIELDS */
    input {
      width: 100%;
      padding: 14px 16px;
      margin: 10px 0;
      border: 2px solid rgba(45, 51, 107, 0.2);
      border-radius: 10px;
      font-family: "DM Sans", sans-serif;
      font-size: clamp(0.95rem, 2.5vw, 1.05rem);
      transition: all 0.3s ease;
      box-sizing: border-box;
    }

    input:focus {
      outline: none;
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
      padding: 16px 20px;
      border: none;
      border-radius: 10px;
      cursor: pointer;
      font-family: "DM Sans", sans-serif;
      font-size: clamp(1.05rem, 3vw, 1.15rem);
      font-weight: 600;
      margin-top: 25px;
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
      font-size: clamp(0.95rem, 2.5vw, 1.05rem);
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
      padding: 12px 24px;
      border: 2px solid var(--second);
      border-radius: 10px;
      transition: all 0.3s ease;
      font-size: clamp(0.95rem, 2.5vw, 1.05rem);
    }

    .link:hover {
      background: var(--second);
      color: white;
    }

    .hidden {
      display: none;
    }

    /* Responsive design for tablets and medium screens */
    @media (max-width: 768px) and (min-width: 481px) {
      body {
        padding: 20px;
      }

      .container {
        padding: 35px;
        max-width: 500px;
      }

      .header-icon {
        width: 55px;
        height: 55px;
      }

      .header-icon svg {
        width: 32px;
        height: 32px;
      } 
    }

    /* Responsive design for phones (small to medium) */
    @media (max-width: 480px) {
      body {
        padding: 0;
        justify-content: flex-start;
      }

      .container {
        padding: 30px 25px;
        max-width: 100%;
        min-height: 100vh;
        margin: 0;
        border-radius: 0;
        display: flex;
        flex-direction: column;
        justify-content: center;
      }

      .close-btn {
        top: 12px;
        right: 12px;
        width: 34px;
        height: 34px;
        font-size: 1rem;
      }

      .header-icon {
        width: 55px;
        height: 55px;
        margin-bottom: 25px;
      }

      .header-icon svg {
        width: 32px;
        height: 32px;
      }

      input {
        padding: 15px 16px;
        margin: 12px 0;
      }

      button {
        padding: 16px 20px;
        margin-top: 30px;
      }

      h3 {
        margin-top: 25px;
      }
    }

    /* Extra small phones */
    @media (max-width: 360px) {
      .container {
        padding: 25px 20px;
      }

      .header-icon {
        width: 50px;
        height: 50px;
        margin-bottom: 20px;
      }

      .header-icon svg {
        width: 28px;
        height: 28px;
      }

      input {
        padding: 13px 14px;
      }

      button {
        padding: 14px 18px;
      }
    }

    /* Large screens */
    @media (min-width: 1024px) {
      .container {
        padding: 45px;
        max-width: 650px;
      }

      .header-icon {
        width: 70px;
        height: 70px;
        margin-bottom: 25px;
      }

      .header-icon svg {
        width: 40px;
        height: 40px;
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
  server.onNotFound(handleNotFound);

  server.begin();
  Serial.println("Captive Portal started");
  Serial.print("AP IP: ");
  Serial.println(WiFi.softAPIP());

}

bool discoverLaptopIP(){
  WiFiUDP discIP;
  const uint16_t srcPort = 61234;
  const char* msg = "DISCOVER_MAIN_DEVICE";

  if (!discIP.begin(srcPort)) {
    Serial.println("UDP begin failed");
    return false;
  }

  IPAddress local = WiFi.localIP();
  IPAddress mask  = WiFi.subnetMask();
  IPAddress bcast;

  for (int i = 0; i < 4; ++i) bcast[i] = local[i] | ~mask[i];
  Serial.printf("Calling %s:%d\n", bcast.toString().c_str(), disc_port);
  discIP.beginPacket(bcast, disc_port);
  discIP.write((const uint8_t*)msg, strlen(msg));
  discIP.endPacket();

  unsigned long start = millis();
  char buf[128];

  while (millis() - start < timeoutMs) {
    int packetSize = discIP.parsePacket();
    if (packetSize > 0) {
      int n = discIP.read(buf, sizeof(buf) - 1);
      if (n < 0) continue;
      buf[n] = 0;

      String s(buf);
      if (s.startsWith("MAIN_DEVICE_HERE:")) {
        String ip = s.substring(strlen("MAIN_DEVICE_HERE:"));
        ip.trim();
        if (ip.length()) {
          laptop_ip = ip;
          EEPROM.writeString(IP_ADDR, laptop_ip);
          EEPROM.commit();
          Serial.printf("Discovery: Saving %s as Laptop IP.\n", laptop_ip.c_str());
          discIP.stop();
          return true;
        }
      } else {
        Serial.printf("Ignoring reply: '%s'\n", s.c_str());
      }
    }
    delay(10);
  }

  discIP.stop();
  Serial.println("Timeout, no reply");
  return false;
}

/////////////////////////////////////////////////////////

void setup() { // esp setup
  Serial.begin(115200);
  EEPROM.begin(512);

  delay(500);

  loadWiFiCredentials();

  if (wifi_configured && connectToWiFi()) {
    
    Serial.println("Connected to saved WiFi.");
    Serial.println("Room ID: " + String(room_id));

    delay(500);
    if (laptop_ip.length() == 0 || laptop_ip == "0.0.0.0") {
      if (!discoverLaptopIP()){
        Serial.println("Discovering Laptop IP failed.");
      }
    } else {
      if (!discoverLaptopIP()) {
        Serial.println("Discovery failed... trying again.");
        discoverLaptopIP();
      }
    }

    Serial.println("Laptop IP: " + laptop_ip);
    udp.begin(audio_port);
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

    size_t maxPacketSize = 1400;
    size_t maxAudioSize = maxPacketSize - 12;
    size_t maxSamplesPerPacket = maxAudioSize / sizeof(int16_t);

    size_t totalSamples = audioRecording.sampleCount;
    size_t samplesSent = 0;

    while (samplesSent < totalSamples) {
      size_t samplesToSend = min(maxSamplesPerPacket, totalSamples - samplesSent);
      size_t packetSize = 12 + samplesToSend * sizeof(int16_t);

      uint8_t* buffer = (uint8_t*)malloc(packetSize);
      if (!buffer) {
        Serial.println("Failed to allocate buffer");
        return;
      }

      memcpy(buffer, &room_id, 4);
      memcpy(buffer + 4, &audioRecording.timestamp, 4);
      memcpy(buffer + 8, &samplesToSend, 4);
      memcpy(buffer + 12, audioRecording.audioData + samplesSent, samplesToSend * sizeof(int16_t));

      udp.beginPacket(laptop_ip.c_str(), audio_port);
      udp.write(buffer, packetSize);
      udp.endPacket();
      
      free(buffer);

      samplesSent += samplesToSend;
          
      delay(0);
    }

    Serial.println("All packets sent");
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
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected!");
    return;
  }
  if (laptop_ip.length() == 0 || laptop_ip == "0.0.0.0") {
    Serial.println("Invalid laptop IP!");
    return;
  }
  
  udp.beginPacket(laptop_ip.c_str(), reset_port);
  
  JsonDocument jsonDoc;
  jsonDoc["roomID"] = room_id;
  jsonDoc["action"] = "reset";

  String sendResetData;
  serializeJson(jsonDoc, sendResetData);

  udp.print(sendResetData);
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
    processAudioRecording();
    sendData();
    if (processResetButton()) {
      sendResetSignal();
    }

    delay(1); 
  }
}