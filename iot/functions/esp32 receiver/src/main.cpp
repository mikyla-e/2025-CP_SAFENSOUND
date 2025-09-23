#include <Arduino.h>

#define BUZZER_PIN 25
#define LED_PIN 26
#define RESET_BUTTON_PIN 27

bool alarmActive = false;
unsigned long alarmStartTime = 0;
const unsigned long ALARM_DURATION = 60000; // 1 minute


void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  pinMode(RESET_BUTTON_PIN, INPUT_PULLUP);

  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  Serial.println("Receiver ready!");
}

void triggerAlarm() {
  alarmActive = true;
  alarmStartTime = millis();
  digitalWrite(LED_PIN, HIGH);
  digitalWrite(BUZZER_PIN, HIGH);
  Serial.println("Alarm activated!");
}

void resetAlarm() {
  alarmActive = false;
  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
  Serial.println("Alarm reset!");
} 

void processCommand() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command == "ALARM") {
      int roomID = command.substring(6).toInt();
      Serial.printf("Alarm triggered from room %d\n", roomID);
      triggerAlarm();
    } else if (command == "RESET") {
      resetAlarm();
    } else if (command == "STATUS") {
      Serial.println(alarmActive ? "ALARM_ACTIVE" : "NO_ALARM");
    }
  }
}

void checkResetButton() {
  static unsigned long lastCheck = 0;
  static bool lastState = HIGH;
  
  if (millis() - lastCheck > 50) {
    bool currentState = digitalRead(RESET_BUTTON_PIN);
    if (lastState == HIGH && currentState == LOW) {
        resetAlarm();
    }
    lastState = currentState;
    lastCheck = millis();
  }
}

void autoReset() {
  if (alarmActive && (millis() - alarmStartTime > ALARM_DURATION)) {
    Serial.println("AUTO_RESET");
    resetAlarm();
  }
}



void loop() {
  checkResetButton();
  autoReset();
  processCommand();
  delay(100);
}