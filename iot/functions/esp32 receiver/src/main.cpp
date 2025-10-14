#include <Arduino.h>

#define BUZZER_PIN 23
#define LED_PIN 16

bool alarmActive = false;
unsigned long alarmStartTime = 0;
const unsigned long ALARM_DURATION = 5000; // 5 seconds


void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);

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
    Serial.println("Received command: " + command);

    if (command.startsWith("ALERT: ")) {
      int roomID = command.substring(7).toInt();
      Serial.printf("Alarm triggered from room %d\n", roomID);
      triggerAlarm();
    }
    else if (command.startsWith("RESET: ")) {
      int roomID = command.substring(7).toInt();
      Serial.printf("Reset command received from room %d\n", roomID);
      resetAlarm();
    } else {
      Serial.println("UNKNOWN COMMAND: " + command + ". Ignored.");
    }

  }
}

void autoReset() {
  if (alarmActive && (millis() - alarmStartTime > ALARM_DURATION)) {
    Serial.println("AUTO RESET");
    resetAlarm();
  }
}

///////////////////////////////////////////////////////////////

void loop() { 
  processCommand();
  autoReset();
  delay(5);
}