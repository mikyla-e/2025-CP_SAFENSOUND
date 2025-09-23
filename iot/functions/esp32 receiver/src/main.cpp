#include <Arduino.h>

#define BUZZER_PIN 25
#define LED_PIN 26

bool alarmActive = false;
unsigned long alarmStartTime = 0;
const unsigned long ALARM_DURATION = 120000; // 1 minute


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
    command.trim();

    if (command.startsWith("ALARM: ")) {
      int roomID = command.substring(7).toInt();
      Serial.printf("Alarm triggered from room %d\n", roomID);
      triggerAlarm();
    } else if (command == "STATUS") {
      Serial.println(alarmActive ? "ALARM_ACTIVE" : "NO_ALARM");
    }

  }
}

void autoReset() {
  if (alarmActive && (millis() - alarmStartTime > ALARM_DURATION)) {
    Serial.println("AUTO_RESET");
    resetAlarm();
  }
}



void loop() {
  processCommand();
  autoReset();
  delay(100);
}