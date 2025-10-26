#include <Arduino.h>

#define BUZZER_PIN 23
#define LED_PIN_1 22
#define LED_PIN_2 21
#define LED_PIN_3 19

bool alarmActive = false;
bool led1Active = false;
bool led2Active = false; 
bool led3Active = false;
const unsigned long ALARM_DURATION = 120000; // 120 seconds

unsigned long currentMS = 0;
unsigned long lastBlink = 0;
const unsigned long BLINK_INTERVAL = 500;
bool lowState = false;
int roomID;



void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN_1, OUTPUT);
  pinMode(LED_PIN_2, OUTPUT);
  pinMode(LED_PIN_3, OUTPUT);

  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN_1, LOW);
  digitalWrite(LED_PIN_2, LOW);
  digitalWrite(LED_PIN_3, LOW);

  Serial.println("Receiver ready!");
}

void resetAlarm() {
  switch(roomID) {
    case 1:
      led1Active = false;
      break;
    case 2:
      led2Active = false;
      break;
    case 3:
      led3Active = false;
      break;
  }

  if (!led1Active && !led2Active && !led3Active) {
    alarmActive = false;
    Serial.println("Alarm reset.");
  } else {
    Serial.println("Alarm still active in other rooms.");
  }
}


void triggerAlarm() {
  alarmActive = true;

  switch(roomID) {
    case 1:
      led1Active = true;
      Serial.printf("Alarm triggered from room %d\n", roomID);
      break;
    case 2:
      led2Active = true;
      Serial.printf("Alarm triggered from room %d\n", roomID);
      break;
    case 3:
      led3Active = true;
      Serial.printf("Alarm triggered from room %d\n", roomID);
      break;
  }
}

void processCommand() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');

    if (command.startsWith("ALERT: ")) {
      roomID = command.substring(7).toInt();
      
      triggerAlarm();
    }
    
    else if (command.startsWith("RESET: ")) {
      roomID = command.substring(7).toInt();
      Serial.printf("Reset command received from room %d\n", roomID);
      resetAlarm();
      // resetSignal = true;
    }
    
    else {
      Serial.println("UNKNOWN COMMAND: " + command + ".");
    }
    return;
  }
}


// unsigned long alarmStartTime = 0;
// void autoReset() {
//   if (alarmActive && (millis() - alarmStartTime > ALARM_DURATION)) {
//     Serial.println("AUTO RESET");
//     resetAlarm();
//   }
// }

///////////////////////////////////////////////////////////////

void loop() { 
  processCommand();
  // autoReset();
  if(alarmActive){
    currentMS = millis();
    if (currentMS - lastBlink >= BLINK_INTERVAL){
      lastBlink = currentMS;
      lowState = !lowState;
      digitalWrite(BUZZER_PIN, lowState ? HIGH : LOW);

      if (led1Active) {
        digitalWrite(LED_PIN_1, lowState ? HIGH : LOW);
      } else {
        digitalWrite(LED_PIN_1, LOW);
      }

      if (led2Active) {
        digitalWrite(LED_PIN_2, lowState ? HIGH : LOW);
      } else {
        digitalWrite(LED_PIN_2, LOW);
      }
      
      if (led3Active) {
        digitalWrite(LED_PIN_3, lowState ? HIGH : LOW);
      } else {
        digitalWrite(LED_PIN_3, LOW);
      }
    }
  } else {
    digitalWrite(BUZZER_PIN, LOW);
    digitalWrite(LED_PIN_1, LOW);
    digitalWrite(LED_PIN_2, LOW);
    digitalWrite(LED_PIN_3, LOW);
  }
  delay(50);
}