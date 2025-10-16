#include <Arduino.h>
#include "reset.h"

#define RESET_BUTTON 15

bool resetButtonPressed = false;

void setupResetButton() {
  pinMode(RESET_BUTTON, INPUT_PULLUP);
}

bool processResetButton() {
  if(digitalRead(RESET_BUTTON) == LOW) {
    Serial.println("Button pressed");
    resetButtonPressed = true;

    while (resetButtonPressed) {
      if (digitalRead(RESET_BUTTON) == HIGH) {
        resetButtonPressed = false;
        return true;
      }
      delay(50);
    }
  } else {
    return false;
  }
}
