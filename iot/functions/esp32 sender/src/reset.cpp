#include <Arduino.h>

#define RESET_BUTTON 21

bool resetButtonPressed = false;

void setupResetButton() {
  pinMode(RESET_BUTTON, INPUT_PULLUP);
}

void pressResetButton() {
  if (digitalRead(RESET_BUTTON) == LOW) {

    delay(50);

    if (!resetButtonPressed) {
      resetButtonPressed = true;
      Serial.println("Reset button pressed");

    }

  } else {
    resetButtonPressed = false;
  }
}
