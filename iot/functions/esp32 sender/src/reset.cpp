#include <Arduino.h>
#include "reset.h"

#define RESET_BUTTON 21

bool resetButtonPressed = false;

void setupResetButton() {
  pinMode(RESET_BUTTON, INPUT_PULLUP);
}

bool processResetButton() {
  return digitalRead(RESET_BUTTON) == LOW ? true : false;
}
