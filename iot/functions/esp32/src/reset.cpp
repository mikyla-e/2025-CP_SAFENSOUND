#include <Arduino.h>

#define RESET_BUTTON 2

bool resetButtonPressed = false;

void setup() {
  pinMode(RESET_BUTTON, INPUT_PULLUP);
}

