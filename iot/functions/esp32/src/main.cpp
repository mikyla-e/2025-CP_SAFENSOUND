#include <Arduino.h>
#include "audio.h"
#include "reset.h"

// put function declarations here:
int myFunction(int, int);

void setup() {

  Serial.begin(115200);
  setupAudio();
  setupResetButton();

}

void loop() {

  processAudioRecording();
  // pressResetButton();

  delay(10);

}