#ifndef AUDIO_H
#define AUDIO_H

#include <Arduino.h>

void testMicrophoneHardware();
void checkClocksWhileReading();
void setupAudio();
void processAudioRecording();

extern void prepareAudio(int16_t* audio, size_t sampleCount);

#endif