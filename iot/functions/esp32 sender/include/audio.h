#ifndef AUDIO_H
#define AUDIO_H

#include <Arduino.h>

void setupAudio();
void processAudioRecording();

extern void prepareAudio(float avg, float rms, int16_t* audio, size_t sampleCount);

#endif