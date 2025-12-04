#ifndef AUDIO_H
#define AUDIO_H

#include <Arduino.h>

void setupAudio();
void processAudioRecording();
float calculateAverageAmplitude(int16_t* samples, size_t length);
bool hasAmplitude(int16_t* audio, size_t sampleCount);


extern void prepareAudio(int16_t* audio, size_t sampleCount);

#endif