#ifndef AUDIO_H
#define AUDIO_H

#include <Arduino.h>

void setupAudio();
void processAudioRecording();
float analyzeAudioActivity(int16_t* audio_data, size_t sample_count);
float calculatePercentile(float* data, int size, float percentile);

void prepareAudio(int16_t* audio, size_t sampleCount);

#endif