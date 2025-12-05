#include "driver/i2s.h"
#include <WiFi.h>
#include "audio.h"
#include "esp_heap_caps.h"
#include <esp32-hal-psram.h>
#include "soc/i2s_reg.h"
#include "soc/gpio_sig_map.h"
#include "driver/gpio.h"
#include "driver/periph_ctrl.h"
#include <math.h>

#define I2S_PORT I2S_NUM_0
#define I2S_SCK 26 //BCLK
#define I2S_WS 25 //LRCLK
#define I2S_SD 32 //DOUT

static const int SAMPLE_RATE = 16000;
#define BUFFER_SIZE 1024

#define AUDIO_DURATION 5
#define CHUNK_SAMPLES 1600
#define CHUNKS_PER_RECORDING (SAMPLE_RATE * AUDIO_DURATION / CHUNK_SAMPLES)

static int16_t* full_audio = NULL;

extern void prepareAudio(int16_t* audio, size_t sampleCount);

// void printI2SRegisters() {
//     Serial.println("\n=== I2S Register Dump ===");
    
//     volatile uint32_t* I2S_CONF_REG = (volatile uint32_t*)0x3FF4F008;
//     volatile uint32_t* I2S_CLKM_CONF_REG = (volatile uint32_t*)0x3FF4F0AC;
//     volatile uint32_t* I2S_SAMPLE_RATE_CONF_REG = (volatile uint32_t*)0x3FF4F0B0;
//     volatile uint32_t* I2S_FIFO_CONF_REG = (volatile uint32_t*)0x3FF4F018;
    
//     Serial.printf("I2S_CONF_REG: 0x%08X\n", *I2S_CONF_REG);
//     Serial.printf("I2S_CLKM_CONF_REG: 0x%08X\n", *I2S_CLKM_CONF_REG);
//     Serial.printf("I2S_SAMPLE_RATE_CONF_REG: 0x%08X\n", *I2S_SAMPLE_RATE_CONF_REG);
//     Serial.printf("I2S_FIFO_CONF_REG: 0x%08X\n", *I2S_FIFO_CONF_REG);
    
//     bool rx_start = (*I2S_CONF_REG) & (1 << 11);
//     bool tx_start = (*I2S_CONF_REG) & (1 << 10);
//     Serial.printf("RX_START bit: %d, TX_START bit: %d\n", rx_start, tx_start);
// }

void setupAudio(){
    i2s_config_t cfg = {
		.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
		.sample_rate = SAMPLE_RATE,
		.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
		.channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
		.communication_format = I2S_COMM_FORMAT_STAND_I2S,
		.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
		.dma_buf_count = 8,
		.dma_buf_len = 256,
		.use_apll = true,
        .tx_desc_auto_clear = true,
        .fixed_mclk = SAMPLE_RATE * 256
	};

	i2s_pin_config_t pins = {
		.bck_io_num = I2S_SCK,
		.ws_io_num = I2S_WS,
		.data_out_num = I2S_PIN_NO_CHANGE,
		.data_in_num = I2S_SD
	};

    esp_err_t err;

	err = i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
	if (err != ESP_OK) {Serial.println(esp_err_to_name(err));};

	err = i2s_set_pin(I2S_PORT, &pins);
	if (err != ESP_OK) {Serial.println(esp_err_to_name(err));};

	err = i2s_set_clk(I2S_PORT, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_32BIT, I2S_CHANNEL_MONO);
	if (err != ESP_OK) {Serial.println(esp_err_to_name(err));};

	i2s_zero_dma_buffer(I2S_PORT);
    i2s_start(I2S_PORT);
}

static const int FRAME_LENGTH = 1024;
static const int HOP_LENGTH = 200;
static const float EMERGENCY_RMS_THRESHOLD = 1200.0f;
static const int32_t AMP_THRESHOLD = 1000;
static const float ACTIVITY_THRESHOLD = 600.0f;
static const float MIN_ACTIVE_RATIO = 0.15f;

static int32_t dc_offset = 0;
static const float DC_FILTER_ALPHA = 0.995f;

enum RecordingState {
    STATE_LISTENING,    // Waiting for amplitude trigger
    STATE_RECORDING     // Recording 5 seconds of audio
};

static RecordingState currentState = STATE_LISTENING;
static int chunks_sent = 0;

void removeDCOffset(int16_t* samples, size_t length) {
    for (size_t i = 0; i < length; i++) {
        dc_offset = (int32_t)(DC_FILTER_ALPHA * dc_offset + (1.0f - DC_FILTER_ALPHA) * samples[i]);
        samples[i] = samples[i] - (int16_t)dc_offset;
    }
}

float calculateAverageAmplitude(int16_t* samples, size_t length) {
    if (length == 0) return 0.0f;
    
    float sum = 0.0f;
    for (size_t i = 0; i < length; i++) {
        float val = abs(samples[i]);
        sum += val * val;
    }
    
    return sqrtf(sum / length);
}

bool hasAmplitude(int16_t* audio, size_t sampleCount) {
    float avgAmplitude = calculateAverageAmplitude(audio, sampleCount);
    
    int samplesAboveThreshold = 0;
    for (size_t i = 0; i < sampleCount; i++) {
        if (abs(audio[i]) > AMP_THRESHOLD) {
            samplesAboveThreshold++;
        }
    }
    
    float activeRatio = (float)samplesAboveThreshold / sampleCount;
    
    Serial.printf("Avg Amplitude: %.2f | Samples above threshold: %d/%d (%.1f%%)\n", 
                  avgAmplitude, samplesAboveThreshold, sampleCount, activeRatio * 100);
    
    bool shouldSend = (avgAmplitude > AMP_THRESHOLD) || 
                      (activeRatio > MIN_ACTIVE_RATIO);
    
    if (shouldSend) {
        Serial.println("Amplitude above threshold, will send data");
    } else {
        Serial.println("Amplitude below threshold, will NOT send data");
    }
    
    return shouldSend;
}


void processAudioRecording(){
    static int16_t audio_buffer[CHUNK_SAMPLES];
    static size_t samples_collected = 0;

    int32_t audio[BUFFER_SIZE];
    size_t bytes_read;


    esp_err_t result = i2s_read(I2S_PORT, audio, sizeof(audio), &bytes_read, portMAX_DELAY);
    if (result == ESP_OK && bytes_read > 0) {
        int samples_read = bytes_read / sizeof(int32_t);
        int idx = 0;

        while (idx < samples_read) {
            int space = CHUNK_SAMPLES - (int)samples_collected;
            int to_copy = min(space, samples_read - idx);

            for (int i = 0; i < to_copy; ++i) {
                const int SHIFT = 14;
                int32_t s32 = audio[idx + i] >> SHIFT;
                if (s32 > 32767) s32 = 32767;
                if (s32 < -32768) s32 = -32768;

                int16_t s16 = (int16_t)s32;
                audio_buffer[samples_collected + i] = s16;
            }
            
            samples_collected += to_copy;
            idx += to_copy;

            if (samples_collected == CHUNK_SAMPLES) {
                removeDCOffset(audio_buffer, CHUNK_SAMPLES);
                switch (currentState) {
                    case STATE_LISTENING: {
                        bool triggered = hasAmplitude(audio_buffer, CHUNK_SAMPLES);
                        
                        if (triggered) {
                            Serial.println("\nTriggered: Start 5s recording");
                            currentState = STATE_RECORDING;
                            chunks_sent = 0;
                            
                            prepareAudio(audio_buffer, CHUNK_SAMPLES);
                            chunks_sent++;
                            Serial.printf("Sent chunk %d/%d\n", chunks_sent, CHUNKS_PER_RECORDING);
                        }
                        break;
                    }
                    
                    case STATE_RECORDING: {
                        // Send all chunks for the 5-second recording period
                        prepareAudio(audio_buffer, CHUNK_SAMPLES);
                        chunks_sent++;
                        Serial.printf("Sent chunk %d/%d\n", chunks_sent, CHUNKS_PER_RECORDING);
                        
                        // Check if 5 seconds complete
                        if (chunks_sent >= CHUNKS_PER_RECORDING) {
                            Serial.println("Finished 5s recording\n");
                            currentState = STATE_LISTENING;
                            chunks_sent = 0;
                        }
                        break;
                    }
                }

                samples_collected = 0;

            }
        }
    }
}