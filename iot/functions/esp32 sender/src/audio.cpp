#include "driver/i2s.h"
#include <WiFi.h>
#include "audio.h"
#include "esp_heap_caps.h"
#include "soc/i2s_reg.h"
#include "soc/gpio_sig_map.h"
#include "driver/gpio.h"
#include "driver/periph_ctrl.h"

#define I2S_PORT I2S_NUM_0
#define I2S_SCK 26 //BCLK
#define I2S_WS 25 //LRCLK
#define I2S_SD 32 //DOUT

// #define I2S_SCK 14 //BCLK
// #define I2S_WS 15 //LRCLK
// #define I2S_SD 32 //DOUT

#define SAMPLE_RATE 16000
#define BUFFER_SIZE 1024

#define AUDIO_DURATION 5
#define CHUNK_SAMPLES 1600
#define CHUNKS_PER_RECORDING (SAMPLE_RATE * AUDIO_DURATION / CHUNK_SAMPLES)

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
    // Serial.println("\n=== I2S MICROPHONE INITIALIZATION ===");
    
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

    // printI2SRegisters();
    // Serial.println("\nForcing initial read...");
    // int32_t dummy[256];
    // size_t bytes_read;
    // for(int i = 0; i < 5; i++) {
    //     esp_err_t result = i2s_read(I2S_PORT, dummy, sizeof(dummy), &bytes_read, 1000);
    //     Serial.printf("Read %d: %s, %u bytes, first sample: 0x%08X\n", 
    //                  i, esp_err_to_name(result), bytes_read, dummy[0]);
    //     delay(100);
    // }
}

#define FRAME_LENGTH 1024
#define HOP_LENGTH 200
#define ACTIVITY_THRESHOLD_MS 600
// #define NOISE_PERCENTILE 30
// #define MARGIN_DB 8.0
#define EMERGENCY_RMS_THRESHOLD 8000.0

float computeRMS(int16_t* buffer, int length) {
    int64_t sum = 0;
    for (int i = 0; i < length; i++) {
        int32_t sample = buffer[i];
        sum += sample * sample;
    }
    return sqrt((float)sum / (float)length);
}

bool checkAudio(int16_t* full_recording, size_t total_samples) {
    const int num_frames = (total_samples - FRAME_LENGTH) / HOP_LENGTH + 1;
    if (num_frames <= 0) return false;

    int active_frames = 0;

    for (int i = 0; i< num_frames; i++) {
        int start = i * HOP_LENGTH;
        float rms = computeRMS(&full_recording[start], FRAME_LENGTH);

        if (rms > EMERGENCY_RMS_THRESHOLD) {
            active_frames++;
        }
    }

    float active_ms = (float)active_frames * (float)HOP_LENGTH / (float)SAMPLE_RATE * 1000.0;

    return (active_ms >= ACTIVITY_THRESHOLD_MS);
}

void processAudioRecording(){
    static int16_t audio_buffer[CHUNK_SAMPLES];
    static int16_t* full_audio = NULL;
    static size_t samples_collected = 0;
    static int total_samples = 0;
    static int chunks_sent = 0;
    static bool recording_checked = false;

    if (full_audio == NULL) {
        full_audio = (int16_t*)heap_caps_malloc((SAMPLE_RATE * AUDIO_DURATION) * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!full_audio) {
            Serial.println("Failed to allocate full_audio buffer!");
            return;
        }
    }

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

                if(total_samples < (SAMPLE_RATE * AUDIO_DURATION)) {
                    full_audio[total_samples + i] = s16;
                }
            }

            samples_collected += to_copy;
            total_samples += to_copy;
            idx += to_copy;

            if (total_samples == SAMPLE_RATE * AUDIO_DURATION && !recording_checked) {
                recording_checked = checkAudio(full_audio, total_samples) ? true : false;

                if (recording_checked) {
                    Serial.println("Activity detected, sending audio chunks!");
                } else {
                    Serial.println("No significant activity detected.");
                }
            } 

            if (samples_collected == CHUNK_SAMPLES) {
                if (recording_checked) {
                    prepareAudio(audio_buffer, CHUNK_SAMPLES);
                    chunks_sent++;
                    Serial.printf("Sent chunk %d/%d\n", chunks_sent, CHUNKS_PER_RECORDING);
                }
                samples_collected = 0;

                if (chunks_sent >= CHUNKS_PER_RECORDING || (total_samples >= SAMPLE_RATE * AUDIO_DURATION && !recording_checked)) {
                    samples_collected = 0;
                    total_samples = 0;
                    chunks_sent = 0;
                    recording_checked = false;
                    Serial.printf("Completed processing audio recording.\n");
                }
            }
        }
    }
}