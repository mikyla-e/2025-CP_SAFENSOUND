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

#define CHUNK_DURATION 1
#define CHUNK_SAMPLES (SAMPLE_RATE * CHUNK_DURATION)
#define TOTAL_CHUNKS 5

static int16_t* full_audio = NULL;

extern void sendData(int16_t* audio, size_t sampleCount, int chunkIndex, int totalChunks);

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
}

static const int FRAME_LENGTH = 1024;
static const int HOP_LENGTH = 200;
static const float ACTIVITY_THRESHOLD_MS = 400.0f;
static float EMERGENCY_RMS_THRESHOLD = 400.0f;

static int16_t* chunk_buffer = NULL;
static size_t chunk_samples_recorded = 0;
static int current_chunk = 0;
static uint32_t recording_session_id = 0;

float analyzeAudioActivity(int16_t* audio_data, size_t sample_count) {
    // Convert to float and normalize
    float* y = (float*)malloc(sample_count * sizeof(float));
    if (!y) {
        Serial.println("Failed to allocate float buffer");
        return 0.0f;
    }
    
    // Convert to float and remove DC offset
    float mean = 0.0f;
    for (size_t i = 0; i < sample_count; i++) {
        y[i] = (float)audio_data[i] / 32768.0f;
        mean += y[i];
    }
    mean /= sample_count;
    
    for (size_t i = 0; i < sample_count; i++) {
        y[i] -= mean;
    }
    
    // Calculate RMS for each frame
    int num_frames = (sample_count - FRAME_LENGTH) / HOP_LENGTH + 1;
    float* rms = (float*)malloc(num_frames * sizeof(float));
    if (!rms) {
        free(y);
        Serial.println("Failed to allocate RMS buffer");
        return 0.0f;
    }
    
    for (int frame = 0; frame < num_frames; frame++) {
        int start = frame * HOP_LENGTH;
        float sum_sq = 0.0f;
        
        for (int i = 0; i < FRAME_LENGTH && (start + i) < sample_count; i++) {
            float val = y[start + i];
            sum_sq += val * val;
        }
        
        rms[frame] = sqrtf(sum_sq / FRAME_LENGTH);
    }

    // Convert RMS to dB
    float* rms_db = (float*)malloc(num_frames * sizeof(float));
    if (!rms_db) {
        free(y);
        free(rms);
        Serial.println("Failed to allocate RMS_dB buffer");
        return 0.0f;
    }
    
    for (int i = 0; i < num_frames; i++) {
        rms_db[i] = 20.0f * log10f(fmaxf(rms[i], 1e-6f));
    }

    // Calculate noise floor (30th percentile)
    float noise_floor = calculatePercentile(rms_db, num_frames, 30.0f);
    float margin_db = 8.0f;
    float threshold = noise_floor + margin_db;

    // Count active frames
    int active_count = 0;
    for (int i = 0; i < num_frames; i++) {
        if (rms_db[i] > threshold) {
            active_count++;
        }
    }

    // Calculate activity duration in milliseconds
    float active_ms = active_count * (HOP_LENGTH / (float)SAMPLE_RATE) * 1000.0f;

    // Cleanup
    free(y);
    free(rms);
    free(rms_db);
    
    return active_ms;
}

float calculatePercentile(float* data, int size, float percentile) {
    // Simple percentile calculation using sorting
    float* sorted = (float*)malloc(size * sizeof(float));
    if (!sorted) return 0.0f;
    
    memcpy(sorted, data, size * sizeof(float));
    
    // Bubble sort (simple for small arrays)
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (sorted[j] > sorted[j + 1]) {
                float temp = sorted[j];
                sorted[j] = sorted[j + 1];
                sorted[j + 1] = temp;
            }
        }
    }
    
    int index = (int)((percentile / 100.0f) * size);
    if (index >= size) index = size - 1;
    
    float result = sorted[index];
    free(sorted);
    
    return result;
}

void processAudioRecording(){
    static int16_t audio_buffer[BUFFER_SIZE];
    int32_t audio[BUFFER_SIZE];
    size_t bytes_read;

     if (chunk_buffer == NULL) {
        chunk_buffer = (int16_t*)malloc(CHUNK_SAMPLES * sizeof(int16_t));
        if (!chunk_buffer) {
            Serial.println("Failed to allocate recording buffer");
            return;
        }
        Serial.printf("Allocated chunk buffer of %d samples (%d bytes)\n", CHUNK_SAMPLES, CHUNK_SAMPLES * sizeof(int16_t));
    }

    esp_err_t result = i2s_read(I2S_PORT, audio, sizeof(audio), &bytes_read, portMAX_DELAY);
    if (result == ESP_OK && bytes_read > 0) {
        int samples_read = bytes_read / sizeof(int32_t);

        for (int i = 0; i < samples_read; ++i) {
            const int SHIFT = 14;
            int32_t s32 = audio[i] >> SHIFT;
            if (s32 > 32767) s32 = 32767;
            if (s32 < -32768) s32 = -32768;
            audio_buffer[i] = s32;
        }

         if (chunk_samples_recorded < CHUNK_SAMPLES) {
            size_t to_copy = min((size_t)samples_read, (size_t)(CHUNK_SAMPLES - chunk_samples_recorded));
            memcpy(chunk_buffer + chunk_samples_recorded, audio_buffer, to_copy * sizeof(int16_t));
            chunk_samples_recorded += to_copy;
        }

        if (chunk_samples_recorded >= CHUNK_SAMPLES) {

            if (current_chunk == 0) {
                float activity_ms = analyzeAudioActivity(chunk_buffer, chunk_samples_recorded);
                 Serial.printf("Chunk %d Activity=%.0f ms\n", current_chunk, activity_ms);
            
                if (activity_ms < ACTIVITY_THRESHOLD_MS) {
                    Serial.println("Skipping recording");
                    chunk_samples_recorded = 0;
                    current_chunk = 0;
                    return;
                }
                
                // Reset for next recording
                recording_session_id = millis();
                Serial.println("Activity detected!");
            }            

            sendData(chunk_buffer, chunk_samples_recorded, current_chunk, TOTAL_CHUNKS);
            current_chunk++;
            chunk_samples_recorded = 0;

            if (current_chunk >= TOTAL_CHUNKS) {
                current_chunk = 0;
                Serial.println("Recording session complete");
            }

        }
        
    }
}