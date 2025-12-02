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

// #define I2S_SCK 14 //BCLK
// #define I2S_WS 15 //LRCLK
// #define I2S_SD 32 //DOUT

static const int SAMPLE_RATE = 16000;
#define BUFFER_SIZE 1024

#define AUDIO_DURATION 5
#define CHUNK_SAMPLES 1600
#define CHUNKS_PER_RECORDING (SAMPLE_RATE * AUDIO_DURATION / CHUNK_SAMPLES)

static int16_t* full_audio = NULL;

extern void prepareAudio(int16_t* audio, size_t sampleCount);

static int16_t* allocateFullAudio(size_t samples) {
    size_t bytes = samples * sizeof(int16_t);
    size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t free_dram  = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    Serial.printf("Alloc full_audio (%u bytes). Free PSRAM=%u, DRAM=%u\n",
                  (unsigned)bytes, (unsigned)free_psram, (unsigned)free_dram);

    int16_t* ptr = nullptr;

    if (psramFound() && free_psram >= bytes) {
        ptr = (int16_t*)ps_malloc(bytes);
        if (ptr) {
            Serial.println("full_audio allocated in PSRAM");
            return ptr;
        }
    }

    if (free_dram >= bytes) {
        ptr = (int16_t*)heap_caps_malloc(bytes, MALLOC_CAP_8BIT); // DRAM
        if (ptr) {
            Serial.println("full_audio allocated in DRAM");
            return ptr;
        }
    }

    Serial.println("full_audio allocation failed");
    return nullptr;
}

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

    size_t rb_samples = (size_t)SAMPLE_RATE * (size_t)AUDIO_DURATION;
    if (!full_audio) {
        full_audio = allocateFullAudio(rb_samples);
        if (full_audio) {
            memset(full_audio, 0, rb_samples * sizeof(int16_t));
        }
    }
}

static const int FRAME_LENGTH = 1024;
static const int HOP_LENGTH = 200;
static const float ACTIVITY_THRESHOLD_MS = 600.0f;
static float EMERGENCY_RMS_THRESHOLD = 400.0f;

static float dc_mean = 0.0f;
static const float DC_ALPHA = 0.001f;
static float noise_floor_rms = 0.0f;
static const float NOISE_ALPHA = 0.001f;
static const float RMS_MARGIN_MULT = 3.0f;

static int16_t frame_buffer[FRAME_LENGTH];
static int frame_fill = 0;
static int active_frames = 0;
static int total_frames = 0;
static bool activity_confirmed = false;

static const size_t RING_BUFFER_SAMPLES = (size_t)SAMPLE_RATE * (size_t)AUDIO_DURATION;
static size_t rb_write_idx = 0;
static size_t rb_filled = 0;
static bool activity_triggered = false;

static bool flush_in_progress = false;
static size_t flush_read_idx = 0;
static size_t flush_samples_remaining = 0;

float computeRMS(int16_t* buffer, int length) {
    double sum_sq = 0.0;
    for (int i = 0; i < length; i++) {
        float x = (float)buffer[i];
        dc_mean = dc_mean + DC_ALPHA * (x - dc_mean);
        float y = x -dc_mean;
        sum_sq += (double)y * (double)y;
    }
    float rms = sqrtf((float)(sum_sq / (double)length));

    noise_floor_rms = noise_floor_rms + NOISE_ALPHA * (rms - noise_floor_rms);
    EMERGENCY_RMS_THRESHOLD = max(300.0f, noise_floor_rms * RMS_MARGIN_MULT);
    return rms;
}

static void pushSample(int16_t sample) {
    frame_buffer[frame_fill++] = sample;

    if (frame_fill >= FRAME_LENGTH) {
        float rms = computeRMS(frame_buffer, FRAME_LENGTH);
        if (rms > EMERGENCY_RMS_THRESHOLD) {
            active_frames++;
        }

        total_frames++;
        float per_frame_ms = (float)HOP_LENGTH / (float)SAMPLE_RATE * 1000.0f;
        float active_ms = (float)active_frames * per_frame_ms;
        if (!activity_confirmed && active_ms >= ACTIVITY_THRESHOLD_MS) {
            activity_confirmed = true;
            Serial.printf("Activity >= %.1f ms detected (per-frame=%.2f ms, frames=%d)\n", ACTIVITY_THRESHOLD_MS, per_frame_ms, active_frames);
        }

        memmove(frame_buffer, frame_buffer + HOP_LENGTH, (FRAME_LENGTH - HOP_LENGTH) * sizeof(int16_t));
        frame_fill = FRAME_LENGTH - HOP_LENGTH;
    }
}

static void beginAudioFlush() {
    if (!full_audio) return;

    flush_in_progress = true;
    flush_read_idx = rb_write_idx % RING_BUFFER_SAMPLES;
    flush_samples_remaining = RING_BUFFER_SAMPLES;

    Serial.printf("Starting audio flush: %u samples (%ds)\n", (unsigned)flush_samples_remaining, AUDIO_DURATION);
}

static void sendNextFlushChunk() {
    if (!flush_in_progress || !full_audio || flush_samples_remaining == 0) return;

    static int16_t chunk[CHUNK_SAMPLES];
    size_t to_send = min((size_t)CHUNK_SAMPLES, flush_samples_remaining);
    for (size_t i = 0; i < to_send; ++i) {
        chunk[i] = full_audio[(flush_read_idx + i) % RING_BUFFER_SAMPLES];
    }

    prepareAudio(chunk, to_send);

    flush_samples_remaining -= to_send;
    if (flush_samples_remaining == 0) {
        Serial.println("Audio flush completed.");
        flush_in_progress = false;
        active_frames = 0;
        total_frames = 0;
        frame_fill = 0;
        activity_confirmed = false;
    }
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
    int32_t audio[BUFFER_SIZE];
    size_t bytes_read;

    bool prepared_audio = false;

    esp_err_t result = i2s_read(I2S_PORT, audio, sizeof(audio), &bytes_read, portMAX_DELAY);
    if (result == ESP_OK && bytes_read > 0) {
        int samples_read = bytes_read / sizeof(int32_t);

        for (int idx = 0; idx < samples_read; ++idx) {
            const int SHIFT = 14;
            int32_t s32 = audio[idx] >> SHIFT;
            if (s32 > 32767) s32 = 32767;
            if (s32 < -32768) s32 = -32768;
            int16_t s16 = (int16_t)s32;

            pushSample(s16);
            if (full_audio) {
                full_audio[rb_write_idx] = s16;
                rb_write_idx = (rb_write_idx + 1) % RING_BUFFER_SAMPLES;
                if (rb_filled < RING_BUFFER_SAMPLES) {
                    rb_filled++;
                }
            }

        }
    }

    if (activity_confirmed && !activity_triggered) {
        activity_triggered = true;
        Serial.println("Audio activity confirmed, beginning flush...");
        beginAudioFlush();
    }

    if (flush_in_progress && !prepared_audio) {
        sendNextFlushChunk();
        prepared_audio = true;
    }
}