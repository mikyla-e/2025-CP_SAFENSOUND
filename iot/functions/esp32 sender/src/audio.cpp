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

void processAudioRecording(){
    static int16_t audio_buffer[CHUNK_SAMPLES];
    static size_t samples_collected = 0;
    static int samples_sent = 0;
    static int64_t sum_abs = 0;
    static int64_t sum_sq = 0;

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

                sum_abs += (uint16_t)abs(s16);
                sum_sq += (int32_t)s16 * (int32_t)s16;
            }

            samples_collected += to_copy;
            idx += to_copy;

            if (samples_collected == CHUNK_SAMPLES) {
                const float EMERGENCY_RMS_THRESHOLD = 8000.0;

                samples_sent += samples_collected;
                uint16_t avg_amplitude = (float)sum_abs / (float)samples_sent;
                uint16_t rms_amplitude = sqrt((float)sum_sq / (float)samples_sent);

                if (rms_amplitude > EMERGENCY_RMS_THRESHOLD) {
                    prepareAudio(avg_amplitude, rms_amplitude, audio_buffer, CHUNK_SAMPLES);

                    samples_collected = 0;

                    if (samples_sent >= SAMPLE_RATE * AUDIO_DURATION) {
                        Serial.printf("5s metrics: avg_abs=%.2f rms=%.2f\n", avg_amplitude, rms_amplitude);
                        Serial.println("Full 5-second recording sent in chunks.");
                        
                        samples_sent = 0;
                        sum_abs = 0;
                        sum_sq = 0;
                    }
                } else {
                    samples_collected = 0;
                    samples_sent = 0;
                    sum_abs = 0;
                    sum_sq = 0;
                    Serial.printf("Chunk skipped due to low RMS: %.2f\n", rms_amplitude);
                }
            }
        }
    }
}