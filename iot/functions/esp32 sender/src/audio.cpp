#include "driver/i2s.h"
#include <WiFi.h>
#include "audio.h"
#include "esp_heap_caps.h"

#define I2S_PORT I2S_NUM_0
#define I2S_SCK 26
#define I2S_WS 25
#define I2S_SD 32

#define AUDIO_DURATION 2
#define SAMPLE_RATE 16000
#define TOTAL_SAMPLES (SAMPLE_RATE * AUDIO_DURATION)
#define BUFFER_SIZE 160

extern void prepareAudio(int16_t* audio, size_t sampleCount);

i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = BUFFER_SIZE,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
};


i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
};

void setupAudio(){
    esp_err_t result = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    if (result != ESP_OK) {
        Serial.printf("Error installing I2S driver: %d\n", result);
        return;
    } else {
        Serial.println("I2S driver installed.");
    }

    result = i2s_set_pin(I2S_PORT, &pin_config);
    if (result != ESP_OK) {
        Serial.printf("Error setting I2S pin: %d\n", result);
        return;
    } else {
        Serial.println("I2S pin set.");
    }
    
    int16_t test_buffer[BUFFER_SIZE];
    size_t bytes_read = 0;

    Serial.println("Checking microphone connection...");
    result = i2s_read(I2S_PORT, test_buffer, sizeof(test_buffer), &bytes_read, 100 / portTICK_PERIOD_MS); // 100ms timeout

    if (result != ESP_OK || bytes_read == 0) {
        Serial.println("Error: No data read from microphone. Please check the connection.");
        return;
    }

    while(true) {
        result = i2s_read(I2S_PORT, test_buffer, sizeof(test_buffer), &bytes_read, 100 / portTICK_PERIOD_MS); // 100ms timeout

        if (result == ESP_OK && bytes_read > 0) {
            bool valid_data = false;
            for (int i = 0; i < bytes_read / sizeof(int16_t); i++) {
            if (test_buffer[i] != 0) {
                    valid_data = true;
                    break;
                }
            }

            if (valid_data) {
                Serial.println("Microphone connected and ready.");
                break;
            }
        } 
        Serial.println("Waiting for valid microphone data...");
        delay(1000);
    }

}

void processAudioRecording(){
    static int16_t audio_buffer[TOTAL_SAMPLES];

    static size_t samples_collected = 0;

    int16_t audio[BUFFER_SIZE];
    size_t bytes_read;

    esp_err_t result = i2s_read(I2S_PORT, audio, sizeof(audio), &bytes_read, portMAX_DELAY);
    if (result == ESP_OK && bytes_read > 0) {
        int samples_read = bytes_read / sizeof(int16_t);
        Serial.printf("Read %d samples from I2S\n", samples_read);
    
        for (int i = 0; i < samples_read && samples_collected < TOTAL_SAMPLES; i++) {
            Serial.printf("%d ", audio[i]);
            audio_buffer[samples_collected++] = audio[i];
        }

        if (samples_collected >= TOTAL_SAMPLES) {
            Serial.println("Audio recording complete.");

            long sum = 0;
            for (size_t i = 0; i < TOTAL_SAMPLES; i++) {
                sum += abs(audio_buffer[i]);
            }
            int average_amplitude = sum / TOTAL_SAMPLES;
            Serial.printf("Average amplitude: %d\n", average_amplitude);

            samples_collected = 0;
            prepareAudio(audio_buffer, TOTAL_SAMPLES);
        }
    }

    delay(100);
}