#include "driver/i2s.h"
#include <WiFi.h>
#include "audio.h"
#include "esp_heap_caps.h"

#define I2S_PORT I2S_NUM_0
#define I2S_SCK 26 //BCLK
#define I2S_WS 25 //LRCLK
#define I2S_SD 32 //DOUT

#define AUDIO_DURATION 5
#define SAMPLE_RATE 16000
#define TOTAL_SAMPLES (SAMPLE_RATE * AUDIO_DURATION)
#define BUFFER_SIZE 256

#define MIC_CHANNEL_RIGHT 0 // Set to 1 if using right channel, 0 for left channel

extern void prepareAudio(int16_t* audio, size_t sampleCount);

i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 1024,
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

void debug_print_raw() {
  const int WORDS = 32;
    int32_t audio_buf[WORDS];
    size_t br = 0;
    esp_err_t res = i2s_read(I2S_PORT, audio_buf, sizeof(audio_buf), &br, 200 / portTICK_PERIOD_MS);
    Serial.printf("i2s_read res=%d bytes=%u words=%u\n", (int)res, (unsigned)br, (unsigned)(br/4));

    if (res == ESP_OK && br > 0) {
        int words = br / sizeof(int32_t);
        Serial.printf("raw %d words (pairs shown as L,R):\n", words);
        
        // Calculate RMS for signal strength indication
        int64_t sum = 0;
        int valid_samples = 0;
        
        for (int i = 0; i < words; i += 2) {
            uint32_t L = (uint32_t)audio_buf[i];
            uint32_t R = (i+1 < words) ? (uint32_t)audio_buf[i+1] : 0;
            
            // Check which channel has data (some mics use only one channel)
            int32_t left_sample = (int32_t)L;
            int32_t right_sample = (int32_t)R;
            
            if (abs(left_sample) > 1000 || abs(right_sample) > 1000) {
                valid_samples++;
                sum += (int64_t)left_sample * left_sample + (int64_t)right_sample * right_sample;
            }
            
            Serial.printf("[%02d] L=0x%08X %d   R=0x%08X %d\n", 
                            i/2, L, left_sample, R, right_sample);
        }
        
        if (valid_samples > 0) {
            double rms = sqrt((double)sum / valid_samples);
            Serial.printf("RMS Signal Level: %.2f\n", rms);
        }
    } else {
        Serial.println("No data or read failed.");
    }
}

void print_pin_levels() {
  int sck = gpio_get_level((gpio_num_t)I2S_SCK);
  int ws  = gpio_get_level((gpio_num_t)I2S_WS);
  int sd  = gpio_get_level((gpio_num_t)I2S_SD);
  Serial.printf("GPIO levels SCK=%d WS=%d SD=%d\n", sck, ws, sd);
}

void verify_i2s_clocks() {
    // Set up interrupt to check if clock is toggling
    static volatile int sck_toggles = 0;
    static volatile int ws_toggles = 0;
    
    Serial.println("Checking I2S clock signals for 100ms...");
    
    // Sample the pins rapidly
    int last_sck = gpio_get_level((gpio_num_t)I2S_SCK);
    int last_ws = gpio_get_level((gpio_num_t)I2S_WS);
    
    unsigned long start = millis();
    while (millis() - start < 100) {
        int cur_sck = gpio_get_level((gpio_num_t)I2S_SCK);
        int cur_ws = gpio_get_level((gpio_num_t)I2S_WS);
        
        if (cur_sck != last_sck) {
            sck_toggles++;
            last_sck = cur_sck;
        }
        if (cur_ws != last_ws) {
            ws_toggles++;
            last_ws = cur_ws;
        }
        delayMicroseconds(10);
    }
    
    Serial.printf("Clock toggles detected - SCK: %d, WS: %d\n", sck_toggles, ws_toggles);
    
    if (sck_toggles < 100) {
        Serial.println("WARNING: SCK clock not toggling properly!");
    }
    if (ws_toggles < 10) {
        Serial.println("WARNING: WS clock not toggling properly!");
    }
}

void test_pin_output() {
    // Temporarily configure as GPIO output
    pinMode(I2S_SCK, OUTPUT);
    pinMode(I2S_WS, OUTPUT);
    pinMode(I2S_SD, INPUT);
    
    Serial.println("Testing pin output capability:");
    for (int i = 0; i < 5; i++) {
        digitalWrite(I2S_SCK, HIGH);
        digitalWrite(I2S_WS, HIGH);
        delay(100);
        print_pin_levels();
        
        digitalWrite(I2S_SCK, LOW);
        digitalWrite(I2S_WS, LOW);
        delay(100);
        print_pin_levels();
    }
}

void setupAudio(){
    gpio_reset_pin((gpio_num_t)I2S_SCK);
    gpio_reset_pin((gpio_num_t)I2S_WS);
    gpio_reset_pin((gpio_num_t)I2S_SD);
    
    esp_err_t result = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    if (result != ESP_OK) {
        Serial.printf("Error installing I2S driver: %s (0x%x)\n", esp_err_to_name(result), result);
        return;
    } else {
        Serial.println("I2S driver installed.");
    }

    result = i2s_set_pin(I2S_PORT, &pin_config);
    if (result != ESP_OK) {
        Serial.printf("Error setting I2S pins: %s (0x%x)\n", esp_err_to_name(result), result);
        return;
    } else {
        Serial.println("I2S pin set.");
    }

    result = i2s_set_clk(I2S_PORT, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_32BIT, I2S_CHANNEL_MONO);
    if (result != ESP_OK) {
        Serial.printf("Error setting I2S clock: %s (0x%x)\n", esp_err_to_name(result), result);
        return;
    } else {
        Serial.println("I2S clock set.");
    }

    i2s_zero_dma_buffer(I2S_PORT);
    i2s_start(I2S_PORT);
    delay(100);

    verify_i2s_clocks();
    print_pin_levels();

    size_t bytes_read = 0;

    int32_t discard[1024];
    size_t br = 0;
    for (int i = 0; i < 15; ++i) {
        result = i2s_read(I2S_PORT, discard, sizeof(discard), &br, 200 / portTICK_PERIOD_MS);
        Serial.printf("Warm read %d: res=%d bytes=%u\n", i, (int)result, (unsigned)br);
        
        if (br > 0 && i > 5) {
            Serial.printf("  Sample data: 0x%08X 0x%08X 0x%08X\n", 
                         (uint32_t)discard[0], (uint32_t)discard[1], (uint32_t)discard[2]);
        }
        delay(20);
    }

    print_pin_levels();
    Serial.println("I2S microphone ready.");

    debug_print_raw();
}

#define CHUNK_SAMPLES 16000
#define CHUNKS_PER_RECORDING 5

void processAudioRecording(){
    static int16_t audio_buffer[CHUNK_SAMPLES];
    static size_t samples_collected = 0;
    static int chunk_count = 0;

    int32_t audio[BUFFER_SIZE];
    size_t bytes_read;

    esp_err_t result = i2s_read(I2S_PORT, audio, sizeof(audio), &bytes_read, portMAX_DELAY);
    if (result == ESP_OK && bytes_read > 0) {
        int samples_read = bytes_read / sizeof(int32_t);
    
        for (int i = 0; i < samples_read && samples_collected < CHUNK_SAMPLES; i++) {
            int32_t s32 = audio[i] >> 8;
            if (s32 > 32767) s32 = 32767;
            if (s32 < -32768) s32 = -32768;
            audio_buffer[samples_collected++] = (int16_t)s32;
        }

        if (samples_collected >= CHUNK_SAMPLES) {
            prepareAudio(audio_buffer, CHUNK_SAMPLES);
            
            samples_collected = 0;
            chunk_count++;
            
            if (chunk_count >= CHUNKS_PER_RECORDING) {
                Serial.println("Full 5-second recording sent in chunks.");
                chunk_count = 0;
            }
        }
    }
}