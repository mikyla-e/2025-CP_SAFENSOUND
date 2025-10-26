#include "driver/i2s.h"
#include <WiFi.h>
#include "audio.h"
#include "esp_heap_caps.h"
#include "soc/i2s_reg.h"
#include "driver/periph_ctrl.h"

#define I2S_PORT I2S_NUM_0
#define I2S_SCK 26 //BCLK
#define I2S_WS 25 //LRCLK
#define I2S_SD 32 //DOUT

// #define I2S_SCK 14 //BCLK
// #define I2S_WS 15 //LRCLK
// #define I2S_SD 32 //DOUT

#define AUDIO_DURATION 5
#define SAMPLE_RATE 16000
#define TOTAL_SAMPLES (SAMPLE_RATE * AUDIO_DURATION)
#define BUFFER_SIZE 256

extern void prepareAudio(int16_t* audio, size_t sampleCount);

i2s_config_t i2s_config = {
    mode: (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    sample_rate: SAMPLE_RATE,
    bits_per_sample: I2S_BITS_PER_SAMPLE_32BIT,
    channel_format: I2S_CHANNEL_FMT_ONLY_LEFT,
    communication_format: I2S_COMM_FORMAT_STAND_I2S,
    intr_alloc_flags: ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 1024,
    .use_apll = true,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
};

i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
};

// void debugPrintRaw() {
//   const int WORDS = 32;
//     int32_t audio_buf[WORDS];
//     size_t br = 0;
//     esp_err_t res = i2s_read(I2S_PORT, audio_buf, sizeof(audio_buf), &br, 200 / portTICK_PERIOD_MS);
//     Serial.printf("i2s_read res=%d bytes=%u words=%u\n", (int)res, (unsigned)br, (unsigned)(br/4));

//     if (res == ESP_OK && br > 0) {
//         int words = br / sizeof(int32_t);
//         Serial.printf("raw %d words (pairs shown as L,R):\n", words);
        
//         // Calculate RMS for signal strength indication
//         int64_t sum = 0;
//         int valid_samples = 0;
        
//         for (int i = 0; i < words; i += 2) {
//             uint32_t L = (uint32_t)audio_buf[i];
//             uint32_t R = (i+1 < words) ? (uint32_t)audio_buf[i+1] : 0;
            
//             // Check which channel has data (some mics use only one channel)
//             int32_t left_sample = (int32_t)L;
//             int32_t right_sample = (int32_t)R;
            
//             if (L != 0xFFFFFFFF && R != 0xFFFFFFFF) {
//                 valid_samples++;
//                 sum += (int64_t)left_sample * left_sample;
//             }
            
//             Serial.printf("[%02d] L=0x%08X %d   R=0x%08X %d\n", 
//                             i/2, L, left_sample, R, right_sample);
//         }
        
//         if (valid_samples > 0) {
//             double rms = sqrt((double)sum / valid_samples);
//             Serial.printf("RMS Signal Level: %.2f\n", rms);
//         } else {
//             Serial.println("No valid samples detected (all 0xFFFFFFFF)");
//         }
//     } else {
//         Serial.println("No data or read failed.");
//     }
// }

// void printPinLevels() {
//   int sck = gpio_get_level((gpio_num_t)I2S_SCK);
//   int ws  = gpio_get_level((gpio_num_t)I2S_WS);
//   int sd  = gpio_get_level((gpio_num_t)I2S_SD);
//   Serial.printf("GPIO levels SCK=%d WS=%d SD=%d\n", sck, ws, sd);
// }

void testMicrophoneHardware() {
    Serial.println("\n=== MICROPHONE HARDWARE TEST ===");
    
    // Test 1: Can we read SD pin as digital input?
    pinMode(I2S_SD, INPUT);
    Serial.print("SD pin reads: ");
    for(int i = 0; i < 20; i++) {
        Serial.print(digitalRead(I2S_SD));
        delay(10);
    }
    Serial.println();
    
    delay(5000);
}

void printI2SRegisters() {
    Serial.println("\n=== I2S Register Dump ===");
    
    volatile uint32_t* I2S_CONF_REG = (volatile uint32_t*)0x3FF4F008;
    volatile uint32_t* I2S_CLKM_CONF_REG = (volatile uint32_t*)0x3FF4F0AC;
    volatile uint32_t* I2S_SAMPLE_RATE_CONF_REG = (volatile uint32_t*)0x3FF4F0B0;
    volatile uint32_t* I2S_FIFO_CONF_REG = (volatile uint32_t*)0x3FF4F018;
    
    Serial.printf("I2S_CONF_REG: 0x%08X\n", *I2S_CONF_REG);
    Serial.printf("I2S_CLKM_CONF_REG: 0x%08X\n", *I2S_CLKM_CONF_REG);
    Serial.printf("I2S_SAMPLE_RATE_CONF_REG: 0x%08X\n", *I2S_SAMPLE_RATE_CONF_REG);
    Serial.printf("I2S_FIFO_CONF_REG: 0x%08X\n", *I2S_FIFO_CONF_REG);
    
    bool rx_start = (*I2S_CONF_REG) & (1 << 11);
    bool tx_start = (*I2S_CONF_REG) & (1 << 10);
    Serial.printf("RX_START bit: %d, TX_START bit: %d\n", rx_start, tx_start);
}

void checkClocksWhileReading() {
    Serial.println("\n=== REAL-TIME CLOCK CHECK ===");
    
    // Create a task that continuously reads
    int32_t samples[1024];
    size_t bytes_read;
    
    Serial.println("Starting continuous read...");
    
    for(int attempt = 0; attempt < 10; attempt++) {
        // Start a read (this should activate clocks)
        unsigned long start = micros();
        esp_err_t err = i2s_read(I2S_PORT, samples, sizeof(samples), &bytes_read, 100);
        unsigned long duration = micros() - start;
        
        // Check pins immediately while read might still be active
        int sck_high = 0, sck_low = 0;
        for(int i = 0; i < 100; i++) {
            if (gpio_get_level((gpio_num_t)I2S_SCK)) sck_high++;
            else sck_low++;
            delayMicroseconds(1);
        }
        
        Serial.printf("Attempt %d: read=%s bytes=%u time=%luus SCK high=%d low=%d sample[0]=0x%08X\n",
                     attempt, esp_err_to_name(err), bytes_read, duration, 
                     sck_high, sck_low, samples[0]);
        
        delay(100);
    }
}

void setupAudio(){
    Serial.println("\n=== I2S MICROPHONE INITIALIZATION ===");

    esp_err_t result = i2s_driver_uninstall(I2S_PORT);
    if (result == ESP_ERR_INVALID_STATE) {
        Serial.println("No previous I2S driver (OK)");
    }
    delay(100);

    periph_module_disable(PERIPH_I2S0_MODULE);
    delay(100);
    periph_module_enable(PERIPH_I2S0_MODULE);
    delay(100);

    i2s_config_t i2s_config_local = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 16000,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = 0,
        .dma_buf_count = 4,
        .dma_buf_len = 512,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config_local = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };
    
    Serial.println("Installing I2S driver...");
    result = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    if (result != ESP_OK) {
        Serial.printf("Error installing I2S driver: %s (0x%x)\n", esp_err_to_name(result), result);
        return;
    } else {
        Serial.println("I2S driver installed successfully");
    }

    Serial.println("Setting I2S pins...");
    result = i2s_set_pin(I2S_PORT, &pin_config);
    if (result != ESP_OK) {
        Serial.printf("Error setting I2S pins: %s (0x%x)\n", esp_err_to_name(result), result);
        return;
    } else {
        Serial.println("I2S pins configured");
    }

    Serial.println("Setting clock explicitly...");
    result = i2s_set_clk(I2S_PORT, 16000, I2S_BITS_PER_SAMPLE_32BIT, I2S_CHANNEL_MONO);
    if (result != ESP_OK) {
        Serial.printf("Error: %s\n", esp_err_to_name(result));
    } else {
        Serial.println("Clock set");
    }

    Serial.println("Clearing DMA buffers...");
    i2s_zero_dma_buffer(I2S_PORT);
    delay(100);

    Serial.println("Starting I2S peripheral...");
    result = i2s_start(I2S_PORT);
    if (result != ESP_OK) {
        Serial.printf("Error starting I2S: %s (0x%x)\n", esp_err_to_name(result));
        return;
    } else {
        Serial.println("I2S started");
    }
    
    delay(500);
    printI2SRegisters();

    Serial.println("\n7. Forcing initial read...");
    int32_t dummy[256];
    size_t bytes_read;
    for(int i = 0; i < 5; i++) {
        result = i2s_read(I2S_PORT, dummy, sizeof(dummy), &bytes_read, 1000);
        Serial.printf("Read %d: %s, %u bytes, first sample: 0x%08X\n", 
                     i, esp_err_to_name(result), bytes_read, dummy[0]);
        delay(100);
    }

    Serial.println("\nChecking clocks NOW (during active read):");
    
    for(int i = 0; i < 100; i++) {
        int sck = gpio_get_level((gpio_num_t)I2S_SCK);
        int ws = gpio_get_level((gpio_num_t)I2S_WS);
        
        if (i < 20) {
            Serial.printf("[%02d] SCK=%d WS=%d\n", i, sck, ws);
        }
        
        delayMicroseconds(10);
        
        // Do another quick read to keep clocks active
        if (i % 50 == 0) {
            i2s_read(I2S_PORT, dummy, 64, &bytes_read, 10);
        }
    }


    Serial.println("\n=== I2S INITIALIZATION COMPLETE ===\n");
}

#define CHUNK_SAMPLES 16000
#define CHUNKS_PER_RECORDING 5

void processAudioRecording(){
    static int16_t audio_buffer[CHUNK_SAMPLES];
    static size_t samples_collected = 0;
    static int chunk_count = 0;

    int32_t audio[BUFFER_SIZE*2];
    size_t bytes_read;

    esp_err_t result = i2s_read(I2S_PORT, audio, sizeof(audio), &bytes_read, portMAX_DELAY);
    if (result == ESP_OK && bytes_read > 0) {
        int samples_read = bytes_read / sizeof(int32_t);
    
        for (int i = 0; i < samples_read && samples_collected < CHUNK_SAMPLES; i += 2) {
            int32_t s32 = audio[i] >> 14;
            s32 = s32 >> 2;

            if (s32 > 32767) s32 = 32767;
            if (s32 < -32768) s32 = -32768;

            audio_buffer[samples_collected++] = (int16_t)s32;

            // int16_t sample = (int16_t)(audio[i] >> 16);
            // audio_buffer[samples_collected++] = sample;
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