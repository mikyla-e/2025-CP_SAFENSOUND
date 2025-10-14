#include <Arduino.h>
#include "driver/i2s.h"

#define I2S_WS   25  // LRCLK (Word Select)
#define I2S_SD   22  // DOUT (Data In)
#define I2S_SCK  26  // BCLK (Bit Clock)

void setup() {
    Serial.begin(115200);
    delay(1000);

    // I2S configuration
    const i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 16000,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .intr_alloc_flags = 0,
        .dma_buf_count = 8,
        .dma_buf_len = 64,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    // Pin configuration
    const i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = -1,      // Not used for microphone
        .data_in_num = I2S_SD
    };

    // Install and start I2S driver
    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_config);

    Serial.println("I2S initialized. Reading microphone...");
}

void loop() {
    int32_t sample = 0;
    size_t bytes_read = 0;
    i2s_read(I2S_NUM_0, &sample, sizeof(sample), &bytes_read, portMAX_DELAY);
    Serial.println(sample);
    delay(100);
}