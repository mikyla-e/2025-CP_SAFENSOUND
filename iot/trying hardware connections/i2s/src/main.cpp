#include <Arduino.h>
#include "driver/i2s.h"

#define I2S_SCK 26
#define I2S_WS  25
#define I2S_SD  33

void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("=== SIMPLE I2S TEST ===");
  
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 4,
    .dma_buf_len = 64,
    .use_apll = false
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

  esp_err_t err = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  Serial.printf("i2s_driver_install: %s\n", esp_err_to_name(err));
  
  err = i2s_set_pin(I2S_NUM_0, &pin_config);
  Serial.printf("i2s_set_pin: %s\n", esp_err_to_name(err));
  
  delay(100);
  
  // Trigger clock generation with a read
  int32_t samples[64];
  size_t bytes_read = 0;
  err = i2s_read(I2S_NUM_0, samples, sizeof(samples), &bytes_read, 1000);
  Serial.printf("i2s_read: %s, bytes: %d\n", esp_err_to_name(err), bytes_read);
  
  delay(500);
  
  // Check pins
  Serial.println("\nChecking GPIO levels (should toggle):");
  for(int i = 0; i < 50; i++) {
    int sck = digitalRead(I2S_SCK);
    int ws = digitalRead(I2S_WS);
    if (i % 10 == 0) {
      Serial.printf("SCK=%d WS=%d\n", sck, ws);
    }
    delayMicroseconds(50);
  }
}

void loop() {
  // Continuously read to keep clocks running
  int32_t samples[64];
  size_t bytes_read;
  i2s_read(I2S_NUM_0, samples, sizeof(samples), &bytes_read, 100);
  
  // Check first few samples
  Serial.printf("Sample[0]=0x%08X Sample[1]=0x%08X\n", samples[0], samples[1]);
  delay(1000);
}