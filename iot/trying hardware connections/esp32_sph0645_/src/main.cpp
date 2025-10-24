#include <Arduino.h>
#include <driver/i2s.h>

#define I2S_WS 25 //LRCL
#define I2S_SD 22 //DOUT
#define I2S_SCK 26 //BCLK

void setupI2S() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = 64,
    .use_apll = false
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);
}

bool checkMicConnection() {
  int32_t buffer[256];
  size_t bytes_read = 0;
  
  delay(100);
  
  for (int attempt = 0; attempt < 3; attempt++) {
    i2s_read(I2S_NUM_0, buffer, sizeof(buffer), &bytes_read, 1000 / portTICK_PERIOD_MS);
    
    if (bytes_read > 0) {
      int32_t sum = 0;
      int non_zero = 0;
      
      for (size_t i = 0; i < bytes_read / 4; i++) {
        if (buffer[i] != 0) {
          non_zero++;
          sum += abs(buffer[i]);
        }
      }
      
      if (non_zero > 10) {
        // Remove Serial.printf that corrupts audio data
        return true;
      }
    }
    delay(100);
  }
  
  return false;
}

void setup() {
  Serial.begin(115200);
  setupI2S();

  // Wait for mic without printing during check
  while (!checkMicConnection()) {
    delay(1000);
  }

  // Small delay to ensure serial is ready
  delay(500);
  
  // NO TEXT OUTPUT - go straight to recording
}

void loop() {
  int32_t buffer[256];  // Use int32_t to match I2S config
  size_t bytes_read;

  unsigned long start = millis();
  while (millis() - start < 10000) {
    i2s_read(I2S_NUM_0, buffer, sizeof(buffer), &bytes_read, portMAX_DELAY);
    // Send only raw audio data
    Serial.write((uint8_t*)buffer, bytes_read);
  }

  // Stop forever - no text output
  while (1) {
    delay(1000);
  }
}