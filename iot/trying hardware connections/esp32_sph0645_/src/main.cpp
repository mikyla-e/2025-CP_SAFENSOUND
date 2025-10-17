#include <Arduino.h>
#include <driver/i2s.h>

#define I2S_WS 25
#define I2S_SD 22
#define I2S_SCK 26

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
  uint8_t buffer[1024];
  size_t bytes_read = 0;
  i2s_read(I2S_NUM_0, &buffer, sizeof(buffer), &bytes_read, 100 / portTICK_PERIOD_MS);
  for (size_t i = 0; i < bytes_read; i++) {
    if (buffer[i] != 0) return true;
  }
  return false;
}

void setup() {
  Serial.begin(115200);
  setupI2S();

  while (!checkMicConnection()) {
    Serial.println("Failed connection: SPH0645LM4H not detected! Retrying...");
    delay(1000);
  }

  Serial.println("Start recording...");
}

void loop() {
  uint32_t buffer[256]; // 256 * 4 bytes = 1024 bytes
  size_t bytes_read;

  unsigned long start = millis();
  while (millis() - start < 15000) {
    i2s_read(I2S_NUM_0, buffer, sizeof(buffer), &bytes_read, portMAX_DELAY);
    Serial.write((uint8_t*)buffer, bytes_read);
  }
  Serial.println("Recording finished. Save the output from Serial Monitor.");

  while (1);
}