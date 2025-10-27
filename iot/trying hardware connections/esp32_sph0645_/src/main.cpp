#include <Arduino.h>
#include <driver/i2s.h>

// I2S Configuration for SPH0645LM4H-B microphone breakout
#define I2S_WS 15    // Word Select (LRCL) - GPIO 15
#define I2S_SD 32    // Serial Data (DOUT) - GPIO 32
#define I2S_SCK 14   // Serial Clock (BCLK) - GPIO 14

#define I2S_PORT I2S_NUM_0
#define SAMPLE_RATE 44100  // Try higher sample rate for -B version
#define BUFFER_SIZE 512

int32_t samples[BUFFER_SIZE];

void i2s_install() {
  const i2s_config_t i2s_config = {
    .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 64,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  esp_err_t err = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("ERROR: Failed to install I2S driver: %d\n", err);
  } else {
    Serial.println("✓ I2S driver installed");
  }
}

void i2s_setpin() {
  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

  esp_err_t err = i2s_set_pin(I2S_PORT, &pin_config);
  if (err != ESP_OK) {
    Serial.printf("ERROR: Failed to set I2S pins: %d\n", err);
  } else {
    Serial.println("✓ I2S pins configured");
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n\n=== SPH0645LM4H-B Microphone Test ===");
  Serial.println("Breakout Board Version");
  Serial.println("Pin Configuration:");
  Serial.printf("  BCLK:  GPIO %d\n", I2S_SCK);
  Serial.printf("  LRCL:  GPIO %d\n", I2S_WS);
  Serial.printf("  DOUT:  GPIO %d\n", I2S_SD);
  Serial.println("  VDD:   3.3V");
  Serial.println("  GND:   GND");
  Serial.println("\nInitializing I2S...");
  
  i2s_install();
  i2s_setpin();
  
  // Start I2S
  i2s_start(I2S_PORT);
  
  // Clear I2S buffer - discard first readings
  i2s_zero_dma_buffer(I2S_PORT);
  delay(100);
  
  Serial.println("\n✓ I2S initialized successfully!");
  Serial.println("Listening for audio...");
  Serial.println("Make sounds near the microphone!");
  Serial.println("=====================================\n");
  
  delay(500);
}

void loop() {
  size_t bytes_read = 0;
  
  // Read data from I2S
  esp_err_t result = i2s_read(I2S_PORT, &samples, sizeof(samples), &bytes_read, portMAX_DELAY);
  
  if (result != ESP_OK) {
    Serial.printf("ERROR: I2S read failed: %d\n", result);
    delay(1000);
    return;
  }
  
  int samples_read = bytes_read / sizeof(int32_t);
  
  if (samples_read > 0) {
    // Check if we're getting actual data (not all -1 or 0)
    bool hasValidData = false;
    int validCount = 0;
    
    for (int i = 0; i < min(10, samples_read); i++) {
      // For SPH0645LM4H-B, valid data should vary
      if (samples[i] != -1 && samples[i] != 0 && samples[i] != 0xFFFFFFFF) {
        hasValidData = true;
        validCount++;
      }
    }
    
    // Print raw sample values for debugging
    static unsigned long lastPrint = 0;
    if (millis() - lastPrint > 1000) {
      Serial.print("Raw samples (first 10): ");
      for (int i = 0; i < min(10, samples_read); i++) {
        Serial.printf("0x%08X ", samples[i]);
      }
      Serial.printf("\nValid samples: %d/10 ", validCount);
      if (hasValidData) {
        Serial.println("✓ VALID DATA!");
      } else {
        Serial.println("✗ NO VALID DATA - Check wiring!");
      }
      lastPrint = millis();
    }
    
    if (hasValidData) {
      // Calculate audio level (RMS)
      int64_t sum = 0;
      for (int i = 0; i < samples_read; i++) {
        // SPH0645LM4H-B uses 18-bit data in 32-bit words
        int32_t sample = samples[i] >> 14;
        sum += (int64_t)sample * sample;
      }
      
      float rms = sqrt(sum / samples_read);
      int level = (int)(rms / 100);
      
      // Print audio level as a bar graph
      if (level > 2) {  // Lower threshold for breakout board
        Serial.print("Audio Level: ");
        Serial.print(level);
        Serial.print(" [");
        for (int i = 0; i < min(level, 50); i++) {
          Serial.print("=");
        }
        Serial.println("]");
      }
    }
  }
  
  delay(50);
}