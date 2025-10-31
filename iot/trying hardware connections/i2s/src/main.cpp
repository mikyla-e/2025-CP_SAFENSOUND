// Fixed: replace accidental quoted content with valid C++ code
#include <Arduino.h>
#include "driver/i2s.h"

// ========== Pin mapping (adjust to your wiring) ==========
// SPH0645LM4H-B (no MCLK). L/R (SEL) -> GND for LEFT channel.
#define I2S_SCK 26   // BCLK
#define I2S_WS  25   // LRCLK / WS
// We'll probe both 32 and 33 at runtime; default to 33 for now
#define I2S_SD_DEFAULT  33   // DOUT from mic (default)

// ========== Audio/I2S parameters ==========
#define SAMPLE_RATE       16000
#define I2S_PORT          I2S_NUM_0
#define DMA_BUF_COUNT     8
#define DMA_BUF_LEN       256   // DMA buffer length in 32-bit words
#define CHUNK_SAMPLES     1024  // samples we read/process per loop (32-bit words)

static int32_t i2s_buf[CHUNK_SAMPLES];  // raw 32-bit from mic
static int16_t pcm_buf[CHUNK_SAMPLES];  // converted to 16-bit

static esp_err_t configureI2S(int dataPin, i2s_comm_format_t commFmt, i2s_channel_fmt_t chanFmt) {
	i2s_driver_uninstall(I2S_PORT); // safe even if not installed

	i2s_config_t cfg = {
		.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
		.sample_rate = SAMPLE_RATE,
		.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
		.channel_format = chanFmt,
		.communication_format = commFmt,
		.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
		.dma_buf_count = DMA_BUF_COUNT,
		.dma_buf_len = DMA_BUF_LEN,
		.use_apll = true
	};

	i2s_pin_config_t pins = {
		.bck_io_num = I2S_SCK,
		.ws_io_num = I2S_WS,
		.data_out_num = I2S_PIN_NO_CHANGE,
		.data_in_num = dataPin
	};

	esp_err_t err;
	err = i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
	if (err != ESP_OK) return err;
	err = i2s_set_pin(I2S_PORT, &pins);
	if (err != ESP_OK) return err;
	err = i2s_set_clk(I2S_PORT, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_32BIT, chanFmt == I2S_CHANNEL_FMT_ONLY_RIGHT || chanFmt == I2S_CHANNEL_FMT_ONLY_LEFT ? I2S_CHANNEL_MONO : I2S_CHANNEL_STEREO);
	if (err != ESP_OK) return err;
	i2s_zero_dma_buffer(I2S_PORT);
	return i2s_start(I2S_PORT);
}

static void setupI2S() {
	// Auto-probe common combinations to find a live signal
	struct Cand { int pin; i2s_comm_format_t fmt; i2s_channel_fmt_t chan; const char* name; };
	Cand cands[] = {
		{32, (i2s_comm_format_t)I2S_COMM_FORMAT_STAND_I2S, I2S_CHANNEL_FMT_ONLY_LEFT,  "PIN32 STAND LEFT"},
		{33, (i2s_comm_format_t)I2S_COMM_FORMAT_STAND_I2S, I2S_CHANNEL_FMT_ONLY_LEFT,  "PIN33 STAND LEFT"},
		{32, (i2s_comm_format_t)I2S_COMM_FORMAT_STAND_I2S, I2S_CHANNEL_FMT_ONLY_RIGHT, "PIN32 STAND RIGHT"},
		{33, (i2s_comm_format_t)I2S_COMM_FORMAT_STAND_I2S, I2S_CHANNEL_FMT_ONLY_RIGHT, "PIN33 STAND RIGHT"},
		{32, (i2s_comm_format_t)I2S_COMM_FORMAT_I2S,       I2S_CHANNEL_FMT_ONLY_LEFT,  "PIN32 I2S LEFT"},
		{33, (i2s_comm_format_t)I2S_COMM_FORMAT_I2S,       I2S_CHANNEL_FMT_ONLY_LEFT,  "PIN33 I2S LEFT"},
		{32, (i2s_comm_format_t)I2S_COMM_FORMAT_I2S,       I2S_CHANNEL_FMT_ONLY_RIGHT, "PIN32 I2S RIGHT"},
		{33, (i2s_comm_format_t)I2S_COMM_FORMAT_I2S,       I2S_CHANNEL_FMT_ONLY_RIGHT, "PIN33 I2S RIGHT"},
	};

	uint32_t bestScore = 0;
	int bestIdx = -1;
	for (size_t i = 0; i < sizeof(cands)/sizeof(cands[0]); ++i) {
		auto &c = cands[i];
		esp_err_t err = configureI2S(c.pin, c.fmt, c.chan);
		Serial.printf("Probe %s -> %s\n", c.name, esp_err_to_name(err));
		if (err != ESP_OK) continue;

		uint32_t score = 0;
		for (int t = 0; t < 3; ++t) {
			size_t br = 0;
			if (i2s_read(I2S_PORT, (void*)i2s_buf, sizeof(i2s_buf), &br, 200) == ESP_OK && br > 0) {
				size_t n = br / sizeof(int32_t);
				for (size_t k = 0; k < n; ++k) {
					if (i2s_buf[k] != 0) score++;
				}
			}
			delay(20);
		}

		Serial.printf("  score=%u\n", (unsigned)score);
		if (score > bestScore) { bestScore = score; bestIdx = (int)i; }
	}

	if (bestIdx >= 0) {
		auto &b = cands[bestIdx];
		esp_err_t err = configureI2S(b.pin, b.fmt, b.chan);
		Serial.printf("Selected %s -> %s (score=%u)\n", b.name, esp_err_to_name(err), (unsigned)bestScore);
	} else {
		// Fallback to default
		esp_err_t err = configureI2S(I2S_SD_DEFAULT, (i2s_comm_format_t)I2S_COMM_FORMAT_STAND_I2S, I2S_CHANNEL_FMT_ONLY_LEFT);
		Serial.printf("Fallback configure -> %s\n", esp_err_to_name(err));
	}
}

void setup() {
	Serial.begin(115200);
	delay(500);
	Serial.println("=== SPH0645 I2S MIC TEST (ESP32) ===");
	Serial.println("- Pins: BCLK=26, LRCLK=25, DOUT=33 (adjust if needed)");
	Serial.println("- Ensure SEL/LR pin on mic is tied to GND for LEFT channel");

	setupI2S();
}

void loop() {
	size_t bytes_read = 0;

	// Blocking read: keeps I2S clocks running
	esp_err_t res = i2s_read(I2S_PORT, (void*)i2s_buf, sizeof(i2s_buf), &bytes_read, 1000);
	if (res != ESP_OK || bytes_read == 0) {
		Serial.printf("i2s_read err=%s bytes=%u\n", esp_err_to_name(res), (unsigned)bytes_read);
		delay(100);
		return;
	}

	size_t samples_read = bytes_read / sizeof(int32_t);
	if (samples_read == 0) return;

	// Convert 32-bit left-justified mic data to 16-bit PCM
	// SPH0645 gives 24-bit data in 32-bit word; right-shift to scale.
		// Typical shift 11–14; increase to reduce clipping if peaks hit 32767 frequently.
			const int SHIFT = 14; // adjust 12..15 as needed
			int64_t acc_sq = 0;
			int64_t acc_sum = 0; // for DC offset estimate
			int16_t peak = 0;

	for (size_t i = 0; i < samples_read; i++) {
		int32_t s32 = i2s_buf[i] >> SHIFT;
		if (s32 > 32767) s32 = 32767;
		if (s32 < -32768) s32 = -32768;
		int16_t s16 = (int16_t)s32;
		pcm_buf[i] = s16;
		acc_sum += s16;

			// Use 32-bit absolute to avoid INT16_MIN overflow
			int32_t a32 = s16 < 0 ? -(int32_t)s16 : (int32_t)s16;
			if (a32 > peak) peak = (int16_t) (a32 > 32767 ? 32767 : a32);
		acc_sq += (int32_t)s16 * (int32_t)s16;
	}

		float rms = sqrtf((float)acc_sq / (float)samples_read);

		// Compute DC-removed metrics for better visibility of true audio content
		int16_t mean16 = (int16_t)(acc_sum / (int64_t)samples_read);
		int64_t acc_sq_dc = 0;
		int16_t peak_dc = 0;
		for (size_t i = 0; i < samples_read; i++) {
			int32_t d = (int32_t)pcm_buf[i] - (int32_t)mean16;
			if (d > 32767) d = 32767; if (d < -32768) d = -32768;
			int16_t dd = (int16_t)d;
			int32_t ad = dd < 0 ? - (int32_t)dd : (int32_t)dd;
			if (ad > peak_dc) peak_dc = (int16_t)(ad > 32767 ? 32767 : ad);
			acc_sq_dc += (int32_t)dd * (int32_t)dd;
		}
		float rms_dc = sqrtf((float)acc_sq_dc / (float)samples_read);

			// Print a concise line once per chunk; suitable for Serial Monitor or Plotter
		static uint32_t lastDump = 0;
		if (millis() - lastDump > 1000) {
			// Brief hex preview of raw words to confirm activity/alignment
			Serial.printf("samples=%u rms=%.1f peak=%d mean=%d rms_dc=%.1f peak_dc=%d raw[0..3]=%08X %08X %08X %08X\n",
						  (unsigned)samples_read, rms, peak, mean16, rms_dc, peak_dc,
										(unsigned)i2s_buf[0], (unsigned)i2s_buf[1], (unsigned)i2s_buf[2], (unsigned)i2s_buf[3]);
			lastDump = millis();
		} else {
			Serial.printf("samples=%u rms=%.1f peak=%d mean=%d rms_dc=%.1f peak_dc=%d\n",
						  (unsigned)samples_read, rms, peak, mean16, rms_dc, peak_dc);
		}

	// Optional: slow down logs a bit
	delay(50);
}