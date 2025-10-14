#include <Arduino.h>

const int buttonPin = 15;    // Button GPIO pin
const int buzzerPin = 23;    // Buzzer control pin (via 2N2222 base)
const int ledPin = 16;       // LED pin (change to your actual GPIO for LED)

bool buzzerState = false;    // Buzzer OFF by default
bool ledState = false;       // LED OFF by default
bool lastButtonState = HIGH;

void setup() {
    Serial.begin(115200);
    pinMode(buttonPin, INPUT_PULLUP); // Internal pull-up for button
    pinMode(buzzerPin, OUTPUT);
    pinMode(ledPin, OUTPUT);
    digitalWrite(buzzerPin, LOW);     // Buzzer OFF by default
    digitalWrite(ledPin, LOW);        // LED OFF by default
    Serial.println("ESP32-powered buzzer and LED ready!");
}

void loop() {
    bool buttonState = digitalRead(buttonPin);

    // Detect button press (transition from HIGH to LOW)
    if (lastButtonState == HIGH && buttonState == LOW) {
        buzzerState = !buzzerState; // Toggle buzzer state
        ledState = !ledState;       // Toggle LED state
        digitalWrite(buzzerPin, buzzerState ? HIGH : LOW);
        digitalWrite(ledPin, ledState ? HIGH : LOW);

        if (buzzerState) {
            Serial.println("Buzzer ON, LED ON");
        } else {
            Serial.println("Buzzer OFF, LED OFF");
        }

        delay(300); // Debounce delay
    }

    lastButtonState = buttonState;
}