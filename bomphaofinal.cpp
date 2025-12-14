#include <Arduino.h>

// Pin definitions (YOLO Uno / ESP32-S3)
#define RELAY_PIN 10   // D7 -> GPIO10
#define LED1_PIN  18   // D9 -> GPIO18
#define LED2_PIN  17   // D8 -> GPIO17

void setup() {
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(LED1_PIN, OUTPUT);
  pinMode(LED2_PIN, OUTPUT);

  // Start OFF
  digitalWrite(RELAY_PIN, LOW);
  digitalWrite(LED1_PIN, LOW);
  digitalWrite(LED2_PIN, LOW);
}

void loop() {
  // Relay ON → LEDs ON
  digitalWrite(RELAY_PIN, HIGH);
  digitalWrite(LED1_PIN, HIGH);
  digitalWrite(LED2_PIN, HIGH);
  delay(2000);

  // Relay OFF → LEDs OFF
  digitalWrite(RELAY_PIN, LOW);
  digitalWrite(LED1_PIN, LOW);
  digitalWrite(LED2_PIN, LOW);
  delay(2000);
}
