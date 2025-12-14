#include <Arduino.h>

// Mapping đúng trên Yolo UNO (ESP32-S3)
const int PHAO_PIN  = 5;   // D2
const int RELAY_PIN = 6;   // D3

// Phao: LOW = cạn, HIGH = đầy
const int MUC_CAN   = HIGH;
const int MUC_DAY   = LOW;

// Relay active LOW
// LOW  = bật bơm
// HIGH = tắt bơm
const int RELAY_ON  = LOW;
const int RELAY_OFF = HIGH;

// Debounce phao
const uint8_t  PHAO_SAMPLE_COUNT = 5;
const uint16_t PHAO_SAMPLE_DELAY = 10;

// Lưu trạng thái cũ
int lastPhaoState  = -1;
int lastRelayState = -1;

int readPhaoStable() {
  uint8_t lowCount = 0;
  uint8_t highCount = 0;

  for (uint8_t i = 0; i < PHAO_SAMPLE_COUNT; i++) {
    int v = digitalRead(PHAO_PIN);
    if (v == LOW) lowCount++;
    else          highCount++;
    delay(PHAO_SAMPLE_DELAY);
  }

  return (lowCount > highCount) ? LOW : HIGH;
}

void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(PHAO_PIN, INPUT_PULLUP);
  pinMode(RELAY_PIN, OUTPUT);

  // Ban đầu tắt bơm
  digitalWrite(RELAY_PIN, RELAY_OFF);

  Serial.println("=== Gardeny Mini (Logic Moi) ===");
  Serial.println("HIGH  (Phao dong)  = Du nuoc  => Tat bom");
  Serial.println("LOW (Phao mo)    = Can nuoc   => Bat bom");
  Serial.println("Relay active LOW: HIGH=OFF, LOW=ON");
}

void loop() {
  int phaoState = readPhaoStable();

  // Áp dụng logic mới
  if (phaoState == MUC_CAN) {
    // Nước cạn → bật bơm
    digitalWrite(RELAY_PIN, RELAY_ON);
  } else {
    // Nước đầy → tắt bơm
    digitalWrite(RELAY_PIN, RELAY_OFF);
  }

  int relayState = digitalRead(RELAY_PIN);

  // Log khi trạng thái thay đổi
  if (phaoState != lastPhaoState || relayState != lastRelayState) {
    Serial.print("Phao = ");
    Serial.print(phaoState == LOW ? "LOW (can nuoc)" : "HIGH (du nuoc)");
    Serial.print(" | Bom = ");
    Serial.println(relayState == RELAY_ON ? "OFF" : "ON");

    lastPhaoState  = phaoState;
    lastRelayState = relayState;
  }

  delay(200);
}
