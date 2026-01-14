// Arduino: safe_serial.ino

void setup() {
  Serial.begin(115200); // Must match Python baud
  while (!Serial) {
    ; // Wait for Serial port to be ready (needed for some boards)
  }
  Serial.println("Arduino ready!");
}

void loop() {
  // Check if data is available from Python
  if (Serial.available() > 0) {
    char received = Serial.read(); // read one byte
    Serial.print("Arduino received: ");
    Serial.println(received);

    // Example reaction: blink onboard LED for '1', off for '0'
    if (received == '1') {
      digitalWrite(LED_BUILTIN, HIGH);
    } else if (received == '0') {
      digitalWrite(LED_BUILTIN, LOW);
    }
  }

  // Do other tasks here without blocking
  // e.g., sensor reading, timers, etc.
}

