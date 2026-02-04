const int MIC_PIN = A0;

void setup() {
  Serial.begin(115200);
}

void loop() {
  int micValue = analogRead(MIC_PIN);
  Serial.println(micValue);
  delay(10);  // ~100 samples/sec
}

