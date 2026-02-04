const int MIC_PIN = A0;

void setup() {
  Serial.begin(9600);
}

void loop() {
  int micValue = analogRead(MIC_PIN);
  Serial.println(micValue);
  delay(10);  // ~100 samples/sec
}

