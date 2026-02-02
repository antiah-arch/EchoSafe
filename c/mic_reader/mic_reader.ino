const int micPin = A0;
const int ledPin = 9;

void setup() {
  Serial.begin(115200); // Match Python
  pinMode(ledPin, OUTPUT);
}

void loop() {
  int micValue = analogRead(micPin);
  Serial.println(micValue);

  if (Serial.available()) {
    char cmd = Serial.read();
    digitalWrite(ledPin, (cmd == '1') ? HIGH : LOW);
  }
  delay(5);
}
