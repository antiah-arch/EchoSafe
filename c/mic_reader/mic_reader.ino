const int micPin = A0;
const int ledPin = 13;

void setup() {
  Serial.begin(9600);
  pinMode(ledPin, OUTPUT);
}

void loop() {
  int micValue = analogRead(micPin);
  Serial.println(micValue);

  if (Serial.available()) {
    char c = Serial.read();
    if (c == '1') digitalWrite(ledPin, HIGH);
    if (c == '0') digitalWrite(ledPin, LOW);
  }

  delay(5);
}
