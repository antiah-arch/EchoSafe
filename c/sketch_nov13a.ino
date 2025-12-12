int ledPin = 9;

void setup() {
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
  Serial.println("Arduino ready!");  // debug message
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    Serial.print("Received: ");      // debug message
    Serial.println(command);

    if (command == '1') {
      digitalWrite(ledPin, HIGH);
      Serial.println("LED ON");
    } else if (command == '0') {
      digitalWrite(ledPin, LOW);
      Serial.println("LED OFF");
    } else {
      Serial.println("Unknown command!");
    }
  }
}


