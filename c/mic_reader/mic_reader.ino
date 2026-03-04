// MAX9814 Clap / Sound Detector for Arduino
// Automatically tracks baseline and detects loud sounds

const int micPin = A0;       // MAX9814 analog output connected to A0
const int sampleWindow = 50; // milliseconds for baseline averaging
const int threshold = 345;    // minimum rise above baseline to detect clap

int baseline = 0;            // running baseline

void setup() {
  Serial.begin(9600);
}

void loop() {
  unsigned long startMillis = millis();
  int sum = 0;
  int count = 0;

  // Take readings over the sample window to calculate baseline
  while (millis() - startMillis < sampleWindow) {
    int val = analogRead(micPin);
    sum += val;
    count++;
  }

  baseline = sum / count; // update baseline

  // Read current value
  int currentVal = analogRead(micPin);

  // Detect loud sound (clap)
  if (currentVal > baseline + threshold) {
    Serial.println("Clap detected!");
  }

  // Optional: print baseline and current value for debugging
  // Serial.print("Baseline: "); Serial.print(baseline);
  // Serial.print(" Current: "); Serial.println(currentVal);

  delay(10);
}
