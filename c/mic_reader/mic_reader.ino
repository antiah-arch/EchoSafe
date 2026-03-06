// EchoSafe — Arduino Sketch
// MAX9814 microphone → raw ADC values → Python over Serial
//
// Protocol:
//   Arduino → Python : one integer per line, e.g. "512\n"  (~100 Hz)
//   Python  → Arduino: '1' = LED on   '0' = LED off
//
// Wiring:
//   MAX9814 OUT → A0
//   LED (or relay) → pin 13 (built-in LED)
//   MAX9814 VDD → 3.3V or 5V (check your module)
//   MAX9814 GND → GND

// ── Pin config ────────────────────────────────────────────────────────────────
const int MIC_PIN = A0;
const int LED_PIN = 13;

// ── Timing ────────────────────────────────────────────────────────────────────
// Target ~100 samples/second so Python's 10ms comment in config stays accurate.
// We use a non-blocking timer rather than delay() so LED commands are never missed.
const unsigned long SAMPLE_INTERVAL_MS = 10;

unsigned long lastSampleTime = 0;

void setup() {
  Serial.begin(9600);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
}

void loop() {

  // ── 1. Read LED commands from Python (non-blocking) ───────────────────────
  // Check before AND after sampling so commands are never held up by the ADC read.
  readCommands();

  // ── 2. Send ADC sample at target rate ─────────────────────────────────────
  unsigned long now = millis();
  if (now - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = now;

    // Send raw ADC value (0–1023) as plain integer + newline.
    // Python's _read_int() / isdigit() expects exactly this format.
    Serial.println(analogRead(MIC_PIN));

    // Check for commands again immediately after sending so we stay responsive
    readCommands();
  }
}

// ── LED command handler ───────────────────────────────────────────────────────
// Python sends a single byte: '1' (0x31) = LED on, '0' (0x30) = LED off.
// Called twice per loop iteration to minimise latency.
void readCommands() {
  while (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == '1') {
      digitalWrite(LED_PIN, HIGH);
    } else if (cmd == '0') {
      digitalWrite(LED_PIN, LOW);
    }
    // Any other byte is silently ignored (handles \n, \r from serial monitors)
  }
}
