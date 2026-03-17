// EchoSafe — Arduino Sketch
// MAX4466 or MAX9814 microphone → raw ADC values → Python over Serial
//
// Sample rate: ~8000 Hz (125µs between samples)
// Baud rate  : 115200 — required to keep up with 8000 samples/sec
//              Each sample is up to 4 digits + '\n' = 5 bytes = 40000 bytes/sec.
//              115200 baud ≈ 11520 bytes/sec — this is tight, so we pack two
//              samples per line as "A,B\n" to halve serial overhead.
//
// Protocol:
//   Arduino → Python : two comma-separated integers per line, e.g. "512,489\n"
//   Python  → Arduino: '1' = alert on (LED + vibration motor)
//                      '0' = alert off
//
// Why 8000 Hz?
//   analogRead() on Uno/Nano takes ~104µs → max ~9600 Hz.
//   8000 Hz is safe headroom and gives Nyquist of 4000 Hz, which covers:
//     • Smoke alarm tone   : 3150 Hz  (mandated by EN 54-3 / UL 217)
//     • CO alarm tone      : 3100 Hz
//     • Door/smoke pattern : detectable via temporal on/off rhythm
//
// Wiring — microphone:
//   MAX4466 OUT → A0   (preferred — fixed gain, no AGC, cleaner transients)
//   MAX4466 VDD → 3.3V or 5V (check your module)
//   MAX4466 GND → GND
//
// Wiring — LED:
//   LED anode → pin 13 (via 220Ω resistor) → LED cathode → GND
//   (pin 13 has a built-in resistor on most boards, external one is safer)
//
// Wiring — vibration motor:
//   ⚠️  Motors draw 60–150 mA; Arduino GPIO pins are limited to 40 mA.
//       Drive the motor through a transistor, not directly from the pin.
//
//   Recommended circuit (NPN transistor, e.g. 2N2222 or BC547):
//
//     Pin 9 ──[1kΩ]── Base (B)
//                     Emitter (E) ── GND
//                     Collector (C) ── Motor (–) terminal
//     5V ──── Motor (+) terminal
//     Motor (–) ──[1N4007 flyback diode, cathode toward 5V]── 5V
//
//   The flyback diode is essential — without it the motor's inductive
//   kickback will damage the Arduino when the motor switches off.
//
//   If you only have a coin-cell vibration module (ERM type, 3V, ~80mA),
//   the same circuit works with 3.3V instead of 5V on the motor (+) rail.

// ── Pin config ────────────────────────────────────────────────────────────────
const int MIC_PIN   = A0;
const int LED_PIN   = 13;
const int MOTOR_PIN = 9;    // NPN transistor base — see wiring notes above

// ── Timing ────────────────────────────────────────────────────────────────────
// 8000 Hz = 125 µs between samples.
// micros() is used (not millis()) since the interval is sub-millisecond.
const unsigned long SAMPLE_INTERVAL_US = 125;   // 1,000,000 / 8000

unsigned long lastSampleTime = 0;

// Pack two readings per serial line to halve transmission overhead.
bool hasPending = false;
int  pendingVal = 0;

// ── Blink state (non-blocking) ────────────────────────────────────────────────
// When a sound is detected Python sends '1'. Instead of turning the LED on
// solid we blink it 3 times so the user gets a clear visual pulse.
// delay() is never used — that would stall the sampling loop.
const int   BLINK_COUNT    = 3;      // number of blinks per detection
const unsigned long BLINK_ON_MS  = 80;   // ms LED is on per blink
const unsigned long BLINK_OFF_MS = 80;   // ms LED is off between blinks

bool          blinking       = false;
int           blinksLeft     = 0;
bool          blinkPhaseOn   = false;
unsigned long blinkTimer     = 0;

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN,   OUTPUT);
  pinMode(MOTOR_PIN, OUTPUT);
  digitalWrite(LED_PIN,   LOW);
  digitalWrite(MOTOR_PIN, LOW);
  lastSampleTime = micros();
}

void loop() {

  // ── 1. Read commands from Python (non-blocking) ───────────────────────────
  readCommands();

  // ── 2. Non-blocking blink handler ─────────────────────────────────────────
  if (blinking) {
    unsigned long now = millis();
    if (blinkPhaseOn && now - blinkTimer >= BLINK_ON_MS) {
      // End of ON phase — turn LED off
      digitalWrite(LED_PIN, LOW);
      blinkPhaseOn = false;
      blinkTimer   = now;
    } else if (!blinkPhaseOn && now - blinkTimer >= BLINK_OFF_MS) {
      // End of OFF phase
      blinksLeft--;
      if (blinksLeft > 0) {
        // Start next ON phase
        digitalWrite(LED_PIN, HIGH);
        blinkPhaseOn = true;
        blinkTimer   = now;
      } else {
        // All blinks done — restore power indicator
        blinking = false;
        digitalWrite(LED_PIN, HIGH);
      }
    }
  }

  // ── 3. Send ADC samples at 8000 Hz ────────────────────────────────────────
  unsigned long now = micros();
  if (now - lastSampleTime >= SAMPLE_INTERVAL_US) {
    lastSampleTime = now;

    int val = analogRead(MIC_PIN);   // 0–1023

    if (!hasPending) {
      // Hold first sample until we have a pair
      pendingVal = val;
      hasPending = true;
    } else {
      // Send "A,B\n" — two samples, one write
      Serial.print(pendingVal);
      Serial.print(',');
      Serial.println(val);
      hasPending = false;
      readCommands();   // stay responsive to commands after each transmission
    }
  }
}

// ── Command handler ───────────────────────────────────────────────────────────
// Python sends a single byte: '1' = alert on, '0' = alert off.
// '1' triggers a 3-blink sequence on the LED and turns the motor on.
// '0' stops the motor (blink finishes naturally).
void readCommands() {
  while (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == '1') {
      // Start blink sequence (if already blinking, restart it)
      blinking     = true;
      blinksLeft   = BLINK_COUNT;
      blinkPhaseOn = true;
      blinkTimer   = millis();
      digitalWrite(LED_PIN,   HIGH);
      digitalWrite(MOTOR_PIN, HIGH);
    } else if (cmd == '0') {
      digitalWrite(MOTOR_PIN, LOW);
      // Let the blink finish naturally — don't cut it short
    }
    // Any other byte silently ignored (\n, \r from serial monitors)
  }
}