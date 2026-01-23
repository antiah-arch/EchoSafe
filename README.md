Overview

This project uses an Arduino Uno (or compatible board) to create a simple microphone-based LED visualizer. The Arduino reads sound input from a microphone and controls an LED (or multiple LEDs) in response to detected audio levels.

Features

Reads sound levels from a microphone sensor.

Lights up an LED based on sound intensity.

Adjustable threshold and sensitivity in code.

Works with a single LED or can be expanded for multiple LEDs.

Ideal for music visualizations, noise indicators, or DIY projects.

Hardware Requirements

Arduino Uno (or compatible)

Microphone sensor module

LED (or multiple LEDs)

Resistors (220Ω recommended for LEDs)

Breadboard and jumper wires

USB cable to connect Arduino to your computer

Circuit Connections

Microphone Module

VCC → 5V on Arduino

GND → GND on Arduino

OUT → Analog input (e.g., A0) on Arduino

LED

Positive (longer leg) → Digital pin (e.g., D13) via 220Ω resistor

Negative (shorter leg) → GND

⚠️ Make sure GND connections are correct. Incorrect wiring may cause short circuits.

Software Requirements

Arduino IDE (latest version recommended)

Basic knowledge of Arduino sketches