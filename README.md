# Obstacle Avoidance Bot

A robotics project that enables an autonomous mobile robot to detect and avoid obstacles using sensors and movement control logic.

---

## Table of Contents
- [About](#about)
- [Features](#features)
- [Hardware & Components](#hardware--components)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Bot](#running-the-bot)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Contact](#contact)

---

## About
The Obstacle Avoidance Bot navigates autonomously by detecting obstacles in real time. Using sensors such as ultrasonic or infrared modules, the robot reads the environment and decides whether to move forward, stop, reverse, or turn. This simple but effective logic forms the basis of autonomous navigation.

---

## Features
- Real-time obstacle detection
- Automatic avoidance routines (stop, reverse, turn)
- Embedded system-friendly implementation
- Simple modular software structure
- Can be extended for more complex autonomous behavior

---

## Hardware & Components
You may adjust the list based on your actual setup:
- Arduino Uno or compatible microcontroller
- Ultrasonic sensor (HC-SR04) or IR proximity sensor
- Motor driver module (L293D or L298N)
- DC motors and wheels
- Chassis and power supply
- Jumper wires and connectors

---

## Project Structure
```
Obstacle-avoidance-bot/
  src/
  hardware_diagram/
  firmware/
  README.md
```

- `src/` – Contains sensor and motor control logic
- `firmware/` – Microcontroller program files
- `hardware_diagram/` – Circuit diagrams and connections

---

## Prerequisites
- Arduino IDE
- USB cable for uploading firmware
- Motor driver library if required
- Properly wired robot chassis

---

## Installation
Clone the repository:
```bash
git clone https://github.com/GeekySalami/Obstacle-avoidance-bot.git
cd Obstacle-avoidance-bot
```

Open the project in Arduino IDE, then:
- Select the appropriate board
- Select the correct COM port
- Upload the code to your microcontroller

---

## Running the Bot
1. Power ON the robot.
2. The robot begins moving forward.
3. When the sensor detects an obstacle below the threshold distance:
   - Robot stops
   - Reverses
   - Turns left or right
   - Continues moving forward

---

## Usage
- Place the robot in a clear testing area.
- Introduce obstacles to test avoidance behavior.
- Adjust speed, sensor threshold, and turning duration as needed.
- Use different sensor types by modifying the `src/` logic.

---

## Configuration
Editable settings include:
- `THRESHOLD_DISTANCE` – Distance to trigger avoidance
- Motor pin assignments
- Reverse and turn durations
- Sensor type settings (IR/ultrasonic)

---

## Contributing
1. Fork the repository
2. Create a new branch for your feature or fix
3. Commit and test changes
4. Submit a pull request

---

## Contact
Author: **GeekySalami**  
GitHub: https://github.com/GeekySalami
