# AI-Vision Security Agent

AI-powered smart gate access control system using computer vision, embedded hardware, and a Flutter mobile platform.

Smart gate access control system powered by Flutter, Firebase, and AI-driven security workflows.

---

## Overview

AI-Vision Security Agent is a complete intelligent gate automation and access control ecosystem that combines:

* Artificial Intelligence
* Computer Vision
* Embedded Systems
* Mobile Application Development
* Real-Time Cloud Infrastructure

The project was designed as a smart alternative to traditional gate systems that rely on:

* Physical keys
* Static PINs
* Remote controls

Traditional systems are vulnerable because credentials can be:

* Lost
* Shared
* Stolen
* Cloned

The AI-Vision Security Agent introduces intelligent authentication using:

* Face recognition
* Gesture recognition
* OTP verification
* Multi-level access control
* Real-time monitoring and alerts

The system operates using ESP32-CAM hardware integrated with AI-powered verification models and a Flutter mobile application for homeowners.

---

# System Features

## Authentication & Session Management

* Firebase Authentication integration
* Real-time auth session monitoring
* Protected route navigation
* Automatic onboarding/setup flow
* Secure sign out with provider cleanup

## AI & Computer Vision Features

* Real-time face recognition
* Gesture recognition system
* Anti-spoofing detection
* AI-powered access validation
* Live camera processing
* Touch-less gate interaction

### Face Recognition

* HOG-based facial encoding
* Real-time facial matching
* Encoded identity storage system
* CPU-optimized inference

### Gesture Recognition

* Trained using the HaGRID dataset
* EfficientNet-B0 gesture classifier
* MediaPipe hand detection
* Real-time gesture-to-action mapping

### Anti-Spoofing

* Texture analysis verification
* LBP + Laplacian based validation
* Fake face prevention system

---

## Hardware System

The physical security system is powered by embedded hardware including:

* ESP32-S2 / ESP32-S3
* OV7670 Camera Module
* Dual 5V Relay Module
* 4×4 Keypad (KP-200)
* Speaker + amplifier system

### Hardware Responsibilities

* Gate relay control
* Live image capture
* PIN entry processing
* Audio feedback
* AI model execution
* Real-time gate automation

---

## Mobile Application Features

* Create new smart properties
* Join existing properties using unique house tokens
* Multi-user shared access system
* Support for multiple gates
* Real-time gate state synchronization

## User Roles & Permissions

### Two-tier access system:

* `admin`
* `member`

### Admin Capabilities

* Manage users
* Promote/demote members
* Remove unauthorized users
* Transfer ownership/admin role

### Member Capabilities

* Gate access
* Monitor assigned systems
* Use generated secure PINs

---

# Tech Stack

| Technology      | Purpose                         |
| --------------- | ------------------------------- |
| Flutter         | Cross-platform mobile framework |
| Dart            | Programming language            |
| Firebase Auth   | Authentication                  |
| Cloud Firestore | Real-time database              |
| Riverpod        | State management                |
| GoRouter        | Navigation & routing            |
| ESP32-CAM       | Smart gate hardware integration |

---

# Architecture

```text
lib/
│
├── core/
│   ├── routing/
│   ├── services/
│   └── utils/
│
├── features/
│   ├── auth/
│   ├── dashboard/
│   ├── gates/
│   ├── setup/
│   ├── users/
│   └── admin/
│
├── models/
│
├── providers/
│
└── main.dart
```

---

# Authentication Flow

1. User signs in or registers
2. FirebaseAuth listener detects active session
3. Firestore streams live user profile
4. App validates linked house
5. Users without a house are redirected to setup
6. Authorized users enter dashboard

---

# House Setup System

## Create House

* Generates unique property token:

```text
HSE-XXXXXXXX
```

* Initializes:

  * House document
  * Default master PIN
  * Rolling homeowner PIN
  * Gate configuration

* Assigns creator as:

```text
admin
```

---

## Join Existing House

Users can join an existing smart property using a valid house token.

The system:

* Validates token
* Links user to house
* Assigns `member` role
* Syncs permissions in Firestore

---

# Gate Management System

Supports multiple smart gates including:

* Front Gate
* Garage Gate

## Gate States

```text
open
closed
```

## Gate Commands

```text
open_gate
close_gate
idle
```

---

# Security Features

* Rotating secure homeowner PINs
* Temporary guest access codes
* Role-based permission system
* Real-time activity logging
* Atomic Firestore transactions
* Secure route protection
* Provider invalidation on logout

---

# Firebase Setup

The repository already includes Firebase configuration support.

## Install Dependencies

```bash
flutter pub get
npm install -g firebase-tools
dart pub global activate flutterfire_cli
```

---

## Login to Firebase

```bash
firebase login
```

---

## Configure FlutterFire

```bash
flutterfire configure
```

Select:

* Correct Firebase project
* `android` platform
* Overwrite files if requested

---

## Add SHA-1 Fingerprint

Run:

```bash
keytool -list -v -keystore ~/.android/debug.keystore -alias androiddebugkey -storepass android -keypass android
```

Send the generated SHA-1 fingerprint to the Firebase project administrator.

---

## Run Application

```bash
flutter clean
flutter pub get
flutter run
```

---

# System Workflow

## Access Validation Flow

1. User approaches gate
2. Camera captures live feed
3. AI system performs:

   * Face recognition
   * Gesture verification
   * Liveness detection
4. System validates identity
5. Gate access decision is made
6. Relay system opens or locks gate
7. Event is logged in Firebase
8. Homeowner receives notification if required

---

# Alert & Notification System

The system sends real-time alerts to homeowners when:

* A valid OTP is used
* Suspicious activity is detected
* Multiple failed attempts occur
* Emergency lockdown is triggered

### Notifications Include

* Guest photo
* OTP usage details
* Security warnings
* Lockdown alerts
* Activity logs

---

# Multi-Gate Support

The system supports:

* Multiple independent gates
* Different permission levels per gate
* Simultaneous access requests
* Remote gate control
* Emergency gate release

Supported examples:

* Front Gate
* Garage Gate

---

# Access Control Features

| Feature                     | Description                         |
| --------------------------- | ----------------------------------- |
| Multi-factor authentication | Face + gesture + PIN verification   |
| Guest OTP system            | Temporary expiring access codes     |
| Auto-rotating homeowner PIN | Monthly rotating secure credentials |
| PIN fallback                | Backup authentication method        |
| Lockdown mode               | Triggered after failed attempts     |
| Real-time alerts            | Security event notifications        |
| Activity logging            | Access history monitoring           |
| Two-way communication       | Speaker + microphone integration    |

---

# AI System Integration

# Remaining Work

* Deploy optimized AI models to ESP32-CAM
* Finalize firmware integration
* Complete full-scale system testing
* Optimize real-time inference performance
* Expand mobile monitoring capabilities

---

# Future Improvements

* Push notifications
* Live camera streaming
* Voice communication support
* Biometric authentication
* Smart automation routines
* Access analytics dashboard
* Hardware telemetry integration

---

# Access Levels

| Homeowner            | Guest                     |
| -------------------- | ------------------------- |
| Full gate access     | Temporary OTP access      |
| View logs            | Limited gate access       |
| Generate guest OTPs  | Expiring credentials      |
| Manage users         | Restricted permissions    |
| Rotate security PINs | Single-use authentication |

---

# Contributors

* Yahia Tamer
* Nouran Hassan
* Malak Mohamed
* Roaa Khaled
* Daniel Michel

---

# License

This project is intended for educational and research purposes.
