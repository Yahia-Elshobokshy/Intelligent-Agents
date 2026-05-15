# AI-Vision Security Agent — Mobile Application

Smart gate access control system powered by Flutter, Firebase, and AI-driven security workflows.

---

## Overview

The AI-Vision Security Agent Mobile Application is a Flutter-based smart gate management platform designed to work alongside an intelligent AI-powered gate access system.

The application enables homeowners and authorized members to:

* Control gates remotely
* Manage secure PIN access
* Monitor gate activity in real time
* Handle user permissions
* Receive security alerts and notifications
* Manage guest and household access

The project integrates with a larger ecosystem that includes:

* AI face recognition
* Gesture recognition
* ESP32-CAM hardware
* OTP/PIN verification
* Smart gate automation

---

# Features

## Authentication & Session Management

* Firebase Authentication integration
* Real-time auth session monitoring
* Protected route navigation
* Automatic onboarding/setup flow
* Secure sign out with provider cleanup

## Smart House & Gate Management

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

# AI System Integration

The mobile application integrates with the AI-powered backend system that includes:

* Face recognition
* Gesture recognition
* Anti-spoofing verification
* OTP validation
* Emergency lockdown protocols
* Real-time monitoring

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
