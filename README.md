# Firebase Setup Guide

This project already includes the source code and the `google-services.json` file.

Follow the steps below to connect your local environment to Firebase and run the application.

---

# Step 1 — Install Required Tools

Run the following commands:

```bash
flutter pub get
npm install -g firebase-tools
dart pub global activate flutterfire_cli
```

---

# Step 2 — Login to Firebase

Authenticate your machine using the Firebase account that has been added to the project.

```bash
firebase login
```

---

# Step 3 — Configure FlutterFire

Generate the local Firebase configuration files:

```bash
flutterfire configure
```

When prompted:

- Select the correct Firebase project
- Select `android` as the platform
- Choose `Yes` if asked to overwrite files

This generates the required `firebase_options.dart` file.

---

# Step 4 — Generate SHA-1 Fingerprint

Android Firebase services require your local SHA-1 fingerprint.

Run:

```bash
keytool -list -v \
-keystore ~/.android/debug.keystore \
-alias androiddebugkey \
-storepass android \
-keypass android
```

Copy the generated `SHA1` fingerprint and send it to the project administrator.

The administrator must add it to:

```text
Firebase Console
→ Project Settings
→ Android App
→ Add Fingerprint
```

---

# Step 5 — Run the Project

After the SHA-1 fingerprint is added:

```bash
flutter clean
flutter pub get
flutter run
```

---

# Notes

- Make sure Flutter SDK is installed correctly
- Ensure Android Studio / VS Code Flutter extensions are installed
- Firebase Authentication and Firestore must already be enabled in the Firebase Console
- Internet connection is required during initial Firebase setup

---

# Troubleshooting

## FlutterFire command not found

Run:

```bash
dart pub global activate flutterfire_cli
```

Then restart the terminal.

---

## Firebase login issues

Run:

```bash
firebase logout
firebase login
```

---

## SHA-1 not working

Delete and regenerate the debug keystore:

```bash
rm ~/.android/debug.keystore
```

Then rebuild the project and generate the SHA-1 again.

---

# Project Stack

- Flutter
- Dart
- Firebase Authentication
- Cloud Firestore
- Riverpod
- GoRouter
- ESP32-CAM Integration
