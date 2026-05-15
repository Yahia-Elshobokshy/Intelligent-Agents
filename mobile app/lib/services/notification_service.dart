// lib/services/notification_service.dart
import 'dart:convert';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

class NotificationService {
  static final _messaging = FirebaseMessaging.instance;
  static final _localNotifications = FlutterLocalNotificationsPlugin();
  static final _firestore = FirebaseFirestore.instance;
  static final _auth = FirebaseAuth.instance;

  static const _channel = AndroidNotificationChannel(
    'gate_alerts',
    'Gate Alerts',
    description: 'Security alerts for your gates',
    importance: Importance.max,
    playSound: true,
  );

  // Active listeners — kept so we can cancel them on logout
  static final List<Stream> _activeStreams = [];

  static Future<void> initialize() async {
    await _messaging.requestPermission(alert: true, badge: true, sound: true);

    await _localNotifications.initialize(
      const InitializationSettings(
        android: AndroidInitializationSettings('@mipmap/ic_launcher'),
        iOS: DarwinInitializationSettings(),
      ),
    );

    await _localNotifications
        .resolvePlatformSpecificImplementation<AndroidFlutterLocalNotificationsPlugin>()
        ?.createNotificationChannel(_channel);

    // Show notifications when app is in foreground
    FirebaseMessaging.onMessage.listen((message) {
      _showLocal(
        message.notification?.title ?? 'Gate Alert',
        message.notification?.body ?? '',
      );
    });
  }

  // Call this after login to start watching Firestore
static Future<void> startListening() async {
  final uid = _auth.currentUser?.uid;
  print('🔔 startListening called, uid: $uid');
  if (uid == null) return;

  final userDoc = await _firestore.collection('users').doc(uid).get();
  final houseId = userDoc.data()?['houseId'];
  print('🔔 houseId found: $houseId');
  if (houseId == null) return;

  print('🔔 Starting gate listener for house: $houseId');
  _listenToGates(houseId);
  _listenToFailedAttempts(houseId);
  _listenToOTPs(houseId);
}

  // --- LISTENER 1: Gate status changes ---
  static void _listenToGates(String houseId) {
    print('🔔 Listening to: houses/$houseId/gates');
    final stream = _firestore
        .collection('houses')
        .doc(houseId)
        .collection('gates')
        .snapshots();

    // Track previous states to detect changes
    final Map<String, String> previousStates = {};
    bool isFirstSnapshot = true;

    stream.listen((snapshot) {
      // Skip the first snapshot (just initializing state)
      if (isFirstSnapshot) {
        for (final doc in snapshot.docs) {
          previousStates[doc.id] = doc.data()['status'] ?? '';
        }
        isFirstSnapshot = false;
        return;
      }

      for (final change in snapshot.docChanges) {
        if (change.type == DocumentChangeType.modified) {
          final data = change.doc.data()!;
          final newStatus = data['status'] ?? '';
          final oldStatus = previousStates[change.doc.id] ?? '';

          if (newStatus != oldStatus) {
            final gateName = data['name'] ?? 'Gate';
            final triggeredBy = data['lastTriggeredBy'] ?? 'System';
            final isOpen = newStatus == 'open';

            _showLocal(
              
              isOpen ? '🔓 Gate Opened' : '🔒 Gate Closed',
              '$gateName was ${isOpen ? 'opened' : 'closed'} by $triggeredBy',
            );

            previousStates[change.doc.id] = newStatus;
          }
        }
        // Update state for new gates too
        if (change.type == DocumentChangeType.added) {
          previousStates[change.doc.id] =
              change.doc.data()?['status'] ?? '';
        }
      }
    });
  }

  // --- LISTENER 2: Failed attempts ---
  static void _listenToFailedAttempts(String houseId) {
    // Only listen to documents created after app starts
    final startTime = Timestamp.now();

    _firestore
        .collection('houses')
        .doc(houseId)
        .collection('failed_attempts')
        .where('timestamp', isGreaterThan: startTime)
        .snapshots()
        .listen((snapshot) {
      for (final change in snapshot.docChanges) {
        if (change.type == DocumentChangeType.added) {
          final data = change.doc.data()!;
          final gateName = data['gateName'] ?? 'your gate';
          final reason = data['reason'] ?? '';

          _showLocal(
            '⚠️ Failed Access Attempt',
            reason == 'wrong_otp'
                ? 'Someone entered an invalid code at $gateName'
                : 'Unauthorized access attempt at $gateName',
          );
        }
      }
    });
  }

  // --- LISTENER 3: OTP used ---
  static void _listenToOTPs(String houseId) {
    _firestore
        .collection('houses')
        .doc(houseId)
        .collection('otps')
        .snapshots()
        .listen((snapshot) {
      for (final change in snapshot.docChanges) {
        if (change.type == DocumentChangeType.modified) {
          final data = change.doc.data()!;
          // Only notify when usedAt is newly set
          if (data['usedAt'] != null) {
            final code = data['code'] ?? 'Unknown';
            final gateName = data['gateName'] ?? 'a gate';
            _showLocal(
              '🔑 Guest Access Used',
              'OTP $code was used to open $gateName',
            );
          }
        }
      }
    });
  }

  static Future<void> _showLocal(String title, String body) async {
    print('🔔 Showing notification: $title - $body');
    await _localNotifications.show(
      DateTime.now().millisecondsSinceEpoch ~/ 1000,
      title,
      body,
      NotificationDetails(
        android: AndroidNotificationDetails(
          _channel.id,
          _channel.name,
          channelDescription: _channel.description,
          importance: Importance.max,
          priority: Priority.high,
          icon: '@mipmap/ic_launcher',
        ),
        iOS: const DarwinNotificationDetails(
          presentAlert: true,
          presentBadge: true,
          presentSound: true,
        ),
      ),
    );
  }

  static void dispose() {
    _activeStreams.clear();
  }
}

final notificationServiceProvider = Provider((_) => NotificationService());