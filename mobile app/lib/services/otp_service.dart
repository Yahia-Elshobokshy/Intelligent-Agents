// lib/services/otp_service.dart
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../models/otp.dart';
import 'auth_service.dart';

final otpServiceProvider = Provider<OTPService>((ref) {
  // Pass the houseId reactively from the current user profile
  final houseId = ref.watch(currentUserProvider).valueOrNull?.houseId;
  return OTPService(houseId: houseId);
});

// Only shows OTPs for the current user's house
final activeOTPsProvider = StreamProvider<List<OTP>>((ref) {
  final houseId = ref.watch(currentUserProvider).valueOrNull?.houseId;
  if (houseId == null || houseId.isEmpty) return Stream.value([]);
  return ref.watch(otpServiceProvider).watchActiveOTPs();
});

class OTPService {
  final FirebaseFirestore _db = FirebaseFirestore.instance;
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final String? houseId;

  OTPService({required this.houseId});

  // Returns the base reference for this house's OTP subcollection
  CollectionReference get _otpsRef {
    if (houseId == null || houseId!.isEmpty) {
      throw Exception('No house linked. Cannot access OTPs.');
    }
    return _db.collection('houses').doc(houseId).collection('otps');
  }

  // Returns the base reference for this house's access_logs subcollection
  CollectionReference get _logsRef {
    if (houseId == null || houseId!.isEmpty) {
      throw Exception('No house linked. Cannot access logs.');
    }
    return _db.collection('houses').doc(houseId).collection('access_logs');
  }

  Stream<List<OTP>> watchActiveOTPs() {
    return _otpsRef
        .where('expires_at', isGreaterThan: Timestamp.fromDate(DateTime.now()))
        .where('used', isEqualTo: false)
        .orderBy('expires_at', descending: true)
        .snapshots()
        .map((snap) => snap.docs
            .map((doc) => OTP.fromMap(doc.id, doc.data() as Map<String, dynamic>))
            .toList());
  }

  Future<String> generateOTP(String gateId) async {
    final code = _generate6DigitCode();
    final now = DateTime.now();
    final expiresAt = now.add(const Duration(minutes: 15));
    final userEmail = _auth.currentUser?.email ?? 'unknown';

    final docRef = _otpsRef.doc();
    final otp = OTP(
      id: docRef.id,
      code: code,
      gateId: gateId,
      createdAt: now,
      expiresAt: expiresAt,
      used: false,
      createdBy: userEmail,
    );

    // Write OTP to house-isolated subcollection
    await docRef.set(otp.toMap());

    // Write log to house-isolated subcollection
    await _logsRef.add({
      'timestamp': FieldValue.serverTimestamp(),
      'gate_id': gateId,
      'action': 'otp_generated',
      'user_name': userEmail,
    });

    return code;
  }

  String _generate6DigitCode() {
    final random = DateTime.now().microsecondsSinceEpoch % 1000000;
    return (random % 900000 + 100000).toString();
  }

  Future<void> markAsUsed(String otpId) async {
    await _otpsRef.doc(otpId).update({'used': true});
  }
}