// lib/services/pin_service.dart
import 'dart:math';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'auth_service.dart';
import '../models/user_pin.dart';

final pinServiceProvider = Provider<PINService>((ref) => PINService(ref));

final currentUserPINProvider = FutureProvider<UserPIN?>((ref) async {
  final user = ref.watch(currentUserProvider).valueOrNull;
  if (user?.houseId == null || user?.uid == null) return null;
  // Auto-generates if missing or expired — always returns a valid PIN
  return ref.read(pinServiceProvider).getOrCreateCurrentUserPIN(user!.houseId!, user.uid);
});

class PINService {
  final Ref _ref;
  final FirebaseFirestore _db = FirebaseFirestore.instance;

  PINService(this._ref);

  CollectionReference _getPinsRef(String houseId) {
    return _db.collection('houses').doc(houseId).collection('user_pins');
  }

  /// Fetches raw PIN doc without auto-generating.
  Future<UserPIN?> getCurrentUserPIN(String houseId, String userId) async {
    final doc = await _getPinsRef(houseId).doc(userId).get();
    if (!doc.exists) return null;
    return UserPIN.fromMap(userId, doc.data() as Map<String, dynamic>);
  }

  /// Always returns a valid PIN — auto-generates if missing or expired.
  /// This is what [currentUserPINProvider] uses so the 14-day rotation
  /// is fully automatic: no manual action required.
  Future<UserPIN?> getOrCreateCurrentUserPIN(String houseId, String userId) async {
    final existing = await getCurrentUserPIN(houseId, userId);

    if (existing != null && !existing.isExpired && existing.isActive) {
      return existing; // Still valid, return as-is
    }

    // Expired, inactive, or never existed — generate a fresh PIN
    final newPinCode = await _writePIN(houseId, userId);
    return getCurrentUserPIN(houseId, userId);
  }

  /// Rotates PIN for the CURRENT logged-in user (manual regeneration).
  Future<String> rotatePIN() async {
    final userProfile = _ref.read(currentUserProvider).valueOrNull;
    if (userProfile == null) throw Exception("User not found");
    if (userProfile.houseId == null) throw Exception("No house linked");

    return await _writePIN(userProfile.houseId!, userProfile.uid);
  }

  /// Rotates PIN for a specific user (admin use).
  Future<String> rotatePINForUser(String userId) async {
    final userProfile = _ref.read(currentUserProvider).valueOrNull;
    if (userProfile?.houseId == null) throw Exception("No house linked");

    return await _writePIN(userProfile!.houseId!, userId);
  }

  /// Internal: writes a new 4-digit PIN to Firestore and returns the code.
  Future<String> _writePIN(String houseId, String userId) async {
    final newPin = (1000 + Random().nextInt(9000)).toString();
    final now = DateTime.now();
    final expiresAt = now.add(const Duration(days: 14));

    await _getPinsRef(houseId).doc(userId).set({
      'pin_code': newPin,
      'created_at': Timestamp.fromDate(now),
      'expires_at': Timestamp.fromDate(expiresAt),
      'is_active': true,
    });

    return newPin;
  }

  Future<String> getOrCreatePINForUser(String userId) async {
    final userProfile = _ref.read(currentUserProvider).valueOrNull;
    if (userProfile?.houseId == null) throw Exception("No house linked");

    final existing = await getCurrentUserPIN(userProfile!.houseId!, userId);

    if (existing != null && !existing.isExpired && existing.isActive) {
      return existing.pinCode;
    }

    return await rotatePINForUser(userId);
  }

  /// Admin: rotate ALL pins in house.
  Future<void> rotateAllHousePins() async {
    final userProfile = _ref.read(currentUserProvider).valueOrNull;
    if (userProfile?.houseId == null) throw Exception("No house linked");
    if (userProfile?.role != 'admin') throw Exception("Only admin can rotate all pins");

    final usersSnapshot = await _db
        .collection('users')
        .where('house_id', isEqualTo: userProfile!.houseId)
        .get();

    for (final userDoc in usersSnapshot.docs) {
      await rotatePINForUser(userDoc.id);
    }
  }

  Future<bool> verifyPIN(String houseId, String enteredPin) async {
    final pinsSnapshot = await _getPinsRef(houseId)
        .where('is_active', isEqualTo: true)
        .get();

    for (final doc in pinsSnapshot.docs) {
      final pin = UserPIN.fromMap(doc.id, doc.data() as Map<String, dynamic>);
      if (pin.pinCode == enteredPin && !pin.isExpired && pin.isActive) {
        return true;
      }
    }
    return false;
  }
}