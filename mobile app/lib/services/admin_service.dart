// lib/services/admin_service.dart
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/app_user.dart';
import 'auth_service.dart';

final adminServiceProvider = Provider<AdminService>((ref) {
  final houseId = ref.watch(currentUserProvider).valueOrNull?.houseId;
  return AdminService(houseId: houseId, ref: ref);
});

final allUsersProvider = StreamProvider<List<AppUser>>((ref) {
  final houseId = ref.watch(currentUserProvider).valueOrNull?.houseId;
  if (houseId == null || houseId.isEmpty) return Stream.value([]);
  return ref.watch(adminServiceProvider).watchHouseUsers(houseId);
});

class AdminService {
  final FirebaseFirestore _db = FirebaseFirestore.instance;
  final String? houseId;
  final Ref _ref;

  AdminService({required this.houseId, required Ref ref}) : _ref = ref;

  Stream<List<AppUser>> watchHouseUsers(String houseId) {
    return _db
        .collection('users')
        .where('house_id', isEqualTo: houseId)
        .snapshots()
        .map((snap) => snap.docs
            .map((doc) => AppUser.fromMap(doc.id, doc.data()))
            .toList());
  }

  Future<void> promoteToAdmin(String newAdminId) async {
    if (houseId == null) return;
    final batch = _db.batch();

    try {
      final currentAdminQuery = await _db
          .collection('users')
          .where('role', isEqualTo: 'admin')
          .where('house_id', isEqualTo: houseId)
          .get();

      for (var doc in currentAdminQuery.docs) {
        batch.update(doc.reference, {'role': 'member'});
      }

      batch.update(_db.collection('users').doc(newAdminId), {'role': 'admin'});
      await batch.commit();
    } catch (e) {
      debugPrint('Error swapping admin: $e');
    }
  }

  // ✅ FIXED: This method name matches what users_screen.dart calls
  Future<void> deleteUser(String userId) async {
    await _db.collection('users').doc(userId).update({
      'house_id': null,
      'role': 'member',
    });
    
    // Also delete their PIN from the house
    if (houseId != null) {
      await _db
          .collection('houses')
          .doc(houseId)
          .collection('user_pins')
          .doc(userId)
          .delete();
    }
  }

  Future<void> removeFromHouse(String userId) async {
    await _db.collection('users').doc(userId).update({
      'house_id': null,
      'role': 'member',
    });
    
    if (houseId != null) {
      await _db
          .collection('houses')
          .doc(houseId)
          .collection('user_pins')
          .doc(userId)
          .delete();
    }
  }

  Future<bool> isCurrentUserAdmin(String currentUserId) async {
    final doc = await _db.collection('users').doc(currentUserId).get();
    return doc.data()?['role'] == 'admin';
  }
}