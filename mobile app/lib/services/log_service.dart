// lib/services/log_service.dart
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/access_log.dart';
import 'auth_service.dart';

final logServiceProvider = Provider<LogService>((ref) {
  final houseId = ref.watch(currentUserProvider).valueOrNull?.houseId;
  return LogService(houseId: houseId);
});

final accessLogsProvider = StreamProvider<List<AccessLog>>((ref) {
  final houseId = ref.watch(currentUserProvider).valueOrNull?.houseId;
  if (houseId == null || houseId.isEmpty) return Stream.value([]);

  return FirebaseFirestore.instance
      .collection('houses')
      .doc(houseId)
      .collection('access_logs')
      .orderBy('timestamp', descending: true)
      .snapshots()
      .map((snapshot) => snapshot.docs
          .map((doc) => AccessLog.fromMap(doc.id, doc.data()))
          .toList());
});

class LogService {
  final FirebaseFirestore _db = FirebaseFirestore.instance;
  final String? houseId;

  LogService({required this.houseId});

  CollectionReference get _logsRef {
    if (houseId == null || houseId!.isEmpty) {
      throw Exception('No house linked. Cannot access logs.');
    }
    return _db.collection('houses').doc(houseId).collection('access_logs');
  }

  Future<void> addLog({
    required String gateId,
    required String action,
    required String userName,
  }) async {
    await _logsRef.add({
      'timestamp': FieldValue.serverTimestamp(),
      'gate_id': gateId,
      'action': action,
      'user_name': userName,
    });
  }

  Future<void> clearLogs() async {
    final batch = _db.batch();
    final snap = await _logsRef.get();
    for (final doc in snap.docs) {
      batch.delete(doc.reference);
    }
    await batch.commit();
  }
}