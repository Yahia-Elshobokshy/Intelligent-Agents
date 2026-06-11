// lib/services/gate_service.dart
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/gate.dart';
import 'auth_service.dart';

// MQTT
import 'mqtt_gate_service.dart';

final gateServiceProvider = Provider<GateService>((ref) {
  final houseId = ref.watch(currentUserProvider).valueOrNull?.houseId;
  return GateService(houseId: houseId);
});

final gatesStreamProvider = StreamProvider<List<Gate>>((ref) {
  final houseId = ref.watch(currentUserProvider).valueOrNull?.houseId;
  if (houseId == null || houseId.isEmpty) return Stream.value([]);
  return ref.watch(gateServiceProvider).watchGates();
});

class GateService {
  final FirebaseFirestore _db = FirebaseFirestore.instance;
  final String? houseId;

  GateService({required this.houseId});

  CollectionReference get _gatesRef {
    if (houseId == null || houseId!.isEmpty) {
      throw Exception('No house linked. Cannot access gates.');
    }
    return _db.collection('houses').doc(houseId).collection('gates');
  }

  Stream<List<Gate>> watchGates() {
    return _gatesRef.snapshots().map(
      (snap) => snap.docs
          .map(
            (doc) => Gate.fromMap(doc.id, doc.data() as Map<String, dynamic>),
          )
          .toList(),
    );
  }

  Future<void> addGate(String name) async {
    await _gatesRef.add({
      'name': name,
      'status': 'locked',
      'command': 'none',
      'last_updated': FieldValue.serverTimestamp(),
      'is_active': true,
    });
  }

  Future<void> deleteGate(String gateId) async {
    await _gatesRef.doc(gateId).delete();
  }

// MQTT Open Gate + write to firebase
  Future<void> openGate(
    String gateId,
    String houseId,
    MqttGateService mqtt,
    String userName,
  ) async {
    mqtt.sendCommand(houseId, gateId, 'open');
    await _db.collection('houses').doc(houseId).collection('access_logs').add({
      'timestamp': FieldValue.serverTimestamp(),
      'gate_id': gateId,
      'action': 'open_via_remote',
      'user_name': userName,
    });
  }

// MQTT Close Gate + write to firebase
  Future<void> closeGate(
    String gateId,
    String houseId,
    MqttGateService mqtt,
    String userName,
  ) async {
    mqtt.sendCommand(houseId, gateId, 'close');
    await _db.collection('houses').doc(houseId).collection('access_logs').add({
      'timestamp': FieldValue.serverTimestamp(),
      'gate_id': gateId,
      'action': 'close_via_remote',
      'user_name': userName,
    });
  }

  Future<void> openAllGates(List<String> gateIds) async {
    final batch = _db.batch();
    for (final id in gateIds) {
      batch.update(_gatesRef.doc(id), {
        'command': 'open',
        'last_updated': FieldValue.serverTimestamp(),
      });
    }
    await batch.commit();
  }

  Future<void> closeAllGates(List<String> gateIds) async {
    final batch = _db.batch();
    for (final id in gateIds) {
      batch.update(_gatesRef.doc(id), {
        'command': 'close',
        'last_updated': FieldValue.serverTimestamp(),
      });
    }
    await batch.commit();
  }
}
