// lib/services/mock_gate_controller.dart
import 'dart:async';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

final mockGateControllerProvider = Provider<MockGateController>((ref) => MockGateController(ref));

class MockGateController {
  final Ref _ref;
  Timer? _pollingTimer;

  MockGateController(this._ref);

  void startMocking() {
    _pollingTimer = Timer.periodic(const Duration(seconds: 2), (timer) async {
      await _checkAndExecuteCommands();
    });
    debugPrint('✅ Mock ESP32 controller started');
  }

  Future<void> _checkAndExecuteCommands() async {
    // Get all houses that have gates
    final housesSnapshot = await FirebaseFirestore.instance.collection('houses').get();
    
    for (final houseDoc in housesSnapshot.docs) {
      final houseId = houseDoc.id;
      final gatesRef = FirebaseFirestore.instance
          .collection('houses')
          .doc(houseId)
          .collection('gates');
      
      final gatesSnapshot = await gatesRef.get();
      
      for (final gateDoc in gatesSnapshot.docs) {
        final command = gateDoc.data()['command'] as String? ?? 'none';
        final currentStatus = gateDoc.data()['status'] as String? ?? 'closed';
        
        if (command != 'none') {
          debugPrint('🔧 [$houseId] ESP32 executing: $command on ${gateDoc.id}');
          
          String newStatus;
          switch (command) {
            case 'open':
              newStatus = 'open';
              break;
            case 'close':
              newStatus = 'closed';
              break;
            case 'emergency_open':
              newStatus = 'open';
              break;
            default:
              newStatus = currentStatus;
          }
          
          await Future.delayed(const Duration(milliseconds: 500));
          
          await gateDoc.reference.update({
            'status': newStatus,
            'command': 'none',
            'last_updated': FieldValue.serverTimestamp(),
          });
          
          debugPrint('✅ Gate ${gateDoc.id} status changed to $newStatus');
          
          // Log to house-specific logs
          await FirebaseFirestore.instance
              .collection('houses')
              .doc(houseId)
              .collection('access_logs')
              .add({
            'timestamp': FieldValue.serverTimestamp(),
            'gate_id': gateDoc.id,
            'action': '${command}_via_remote',
            'user_name': 'system',
          });
        }
      }
    }
  }

  void dispose() {
    _pollingTimer?.cancel();
  }
}