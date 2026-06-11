// lib/widgets/gate_card.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/gate.dart';
import '../services/gate_service.dart';

// MQTT
import '../services/mqtt_gate_service.dart';
import '../services/auth_service.dart';

class GateCard extends ConsumerWidget {
  final Gate gate;
  const GateCard({super.key, required this.gate});

  Color _statusColor() {
    switch (gate.status) {
      case 'open':
        return Colors.green;
      case 'locked':
        return Colors.red;
      default:
        return Colors.orange;
    }
  }

  IconData _statusIcon() {
    switch (gate.status) {
      case 'open':
        return Icons.lock_open;
      case 'locked':
        return Icons.lock;
      default:
        return Icons.door_front_door;
    }
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // MQTT
    final gateService = ref.read(gateServiceProvider);
    final mqtt = ref.watch(mqttGateServiceProvider);
    final user = ref.read(currentUserProvider).valueOrNull;
    final houseId = user?.houseId ?? '';
    final userName = user?.name ?? 'Unknown';

    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(_statusIcon(), color: _statusColor(), size: 28),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        gate.name,
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Text(
                        gate.status.toUpperCase(),
                        style: TextStyle(color: _statusColor(), fontSize: 12),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    
                    // MQTT
                   onPressed: () async {
                      if (!mqtt.isConnected) await mqtt.connect();
                      gateService.openGate(gate.id, houseId, mqtt, userName);
                    },

                    icon: const Icon(Icons.lock_open, size: 18),
                    label: const Text('Open'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.green,
                      foregroundColor: Colors.white,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: ElevatedButton.icon(

                    // MQTT
                   onPressed: () async {
                      if (!mqtt.isConnected) await mqtt.connect();
                      gateService.closeGate(gate.id, houseId, mqtt, userName);
                    },

                    icon: const Icon(Icons.lock, size: 18),
                    label: const Text('Close'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.orange,
                      foregroundColor: Colors.white,
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
