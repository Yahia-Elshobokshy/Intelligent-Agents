// lib/widgets/debug_panel.dart
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

class DebugPanel extends StatefulWidget {
  const DebugPanel({super.key});

  @override
  State<DebugPanel> createState() => _DebugPanelState();
}

class _DebugPanelState extends State<DebugPanel> {
  @override
  Widget build(BuildContext context) {
    if (!kDebugMode) return const SizedBox.shrink();
    
    return Container(
      margin: const EdgeInsets.all(8),
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.grey[900],
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          Text('🐛 DEBUG', style: TextStyle(color: Colors.green[300], fontSize: 12)),
          _DebugButton(
            label: 'Test Notification',
            onTap: () async {
              await FirebaseFirestore.instance.collection('alerts').add({
                'type': 'failed_attempts',
                'gate_id': 'front_gate',
                'timestamp': FieldValue.serverTimestamp(),
                'photo_url': '',
                'message': '3 failed face recognition attempts at Front Gate',
                'read': false,
              });
              if (mounted) {
                // ignore: use_build_context_synchronously
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Test alert created!')),
                );
              }
            },
          ),
          _DebugButton(
            label: 'Reset Gates',
            onTap: () async {
              final gates = ['front_gate', 'garage_gate'];
              for (var gate in gates) {
                await FirebaseFirestore.instance.collection('gates').doc(gate).update({
                  'status': 'closed',
                  'command': 'none',
                });
              }
              if (mounted) {
                // ignore: use_build_context_synchronously
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Gates reset to closed')),
                );
              }
            },
          ),
        ],
      ),
    );
  }
}

class _DebugButton extends StatelessWidget {
  final String label;
  final VoidCallback onTap;
  
  const _DebugButton({required this.label, required this.onTap});
  
  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        decoration: BoxDecoration(
          color: Colors.blue[800],
          borderRadius: BorderRadius.circular(4),
        ),
        child: Text(label, style: const TextStyle(fontSize: 11, color: Colors.white)),
      ),
    );
  }
}