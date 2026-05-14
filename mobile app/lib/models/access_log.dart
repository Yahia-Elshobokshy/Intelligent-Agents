import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';

class AccessLog {
  final String id;
  final DateTime timestamp;
  final String gateId;
  final String action;
  final String userName;

  AccessLog({
    required this.id,
    required this.timestamp,
    required this.gateId,
    required this.action,
    required this.userName,
  });

  factory AccessLog.fromMap(String id, Map<String, dynamic> map) {
    return AccessLog(
      id: id,
      timestamp: (map['timestamp'] as Timestamp?)?.toDate() ?? DateTime.now(),
      userName: map['user_name'] ?? 'Unknown User', // This fixes point 10
      action: map['action'] ?? 'accessed gate',
      gateId: map['gate_id'] ?? 'Gate',
    );
  }

  // THIS IS THE MISSING PIECE CAUSING YOUR ERROR
  Color get actionColor {
    final act = action.toLowerCase();
    if (act.contains('open')) return Colors.green;
    if (act.contains('close')) return Colors.orange;
    if (act.contains('denied')) return Colors.red;
    return Colors.blue; // Default color
  }

  String get formattedTime {
    final diff = DateTime.now().difference(timestamp);
    if (diff.inMinutes < 1) return 'Just now';
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
    if (diff.inHours < 24) return '${diff.inHours}h ago';
    return '${diff.inDays}d ago';
  }
}
