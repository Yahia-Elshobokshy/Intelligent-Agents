// lib/models/user_pin.dart
import 'package:cloud_firestore/cloud_firestore.dart';

class UserPIN {
  final String userId;
  final String pinCode;
  final DateTime createdAt;
  final DateTime expiresAt;
  final bool isActive;

  UserPIN({
    required this.userId,
    required this.pinCode,
    required this.createdAt,
    required this.expiresAt,
    required this.isActive,
  });

  factory UserPIN.fromMap(String userId, Map<String, dynamic> map) {
    return UserPIN(
      userId: userId,
      pinCode: map['pin_code'] ?? '',
      createdAt: (map['created_at'] as Timestamp?)?.toDate() ?? DateTime.now(),
      expiresAt: (map['expires_at'] as Timestamp?)?.toDate() ?? DateTime.now(),
      isActive: map['is_active'] ?? true,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'pin_code': pinCode,
      'created_at': Timestamp.fromDate(createdAt),
      'expires_at': Timestamp.fromDate(expiresAt),
      'is_active': isActive,
    };
  }

  bool get isExpired => DateTime.now().isAfter(expiresAt);
  int get daysRemaining => expiresAt.difference(DateTime.now()).inDays;
}