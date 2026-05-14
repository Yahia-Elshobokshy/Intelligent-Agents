// lib/models/app_user.dart
import 'package:cloud_firestore/cloud_firestore.dart';

class AppUser {
  final String uid;
  final String email;
  final String phone;
  final String name;
  final String avatarUrl;
  final String role;
  final DateTime createdAt;
  final bool isVerified;
  final String? houseId;

  AppUser({
    required this.uid,
    required this.email,
    required this.phone,
    required this.name,
    required this.avatarUrl,
    required this.role,
    required this.createdAt,
    required this.isVerified,
    this.houseId, 
  });

  Map<String, dynamic> toMap() {
    return {
      'uid': uid,
      'email': email,
      'phone': phone,
      'name': name,
      'avatar_url': avatarUrl,
      'role': role,
      'created_at': Timestamp.fromDate(createdAt),
      'is_verified': isVerified,
      'house_id': houseId, // Matches Firestore
    };
  }

  factory AppUser.fromMap(String uid, Map<String, dynamic> map) {
    return AppUser(
      uid: uid,
      email: map['email'] ?? '',
      phone: map['phone'] ?? '',
      name: map['name'] ?? '',
      avatarUrl: map['avatar_url'] ?? '',
      role: map['role'] ?? 'member',
      createdAt: (map['created_at'] as Timestamp?)?.toDate() ?? DateTime.now(),
      isVerified: map['is_verified'] ?? false,
      houseId: map['house_id'], // CRITICAL: This must match what you save in setup
    );
  }
}