// lib/models/pin.dart
class HomeownerPIN {
  final String userId;
  final String email;
  final String pin;
  final DateTime createdAt;
  final DateTime expiresAt;
  final bool isActive;

  HomeownerPIN({
    required this.userId,
    required this.email,
    required this.pin,
    required this.createdAt,
    required this.expiresAt,
    required this.isActive,
  });

  factory HomeownerPIN.fromMap(String userId, Map<String, dynamic> map) {
    return HomeownerPIN(
      userId: userId,
      email: map['email'] ?? '',
      pin: map['pin'] ?? '',
      createdAt: (map['created_at'] as dynamic)?.toDate() ?? DateTime.now(),
      expiresAt: (map['expires_at'] as dynamic)?.toDate() ?? DateTime.now(),
      isActive: map['is_active'] ?? true,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'email': email,
      'pin': pin,
      'created_at': createdAt,
      'expires_at': expiresAt,
      'is_active': isActive,
    };
  }

  bool get isExpired => DateTime.now().isAfter(expiresAt);
  int get daysRemaining => expiresAt.difference(DateTime.now()).inDays;
}