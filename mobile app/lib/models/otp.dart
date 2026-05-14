// lib/models/otp.dart
class OTP {
  final String id;
  final String code;
  final String gateId;
  final DateTime createdAt;
  final DateTime expiresAt;
  final bool used;
  final String createdBy; // homeowner email

  OTP({
    required this.id,
    required this.code,
    required this.gateId,
    required this.createdAt,
    required this.expiresAt,
    required this.used,
    required this.createdBy,
  });

  factory OTP.fromMap(String id, Map<String, dynamic> map) {
    return OTP(
      id: id,
      code: map['code'] ?? '',
      gateId: map['gate_id'] ?? '',
      createdAt: (map['created_at'] as dynamic)?.toDate() ?? DateTime.now(),
      expiresAt: (map['expires_at'] as dynamic)?.toDate() ?? DateTime.now(),
      used: map['used'] ?? false,
      createdBy: map['created_by'] ?? '',
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'code': code,
      'gate_id': gateId,
      'created_at': createdAt,
      'expires_at': expiresAt,
      'used': used,
      'created_by': createdBy,
    };
  }

  bool get isExpired => DateTime.now().isAfter(expiresAt);
  Duration get timeRemaining => expiresAt.difference(DateTime.now());
}