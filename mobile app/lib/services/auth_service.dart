// lib/services/auth_service.dart
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:uuid/uuid.dart';
import '../models/app_user.dart';

final authServiceProvider = Provider<AuthService>((ref) => AuthService());

final authStateProvider = StreamProvider<User?>((ref) {
  return ref.watch(authServiceProvider).authStateChanges;
});

final currentUserProvider = StreamProvider<AppUser?>((ref) {
  final authUser = ref.watch(authStateProvider).valueOrNull;
  if (authUser == null) return Stream.value(null);

  return FirebaseFirestore.instance
      .collection('users')
      .doc(authUser.uid)
      .snapshots()
      .map((doc) {
        if (!doc.exists || doc.data() == null) return null;
        return AppUser.fromMap(doc.id, doc.data()!);
      });
});

class AuthService {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final FirebaseFirestore _db = FirebaseFirestore.instance;

  Stream<User?> get authStateChanges => _auth.authStateChanges();

  User? get currentUser => _auth.currentUser;

  Future<UserCredential> signIn(String email, String password) {
    return _auth.signInWithEmailAndPassword(email: email, password: password);
  }
  // lib/services/auth_service.dart - Update signUp method

  Future<UserCredential> signUp({
    required String email,
    required String password,
    required String phone,
    required String name,
  }) async {
    final userCredential = await _auth.createUserWithEmailAndPassword(
      email: email,
      password: password,
    );

    final user = userCredential.user!;
    final avatarUrl = _generateAvatar(name);

    final appUser = AppUser(
      uid: user.uid,
      email: email,
      phone: phone,
      name: name,
      avatarUrl: avatarUrl,
      role: 'member',
      createdAt: DateTime.now(),
      isVerified: false,
      houseId: null,
    );

    // Write to Firestore
    await _db.collection('users').doc(user.uid).set(appUser.toMap());

    // 🔥 CRITICAL: Wait for Firestore to acknowledge the write
    // This ensures the document exists before the stream picks it up
    await Future.delayed(const Duration(milliseconds: 500));

    // 🔥 Also trigger a manual refresh of the user's profile
    // Force the stream to re-emit
    final docSnapshot = await _db.collection('users').doc(user.uid).get();
    if (docSnapshot.exists) {
      print('✅ Profile confirmed in Firestore for ${user.email}');
    } else {
      print('⚠️ Profile not found immediately after write!');
    }

    return userCredential;
  }

  Future<void> createHouse(String houseName) async {
    final user = _auth.currentUser;
    if (user == null) return;

    final String houseId =
        'HSE-${const Uuid().v4().substring(0, 8).toUpperCase()}';
    final now = FieldValue.serverTimestamp();
    final batch = _db.batch();

    // 1. Create House document
    final houseRef = _db.collection('houses').doc(houseId);
    batch.set(houseRef, {
      'id': houseId,
      'house_name': houseName,
      'admin_uid': user.uid,
      'created_at': now,
      'homeowner_pin': {
        'code': '1234',
        'updated_at': now,
        'updated_by': user.uid,
      },
    });

    // 2. Update user as admin
    final userRef = _db.collection('users').doc(user.uid);
    batch.set(userRef, {
      'house_id': houseId,
      'role': 'admin',
    }, SetOptions(merge: true));

    await batch.commit();

    // 3. Create default gates for this house
    await _createDefaultGates(houseId);
  }

  Future<void> _createDefaultGates(String houseId) async {
    final gatesCollection = _db
        .collection('houses')
        .doc(houseId)
        .collection('gates');

    await gatesCollection.add({
      'name': 'Main Gate',
      'status': 'closed',
      'command': 'none',
      'last_updated': FieldValue.serverTimestamp(),
    });

    await gatesCollection.add({
      'name': 'Garage Gate',
      'status': 'closed',
      'command': 'none',
      'last_updated': FieldValue.serverTimestamp(),
    });
  }

  Future<bool> joinHouse(String houseId) async {
    final user = _auth.currentUser;
    if (user == null) return false;

    final houseDoc = await _db.collection('houses').doc(houseId).get();
    if (!houseDoc.exists) return false;

    await _db.collection('users').doc(user.uid).set({
      'house_id': houseId,
      'role': 'member',
    }, SetOptions(merge: true));

    return true;
  }

  Future<void> signOut() => _auth.signOut();

  String _generateAvatar(String name) {
    final initial = name.isNotEmpty ? name[0].toUpperCase() : 'U';
    return 'https://ui-avatars.com/api/?name=$initial&background=006A6A&color=fff&size=128';
  }
}
