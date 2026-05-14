// lib/main.dart
import 'package:ai_vision_gate/core/router.dart';
import 'package:ai_vision_gate/services/mock_gate_controller.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'firebase_options.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  
  // 🔥 OPTIONAL: Uncomment to wipe ALL data on app start (DEBUG ONLY!)
  // await wipeAllFirebaseData();
  
  runApp(const ProviderScope(child: MyApp()));
}

// 🧹 Emergency wipe function - DEBUG ONLY!
Future<void> wipeAllFirebaseData() async {
  print('⚠️ WARNING: Wiping all Firebase data...');
  
  // 1. Delete all users
  final auth = FirebaseAuth.instance;
  final user = auth.currentUser;
  if (user != null) {
    await user.delete().catchError((e) => print('Could not delete current user: $e'));
  }
  
  // 2. Delete all Firestore collections
  final firestore = FirebaseFirestore.instance;
  final collections = ['users', 'houses', 'gates', 'access_logs'];
  
  for (final collection in collections) {
    final snapshot = await firestore.collection(collection).get();
    for (final doc in snapshot.docs) {
      await doc.reference.delete();
    }
    print('✅ Deleted collection: $collection');
  }
  
  print('✅ All Firebase data wiped!');
}

class MyApp extends ConsumerStatefulWidget {
  const MyApp({super.key});

  @override
  ConsumerState<MyApp> createState() => _MyAppState();
}

class _MyAppState extends ConsumerState<MyApp> {
  @override
  void initState() {
    super.initState();
    Future.microtask(() {
      ref.read(mockGateControllerProvider).startMocking();
    });
  }

  @override
  void dispose() {
    ref.read(mockGateControllerProvider).dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final router = ref.watch(routerProvider);
    
    return MaterialApp.router(
      title: 'AI Vision Gate',
      debugShowCheckedModeBanner: false,
      routerConfig: router,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF006A6A)),
        useMaterial3: true,
      ),
    );
  }
}