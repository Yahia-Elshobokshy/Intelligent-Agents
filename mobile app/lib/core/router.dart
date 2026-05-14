// lib/core/router.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../screens/auth/login_screen.dart';
import '../screens/auth/register_screen.dart';
import '../screens/auth/gate_setup_screen.dart';
import '../screens/main_navigation_screen.dart';
import '../services/auth_service.dart';

class _RouterNotifier extends AsyncNotifier<void> implements Listenable {
  VoidCallback? _routerListener;

  @override
  Future<void> build() async {
    ref.listen(authStateProvider, (_, __) => notifyListeners());
    ref.listen(currentUserProvider, (_, __) => notifyListeners());
  }

  @override
  void addListener(VoidCallback listener) {
    _routerListener = listener;
  }

  @override
  void removeListener(VoidCallback listener) {
    _routerListener = null;
  }

  void notifyListeners() => _routerListener?.call();
}

final _routerNotifierProvider =
    AsyncNotifierProvider<_RouterNotifier, void>(_RouterNotifier.new);

final routerProvider = Provider<GoRouter>((ref) {
  final notifier = ref.watch(_routerNotifierProvider.notifier);

  return GoRouter(
    initialLocation: '/login',
    refreshListenable: notifier,
    debugLogDiagnostics: true,
    redirect: (context, state) {
      final authState = ref.read(authStateProvider);
      final profileAsync = ref.read(currentUserProvider);
      
      print('🔀 ROUTER: Checking redirect...');
      print('   authState: ${authState.valueOrNull?.uid ?? 'null'}');
      print('   profileAsync: ${profileAsync.valueOrNull?.uid ?? 'null'}');
      print('   profileAsync loading: ${profileAsync.isLoading}');
      print('   current location: ${state.matchedLocation}');

      final isOnLogin = state.matchedLocation == '/login';
      final isOnRegister = state.matchedLocation == '/register';
      final isOnSetup = state.matchedLocation == '/setup';

      // CASE 1: Auth is still loading → wait
      if (authState.isLoading) {
        print('   ⏳ Auth loading, waiting...');
        return null;
      }

      final user = authState.valueOrNull;

      // CASE 2: No user → go to login
      if (user == null) {
        print('   ❌ No user, redirecting to /login');
        if (isOnLogin || isOnRegister) return null;
        return '/login';
      }

      // CASE 3: User exists but profile is STILL LOADING → wait (DON'T sign out!)
      if (profileAsync.isLoading) {
        print('   ⏳ Profile still loading, waiting...');
        return null;
      }

      // CASE 4: Profile failed to load (error)
      if (profileAsync.hasError) {
        print('   ❌ Profile error: ${profileAsync.error}');
        // Don't sign out automatically, just wait or show error
        return null;
      }

      final profile = profileAsync.valueOrNull;

      // CASE 5: User exists but NO profile document found → This is the problem!
      if (profile == null) {
        print('   ⚠️ User exists but NO Firestore profile!');
        print('   ⚠️ This should not happen - profile should be created on signup');
        // Instead of signing out, show an error dialog
        WidgetsBinding.instance.addPostFrameCallback((_) {
          // Navigate to an error screen or show snackbar
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Profile not found. Please contact support.'), backgroundColor: Colors.red),
          );
        });
        return null; // Stay where you are
      }

      // CASE 6: Logged in but no house linked → force setup
      final hasNoHouse = profile.houseId == null || profile.houseId!.trim().isEmpty;
      if (hasNoHouse) {
        print('   🏠 No house linked (houseId: ${profile.houseId}), redirecting to /setup');
        if (isOnSetup) return null;
        return '/setup';
      }

      // CASE 7: Fully set up → go to home
      print('   ✅ Fully set up, user: ${profile.name}, house: ${profile.houseId}');
      if (isOnSetup || isOnLogin || isOnRegister) {
        print('   ➡️ Redirecting to /');
        return '/';
      }

      return null;
    },
    routes: [
      GoRoute(
        path: '/',
        name: 'home',
        builder: (context, state) => const MainNavigationScreen(),
      ),
      GoRoute(
        path: '/login',
        name: 'login',
        builder: (context, state) => const LoginScreen(),
      ),
      GoRoute(
        path: '/register',
        name: 'register',
        builder: (context, state) => const RegisterScreen(),
      ),
      GoRoute(
        path: '/setup',
        name: 'setup',
        builder: (context, state) => const GateSetupScreen(),
      ),
    ],
  );
});