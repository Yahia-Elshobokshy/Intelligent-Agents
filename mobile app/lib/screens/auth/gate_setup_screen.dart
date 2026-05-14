// lib/screens/auth/gate_setup_screen.dart
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/app_theme.dart';
import '../../services/auth_service.dart';
import 'package:go_router/go_router.dart';

class GateSetupScreen extends ConsumerWidget {
  const GateSetupScreen({super.key});

  String _generateHouseId() {
    final random = Random();
    final id = 10000000 + random.nextInt(90000000);
    return 'HSE-$id';
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Align(
                alignment: Alignment.topRight,
                child: IconButton(
                  icon: const Icon(Icons.logout, color: AppTheme.grey900),
                  onPressed: () async =>
                      await ref.read(authServiceProvider).signOut(),
                ),
              ),
            ),
            Expanded(
              child: SingleChildScrollView(
                child: Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 24.0,
                    vertical: 40.0,
                  ),
                  child: Column(
                    children: [
                      const Icon(
                        Icons.security_update_good_rounded,
                        size: 80,
                        color: AppTheme.primary,
                      ),
                      const SizedBox(height: 24),
                      const Text(
                        "SECURE YOUR PERIMETER",
                        style: TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.w900,
                          letterSpacing: 1.5,
                        ),
                      ),
                      const SizedBox(height: 48),
                      _buildSetupOption(
                        context: context,
                        title: "CREATE NEW HOUSE",
                        subtitle: "Start fresh and manage your gates",
                        icon: Icons.add_home_work_rounded,
                        color: AppTheme.primary,
                        onTap: () => _handleCreateHouse(context, ref),
                      ),
                      const SizedBox(height: 16),
                      _buildSetupOption(
                        context: context,
                        title: "JOIN EXISTING HOUSE",
                        subtitle: "Use a code from your admin",
                        icon: Icons.group_add_rounded,
                        color: AppTheme.secondary,
                        onTap: () => _showJoinDialog(context, ref),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
  // lib/screens/auth/gate_setup_screen.dart
  // lib/screens/auth/gate_setup_screen.dart
  // Fix _handleCreateHouse method:

  Future<void> _handleCreateHouse(BuildContext context, WidgetRef ref) async {
    final messenger = ScaffoldMessenger.of(context);
    final router = GoRouter.of(context);

    try {
      await ref.read(authServiceProvider).createHouse("Home");

      // Force refresh
      ref.invalidate(currentUserProvider);

      if (context.mounted) {
        // Use go instead of router.go for consistency
        context.go('/');
        messenger.showSnackBar(
          const SnackBar(content: Text('House created successfully!')),
        );
      }
    } catch (e) {
      if (context.mounted) {
        messenger.showSnackBar(
          SnackBar(content: Text("Error: $e"), backgroundColor: Colors.red),
        );
      }
    }
  }
  // lib/screens/auth/gate_setup_screen.dart
  // Fix the _showJoinDialog method:

  void _showJoinDialog(BuildContext context, WidgetRef ref) {
    final controller = TextEditingController();
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Join House'),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(
            hintText: 'Enter HSE-XXXXXXXX',
            border: OutlineInputBorder(),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('CANCEL'),
          ),
          ElevatedButton(
            onPressed: () async {
              final code = controller.text.trim();
              Navigator.pop(ctx); // Close dialog FIRST

              // Small delay to allow dialog to close
              await Future.delayed(const Duration(milliseconds: 100));

              final success = await ref
                  .read(authServiceProvider)
                  .joinHouse(code);

              if (success) {
                // Force refresh the user profile
                ref.invalidate(currentUserProvider);

                if (context.mounted) {
                  // Use go_router's go method instead of Navigator
                  context.go('/'); // This will trigger router redirect

                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Successfully joined house!')),
                  );
                }
              } else if (context.mounted) {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text("Invalid House Code"),
                    backgroundColor: Colors.red,
                  ),
                );
              }
            },
            child: const Text('JOIN'),
          ),
        ],
      ),
    );
  }

  Widget _buildSetupOption({
    required BuildContext context,
    required String title,
    required String subtitle,
    required IconData icon,
    required Color color,
    required VoidCallback onTap,
  }) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
        child: Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            border: Border.all(color: AppTheme.grey200),
            borderRadius: BorderRadius.circular(16),
          ),
          child: Row(
            children: [
              CircleAvatar(
                backgroundColor: color.withOpacity(0.1),
                child: Icon(icon, color: color),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: const TextStyle(fontWeight: FontWeight.bold),
                    ),
                    Text(
                      subtitle,
                      style: const TextStyle(
                        fontSize: 12,
                        color: AppTheme.grey600,
                      ),
                    ),
                  ],
                ),
              ),
              const Icon(Icons.chevron_right, color: AppTheme.grey400),
            ],
          ),
        ),
      ),
    );
  }
}
