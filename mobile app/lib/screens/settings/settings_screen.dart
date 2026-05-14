// lib/screens/settings/settings_screen.dart
import '../../screens/admin/users_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/app_theme.dart';
import '../../services/auth_service.dart';
import '../../services/pin_service.dart';
import '../../core/router.dart';
import '../../models/user_pin.dart';

class SettingsScreen extends ConsumerWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final userProfileAsync = ref.watch(currentUserProvider);
    final userProfile = userProfileAsync.valueOrNull;
    final isAdmin = userProfile?.role == 'admin';

    final currentPinAsync = ref.watch(currentUserPINProvider);

    return Scaffold(
      backgroundColor: AppTheme.grey50,
      appBar: AppBar(
        title: const Text('SYSTEM MANAGEMENT'),
        centerTitle: true,
        titleTextStyle: const TextStyle(
          color: AppTheme.grey900,
          fontSize: 14,
          fontWeight: FontWeight.w900,
          letterSpacing: 2.0,
        ),
      ),
      body: userProfileAsync.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (e, _) => Center(child: Text('Error loading profile: $e')),
        data: (user) => ListView(
          padding: const EdgeInsets.fromLTRB(20, 20, 20, 120),
          children: [
            _buildUserProfile(user),
            const SizedBox(height: 32),

            if (isAdmin) ...[
              _buildSectionHeader('ACCESS CONTROL'),
              const SizedBox(height: 16),
              _buildMenuCard([
                _buildMenuItem(
                  icon: Icons.people_alt_rounded,
                  title: 'Manage Residents',
                  subtitle: 'Manage roles and gate access',
                  onTap: () => Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => const UsersScreen(),
                    ),
                  ),
                ),
              ]),
              const SizedBox(height: 32),
            ],

            _buildSectionHeader('MY ACCESS PIN'),
            const SizedBox(height: 16),
            _buildPINCard(context, ref, currentPinAsync),
            const SizedBox(height: 32),

            _buildSectionHeader('ACCOUNT'),
            const SizedBox(height: 16),
            _buildMenuCard([
              _buildMenuItem(
                icon: Icons.home_work_rounded,
                title: 'House ID',
                subtitle: user?.houseId ?? 'Not Linked',
                onTap: () {},
              ),
            ]),
            const SizedBox(height: 32),
            Center(child: _buildLogoutButton(context, ref)),
          ],
        ),
      ),
    );
  }

  Widget _buildPINCard(
    BuildContext context,
    WidgetRef ref,
    AsyncValue<UserPIN?> pinAsync,
  ) {
    return pinAsync.when(
      loading: () => Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: AppTheme.grey200),
        ),
        child: const Center(child: CircularProgressIndicator()),
      ),
      error: (e, _) => Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: AppTheme.grey200),
        ),
        child: Column(
          children: [
            const Icon(Icons.error_outline, color: AppTheme.danger),
            const SizedBox(height: 8),
            Text('Error: $e'),
            ElevatedButton(
              onPressed: () => ref.invalidate(currentUserPINProvider),
              child: const Text('Retry'),
            ),
          ],
        ),
      ),
      data: (pin) {
        final isExpired = pin?.isExpired ?? true;
        final daysRemaining = pin?.daysRemaining ?? 0;
        final pinCode = pin?.pinCode ?? '----';

        return Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: isExpired ? AppTheme.danger : AppTheme.grey200,
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Column(
                  children: [
                    Text(
                      pinCode,
                      style: const TextStyle(
                        fontSize: 36,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 4,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      isExpired
                          ? 'EXPIRED'
                          : 'Expires in $daysRemaining days',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        color: isExpired ? AppTheme.danger : AppTheme.primary,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: () => _generatePin(context, ref),
                  icon: const Icon(Icons.refresh),
                  label: const Text('GENERATE NEW PIN'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.primary,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 12),
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Future<void> _generatePin(BuildContext context, WidgetRef ref) async {
    try {
      await ref.read(pinServiceProvider).rotatePIN();
      ref.invalidate(currentUserPINProvider);
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e'), backgroundColor: Colors.red),
        );
      }
    }
  }

  Widget _buildUserProfile(dynamic user) {
    final bool isAdmin = user?.role == 'admin';
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: isAdmin ? AppTheme.grey900 : Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: isAdmin ? null : Border.all(color: AppTheme.grey200),
      ),
      child: Row(
        children: [
          CircleAvatar(
            radius: 25,
            backgroundColor: isAdmin
                ? AppTheme.primary.withValues(alpha: 0.2)
                : AppTheme.grey100,
            child: Icon(
              isAdmin ? Icons.shield_rounded : Icons.person_rounded,
              color: isAdmin ? AppTheme.primary : AppTheme.grey600,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  isAdmin ? 'SYSTEM ADMINISTRATOR' : 'RESIDENT MEMBER',
                  style: TextStyle(
                    color: isAdmin ? AppTheme.primary : AppTheme.grey600,
                    fontSize: 10,
                    fontWeight: FontWeight.w900,
                    letterSpacing: 1,
                  ),
                ),
                Text(
                  user?.name ?? 'User',
                  style: TextStyle(
                    color: isAdmin ? Colors.white : AppTheme.grey900,
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
                Text(
                  user?.email ?? '',
                  style: TextStyle(
                    color: isAdmin ? AppTheme.grey400 : AppTheme.grey600,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionHeader(String title) {
    return Text(
      title,
      style: const TextStyle(
        fontSize: 11,
        fontWeight: FontWeight.w800,
        color: AppTheme.grey600,
        letterSpacing: 1.5,
      ),
    );
  }

  Widget _buildMenuCard(List<Widget> children) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.grey200),
      ),
      child: Column(children: children),
    );
  }

  Widget _buildMenuItem({
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return ListTile(
      onTap: onTap,
      leading: Icon(icon, color: AppTheme.grey900, size: 20),
      title: Text(
        title,
        style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
      ),
      subtitle: Text(
        subtitle,
        style: const TextStyle(fontSize: 12, color: AppTheme.grey600),
      ),
      trailing: const Icon(
        Icons.chevron_right_rounded,
        color: AppTheme.grey400,
      ),
    );
  }

  Widget _buildLogoutButton(BuildContext context, WidgetRef ref) {
    return TextButton.icon(
      onPressed: () => _showLogoutDialog(context, ref),
      icon: const Icon(Icons.logout_rounded, size: 18, color: AppTheme.danger),
      label: const Text(
        'LOG OUT',
        style: TextStyle(color: AppTheme.danger, fontWeight: FontWeight.bold),
      ),
    );
  }

  void _showLogoutDialog(BuildContext context, WidgetRef ref) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Confirm Logout'),
        content: const Text(
          'Are you sure you want to sign out?',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('CANCEL'),
          ),
          TextButton(
            onPressed: () async {
              Navigator.pop(ctx);
              ref.invalidate(currentUserProvider);
              ref.invalidate(routerProvider);
              await ref.read(authServiceProvider).signOut();
            },
            child: const Text(
              'LOG OUT',
              style: TextStyle(color: AppTheme.danger),
            ),
          ),
        ],
      ),
    );
  }
}