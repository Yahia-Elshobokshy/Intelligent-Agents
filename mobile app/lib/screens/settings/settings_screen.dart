// lib/screens/settings/settings_screen.dart
import 'package:flutter/services.dart';

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
    // 1. Get the real user profile from Firestore
    final userProfileAsync = ref.watch(currentUserProvider);
    final userProfile = userProfileAsync.valueOrNull;
    final isAdmin = userProfile?.role == 'admin';
    
    // Get current user's PIN info (for ALL users)
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
            // 2. Dynamic Profile Card
            _buildUserProfile(user),
            const SizedBox(height: 32),

            // 3. Admin-Only Sections
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

            // 4. PIN Section - FOR ALL USERS (members + admins)
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
                onTap: () {
                  // Optional: Copy to clipboard logic
                },
              ),
            ]),
            const SizedBox(height: 32),
            Center(child: _buildLogoutButton(context, ref)),
          ],
        ),
      ),
    );
  }

  // NEW: PIN Card for all users
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
              // PIN Display
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
              // Generate New Button
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: () => _showGeneratePinDialog(context, ref),
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

  // Dialog for generating new PIN
  Future<void> _showGeneratePinDialog(BuildContext context, WidgetRef ref) async {
    final shouldGenerate = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Generate New PIN'),
        content: const Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.security_update_good, size: 48, color: AppTheme.primary),
            SizedBox(height: 16),
            Text('This will generate a new 4-digit PIN.'),
            SizedBox(height: 8),
            Text(
              'Old PIN will expire immediately.',
              style: TextStyle(fontSize: 12, color: AppTheme.grey600),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('CANCEL'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: ElevatedButton.styleFrom(backgroundColor: AppTheme.primary),
            child: const Text('GENERATE', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );

    if (shouldGenerate == true && context.mounted) {
      try {
        final newPin = await ref.read(pinServiceProvider).rotatePIN();
        ref.invalidate(currentUserPINProvider);
        
        if (context.mounted) {
          showDialog(
            context: context,
            builder: (ctx) => AlertDialog(
              title: const Text('NEW PIN GENERATED'),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text('Your new PIN is:'),
                  const SizedBox(height: 16),
                  Text(
                    newPin,
                    style: const TextStyle(
                      fontSize: 32,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 4,
                      color: AppTheme.primary,
                    ),
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    'Valid for 14 days',
                    style: TextStyle(fontSize: 12, color: AppTheme.grey600),
                  ),
                ],
              ),
              actions: [
                TextButton(
                  onPressed: () {
                    Clipboard.setData(ClipboardData(text: newPin));
                    Navigator.pop(ctx);
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('PIN copied!')),
                    );
                  },
                  child: const Text('COPY'),
                ),
                ElevatedButton(
                  onPressed: () => Navigator.pop(ctx),
                  child: const Text('OK'),
                ),
              ],
            ),
          );
        }
      } catch (e) {
        if (context.mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Error: $e'), backgroundColor: Colors.red),
          );
        }
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
          'Are you sure you want to sign out? This will clear your current session.',
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