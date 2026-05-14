import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/app_theme.dart';
import '../../models/app_user.dart';
import '../../services/admin_service.dart';

class UsersScreen extends ConsumerWidget {
  const UsersScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // Watching the real stream from Firestore
    final usersAsync = ref.watch(allUsersProvider);

    return Scaffold(
      backgroundColor: AppTheme.grey50,
      appBar: AppBar(
        title: const Text('RESIDENT DIRECTORY'),
        centerTitle: true,
        titleTextStyle: const TextStyle(
          color: AppTheme.grey900,
          fontSize: 13,
          fontWeight: FontWeight.w900,
          letterSpacing: 2.0,
        ),
      ),
      body: usersAsync.when(
        data: (residents) => ListView(
          padding: const EdgeInsets.fromLTRB(20, 20, 20, 120),
          children: [
            ...residents.map((user) => _buildResidentTile(context, ref, user)),
          ],
        ),
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (err, stack) => Center(child: Text('Error: $err')),
      ),
    );
  }

  Widget _buildResidentTile(BuildContext context, WidgetRef ref, AppUser user) {
    final bool isAdmin = user.role == 'admin';
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.grey200),
      ),
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: isAdmin ? AppTheme.grey900 : AppTheme.grey100,
          child: Icon(
            isAdmin ? Icons.shield_rounded : Icons.person_rounded,
            color: isAdmin ? AppTheme.primary : AppTheme.grey600,
          ),
        ),
        title: Text(
          user.name,
          style: const TextStyle(fontWeight: FontWeight.bold),
        ),
        subtitle: Text(
          user.role.toUpperCase(),
          style: const TextStyle(fontSize: 12),
        ),
        trailing: IconButton(
          icon: const Icon(Icons.more_vert_rounded, color: AppTheme.grey400),
          onPressed: () => _showUserOptions(context, ref, user),
        ),
      ),
    );
  }

  void _showUserOptions(BuildContext context, WidgetRef ref, AppUser user) {
    showModalBottomSheet(
      context: context,
      builder: (ctx) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (user.role != 'admin')
              ListTile(
                leading: const Icon(Icons.verified_user_outlined),
                title: const Text('Promote to Admin'),
                onTap: () async {
                  await ref.read(adminServiceProvider).promoteToAdmin(user.uid);
                  if (context.mounted) Navigator.pop(context);
                },
              ),
            ListTile(
              leading: const Icon(
                Icons.delete_outline_rounded,
                color: AppTheme.danger,
              ),
              title: const Text(
                'Revoke Access',
                style: TextStyle(color: AppTheme.danger),
              ),
              onTap: () async {
                await ref.read(adminServiceProvider).deleteUser(user.uid);
                if (context.mounted) Navigator.pop(context);
              },
            ),
          ],
        ),
      ),
    );
  }
}
