// lib/screens/logs/logs_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../services/log_service.dart';
import '../../widgets/log_tile.dart';
import '../../core/app_theme.dart';

class LogsScreen extends ConsumerWidget {
  const LogsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final logsAsync = ref.watch(accessLogsProvider);

    return Scaffold(
      backgroundColor: AppTheme.grey50,
      appBar: AppBar(
        title: const Text('ACTIVITY LOGS'),
        centerTitle: true,
        titleTextStyle: const TextStyle(
          color: AppTheme.grey900,
          fontSize: 14,
          fontWeight: FontWeight.w900,
          letterSpacing: 2.0,
        ),
      ),
      body: logsAsync.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (e, _) => Center(child: Text('Error: $e')),
        data: (logs) {
          if (logs.isEmpty) return const Center(child: Text('No activity.'));

          return ListView.builder(
            // The bottom padding (120) ensures the last item stops BEFORE the Nav Bar
            padding: const EdgeInsets.fromLTRB(20, 20, 20, 120),
            itemCount: logs.length,
            itemBuilder: (context, index) {
              return Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: LogTile(log: logs[index]),
              );
            },
          );
        },
      ),
    );
  }
}