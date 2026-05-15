// lib/widgets/log_tile.dart
import 'package:flutter/material.dart';
import '../models/access_log.dart';
import '../core/app_theme.dart';

class LogTile extends StatelessWidget {
  final AccessLog log;
  const LogTile({super.key, required this.log});

  @override
  Widget build(BuildContext context) {
    // Dynamically display the gate name. If it's a custom name from Firestore, it shows that.
    final String gateDisplay = log.gateId.toUpperCase();

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.grey200),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            // Status Icon: Changes color based on whether it was an 'open' or 'close' action
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: log.actionColor.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(
                log.action.contains('open') 
                    ? Icons.lock_open_rounded 
                    : Icons.lock_outline_rounded, 
                color: log.actionColor, 
                size: 22
              ),
            ),
            const SizedBox(width: 16),
            
            // Log Content
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Wrap(
                    spacing: 6,
                    runSpacing: 4,
                    crossAxisAlignment: WrapCrossAlignment.center,
                    children: [
                      Text(
                        gateDisplay,
                        style: const TextStyle(
                          fontWeight: FontWeight.w900, 
                          fontSize: 13, 
                          letterSpacing: 0.5
                        ),
                      ),
                      _buildActionBadge(),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Triggered by: ${log.userName}',
                    style: const TextStyle(
                      fontSize: 12, 
                      color: AppTheme.grey600, 
                      fontWeight: FontWeight.w500
                    ),
                  ),
                ],
              ),
            ),

            // Time Info
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text(
                  log.formattedTime,
                  style: const TextStyle(
                    fontSize: 11, 
                    fontWeight: FontWeight.bold, 
                    color: AppTheme.grey400
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildActionBadge() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: log.actionColor,
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(
        log.action.toUpperCase(),
        style: const TextStyle(
          color: Colors.white, 
          fontSize: 9, 
          fontWeight: FontWeight.w900
        ),
      ),
    );
  }
}