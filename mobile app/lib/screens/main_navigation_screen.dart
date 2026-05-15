// lib/screens/main_navigation_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../core/app_theme.dart';
import 'dashboard/dashboard_screen.dart';
import 'otp/otp_screen.dart';
import 'logs/logs_screen.dart';
import 'settings/settings_screen.dart';
import 'intercom/intercom_screen.dart';          // ← new

class MainNavigationScreen extends ConsumerStatefulWidget {
  const MainNavigationScreen({super.key});

  @override
  ConsumerState<MainNavigationScreen> createState() =>
      _MainNavigationScreenState();
}

class _MainNavigationScreenState
    extends ConsumerState<MainNavigationScreen> {
  int _selectedIndex = 0;

  final List<Widget> _screens = const [
    DashboardScreen(),
    OTPScreen(),
    IntercomScreen(),      // ← new (index 2)
    LogsScreen(),
    SettingsScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.grey50,
      body: Column(
        children: [
          Expanded(
            child: IndexedStack(
              index: _selectedIndex,
              children: _screens,
            ),
          ),
          _buildBottomZone(),
        ],
      ),
    );
  }

  Widget _buildBottomZone() {
    return Container(
      padding: const EdgeInsets.fromLTRB(24, 0, 24, 30),
      color: AppTheme.grey50,
      child: Container(
        height: 70,
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(25),
          border: Border.all(color: AppTheme.grey200),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.05),
              blurRadius: 15,
              offset: const Offset(0, 5),
            ),
          ],
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: [
            _navItem(0, Icons.grid_view_rounded,      'Home'),
            _navItem(1, Icons.vpn_key_rounded,         'Access'),
            _navItem(2, Icons.videocam_rounded,        'Intercom'),  // ← new
            _navItem(3, Icons.assignment_rounded,      'Logs'),
            _navItem(4, Icons.manage_accounts_rounded, 'Setup'),
          ],
        ),
      ),
    );
  }

  Widget _navItem(int index, IconData icon, String label) {
    final bool isSelected = _selectedIndex == index;
    final Color color =
        isSelected ? AppTheme.primary : AppTheme.grey400;

    // Intercom tab gets a subtle live-dot badge when not selected
    final bool isIntercom = index == 2;

    return InkWell(
      onTap: () => setState(() => _selectedIndex = index),
      child: Stack(
        clipBehavior: Clip.none,
        children: [
          Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, color: color, size: 24),
              const SizedBox(height: 2),
              Text(
                label,
                style: TextStyle(
                  color: color,
                  fontSize: 10,
                  fontWeight:
                      isSelected ? FontWeight.bold : FontWeight.w500,
                ),
              ),
            ],
          ),
          // Live indicator dot on Intercom tab when not active
          if (isIntercom && !isSelected)
            Positioned(
              top: -2,
              right: -4,
              child: Container(
                width: 8,
                height: 8,
                decoration: const BoxDecoration(
                  shape: BoxShape.circle,
                  color: Color(0xFF4ADE80),
                ),
              ),
            ),
        ],
      ),
    );
  }
}