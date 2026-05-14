// lib/screens/dashboard/dashboard_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../services/gate_service.dart';
import '../../services/auth_service.dart'; 
import '../../widgets/gate_card.dart';
import '../../models/gate.dart';
import '../../core/app_theme.dart';

class DashboardScreen extends ConsumerStatefulWidget {
  const DashboardScreen({super.key});

  @override
  ConsumerState<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends ConsumerState<DashboardScreen> {
  bool _isGlobalAction = false;

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: AppTheme.danger),
    );
  }

  // --- GATE MANAGEMENT METHODS ---

  void _showAddGateDialog(BuildContext context, WidgetRef ref) {
    final controller = TextEditingController();
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Add New Gate'),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(
            hintText: 'e.g., Main Entrance, Garage, Pool',
            labelText: 'Gate Name',
            border: OutlineInputBorder(),
          ),
          textCapitalization: TextCapitalization.words,
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('CANCEL')),
          ElevatedButton(
            onPressed: () {
              final name = controller.text.trim();
              if (name.isNotEmpty) {
                ref.read(gateServiceProvider).addGate(name);
                Navigator.pop(ctx);
              }
            },
            style: ElevatedButton.styleFrom(backgroundColor: AppTheme.primary),
            child: const Text('CREATE', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }

  void _showDeleteGateDialog(BuildContext context, WidgetRef ref, String gateId, String gateName) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Delete Gate?'),
        content: Text('Are you sure you want to remove "$gateName"? This cannot be undone.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('CANCEL')),
          TextButton(
            onPressed: () {
              ref.read(gateServiceProvider).deleteGate(gateId);
              Navigator.pop(ctx);
            },
            child: const Text('DELETE', style: TextStyle(color: AppTheme.danger)),
          ),
        ],
      ),
    );
  }

  // --- GLOBAL COMMANDS ---

  Future<void> _openAllGates(List<Gate> gates) async {
    if (_isGlobalAction || gates.isEmpty) return;
    setState(() => _isGlobalAction = true);
    try {
      final gateIds = gates.map((g) => g.id).toList();
      await ref.read(gateServiceProvider).openAllGates(gateIds);
    } catch (e) {
      _showError(e.toString());
    } finally {
      if (mounted) setState(() => _isGlobalAction = false);
    }
  }

  Future<void> _closeAllGates(List<Gate> gates) async {
    if (_isGlobalAction || gates.isEmpty) return;
    setState(() => _isGlobalAction = true);
    try {
      final gateIds = gates.map((g) => g.id).toList();
      await ref.read(gateServiceProvider).closeAllGates(gateIds);
    } catch (e) {
      _showError(e.toString());
    } finally {
      if (mounted) setState(() => _isGlobalAction = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final gatesAsync = ref.watch(gatesStreamProvider);
    final userProfile = ref.watch(currentUserProvider).value;
    final isAdmin = userProfile?.role == 'admin';

    return Scaffold(
      backgroundColor: Colors.transparent, 
      floatingActionButton: isAdmin
          ? FloatingActionButton.extended(
              onPressed: () => _showAddGateDialog(context, ref),
              backgroundColor: AppTheme.primary,
              icon: const Icon(Icons.add, color: Colors.white),
              label: const Text("ADD GATE", style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
            )
          : null,
      body: gatesAsync.when(
        loading: () => const Center(child: CircularProgressIndicator(color: AppTheme.primary)),
        error: (err, stack) => Center(child: Text("Connection Error: $err")),
        data: (gates) {
          return RefreshIndicator(
            onRefresh: () async => ref.refresh(gatesStreamProvider),
            child: ListView(
              padding: const EdgeInsets.fromLTRB(20, 20, 20, 100),
              children: [
                _buildHeader(userProfile?.name ?? "User"),
                const SizedBox(height: 24),
                
                if (gates.isEmpty) 
                  _buildEmptyState(isAdmin)
                else ...[
                  _buildGlobalControls(gates),
                  const SizedBox(height: 24),
                  const Text("INDIVIDUAL GATES", 
                    style: TextStyle(
                      fontSize: 11, 
                      fontWeight: FontWeight.w800, 
                      color: AppTheme.grey600, 
                      letterSpacing: 1.5
                    )),
                  const SizedBox(height: 16),
                  ...gates.map((gate) => Padding(
                    padding: const EdgeInsets.only(bottom: 12.0),
                    child: GestureDetector(
                      onLongPress: isAdmin ? () => _showDeleteGateDialog(context, ref, gate.id, gate.name) : null,
                      child: GateCard(gate: gate),
                    ),
                  )),
                ],
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildHeader(String name) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text("WELCOME HOME,", 
          style: TextStyle(
            fontSize: 12, 
            fontWeight: FontWeight.w800, 
            color: AppTheme.primary.withValues(alpha: 0.7), 
            letterSpacing: 2
          )),
        Text(name.toUpperCase(), 
          style: const TextStyle(fontSize: 28, fontWeight: FontWeight.w900, color: AppTheme.grey900)),
      ],
    );
  }

  Widget _buildEmptyState(bool isAdmin) {
    return Center(
      child: Column(
        children: [
          const SizedBox(height: 60),
          const Icon(Icons.door_sliding_outlined, size: 64, color: AppTheme.grey400),
          const SizedBox(height: 16),
          Text(isAdmin ? "No gates added yet." : "No gates assigned to your house.", 
            style: const TextStyle(color: AppTheme.grey600, fontWeight: FontWeight.w600)),
          if (isAdmin) const Text("Click 'Add Gate' to get started.", style: TextStyle(color: AppTheme.grey400, fontSize: 12)),
        ],
      ),
    );
  }

  Widget _buildGlobalControls(List<Gate> gates) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.04), blurRadius: 20, offset: const Offset(0, 10))],
      ),
      child: Row(
        children: [
          Expanded(
            child: _buildActionButton(
              label: 'OPEN ALL',
              icon: Icons.lock_open_rounded,
              color: AppTheme.primary,
              onPressed: () => _openAllGates(gates),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: _buildActionButton(
              label: 'CLOSE ALL',
              icon: Icons.lock_rounded,
              color: AppTheme.secondary,
              onPressed: () => _closeAllGates(gates),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildActionButton({required String label, required IconData icon, required Color color, required VoidCallback onPressed}) {
    return SizedBox(
      height: 54,
      child: ElevatedButton(
        onPressed: _isGlobalAction ? null : onPressed,
        style: ElevatedButton.styleFrom(
          backgroundColor: color,
          foregroundColor: Colors.white,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          padding: EdgeInsets.zero,
        ),
        child: _isGlobalAction 
          ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
          : Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(icon, size: 18),
                const SizedBox(width: 8),
                Flexible(child: FittedBox(child: Text(label, style: const TextStyle(fontWeight: FontWeight.bold, letterSpacing: 0.5)))),
              ],
            ),
      ),
    );
  }
}