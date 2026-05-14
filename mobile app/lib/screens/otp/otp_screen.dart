// lib/screens/otp/otp_screen.dart
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:share_plus/share_plus.dart'; 
import '../../services/otp_service.dart';
import '../../models/otp.dart';
import '../../core/app_theme.dart';
import '../../services/gate_service.dart';
import '../../models/gate.dart';

class OTPScreen extends ConsumerStatefulWidget {
  const OTPScreen({super.key});

  @override
  ConsumerState<OTPScreen> createState() => _OTPScreenState();
}

class _OTPScreenState extends ConsumerState<OTPScreen> {
  String? _generatedCode;
  String? _selectedGateId;
  bool _isGenerating = false;
  DateTime? _expiresAt;

  Future<void> _generateOTP() async {
    if (_selectedGateId == null) {
      _showSnackBar('Select a terminal to authorize', AppTheme.secondary);
      return;
    }

    setState(() => _isGenerating = true);

    try {
      final otpService = ref.read(otpServiceProvider);
      final code = await otpService.generateOTP(_selectedGateId!);

      setState(() {
        _generatedCode = code;
        _expiresAt = DateTime.now().add(const Duration(minutes: 15));
      });

      await Clipboard.setData(ClipboardData(text: code));
      _showSnackBar('Access code copied to clipboard', AppTheme.primary);
    } catch (e) {
      _showSnackBar('System Error: $e', AppTheme.danger);
    } finally {
      if (mounted) setState(() => _isGenerating = false);
    }
  }

  void _showSnackBar(String message, Color color) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: color)
    );
  }

  @override
  Widget build(BuildContext context) {
    final gatesAsync = ref.watch(gatesStreamProvider);

    return Scaffold(
      backgroundColor: AppTheme.grey50,
      appBar: AppBar(
        title: const Text('GUEST ACCESS'),
        centerTitle: true,
        titleTextStyle: const TextStyle(
          color: AppTheme.grey900,
          fontSize: 14,
          fontWeight: FontWeight.w900,
          letterSpacing: 2.0,
        ),
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(20, 20, 20, 40),
        children: [
          _buildInfoBanner(),
          const SizedBox(height: 32),
          const Text(
            'SELECT TERMINAL',
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w800,
              color: AppTheme.grey600,
              letterSpacing: 1.5,
            ),
          ),
          const SizedBox(height: 16),
          gatesAsync.when(
            data: (gates) => gates.isEmpty 
              ? const Center(child: Text("No gates found in your house."))
              : Column(children: gates.map((gate) => _buildTerminalCard(gate)).toList()),
            loading: () => const Center(child: CircularProgressIndicator()),
            error: (e, _) => Text('Error loading terminals: $e'),
          ),
          const SizedBox(height: 24),
          _buildActionStack(),
          if (_generatedCode != null) _buildCodeDisplay(),
          const SizedBox(height: 40),
          const Text(
            'ACTIVE AUTHORIZATIONS',
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w800,
              color: AppTheme.grey600,
              letterSpacing: 1.5,
            ),
          ),
          const SizedBox(height: 16),
          const _ActiveOTPsList(),
        ],
      ),
    );
  }

  Widget _buildTerminalCard(Gate gate) {
    final isSelected = _selectedGateId == gate.id;
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: InkWell(
        onTap: () => setState(() {
          _selectedGateId = gate.id;
          _generatedCode = null;
        }),
        borderRadius: BorderRadius.circular(12),
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
color: isSelected ? AppTheme.primary.withValues(alpha: 0.05) : Colors.white,            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: isSelected ? AppTheme.primary : AppTheme.grey200),
          ),
          child: Row(
            children: [
              Icon(
                isSelected ? Icons.radio_button_checked : Icons.radio_button_off,
                color: isSelected ? AppTheme.primary : AppTheme.grey400,
              ),
              const SizedBox(width: 12),
              Text(gate.name, style: TextStyle(fontWeight: isSelected ? FontWeight.bold : FontWeight.normal)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildActionStack() {
    return SizedBox(
      width: double.infinity,
      height: 56,
      child: ElevatedButton(
        onPressed: _isGenerating ? null : _generateOTP,
        style: ElevatedButton.styleFrom(backgroundColor: AppTheme.primary),
        child: _isGenerating ? const CircularProgressIndicator(color: Colors.white) : const Text('GENERATE ACCESS CODE'),
      ),
    );
  }

  Widget _buildCodeDisplay() {
    return Container(
      margin: const EdgeInsets.only(top: 24),
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: AppTheme.grey200),
      ),
      child: Column(
        children: [
          Text(_generatedCode!, style: const TextStyle(fontSize: 42, fontWeight: FontWeight.bold, letterSpacing: 6)),
          const SizedBox(height: 16),
          if (_expiresAt != null) _CountdownTimer(expiresAt: _expiresAt!),
          const SizedBox(height: 24),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              OutlinedButton.icon(
                onPressed: () => Clipboard.setData(ClipboardData(text: _generatedCode!)),
                icon: const Icon(Icons.copy),
                label: const Text('COPY'),
              ),
              const SizedBox(width: 12),
              OutlinedButton.icon(
                onPressed: () => Share.share("Gate Access Code: $_generatedCode"),
                icon: const Icon(Icons.share),
                label: const Text('SHARE'),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildInfoBanner() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16), border: Border.all(color: AppTheme.grey200)),
      child: const Row(
        children: [
          Icon(Icons.security, color: AppTheme.primary),
          SizedBox(width: 12),
          Text("Codes valid for 15 minutes", style: TextStyle(color: AppTheme.grey600)),
        ],
      ),
    );
  }
}

class _ActiveOTPsList extends ConsumerWidget {
  const _ActiveOTPsList();
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final activeOTPsAsync = ref.watch(activeOTPsProvider);
    return activeOTPsAsync.when(
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (e, _) => Text('Error: $e'),
      data: (otps) => ListView.builder(
        shrinkWrap: true, // Fixes Overflow inside ScrollView
        physics: const NeverScrollableScrollPhysics(),
        itemCount: otps.length,
        itemBuilder: (context, index) => _ActiveOtpTile(otp: otps[index]),
      ),
    );
  }
}

class _ActiveOtpTile extends StatelessWidget {
  final OTP otp;
  const _ActiveOtpTile({required this.otp});
  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12), border: Border.all(color: AppTheme.grey200)),
      child: Row(
        children: [
          const Icon(Icons.key, color: AppTheme.grey400),
          const SizedBox(width: 12),
          Text(otp.code, style: const TextStyle(fontWeight: FontWeight.bold)),
          const Spacer(),
          Text('${otp.timeRemaining.inMinutes}m left', style: const TextStyle(color: AppTheme.primary, fontSize: 12)),
        ],
      ),
    );
  }
}

class _CountdownTimer extends StatefulWidget {
  final DateTime expiresAt;
  const _CountdownTimer({required this.expiresAt});
  @override
  State<_CountdownTimer> createState() => _CountdownTimerState();
}

class _CountdownTimerState extends State<_CountdownTimer> {
  late Timer _timer;
  late Duration _timeLeft;

  @override
  void initState() {
    super.initState();
    _timeLeft = widget.expiresAt.difference(DateTime.now());
    _timer = Timer.periodic(const Duration(seconds: 1), (t) {
      if (mounted) setState(() => _timeLeft = widget.expiresAt.difference(DateTime.now()));
    });
  }

  @override
  void dispose() { _timer.cancel(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    final seconds = _timeLeft.inSeconds % 60;
    return Text("EXPIRES IN ${_timeLeft.inMinutes}:${seconds.toString().padLeft(2, '0')}", 
      style: const TextStyle(color: AppTheme.secondary, fontWeight: FontWeight.bold));
  }
}