import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

// ─── Stub imports (replace with real packages) ────────────────────────────────
// import 'package:flutter_vlc_player/flutter_vlc_player.dart';
// import 'package:record/record.dart';
// import 'package:permission_handler/permission_handler.dart';

// ─── ESP32 config ─────────────────────────────────────────────────────────────
// const String _esp32BaseUrl  = 'http://192.168.1.XXX';   // ← ESP32 local IP
// const String _streamUrl     = '$_esp32BaseUrl/stream';   // ← MJPEG stream endpoint
// const String _audioOutUrl   = '$_esp32BaseUrl/audio/in'; // ← POST PCM chunks here
// const String _audioInUrl    = '$_esp32BaseUrl/audio/out';// ← GET PCM stream here
const String _streamUrl = 'http://placeholder-stream';

// ─── Simple state ─────────────────────────────────────────────────────────────
class _IntercomState {
  final bool isMuted;
  final bool isConnected;
  final bool isSpeaking;   // ESP32 visitor is speaking (VU meter placeholder)
  const _IntercomState({
    this.isMuted     = true,
    this.isConnected = false,
    this.isSpeaking  = false,
  });
  _IntercomState copyWith({bool? isMuted, bool? isConnected, bool? isSpeaking}) =>
      _IntercomState(
        isMuted:     isMuted     ?? this.isMuted,
        isConnected: isConnected ?? this.isConnected,
        isSpeaking:  isSpeaking  ?? this.isSpeaking,
      );
}

final _intercomProvider =
    StateNotifierProvider.autoDispose<_IntercomNotifier, _IntercomState>(
  (_) => _IntercomNotifier(),
);

class _IntercomNotifier extends StateNotifier<_IntercomState> {
  _IntercomNotifier() : super(const _IntercomState()) {
    _connect();
  }

  Timer? _speakTimer;

  Future<void> _connect() async {
    // ← ESP32: open MJPEG stream + audio-out stream here
    await Future.delayed(const Duration(milliseconds: 600));
    state = state.copyWith(isConnected: true);

    // Simulate visitor speaking periodically (remove once real audio arrives)
    _speakTimer = Timer.periodic(const Duration(seconds: 3), (_) {
      state = state.copyWith(isSpeaking: true);
      Future.delayed(const Duration(milliseconds: 900),
          () => state = state.copyWith(isSpeaking: false));
    });
  }

  void toggleMute() {
    final muting = !state.isMuted;
    state = state.copyWith(isMuted: muting);
    if (muting) {
      // ← ESP32: stop sending mic chunks
      _stopMicStream();
    } else {
      // ← ESP32: start sending mic chunks to /audio/in
      _startMicStream();
    }
  }

  void _startMicStream() {
    // Replace with: record package → HTTP chunked POST to _audioOutUrl
    debugPrint('[Intercom] Mic → ESP32 started');
  }

  void _stopMicStream() {
    debugPrint('[Intercom] Mic → ESP32 stopped');
  }

  @override
  void dispose() {
    _speakTimer?.cancel();
    _stopMicStream();
    // ← ESP32: close streams
    super.dispose();
  }
}


// ─── Screen ───────────────────────────────────────────────────────────────────
class IntercomScreen extends ConsumerWidget {
  const IntercomScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(_intercomProvider);
    final notifier = ref.read(_intercomProvider.notifier);
    final top = MediaQuery.of(context).padding.top;

    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: SystemUiOverlayStyle.light,
      child: Scaffold(
        backgroundColor: const Color(0xFF0A0E1A),
        body: Stack(
          children: [
            // ── Camera feed (full screen) ──────────────────────────────────
            Positioned.fill(
              child: _CameraFeedWidget(
                streamUrl:   _streamUrl,
                isConnected: state.isConnected,
              ),
            ),

            // ── Top gradient ───────────────────────────────────────────────
            Positioned(
              top: 0, left: 0, right: 0,
              height: top + 100,
              child: DecoratedBox(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                    colors: [
                      Colors.black.withOpacity(0.75),
                      Colors.transparent,
                    ],
                  ),
                ),
              ),
            ),

            // ── Bottom gradient ────────────────────────────────────────────
            const Positioned(
              bottom: 0, left: 0, right: 0,
              height: 260,
              child: DecoratedBox(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.bottomCenter,
                    end: Alignment.topCenter,
                    colors: [Colors.black, Colors.transparent],
                  ),
                ),
              ),
            ),

            // ── Top bar ────────────────────────────────────────────────────
            Positioned(
              top: top + 8,
              left: 16,
              right: 16,
              child: _TopBar(isConnected: state.isConnected),
            ),

            // ── Visitor speaking indicator ──────────────────────────────────
            if (state.isSpeaking)
              Positioned(
                top: top + 70,
                left: 0, right: 0,
                child: const _SpeakingBanner(),
              ),

            // ── Bottom controls ─────────────────────────────────────────────
            Positioned(
              bottom: 36,
              left: 24, right: 24,
              child: _BottomControls(
                isMuted: state.isMuted,
                onToggleMute: notifier.toggleMute,
              ),
            ),
          ],
        ),
      ),
    );
  }
}


// ─── Camera feed ──────────────────────────────────────────────────────────────
class _CameraFeedWidget extends StatelessWidget {
  final String streamUrl;
  final bool isConnected;
  const _CameraFeedWidget({required this.streamUrl, required this.isConnected});

  @override
  Widget build(BuildContext context) {
    if (!isConnected) {
      return const _ConnectingPlaceholder();
    }

    // ← ESP32: swap this Container for VlcPlayer pointing at _streamUrl
    //   VlcPlayerController _ctrl = VlcPlayerController.network(streamUrl,
    //       hwAcc: HwAcc.full, autoPlay: true);
    //   return VlcPlayer(controller: _ctrl, aspectRatio: 4/3, placeholder: ...);
    return Container(
      color: const Color(0xFF0D1220),
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.videocam_rounded,
                size: 64, color: Colors.white.withOpacity(0.12)),
            const SizedBox(height: 12),
            Text(
              'Camera stream\n$streamUrl',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.white.withOpacity(0.18),
                fontSize: 12,
                fontFamily: 'monospace',
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ConnectingPlaceholder extends StatefulWidget {
  const _ConnectingPlaceholder();
  @override
  State<_ConnectingPlaceholder> createState() => _ConnectingPlaceholderState();
}

class _ConnectingPlaceholderState extends State<_ConnectingPlaceholder>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;
  late Animation<double> _pulse;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 1200))
      ..repeat(reverse: true);
    _pulse = Tween(begin: 0.4, end: 1.0).animate(
        CurvedAnimation(parent: _ctrl, curve: Curves.easeInOut));
  }

  @override
  void dispose() { _ctrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return Container(
      color: const Color(0xFF080C16),
      child: Center(
        child: AnimatedBuilder(
          animation: _pulse,
          builder: (_, __) => Opacity(
            opacity: _pulse.value,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.wifi_find_rounded,
                    size: 48, color: Color(0xFF4A90D9)),
                const SizedBox(height: 16),
                Text('Connecting to device…',
                    style: TextStyle(
                        color: Colors.white.withOpacity(0.5),
                        fontSize: 14,
                        letterSpacing: 0.5)),
              ],
            ),
          ),
        ),
      ),
    );
  }
}


// ─── Top bar ──────────────────────────────────────────────────────────────────
class _TopBar extends StatelessWidget {
  final bool isConnected;
  const _TopBar({required this.isConnected});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        // Back button
        GestureDetector(
          onTap: () => Navigator.of(context).maybePop(),
          child: Container(
            width: 40, height: 40,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.12),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(Icons.arrow_back_ios_new_rounded,
                color: Colors.white, size: 18),
          ),
        ),
        const SizedBox(width: 14),

        // Title
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('Front Door',
                  style: TextStyle(
                      color: Colors.white,
                      fontSize: 17,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 0.2)),
              Row(
                children: [
                  AnimatedContainer(
                    duration: const Duration(milliseconds: 400),
                    width: 7, height: 7,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: isConnected
                          ? const Color(0xFF4ADE80)
                          : const Color(0xFFF87171),
                    ),
                  ),
                  const SizedBox(width: 5),
                  Text(
                    isConnected ? 'Live' : 'Connecting…',
                    style: TextStyle(
                        color: Colors.white.withOpacity(0.6),
                        fontSize: 12,
                        fontWeight: FontWeight.w500),
                  ),
                ],
              ),
            ],
          ),
        ),

        // Snapshot button
        _IconBtn(
          icon: Icons.photo_camera_outlined,
          onTap: () {
            // ← ESP32: GET /snapshot  → save to gallery
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('Snapshot saved'),
                behavior: SnackBarBehavior.floating,
                duration: Duration(seconds: 2),
              ),
            );
          },
        ),
      ],
    );
  }
}


// ─── Speaking banner ──────────────────────────────────────────────────────────
class _SpeakingBanner extends StatefulWidget {
  const _SpeakingBanner();
  @override
  State<_SpeakingBanner> createState() => _SpeakingBannerState();
}

class _SpeakingBannerState extends State<_SpeakingBanner>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;
  late Animation<double> _fade;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 300));
    _fade = CurvedAnimation(parent: _ctrl, curve: Curves.easeOut);
    _ctrl.forward();
  }
  @override
  void dispose() { _ctrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _fade,
      child: Center(
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          decoration: BoxDecoration(
            color: Colors.black.withOpacity(0.6),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: Colors.white.withOpacity(0.15)),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              _WaveformIcon(),
              const SizedBox(width: 8),
              Text('Visitor is speaking',
                  style: TextStyle(
                      color: Colors.white.withOpacity(0.85),
                      fontSize: 13,
                      fontWeight: FontWeight.w500)),
            ],
          ),
        ),
      ),
    );
  }
}

class _WaveformIcon extends StatefulWidget {
  @override
  State<_WaveformIcon> createState() => _WaveformIconState();
}

class _WaveformIconState extends State<_WaveformIcon>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 600))
      ..repeat(reverse: true);
  }
  @override
  void dispose() { _ctrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _ctrl,
      builder: (_, __) {
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: List.generate(4, (i) {
            final h = 6.0 + (_ctrl.value * 10) * ((i % 2 == 0) ? 1 : 0.6);
            return Padding(
              padding: const EdgeInsets.symmetric(horizontal: 1.5),
              child: Container(
                width: 3, height: h,
                decoration: BoxDecoration(
                  color: const Color(0xFF4ADE80),
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            );
          }),
        );
      },
    );
  }
}


// ─── Bottom controls ──────────────────────────────────────────────────────────
class _BottomControls extends StatelessWidget {
  final bool isMuted;
  final VoidCallback onToggleMute;
  const _BottomControls({required this.isMuted, required this.onToggleMute});

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        // Hint text
        AnimatedSwitcher(
          duration: const Duration(milliseconds: 250),
          child: Text(
            isMuted
                ? 'Tap the mic to speak to the visitor'
                : 'You are live — visitor can hear you',
            key: ValueKey(isMuted),
            style: TextStyle(
              color: Colors.white.withOpacity(0.55),
              fontSize: 13,
              fontWeight: FontWeight.w400,
            ),
          ),
        ),
        const SizedBox(height: 20),

        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Volume / speaker mute (visitor → homeowner audio)
            _IconBtn(
              icon: Icons.volume_up_rounded,
              size: 52,
              onTap: () {
                // ← ESP32: pause /audio/out stream
              },
            ),
            const SizedBox(width: 24),

            // ── Big mic button ─────────────────────────────────────────────
            GestureDetector(
              onTap: onToggleMute,
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 300),
                curve: Curves.easeOutBack,
                width:  isMuted ? 76 : 88,
                height: isMuted ? 76 : 88,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: isMuted
                      ? Colors.white.withOpacity(0.12)
                      : const Color(0xFF4ADE80),
                  border: Border.all(
                    color: isMuted
                        ? Colors.white.withOpacity(0.2)
                        : const Color(0xFF4ADE80),
                    width: 2,
                  ),
                  boxShadow: isMuted ? [] : [
                    BoxShadow(
                        color: const Color(0xFF4ADE80).withOpacity(0.45),
                        blurRadius: 24,
                        spreadRadius: 4),
                  ],
                ),
                child: Icon(
                  isMuted ? Icons.mic_off_rounded : Icons.mic_rounded,
                  color: isMuted ? Colors.white.withOpacity(0.7) : Colors.black,
                  size: 32,
                ),
              ),
            ),

            const SizedBox(width: 24),

            // Snapshot / extra action
            _IconBtn(
              icon: Icons.photo_camera_outlined,
              size: 52,
              onTap: () {
                // ← ESP32: GET /snapshot
              },
            ),
          ],
        ),

        const SizedBox(height: 4),
        Text(
          isMuted ? 'Mic Off' : 'Mic On',
          style: TextStyle(
            color: isMuted
                ? Colors.white.withOpacity(0.35)
                : const Color(0xFF4ADE80),
            fontSize: 11,
            fontWeight: FontWeight.w600,
            letterSpacing: 0.5,
          ),
        ),
      ],
    );
  }
}

class _IconBtn extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  final double size;
  const _IconBtn({required this.icon, required this.onTap, this.size = 46});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: size, height: size,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: Colors.white.withOpacity(0.10),
          border: Border.all(color: Colors.white.withOpacity(0.15)),
        ),
        child: Icon(icon, color: Colors.white.withOpacity(0.8), size: size * 0.44),
      ),
    );
  }
}