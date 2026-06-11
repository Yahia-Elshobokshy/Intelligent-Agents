// lib/services/mqtt_gate_service.dart
import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';

final mqttGateServiceProvider = Provider<MqttGateService>((ref) {
  final service = MqttGateService();
  service.connect();
  ref.onDispose(() => service.disconnect());
  return service;
});

class MqttGateService {
  static const String _broker = 'test.mosquitto.org';
  static const int _port = 1883;
  static const int _maxRetries = 3;

  MqttServerClient? _client;
  bool _isConnecting = false;

  Future<bool> connect() async {
    if (_isConnecting) return false;
    _isConnecting = true;

    final clientId = 'flutter_gate_${DateTime.now().millisecondsSinceEpoch}';
    _client = MqttServerClient.withPort(_broker, clientId, _port);

    _client!.logging(on: false);
    _client!.keepAlivePeriod = 30;
    _client!.connectTimeoutPeriod = 5000; // 5s timeout per attempt
    _client!.onDisconnected = _onDisconnected;
    _client!.onConnected = _onConnected;

    final connMessage = MqttConnectMessage()
        .withClientIdentifier(clientId)
        .startClean()
        .withWillQos(MqttQos.atLeastOnce);

    _client!.connectionMessage = connMessage;

    for (int attempt = 1; attempt <= _maxRetries; attempt++) {
      try {
        debugPrint('MQTT connecting (attempt $attempt/$_maxRetries)...');
        await _client!.connect();

        if (_client!.connectionStatus?.state == MqttConnectionState.connected) {
          _isConnecting = false;
          return true;
        }
      } catch (e) {
        debugPrint('MQTT attempt $attempt failed: $e');
        if (attempt < _maxRetries) {
          // Exponential backoff: 1s, 2s, 4s
          await Future.delayed(Duration(seconds: 1 << (attempt - 1)));
        }
      }
    }

    debugPrint('MQTT: all connection attempts exhausted');
    _client?.disconnect();
    _isConnecting = false;
    return false;
  }

  bool get isConnected =>
      _client?.connectionStatus?.state == MqttConnectionState.connected;

  /// Publish open/close command to a gate.
  /// Returns false if not connected.
  bool sendCommand(String houseId, String gateId, String command) {
    if (!isConnected) {
      debugPrint('MQTT not connected, cannot send command');
      return false;
    }

    final topic = 'modules/gate/$gateId';
    print('MQTT sending "$command" to $topic');
    final builder = MqttClientPayloadBuilder()..addString(command);

    _client!.publishMessage(topic, MqttQos.atLeastOnce, builder.payload!);
    debugPrint('MQTT published "$command" to $topic');
    return true;
  }

  /// Subscribe to gate status updates.
  /// modules/gate/$gateId/status
  Stream<String> watchStatus(String houseId, String gateId) {
    if (_client == null || !isConnected) {
      debugPrint('MQTT not connected, cannot subscribe');
      return const Stream.empty();
    }

    final topic = 'modules/gate/$gateId';
    _client!.subscribe(topic, MqttQos.atLeastOnce);

    final controller = StreamController<String>.broadcast();

    final subscription =
        _client!.updates?.listen((List<MqttReceivedMessage<MqttMessage>> messages) {
      for (final msg in messages) {
        if (msg.topic == topic) {
          final recMess = msg.payload as MqttPublishMessage;
          final payload =
              MqttPublishPayload.bytesToStringAsString(recMess.payload.message);
          debugPrint('MQTT received "$payload" on $topic');
          controller.add(payload);
        }
      }
    });

    // Clean up the subscription when the stream is cancelled
    controller.onCancel = () {
      subscription?.cancel();
      if (isConnected) {
        _client!.unsubscribe(topic);
      }
    };

    return controller.stream;
  }

  void _onConnected() => debugPrint('MQTT connected to $_broker');
  void _onDisconnected() => debugPrint('MQTT disconnected');

  void disconnect() {
    _client?.disconnect();
    _client = null;
  }
}