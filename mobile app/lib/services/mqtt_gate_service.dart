// lib/services/mqtt_gate_service.dart
import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';

final mqttGateServiceProvider = Provider<MqttGateService>((ref) {
  final service = MqttGateService();
  ref.onDispose(() => service.disconnect());
  return service;
});

class MqttGateService {
  static const String _broker = 'broker.hivemq.com';
  static const int _port = 1883;

  MqttServerClient? _client;

  Future<void> connect() async {
    final clientId = 'flutter_gate_${DateTime.now().millisecondsSinceEpoch}';
    _client = MqttServerClient.withPort(_broker, clientId, _port);
    _client!.logging(on: false);
    _client!.keepAlivePeriod = 30;
    _client!.onDisconnected = _onDisconnected;

    final connMessage = MqttConnectMessage()
        .withClientIdentifier(clientId)
        .startClean()
        .withWillQos(MqttQos.atLeastOnce);
    _client!.connectionMessage = connMessage;

    try {
      await _client!.connect();
      debugPrint('MQTT connected to $_broker');
    } catch (e) {
      debugPrint('MQTT connection failed: $e');
      _client!.disconnect();
    }
  }

  // Call this to open or close a specific gate
  // command should be: 'open' or 'close'
  void sendCommand(String houseId, String gateId, String command) {
    if (_client == null || _client!.connectionStatus?.state != MqttConnectionState.connected) {
      debugPrint('MQTT not connected, cannot send command');
      return;
    }

    final topic = 'gates/$houseId/$gateId/command';
    final builder = MqttClientPayloadBuilder();
    builder.addString(command);

    _client!.publishMessage(topic, MqttQos.atLeastOnce, builder.payload!);
    debugPrint('MQTT published "$command" to $topic');
  }

  // Subscribe to status updates from a gate
  // The ESP32 publishes to: gates/{houseId}/{gateId}/status
  Stream<String> watchStatus(String houseId, String gateId) {
    if (_client == null) return const Stream.empty();

    final topic = 'gates/$houseId/$gateId/status';
    _client!.subscribe(topic, MqttQos.atLeastOnce);

    final controller = StreamController<String>.broadcast();

    _client!.updates!.listen((List<MqttReceivedMessage<MqttMessage>> messages) {
      for (final msg in messages) {
        if (msg.topic == topic) {
          final recMess = msg.payload as MqttPublishMessage;
          final payload = MqttPublishPayload.bytesToStringAsString(recMess.payload.message);
          debugPrint('📥 MQTT received "$payload" on $topic');
          controller.add(payload);
        }
      }
    });

    return controller.stream;
  }

  void _onDisconnected() {
    debugPrint('⚠️ MQTT disconnected');
  }

  void disconnect() {
    _client?.disconnect();
  }
}