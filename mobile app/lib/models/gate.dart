class Gate {
  final String id;
  final String name;
  final String status; // closed / opened / locked
  final String command; // none | open | close | emergency_open
  final DateTime lastUpdated;

  Gate({
    required this.id,
    required this.name,
    required this.status,
    required this.command,
    required this.lastUpdated,
  });

  factory Gate.fromMap(String id, Map<String, dynamic> map) {
    return Gate(
      id: id,
      name: map['name'] ?? '',
      status: map['status'] ?? 'closed',
      command: map['command'] ?? 'none',
      lastUpdated: (map['last_updated'] as dynamic)?.toDate() ?? DateTime.now(),
    );
  }
  Map<String, dynamic> toMap() {
    return {
      'name': name,
      'status': status,
      'command': command,
      'last_updated': lastUpdated,
    };
  }
}
