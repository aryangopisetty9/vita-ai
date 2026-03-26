import 'dart:convert';

/// Global health data store – tracks Vita Health Score and local scan history.
///
/// The dashboard score is the fused Vita Health Score from the backend fusion
/// model, recomputed each time any module completes a scan.  Raw module results
/// are cached so they can all be passed to /predict/final-score together.
class HealthData {
  // --- Per-module raw API results (null = not yet scanned) ---
  static Map<String, dynamic>? faceResult;
  static Map<String, dynamic>? audioResult;
  static Map<String, dynamic>? symptomResult;

  /// Combined Vita Health Score (from backend fusion). Null = nothing computed.
  static int? score;
  static String riskLevel = 'unknown';
  static Map<String, dynamic>? latestFusionResult;

  static List<Map<String, String>> history = [];

  static const List<String> _dashboardModules = [
    'face',
    'breathing',
    'symptom',
  ];

  /// Reset all stored data (call on logout or user switch).
  static void clear() {
    clearModuleResults();
    score = null;
    riskLevel = 'unknown';
    latestFusionResult = null;
    history = [];
  }

  /// Clear only per-module raw results. Useful before rehydrating from backend.
  static void clearModuleResults() {
    faceResult = null;
    audioResult = null;
    symptomResult = null;
  }

  /// Store the latest raw result for [module] ('face', 'breathing', 'symptom').
  /// Call this before invoking /predict/final-score so the fusion call always
  /// includes every module that has been scanned so far.
  static void setModuleResult(String module, Map<String, dynamic> result) {
    switch (module) {
      case 'face':
        faceResult = result;
        break;
      case 'breathing':
        audioResult = result;
        break;
      case 'symptom':
        symptomResult = result;
        break;
    }
  }

  /// Apply a fusion response from /predict/final-score.
  ///
  /// Updates [score] and [riskLevel] and inserts a history entry for the
  /// triggering [module].  [moduleScore] is the per-module vita score (shown in
  /// history), [moduleRisk] is the per-module risk label.
  static void applyFusionScore(
    Map<String, dynamic> fusionResult, {
    String module = '',
    int? moduleScore,
    String moduleRisk = '',
    Map<String, dynamic>? moduleResult,
  }) {
    final vita = fusionResult['vita_health_score'] as int?;
    final overall = fusionResult['overall_risk']?.toString() ?? 'unknown';
    latestFusionResult = Map<String, dynamic>.from(fusionResult);
    if (vita != null && vita > 0) {
      score = vita;
      riskLevel = overall;
    }
    final displayScore = moduleScore != null && moduleScore > 0
        ? '$moduleScore%'
        : (vita != null && vita > 0 ? '$vita%' : '—');
    _insertHistoryEntry(
      scoreLabel: displayScore,
      fusion: vita != null && vita > 0 ? '$vita%' : null,
      module: module,
      risk: moduleRisk.isNotEmpty ? moduleRisk : overall,
      resultPayload: _mergedResultPayload(
        moduleResult,
        moduleScore: moduleScore,
        moduleRisk: moduleRisk.isNotEmpty ? moduleRisk : overall,
      ),
    );
  }

  /// Add a history entry without updating the score.
  /// Used for weak/unreliable scans where no Vita score was computed.
  static void addHistoryEntry({
    String? heartRate,
    String risk = 'unreliable',
    String status = 'score unavailable',
    String module = '',
    Map<String, dynamic>? moduleResult,
  }) {
    _insertHistoryEntry(
      scoreLabel: '—',
      module: module,
      risk: risk,
      extra: heartRate != null ? 'HR $heartRate bpm' : status,
      resultPayload: _mergedResultPayload(moduleResult, moduleRisk: risk),
    );
  }

  static void _insertHistoryEntry({
    required String scoreLabel,
    String? fusion,
    String module = '',
    String risk = '',
    String extra = '',
    Map<String, dynamic>? resultPayload,
  }) {
    final now = DateTime.now();
    final label = _moduleLabel(module);
    history.insert(0, {
      'score': scoreLabel,
      'fusion': fusion ?? '',
      'module': module,
      'risk': risk,
      'result_json': resultPayload != null ? jsonEncode(resultPayload) : '',
      'created_at': now.toIso8601String(),
      'time': '${now.hour}:${now.minute.toString().padLeft(2, '0')}',
      'date': '${now.day}-${now.month}-${now.year}',
      'extra': extra.isNotEmpty ? extra : (label.isNotEmpty ? label : risk),
    });
  }

  static Map<String, dynamic>? _mergedResultPayload(
    Map<String, dynamic>? moduleResult, {
    int? moduleScore,
    String? moduleRisk,
  }) {
    if (moduleResult == null || moduleResult.isEmpty) return null;
    final payload = Map<String, dynamic>.from(moduleResult);
    if (moduleScore != null) payload['vita_health_score'] = moduleScore;
    if (moduleRisk != null && moduleRisk.isNotEmpty) {
      payload['overall_risk'] = moduleRisk;
    }
    return payload;
  }

  /// Returns up to 3 dashboard rows: latest Face, latest Voice/Breathing,
  /// and latest Symptoms in a stable order.
  static List<Map<String, String>> latestPerDashboardModule(
    List<Map<String, String>> records,
  ) {
    if (records.isEmpty) return const [];

    final indexed = records.asMap().entries.map((entry) {
      final record = entry.value;
      final module = normalizeModuleLabel(
        record['module'] ?? record['scan_type'] ?? record['type'] ?? '',
      );
      final scanId = _extractScanId(record);
      return _HistoryCandidate(
        record: record,
        module: module,
        scanId: scanId,
        timestamp: _parseRecordTimestamp(record),
        originalIndex: entry.key,
      );
    }).toList();

    indexed.sort((a, b) {
      final tsCompare = b.timestamp.compareTo(a.timestamp);
      if (tsCompare != 0) return tsCompare;
      return a.originalIndex.compareTo(b.originalIndex);
    });

    final seenIds = <String>{};
    final latestByModule = <String, Map<String, String>>{};

    for (final candidate in indexed) {
      if (!_dashboardModules.contains(candidate.module)) continue;

      if (candidate.scanId != null && candidate.scanId!.isNotEmpty) {
        if (seenIds.contains(candidate.scanId)) continue;
        seenIds.add(candidate.scanId!);
      }

      latestByModule.putIfAbsent(candidate.module, () => candidate.record);
      if (latestByModule.length == _dashboardModules.length) break;
    }

    return _dashboardModules
        .map((m) => latestByModule[m])
        .whereType<Map<String, String>>()
        .toList();
  }

  static List<Map<String, String>> latestDashboardHistory() {
    return latestPerDashboardModule(history);
  }

  static String normalizeModuleLabel(String rawType) {
    final norm = rawType
        .trim()
        .toLowerCase()
        .replaceAll(RegExp(r'[^a-z]'), '');
    if (norm.contains('face')) return 'face';
    if (norm.contains('voice') ||
        norm.contains('audio') ||
        norm.contains('breath')) {
      return 'breathing';
    }
    if (norm.contains('symptom')) return 'symptom';
    return rawType.trim().toLowerCase();
  }

  static DateTime _parseRecordTimestamp(Map<String, String> record) {
    final iso = record['created_at'] ?? record['createdAt'] ?? '';
    if (iso.isNotEmpty) {
      try {
        return DateTime.parse(iso);
      } catch (_) {}
    }

    final date = (record['date'] ?? '').trim();
    final time = (record['time'] ?? '').trim();
    if (date.isNotEmpty) {
      try {
        final dateParts = date.split(RegExp(r'[-/.]'));
        final timeParts = time.isNotEmpty ? time.split(':') : <String>[];
        if (dateParts.length == 3) {
          final p0 = int.tryParse(dateParts[0]);
          final p1 = int.tryParse(dateParts[1]);
          final p2 = int.tryParse(dateParts[2]);
          if (p0 != null && p1 != null && p2 != null) {
            final year = p0 > 1900 ? p0 : p2;
            final month = p1;
            final day = p0 > 1900 ? p2 : p0;
            final hour = timeParts.isNotEmpty
                ? int.tryParse(timeParts[0]) ?? 0
                : 0;
            final minute = timeParts.length > 1
                ? int.tryParse(timeParts[1]) ?? 0
                : 0;
            return DateTime(year, month, day, hour, minute);
          }
        }
      } catch (_) {}
    }

    return DateTime.fromMillisecondsSinceEpoch(0);
  }

  static String? _extractScanId(Map<String, String> record) {
    final raw = (record['scan_id'] ?? record['id'] ?? '').trim();
    return raw.isEmpty ? null : raw;
  }

  static String _moduleLabel(String module) {
    switch (module) {
      case 'face':
        return 'Face scan';
      case 'breathing':
        return 'Breathing scan';
      case 'symptom':
        return 'Symptom check';
      default:
        return '';
    }
  }
}

class _HistoryCandidate {
  final Map<String, String> record;
  final String module;
  final String? scanId;
  final DateTime timestamp;
  final int originalIndex;

  const _HistoryCandidate({
    required this.record,
    required this.module,
    required this.scanId,
    required this.timestamp,
    required this.originalIndex,
  });
}
