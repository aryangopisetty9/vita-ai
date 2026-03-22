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

  static List<Map<String, String>> history = [];

  /// Reset all stored data (call on logout or user switch).
  static void clear() {
    faceResult = null;
    audioResult = null;
    symptomResult = null;
    score = null;
    riskLevel = 'unknown';
    history = [];
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
  }) {
    final vita = fusionResult['vita_health_score'] as int?;
    final overall = fusionResult['overall_risk']?.toString() ?? 'unknown';
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
    );
  }

  /// Add a history entry without updating the score.
  /// Used for weak/unreliable scans where no Vita score was computed.
  static void addHistoryEntry({
    String? heartRate,
    String risk = 'unreliable',
    String status = 'score unavailable',
    String module = '',
  }) {
    _insertHistoryEntry(
      scoreLabel: '—',
      module: module,
      risk: risk,
      extra: heartRate != null ? 'HR $heartRate bpm' : status,
    );
  }

  static void _insertHistoryEntry({
    required String scoreLabel,
    String? fusion,
    String module = '',
    String risk = '',
    String extra = '',
  }) {
    final now = DateTime.now();
    final label = _moduleLabel(module);
    history.insert(0, {
      'score': scoreLabel,
      'fusion': fusion ?? '',
      'module': module,
      'risk': risk,
      'time': '${now.hour}:${now.minute.toString().padLeft(2, '0')}',
      'date': '${now.day}-${now.month}-${now.year}',
      'extra': extra.isNotEmpty ? extra : (label.isNotEmpty ? label : risk),
    });
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
