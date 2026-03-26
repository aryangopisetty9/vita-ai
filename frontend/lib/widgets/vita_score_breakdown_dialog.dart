import 'package:flutter/material.dart';

class VitaScoreBreakdownDialog extends StatelessWidget {
  final int? score;
  final String riskLevel;
  final Map<String, dynamic>? fusionResult;

  const VitaScoreBreakdownDialog({
    super.key,
    required this.score,
    required this.riskLevel,
    required this.fusionResult,
  });

  static Future<void> show(
    BuildContext context, {
    required int? score,
    required String riskLevel,
    required Map<String, dynamic>? fusionResult,
  }) {
    return showGeneralDialog<void>(
      context: context,
      barrierDismissible: true,
      barrierLabel: 'Vita score breakdown',
      barrierColor: Colors.black54,
      transitionDuration: const Duration(milliseconds: 220),
      pageBuilder: (ctx, _, __) {
        return SafeArea(
          child: Center(
            child: VitaScoreBreakdownDialog(
              score: score,
              riskLevel: riskLevel,
              fusionResult: fusionResult,
            ),
          ),
        );
      },
      transitionBuilder: (ctx, animation, _, child) {
        final curve = CurvedAnimation(parent: animation, curve: Curves.easeOutCubic);
        return FadeTransition(
          opacity: curve,
          child: ScaleTransition(
            scale: Tween<double>(begin: 0.96, end: 1.0).animate(curve),
            child: child,
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final subtleSurface = Theme.of(context).brightness == Brightness.dark
        ? colorScheme.surfaceContainerHighest.withValues(alpha: 0.4)
        : const Color(0xFFF7F8FB);
    final hasFusionPayload = fusionResult != null && fusionResult!.isNotEmpty;
    final moduleRows = _buildRows(fusionResult);
    final finalScore = _resolveScore(score, fusionResult);
    final finalRisk = _resolveRisk(riskLevel, fusionResult).toUpperCase();
    final recommendations = _resolveRecommendations(fusionResult, moduleRows);
    final usedModules = _resolveUsedModules(fusionResult, moduleRows);
    final unavailableModules = _resolveUnavailableModules(usedModules);
    final explanation = _buildExplanation(
      finalScore,
      finalRisk,
      moduleRows,
      unavailableModules,
    );

    return Material(
      color: Colors.transparent,
      child: Container(
        width: 620,
        constraints: const BoxConstraints(maxWidth: 620, maxHeight: 760),
        margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 24),
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          color: colorScheme.surface,
          borderRadius: BorderRadius.circular(20),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.18),
              blurRadius: 20,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Expanded(
                  child: Text(
                    'Vita Score Breakdown',
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                ),
                IconButton(
                  tooltip: 'Close',
                  onPressed: () => Navigator.of(context).pop(),
                  icon: const Icon(Icons.close),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 12,
              runSpacing: 8,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: [
                Text(
                  finalScore != null ? '$finalScore%' : 'No score',
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    color: _scoreColor(finalScore),
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: _scoreColor(finalScore).withValues(alpha: 0.12),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    finalRisk,
                    style: TextStyle(
                      fontWeight: FontWeight.w700,
                      color: _scoreColor(finalScore),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            const Divider(height: 1),
            const SizedBox(height: 12),
            Expanded(
              child: SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Breakdown',
                      style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: 10),
                    if (!hasFusionPayload)
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: subtleSurface,
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: const Text(
                          'Breakdown unavailable for the current score. Complete a new scan to refresh detailed module contributions.',
                          style: TextStyle(color: Colors.black54, height: 1.3),
                        ),
                      )
                    else if (moduleRows.isEmpty)
                      const Text(
                        'No usable module contributions were returned for this score.',
                        style: TextStyle(color: Colors.black54),
                      ),
                    if (usedModules.isNotEmpty)
                      Padding(
                        padding: const EdgeInsets.only(bottom: 8),
                        child: Text(
                          _calculatedUsingLine(usedModules),
                          style: const TextStyle(fontSize: 12, color: Colors.black87),
                        ),
                      ),
                    if (unavailableModules.isNotEmpty)
                      Padding(
                        padding: const EdgeInsets.only(bottom: 10),
                        child: Text(
                          '${_joinModuleLabels(unavailableModules)} not available for this calculation.',
                          style: const TextStyle(fontSize: 12, color: Colors.black54),
                        ),
                      ),
                    ...moduleRows.map((row) => _buildBreakdownRow(context, row)),
                    if (hasFusionPayload) ...[
                      const SizedBox(height: 14),
                      const Text(
                        'Why this score?',
                        style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        explanation,
                        style: TextStyle(
                          height: 1.35,
                          color: colorScheme.onSurface,
                        ),
                      ),
                      const SizedBox(height: 14),
                      const Text(
                        'Recommendations',
                        style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
                      ),
                      const SizedBox(height: 6),
                      ...recommendations.map(
                        (tip) => Padding(
                          padding: const EdgeInsets.only(bottom: 6),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Padding(
                                padding: EdgeInsets.only(top: 3),
                                child: Icon(Icons.check_circle, size: 14, color: Colors.green),
                              ),
                              const SizedBox(width: 8),
                              Expanded(child: Text(tip, style: const TextStyle(height: 1.3))),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBreakdownRow(BuildContext context, _ModuleRow row) {
    final rowBg = Theme.of(context).brightness == Brightness.dark
        ? Theme.of(context).colorScheme.surfaceContainerHighest.withValues(alpha: 0.35)
        : const Color(0xFFF7F8FB);
    final contributionPct = (row.contribution).clamp(0.0, 100.0) / 100.0;
    final rawPct = (row.rawScore ?? 0.0).clamp(0.0, 100.0) / 100.0;

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: rowBg,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            row.equation,
            style: const TextStyle(fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              const SizedBox(
                width: 92,
                child: Text('Raw score', style: TextStyle(fontSize: 12, color: Colors.black54)),
              ),
              Expanded(
                child: TweenAnimationBuilder<double>(
                  duration: const Duration(milliseconds: 500),
                  tween: Tween<double>(begin: 0, end: rawPct),
                  builder: (context, value, _) => LinearProgressIndicator(
                    value: value,
                    minHeight: 8,
                    borderRadius: BorderRadius.circular(8),
                    backgroundColor: Colors.grey.shade300,
                    color: Colors.blueGrey,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              SizedBox(
                width: 48,
                child: Text(
                  row.rawScore != null ? row.rawScore!.toStringAsFixed(1) : 'N/A',
                  textAlign: TextAlign.right,
                  style: const TextStyle(fontSize: 12),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              const SizedBox(
                width: 92,
                child: Text('Contribution', style: TextStyle(fontSize: 12, color: Colors.black54)),
              ),
              Expanded(
                child: TweenAnimationBuilder<double>(
                  duration: const Duration(milliseconds: 650),
                  tween: Tween<double>(begin: 0, end: contributionPct),
                  builder: (context, value, _) => LinearProgressIndicator(
                    value: value,
                    minHeight: 8,
                    borderRadius: BorderRadius.circular(8),
                    backgroundColor: Colors.grey.shade300,
                    color: Colors.teal,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              SizedBox(
                width: 48,
                child: Text(
                  row.contribution.toStringAsFixed(1),
                  textAlign: TextAlign.right,
                  style: const TextStyle(fontSize: 12),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  int? _resolveScore(int? fallback, Map<String, dynamic>? fusion) {
    final dynamic v = fusion?['vita_health_score'];
    if (v is int) return v;
    if (v is double) return v.round();
    return fallback;
  }

  String _resolveRisk(String fallback, Map<String, dynamic>? fusion) {
    final fromFusion = fusion?['overall_risk']?.toString();
    if (fromFusion != null && fromFusion.isNotEmpty) {
      return fromFusion;
    }
    return fallback;
  }

  List<_ModuleRow> _buildRows(Map<String, dynamic>? fusion) {
    final scores = (fusion?['component_scores'] is Map)
        ? Map<String, dynamic>.from(fusion!['component_scores'] as Map)
        : <String, dynamic>{};
    final weights = (fusion?['final_weights'] is Map)
        ? Map<String, dynamic>.from(fusion!['final_weights'] as Map)
        : <String, dynamic>{};

    final entries = <_ModuleRow>[
      _toRow('Face', scores['heart_score'], weights['face']),
      _toRow('Breathing', scores['breathing_score'], weights['breathing']),
      _toRow('Symptoms', scores['symptom_score'], weights['symptom']),
    ];

    return entries.where((e) => e.rawScore != null || e.weight > 0).toList();
  }

  _ModuleRow _toRow(String label, dynamic raw, dynamic w) {
    final rawScore = _toDouble(raw);
    final weight = _toDouble(w) ?? 0.0;
    final contribution = rawScore != null ? rawScore * weight : 0.0;
    final rawText = rawScore != null ? '${rawScore.toStringAsFixed(1)}%' : 'N/A';

    return _ModuleRow(
      label: label,
      rawScore: rawScore,
      weight: weight,
      contribution: contribution,
      equation: '$label: $rawText x ${weight.toStringAsFixed(2)} = ${contribution.toStringAsFixed(1)}',
    );
  }

  List<String> _resolveUsedModules(
    Map<String, dynamic>? fusion,
    List<_ModuleRow> rows,
  ) {
    final raw = fusion?['used_modules'];
    if (raw is List) {
      final parsed = raw
          .map((e) => _normalizeModule(e?.toString() ?? ''))
          .where((e) => e.isNotEmpty)
          .toSet()
          .toList();
      if (parsed.isNotEmpty) {
        parsed.sort((a, b) => _moduleOrder(a).compareTo(_moduleOrder(b)));
        return parsed;
      }
    }

    final inferred = rows
        .where((r) => r.rawScore != null && r.weight > 0)
        .map((r) => _normalizeModule(r.label))
        .toSet()
        .toList();
    inferred.sort((a, b) => _moduleOrder(a).compareTo(_moduleOrder(b)));
    return inferred;
  }

  List<String> _resolveUnavailableModules(List<String> usedModules) {
    const all = ['face', 'breathing', 'symptom'];
    return all.where((m) => !usedModules.contains(m)).toList();
  }

  String _joinModuleLabels(List<String> modules) {
    final labels = modules.map(_moduleLabel).toList();
    if (labels.isEmpty) return '';
    if (labels.length == 1) return labels.first;
    if (labels.length == 2) return '${labels[0]} and ${labels[1]}';
    return '${labels[0]}, ${labels[1]}, and ${labels[2]}';
  }

  String _calculatedUsingLine(List<String> usedModules) {
    final label = _joinModuleLabels(usedModules);
    if (usedModules.length == 1) {
      return 'Score calculated using $label only.';
    }
    return 'Score calculated using $label.';
  }

  String _moduleLabel(String module) {
    switch (module) {
      case 'face':
        return 'Face';
      case 'breathing':
        return 'Audio';
      case 'symptom':
        return 'Symptoms';
      default:
        return module;
    }
  }

  String _normalizeModule(String raw) {
    final v = raw.trim().toLowerCase();
    if (v == 'audio' || v == 'breathing' || v == 'voice') return 'breathing';
    if (v == 'symptoms' || v == 'symptom') return 'symptom';
    if (v == 'face') return 'face';
    return '';
  }

  int _moduleOrder(String module) {
    switch (module) {
      case 'face':
        return 0;
      case 'breathing':
        return 1;
      case 'symptom':
        return 2;
      default:
        return 99;
    }
  }

  List<String> _resolveRecommendations(
    Map<String, dynamic>? fusion,
    List<_ModuleRow> rows,
  ) {
    final raw = fusion?['recommendations'];
    if (raw is List) {
      final parsed = raw
          .map((e) => e?.toString().trim() ?? '')
          .where((e) => e.isNotEmpty)
          .toList();
      if (parsed.isNotEmpty) return parsed;
    }

    if (rows.isEmpty) {
      return const ['Complete scans for face, breathing, and symptom modules to generate recommendations.'];
    }

    rows.sort((a, b) => a.contribution.compareTo(b.contribution));
    final weakest = rows.first;

    return [
      'Re-test your ${weakest.label.toLowerCase()} module in a calm and stable environment for better signal quality.',
      'Track changes by repeating scans at similar times each day.',
      'Use this score for screening only and consult a qualified clinician if symptoms persist.',
    ];
  }

  String _buildExplanation(
    int? finalScore,
    String risk,
    List<_ModuleRow> rows,
    List<String> unavailableModules,
  ) {
    if (finalScore == null || rows.isEmpty) {
      return 'There is not enough complete data yet to explain the Vita score. Run more scans to build a full breakdown.';
    }

    final scoredRows = rows.where((r) => r.rawScore != null).toList()
      ..sort((a, b) => b.contribution.compareTo(a.contribution));

    if (scoredRows.isEmpty) {
      return 'No reliable module output was available, so the final score cannot be explained yet.';
    }

    final strongest = scoredRows.first;
    final weakest = scoredRows.last;

    final trend = finalScore >= 70
        ? 'high because most contributing module scores are in a healthy range'
        : finalScore >= 40
            ? 'moderate because at least one module lowered the combined score'
            : 'low because multiple module contributions reduced the combined result';

    final missingText = unavailableModules.isEmpty
      ? ''
      : unavailableModules.length == 1
        ? ' ${_joinModuleLabels(unavailableModules)} was not available for this calculation.'
        : ' ${_joinModuleLabels(unavailableModules)} were not available for this calculation.';

    return 'Your Vita score is $trend. The strongest contributor is ${strongest.label} '
        '(${strongest.contribution.toStringAsFixed(1)} points), while ${weakest.label} '
        'is the weakest contributor (${weakest.contribution.toStringAsFixed(1)} points). '
      'Risk level is $risk based on the final weighted total.$missingText';
  }

  double? _toDouble(dynamic value) {
    if (value is double) return value;
    if (value is int) return value.toDouble();
    if (value is String) return double.tryParse(value);
    return null;
  }

  Color _scoreColor(int? score) {
    if (score == null) return Colors.grey;
    if (score >= 70) return Colors.green;
    if (score >= 40) return Colors.orange;
    return Colors.red;
  }
}

class _ModuleRow {
  final String label;
  final double? rawScore;
  final double weight;
  final double contribution;
  final String equation;

  const _ModuleRow({
    required this.label,
    required this.rawScore,
    required this.weight,
    required this.contribution,
    required this.equation,
  });
}
