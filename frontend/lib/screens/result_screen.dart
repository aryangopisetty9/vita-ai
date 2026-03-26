// Result Screen – from abc/ frontend with real backend data
//
// abc/'s result display: score card with CircularProgressIndicator,
// health status with bullet points, recommendations card.
// Shows real backend results passed from AnalysisScreen.
import 'package:flutter/material.dart';
import '../models/health_data.dart';

class ResultScreen extends StatelessWidget {
  final int userId;
  final Map<String, dynamic> result;
  final String scanType;

  const ResultScreen({
    super.key,
    required this.userId,
    required this.result,
    required this.scanType,
  });

  String _riskLabel(String risk) {
    switch (risk.toLowerCase()) {
      case 'low':
        return 'Good';
      case 'moderate':
        return 'Moderate';
      case 'high':
        return 'Needs Attention';
      default:
        return 'Unknown';
    }
  }

  Color _riskColor(String risk) {
    switch (risk.toLowerCase()) {
      case 'low':
        return Colors.green;
      case 'moderate':
        return Colors.orange;
      case 'high':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }

  List<String> _buildStatusPoints() {
    final points = <String>[];
    final risk = (result['risk'] ?? 'unknown').toString();

    // From symptom analysis
    final condition = result['predicted_condition']?.toString();
    if (condition != null && condition.isNotEmpty) {
      points.add('Predicted condition: $condition');
    }

    final severity = result['severity']?.toString();
    if (severity != null) points.add('Severity: $severity');

    // From face analysis
    final hrRaw = result['heart_rate'];
    final hr = (hrRaw is num) ? hrRaw.toDouble() : double.tryParse(hrRaw?.toString() ?? '');
    final resultAvailable = result['result_available'] == true;
    final tier = (result['hr_result_tier']?.toString().toLowerCase() ?? '').trim();
    final estimatedWeak = result['estimated_from_weak_signal'] == true;
    final confidenceRaw = result['confidence'];
    final hrConfidence = (confidenceRaw is num)
        ? confidenceRaw.toDouble()
        : double.tryParse(confidenceRaw?.toString() ?? '') ?? 0.0;
    final hrReliability = (result['reliability']?.toString() ?? '').toLowerCase();
    final fallbackValid = hr != null && hr > 0 && hrConfidence >= 0.2 && hrReliability != 'unreliable';
    final isStrong = tier == 'strong_accept';
    final isWeak = tier == 'weak_accept' || estimatedWeak;
    final isReject = tier == 'reject' || tier == 'result_unavailable';
    final legacyAvailable = isStrong || isWeak || (tier.isEmpty && fallbackValid);
    final isValidHr = hr != null && hr > 0 && ((result.containsKey('result_available') ? resultAvailable : legacyAvailable)) && !isReject;
    debugPrint('HR DEBUG => value=$hr confidence=$hrConfidence reliability=$hrReliability tier=$tier weak=$isWeak isValid=$isValidHr');
    if (isValidHr) {
      points.add('Heart Rate: ${hr.toStringAsFixed(0)} bpm');
      if (isWeak) {
        points.add('Estimated from weak signal');
        points.add('Retake recommended');
      }
    } else {
      points.add('No reliable pulse detected');
      points.add('Try again with better lighting and less movement');
    }

    if (tier.isNotEmpty) {
      points.add('Result Tier: ${isValidHr ? (isWeak ? 'RESULT_AVAILABLE (ESTIMATED)' : 'RESULT_AVAILABLE') : 'RESULT_UNAVAILABLE'}');
    }

    final stability = result['stability']?.toString();
    if (stability != null) points.add('Stability: $stability');

    // From voice analysis
    final br = result['breathing_rate'];
    if (br != null) points.add('Breathing Rate: $br breaths/min');

    // General
    points.add('Risk Level: ${risk.toUpperCase()}');

    final confidence = result['confidence'];
    if (confidence != null) {
      final pct =
          ((confidence is num ? confidence.toDouble() : 0.0) * 100).round();
      points.add('Confidence: $pct%');
    }

    final reliability = result['reliability']?.toString();
    if (reliability != null) {
      points.add('Reliability: ${reliability.toUpperCase()}');
    }

    final message = result['message']?.toString();
    if (message != null && message.isNotEmpty) {
      points.add(message);
    }

    return points;
  }

  List<String> _buildRecommendations() {
    // Return backend recommendations if present
    if (result['recommendations'] is List) {
      return (result['recommendations'] as List)
          .map((e) => e.toString())
          .toList();
    }

    // Generate basic recommendations based on risk
    final risk = (result['risk'] ?? 'unknown').toString().toLowerCase();
    final recs = <String>[];

    if (risk == 'high') {
      recs.add('Consider consulting a healthcare professional soon.');
      recs.add('Monitor your symptoms closely.');
    } else if (risk == 'moderate') {
      recs.add('Keep tracking your symptoms over the next few days.');
      recs.add('Rest and stay hydrated.');
    } else {
      recs.add('Your results look healthy. Keep it up!');
    }

    recs.add('Stay hydrated and maintain a balanced diet.');
    recs.add('Ensure adequate sleep (7-9 hours).');
    recs.add('If symptoms persist, consult a healthcare provider.');

    return recs;
  }

  @override
  Widget build(BuildContext context) {
    final score = HealthData.score;
    final risk = (result['risk'] ?? HealthData.riskLevel).toString();
    final statusPoints = _buildStatusPoints();
    final recommendations = _buildRecommendations();

    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: SafeArea(
        child: Column(
          children: [
            // ── Header ──
            Padding(
              padding: const EdgeInsets.all(18),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(children: const [
                    Icon(Icons.health_and_safety,
                        color: Colors.green, size: 38),
                    SizedBox(width: 8),
                    Text("Vita AI",
                        style: TextStyle(
                            fontSize: 28,
                            fontWeight: FontWeight.bold,
                            color: Colors.blue)),
                  ]),
                  IconButton(
                    onPressed: () {
                      // Pop back to home
                      Navigator.of(context)
                          .popUntil((route) => route.isFirst);
                    },
                    icon: const Icon(Icons.home, size: 28),
                  ),
                ],
              ),
            ),

            // ── Content ──
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(18, 0, 18, 18),
                child: Column(
                  children: [
                    // ── Score Card ──
                    Card(
                      elevation: 4,
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(20)),
                      child: Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(25),
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(20),
                          gradient: LinearGradient(
                            colors: [
                              _riskColor(risk).withValues(alpha: 0.1),
                              Colors.white,
                            ],
                          ),
                        ),
                        child: Column(
                          children: [
                            const Text("Health Score",
                                style: TextStyle(
                                    fontSize: 20,
                                    fontWeight: FontWeight.bold)),
                            const SizedBox(height: 20),
                            SizedBox(
                              width: 160,
                              height: 160,
                              child: Stack(
                                alignment: Alignment.center,
                                children: [
                                  SizedBox(
                                    width: 160,
                                    height: 160,
                                    child: CircularProgressIndicator(
                                      value: (score ?? 0) / 100,
                                      strokeWidth: 14,
                                      backgroundColor: Colors.grey.shade200,
                                      color: _riskColor(risk),
                                    ),
                                  ),
                                  Column(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      Text(score != null ? '$score%' : '—',
                                          style: TextStyle(
                                              fontSize: 36,
                                              fontWeight: FontWeight.bold,
                                              color: _riskColor(risk))),
                                      Text(_riskLabel(risk),
                                          style: TextStyle(
                                              fontSize: 14,
                                              color: _riskColor(risk))),
                                    ],
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),

                    const SizedBox(height: 16),

                    // ── Health Status ──
                    Card(
                      elevation: 2,
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(16)),
                      child: Padding(
                        padding: const EdgeInsets.all(20),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(children: [
                              Icon(Icons.monitor_heart,
                                  color: _riskColor(risk)),
                              const SizedBox(width: 8),
                              const Text("Health Status",
                                  style: TextStyle(
                                      fontSize: 18,
                                      fontWeight: FontWeight.bold)),
                            ]),
                            const Divider(),
                            ...statusPoints.map((p) => Padding(
                                  padding:
                                      const EdgeInsets.symmetric(vertical: 4),
                                  child: Row(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      const Text("• ",
                                          style: TextStyle(fontSize: 16)),
                                      Expanded(
                                          child: Text(p,
                                              style: const TextStyle(
                                                  fontSize: 15))),
                                    ],
                                  ),
                                )),
                          ],
                        ),
                      ),
                    ),

                    const SizedBox(height: 16),

                    // ── Recommendations ──
                    Card(
                      elevation: 2,
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(16)),
                      child: Padding(
                        padding: const EdgeInsets.all(20),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(children: const [
                              Icon(Icons.tips_and_updates,
                                  color: Colors.amber),
                              SizedBox(width: 8),
                              Text("Recommendations",
                                  style: TextStyle(
                                      fontSize: 18,
                                      fontWeight: FontWeight.bold)),
                            ]),
                            const Divider(),
                            ...recommendations.map((r) => Padding(
                                  padding:
                                      const EdgeInsets.symmetric(vertical: 4),
                                  child: Row(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      const Icon(Icons.check_circle,
                                          size: 18, color: Colors.green),
                                      const SizedBox(width: 8),
                                      Expanded(
                                          child: Text(r,
                                              style: const TextStyle(
                                                  fontSize: 15))),
                                    ],
                                  ),
                                )),
                          ],
                        ),
                      ),
                    ),

                    const SizedBox(height: 20),

                    // ── Action Buttons ──
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton.icon(
                        icon: const Icon(Icons.home),
                        label: const Text("Back to Home",
                            style: TextStyle(fontSize: 16)),
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12)),
                        ),
                        onPressed: () {
                          Navigator.of(context)
                              .popUntil((route) => route.isFirst);
                        },
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
