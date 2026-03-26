// Home Screen – from abc/ frontend with backend data integration
//
// abc/'s card-based layout (score card, quick actions, health tests, history)
// wired to the real backend via ApiService.
import 'dart:convert';
import 'package:flutter/material.dart';
import '../models/health_data.dart';
import '../services/api_service.dart';
import 'symptom_screen.dart';
import 'scan_history_screen.dart';
import 'sos_screen.dart';
import 'settings_screen.dart';
import '../widgets/vita_score_breakdown_dialog.dart';

class HomeScreen extends StatefulWidget {
  final int userId;
  final String userName;

  const HomeScreen({
    super.key,
    required this.userId,
    required this.userName,
  });

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  void initState() {
    super.initState();
    _loadLatestScore();
  }

  Future<void> _loadLatestScore() async {
    if (widget.userId == 0) return;
    try {
      // Rehydrate from backend from a clean module state to avoid stale inputs.
      HealthData.clearModuleResults();
      HealthData.score = null;
      HealthData.riskLevel = 'unknown';
      HealthData.latestFusionResult = null;
      final scans = await ApiService.getScanHistory(widget.userId);
      if (scans.isNotEmpty) {
        // Walk through scans (newest first) and populate per-module raw results
        // from the stored result_json so fusion can be recomputed server-side.
        final seen = <String>{};
        for (final s in scans) {
          if (s is! Map<String, dynamic>) continue;
          final scan = s;
          final scanType = scan['scan_type']?.toString() ?? '';
          final resultJson = scan['result_json']?.toString() ?? '';
          final module = HealthData.normalizeModuleLabel(scanType);

          if (!seen.contains(module) && resultJson.isNotEmpty) {
            seen.add(module);
            try {
              final parsed = jsonDecode(resultJson) as Map<String, dynamic>;
              HealthData.setModuleResult(module, parsed);
            } catch (_) {}
          }
          if (seen.length >= 3) break;
        }

        // Recompute the global fusion score using all stored module results.
        if (HealthData.faceResult != null ||
            HealthData.audioResult != null ||
            HealthData.symptomResult != null) {
          try {
            final fusionResult = await ApiService.predictFinalScore(
              faceResult: HealthData.faceResult,
              audioResult: HealthData.audioResult,
              symptomResult: HealthData.symptomResult,
            );
            final vita = fusionResult['vita_health_score'] as int?;
            final overall =
                fusionResult['overall_risk']?.toString() ?? 'unknown';
            HealthData.latestFusionResult = Map<String, dynamic>.from(fusionResult);
            if (vita != null && vita > 0) {
              HealthData.score = vita;
              HealthData.riskLevel = overall;
            }
          } catch (_) {
            // Fusion unavailable — fallback to latest persisted scan score.
            final latest = scans.whereType<Map<String, dynamic>>().cast<Map<String, dynamic>>().firstWhere(
                  (s) => s['vita_score'] != null,
                  orElse: () => <String, dynamic>{},
                );
            final latestScore = latest['vita_score'] as int?;
            if (latestScore != null && latestScore > 0) {
              HealthData.score = latestScore;
              HealthData.riskLevel = latest['risk_level']?.toString() ?? 'unknown';
            }
          }
        } else {
          // No usable module payloads to recompute fusion; fallback to latest saved score.
          final latest = scans.whereType<Map<String, dynamic>>().cast<Map<String, dynamic>>().firstWhere(
                (s) => s['vita_score'] != null,
                orElse: () => <String, dynamic>{},
              );
          final latestScore = latest['vita_score'] as int?;
          if (latestScore != null && latestScore > 0) {
            HealthData.score = latestScore;
            HealthData.riskLevel = latest['risk_level']?.toString() ?? 'unknown';
          }
        }

        // Re-populate dashboard history from backend scans.
        HealthData.history = scans.whereType<Map<String, dynamic>>().map((scan) {
          final vitaScore = scan['vita_score'];
          final riskLevel = scan['risk_level']?.toString() ?? '';
          final scanType = scan['scan_type']?.toString() ?? '';
          final createdAt = scan['created_at']?.toString() ?? '';
          final rawResultJson = scan['result_json']?.toString() ?? '';
          DateTime? dt;
          try {
            dt = DateTime.parse(createdAt);
          } catch (_) {}
          final module = HealthData.normalizeModuleLabel(scanType);
          final label = module == 'face'
              ? 'Face scan'
              : module == 'breathing'
                  ? 'Breathing scan'
                  : module == 'symptom'
                      ? 'Symptom check'
                      : module;

          String extra = vitaScore == null ? 'score unavailable' : label;
          if (module == 'face' && rawResultJson.isNotEmpty) {
            try {
              final parsed = jsonDecode(rawResultJson) as Map<String, dynamic>;
              final resultAvailable = parsed['result_available'] == true;
              final estimatedWeak = parsed['estimated_from_weak_signal'] == true;
              final tier = (parsed['hr_result_tier']?.toString().toLowerCase() ?? '').trim();
              final hrRaw = parsed['heart_rate'];
              final hr = hrRaw is num ? hrRaw.toDouble() : double.tryParse(hrRaw?.toString() ?? '');
              final legacyStrong = tier == 'strong_accept';
              final legacyWeak = tier == 'weak_accept';
              final legacyReject = tier == 'reject' || tier == 'result_unavailable';
              final available = parsed.containsKey('result_available')
                  ? resultAvailable
                  : (legacyStrong || legacyWeak);
              if (available && (estimatedWeak || legacyWeak) && hr != null && hr > 0) {
                extra = 'Estimated HR ${hr.toStringAsFixed(0)} bpm';
              } else if (available && hr != null && hr > 0) {
                extra = 'HR ${hr.toStringAsFixed(0)} bpm';
              } else if (!available || legacyReject) {
                extra = 'No reliable pulse detected';
              }
            } catch (_) {}
          }

          return <String, String>{
            'scan_id': scan['id']?.toString() ?? '',
            'score': vitaScore != null ? '$vitaScore%' : '—',
            'fusion': '',
            'module': module,
            'risk': riskLevel,
            'created_at': createdAt,
            'time': dt != null
                ? '${dt.hour}:${dt.minute.toString().padLeft(2, '0')}'
                : '',
            'date': dt != null
                ? '${dt.day}-${dt.month}-${dt.year}'
                : '',
            'extra': extra,
          };
        }).toList();
      } else {
        // No scans remain — clear stale score so the gauge shows "no data".
        HealthData.clearModuleResults();
        HealthData.score = null;
        HealthData.riskLevel = 'unknown';
        HealthData.latestFusionResult = null;
        HealthData.history = [];
      }
      if (mounted) setState(() {});
    } catch (_) {}
  }

  Future<void> _openScoreBreakdown() async {
    await VitaScoreBreakdownDialog.show(
      context,
      score: HealthData.score,
      riskLevel: HealthData.riskLevel,
      fusionResult: HealthData.latestFusionResult,
    );
  }

  @override
  Widget build(BuildContext context) {
    final int? score = HealthData.score;
    final latestHistory = HealthData.latestDashboardHistory();

    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(18),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // ── Header ──
              Row(
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
                  CircleAvatar(
                    backgroundColor: Colors.white,
                    child: Text(
                      widget.userName.isNotEmpty
                          ? widget.userName[0].toUpperCase()
                          : 'U',
                      style: const TextStyle(
                          fontWeight: FontWeight.bold, color: Colors.black),
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 18),

              Text(
                "Welcome ${widget.userName}!",
                style: const TextStyle(
                    fontSize: 28, fontWeight: FontWeight.bold),
              ),

              const SizedBox(height: 20),

              // ── Health Score Card ──
              _buildCard(
                title: "Vita Health Score",
                icon: Icons.speed,
                child: InkWell(
                  onTap: _openScoreBreakdown,
                  borderRadius: BorderRadius.circular(18),
                  child: Container(
                    height: 120,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(18),
                      gradient: const LinearGradient(
                        colors: [Color(0xFFE9EDF5), Color(0xFFDDE3F0)],
                      ),
                    ),
                    child: Center(
                      child: score != null
                          ? Stack(
                              alignment: Alignment.center,
                              children: [
                                SizedBox(
                                  height: 90,
                                  width: 90,
                                  child: CircularProgressIndicator(
                                    value: score / 100,
                                    strokeWidth: 12,
                                    color: _scoreColor(score),
                                    backgroundColor: Colors.grey.shade300,
                                  ),
                                ),
                                Column(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Text('$score%',
                                        style: TextStyle(
                                            fontSize: 22,
                                            fontWeight: FontWeight.bold,
                                            color: _scoreColor(score))),
                                    if (HealthData.riskLevel != 'unknown')
                                      Text(
                                        HealthData.riskLevel.toUpperCase(),
                                        style: TextStyle(
                                            fontSize: 10,
                                            color: _scoreColor(score)),
                                      ),
                                  ],
                                ),
                              ],
                            )
                          : Column(
                              mainAxisSize: MainAxisSize.min,
                              children: const [
                                Icon(Icons.health_and_safety_outlined,
                                    size: 40, color: Colors.grey),
                                SizedBox(height: 8),
                                Text(
                                  'No complete score yet',
                                  style: TextStyle(
                                      color: Colors.grey, fontSize: 14),
                                ),
                                SizedBox(height: 2),
                                Text(
                                  'Complete a scan to see your score',
                                  style: TextStyle(
                                      color: Colors.grey, fontSize: 11),
                                ),
                              ],
                            ),
                    ),
                  ),
                ),
              ),

              const SizedBox(height: 18),

              // ── Quick Actions ──
              _buildCard(
                title: "Quick Actions",
                icon: Icons.directions_run,
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    _actionButton(Icons.sos, "SOS", Colors.red, () async {
                      await Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) =>
                              SosScreen(userId: widget.userId),
                        ),
                      );
                    }),
                    _actionButton(Icons.history, "History", Colors.black54,
                        () async {
                      await Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) =>
                              ScanHistoryScreen(userId: widget.userId),
                        ),
                      );
                      _loadLatestScore();
                      if (mounted) setState(() {});
                    }),
                    _actionButton(
                        Icons.search, "Analyse", Colors.black87, () async {
                      await Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) =>
                              SymptomScreen(userId: widget.userId),
                        ),
                      );
                      _loadLatestScore();
                      if (mounted) setState(() {});
                    }),
                    _actionButton(
                        Icons.settings, "Settings", Colors.black54, () async {
                      final messenger = ScaffoldMessenger.of(context);
                      debugPrint('Settings button pressed — opening SettingsScreen');
                      try {
                        await Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (_) => SettingsScreen(
                              userId: widget.userId,
                              userName: widget.userName,
                            ),
                          ),
                        );
                        debugPrint('Returned from SettingsScreen');
                      } catch (e, st) {
                        debugPrint('Navigation to SettingsScreen failed: $e\n$st');
                        if (mounted) {
                          messenger.showSnackBar(
                            SnackBar(content: Text('Could not open Settings: $e')),
                          );
                        }
                      }

                      // reload in case profile changes affected display
                      if (mounted) setState(() {});
                    }),
                  ],
                ),
              ),

              const SizedBox(height: 18),

              // ── Health Tests ──
              _buildCard(
                title: "Health Tests",
                icon: Icons.favorite,
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    _actionButton(Icons.camera_alt, "Face", Colors.blue,
                        () async {
                      await Navigator.pushNamed(
                        context,
                        '/face',
                        arguments: {
                          'userId': widget.userId,
                          'userName': widget.userName,
                        },
                      );
                      _loadLatestScore();
                      if (mounted) setState(() {});
                    }),
                    _actionButton(Icons.mic, "Voice", Colors.green, () async {
                      await Navigator.pushNamed(
                        context,
                        '/voice',
                        arguments: {
                          'userId': widget.userId,
                          'userName': widget.userName,
                        },
                      );
                      _loadLatestScore();
                      if (mounted) setState(() {});
                    }),
                    _actionButton(
                        Icons.monitor_heart, "Symptoms", Colors.deepPurple,
                        () async {
                      await Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) =>
                              SymptomScreen(userId: widget.userId),
                        ),
                      );
                      _loadLatestScore();
                      if (mounted) setState(() {});
                    }),
                  ],
                ),
              ),

              const SizedBox(height: 18),

              // ── Recent Scan History ──
              _buildCard(
                title: "Recent Scan History",
                icon: Icons.history,
                child: Column(
                  children: [
                    if (latestHistory.isEmpty)
                      const Text("No history yet"),
                    ...latestHistory.map((h) => Padding(
                          padding: const EdgeInsets.symmetric(vertical: 4),
                          child: Row(
                            mainAxisAlignment:
                                MainAxisAlignment.spaceBetween,
                            children: [
                              Text(h['score'] ?? '—',
                                  style: const TextStyle(
                                      fontWeight: FontWeight.bold)),
                              Flexible(
                                child: Padding(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 4),
                                  child: Text(
                                    (h['extra']?.isNotEmpty == true
                                        ? h['extra']!
                                        : (h['risk'] ?? '')),
                                    overflow: TextOverflow.ellipsis,
                                    style:
                                        const TextStyle(fontSize: 12),
                                  ),
                                ),
                              ),
                              Text(h['time'] ?? '',
                                  style: const TextStyle(fontSize: 12)),
                              Text(h['date'] ?? '',
                                  style: const TextStyle(fontSize: 12)),
                            ],
                          ),
                        )),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ── Reusable Card (from abc/) ──
  Widget _buildCard(
      {required String title,
      required IconData icon,
      required Widget child}) {
    return Card(
      elevation: 3,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(22)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(title,
                  style: const TextStyle(
                      fontWeight: FontWeight.bold, fontSize: 16)),
              Icon(icon),
            ],
          ),
          const SizedBox(height: 14),
          child,
        ]),
      ),
    );
  }

  // ── Action Button (from abc/) ──
  Widget _actionButton(
      IconData icon, String label, Color color, VoidCallback onTap) {
    return Column(
      children: [
        GestureDetector(
          onTap: () {
            try {
              debugPrint('QuickAction tapped: $label');
              onTap();
            } catch (e, st) {
              debugPrint('QuickAction handler error for $label: $e\n$st');
            }
          },
          child: Container(
            width: 62,
            height: 62,
            decoration: BoxDecoration(
              color: Colors.grey.shade200,
              borderRadius: BorderRadius.circular(16),
            ),
            child: Icon(icon, color: color, size: 30),
          ),
        ),
        const SizedBox(height: 6),
        Text(label),
      ],
    );
  }

  Color _scoreColor(int? score) {
    if (score == null) return Colors.grey;
    if (score >= 70) return Colors.green;
    if (score >= 40) return Colors.orange;
    return Colors.red;
  }
}
