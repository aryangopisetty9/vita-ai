// Home Screen – from abc/ frontend with backend data integration
//
// abc/'s card-based layout (score card, quick actions, health tests, history)
// wired to the real backend via ApiService.
import 'dart:convert';
import 'package:flutter/material.dart';
import '../models/health_data.dart';
import '../services/api_service.dart';
import 'face_scan_screen.dart';
import 'voice_screen.dart';
import 'symptom_screen.dart';
import 'scan_history_screen.dart';
import 'sos_screen.dart';

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
      final scans = await ApiService.getScanHistory(widget.userId);
      if (scans.isNotEmpty) {
        // Walk through scans (newest first) and populate per-module raw results
        // from the stored result_json so fusion can be recomputed server-side.
        final seen = <String>{};
        for (final s in scans) {
          final scan = s as Map<String, dynamic>;
          final scanType = scan['scan_type']?.toString() ?? '';
          final resultJson = scan['result_json']?.toString() ?? '';

          if (!seen.contains(scanType) && resultJson.isNotEmpty) {
            seen.add(scanType);
            final module = scanType == 'audio' ? 'breathing' : scanType;
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
            if (vita != null && vita > 0) {
              HealthData.score = vita;
              HealthData.riskLevel = overall;
            }
          } catch (_) {
            // Fusion endpoint unavailable — leave existing score as-is.
          }
        }

        // Re-populate dashboard history from backend scans.
        HealthData.history = scans.take(10).map((s) {
          final scan = s as Map<String, dynamic>;
          final vitaScore = scan['vita_score'];
          final riskLevel = scan['risk_level']?.toString() ?? '';
          final scanType = scan['scan_type']?.toString() ?? '';
          final createdAt = scan['created_at']?.toString() ?? '';
          DateTime? dt;
          try {
            dt = DateTime.parse(createdAt);
          } catch (_) {}
          final module = scanType == 'audio' ? 'breathing' : scanType;
          final label = module == 'face'
              ? 'Face scan'
              : module == 'breathing'
                  ? 'Breathing scan'
                  : module == 'symptom'
                      ? 'Symptom check'
                      : module;
          return <String, String>{
            'score': vitaScore != null ? '$vitaScore%' : '—',
            'fusion': '',
            'module': module,
            'risk': riskLevel,
            'time': dt != null
                ? '${dt.hour}:${dt.minute.toString().padLeft(2, '0')}'
                : '',
            'date': dt != null
                ? '${dt.day}-${dt.month}-${dt.year}'
                : '',
            'extra': vitaScore == null ? 'score unavailable' : label,
          };
        }).toList();
      } else {
        // No scans remain — clear stale score so the gauge shows "no data".
        HealthData.score = null;
        HealthData.riskLevel = 'unknown';
        HealthData.history = [];
      }
      if (mounted) setState(() {});
    } catch (_) {}
  }

  @override
  Widget build(BuildContext context) {
    final int? score = HealthData.score;

    return Scaffold(
      backgroundColor: const Color(0xFFEDEFF3),
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
                        Icons.settings, "Settings", Colors.black54, () {}),
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
                      await Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) =>
                              FaceScanScreen(userId: widget.userId),
                        ),
                      );
                      _loadLatestScore();
                      if (mounted) setState(() {});
                    }),
                    _actionButton(Icons.mic, "Voice", Colors.green, () async {
                      await Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) =>
                              VoiceScreen(userId: widget.userId),
                        ),
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
                    if (HealthData.history.isEmpty)
                      const Text("No history yet"),
                    ...HealthData.history.take(5).map((h) => Padding(
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
          onTap: onTap,
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
