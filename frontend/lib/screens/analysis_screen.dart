// Analysis Screen – from abc/ frontend with backend call
//
// abc/'s loading/intermediate screen with real backend call to
// POST /predict/symptom. Shows progress animation while waiting,
// then navigates to ResultScreen with real results.
import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../models/health_data.dart';
import 'result_screen.dart';

class AnalysisScreen extends StatefulWidget {
  final int userId;
  /// Structured symptom payload from the symptom form.  Contains
  /// major_symptom, minor_symptoms, age, gender, days_suffering,
  /// symptom_category, fever, pain, difficulty_breathing, severity, text.
  final Map<String, dynamic> symptomData;

  const AnalysisScreen({
    super.key,
    required this.userId,
    required this.symptomData,
  });

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen> {
  String _statusText = 'Analysing symptoms...';
  String? _error;

  @override
  void initState() {
    super.initState();
    _runAnalysis();
  }

  Future<void> _runAnalysis() async {
    try {
      setState(() => _statusText = 'Analysing symptoms...');

      debugPrint('[Analysis] Calling /predict/symptom ...');
      final symptomResult =
          await ApiService.predictSymptom(widget.symptomData);
      debugPrint('[Analysis] Symptom result: ${symptomResult.keys.toList()}');

      if (!mounted) return;

      // Get a proper Vita score from the score engine using all available modules
      final risk = symptomResult['risk']?.toString() ?? 'unknown';
      int? score;
      String riskStr = risk;
      // Store raw result before calling fusion so all modules are captured.
      HealthData.setModuleResult('symptom', symptomResult);
      Map<String, dynamic>? scoreResult;
      try {
        scoreResult = await ApiService.predictFinalScore(
          faceResult: HealthData.faceResult,
          audioResult: HealthData.audioResult,
          symptomResult: HealthData.symptomResult,
        );
        score = scoreResult['vita_health_score'] as int?;
        riskStr = scoreResult['overall_risk']?.toString() ?? risk;
        debugPrint('[Analysis] Vita score: $score, overall_risk: $riskStr');
      } catch (e) {
        debugPrint('[Analysis] Final-score call failed: $e');
        score = _estimateScore(risk);
      }

      // Only update HealthData if we have a real score
      if (score != null && score > 0 && scoreResult != null) {
        HealthData.applyFusionScore(
          scoreResult,
          module: 'symptom',
          moduleScore: score,
          moduleRisk: riskStr,
        );
      } else {
        HealthData.riskLevel = riskStr;
        HealthData.addHistoryEntry(risk: riskStr, status: 'symptom analysis', module: 'symptom');
      }

      // Inject the computed vita score into result for ResultScreen display
      final enrichedResult = Map<String, dynamic>.from(symptomResult);
      if (score != null) enrichedResult['vita_health_score'] = score;
      enrichedResult['overall_risk'] = riskStr;

      // Save scan to backend
      if (widget.userId > 0) {
        try {
          await ApiService.saveScan(widget.userId, 'symptom', enrichedResult);
        } catch (e) {
          debugPrint('[Analysis] Failed to save scan: $e');
        }
      }

      if (!mounted) return;

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => ResultScreen(
            userId: widget.userId,
            result: enrichedResult,
            scanType: 'symptom',
          ),
        ),
      );
    } on ApiException catch (e) {
      debugPrint('[Analysis] API error: ${e.statusCode} ${e.body}');
      if (mounted) {
        setState(() {
          _error = 'Backend error (${e.statusCode}): ${e.body}';
        });
      }
    } catch (e) {
      debugPrint('[Analysis] Error: $e');
      if (mounted) {
        setState(() => _error = 'Analysis failed: $e');
      }
    }
  }

  int _estimateScore(String risk) {
    switch (risk.toLowerCase()) {
      case 'low':
        return 80;
      case 'moderate':
        return 55;
      case 'high':
        return 30;
      default:
        return 50;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFEDEFF3),
      body: SafeArea(
        child: Column(
          children: [
            // ── Header ──
            Padding(
              padding: const EdgeInsets.all(18),
              child: Row(children: const [
                Icon(Icons.health_and_safety, color: Colors.green, size: 38),
                SizedBox(width: 8),
                Text("Vita AI",
                    style: TextStyle(
                        fontSize: 28,
                        fontWeight: FontWeight.bold,
                        color: Colors.blue)),
              ]),
            ),

            // ── Content ──
            Expanded(
              child: Center(
                child: Padding(
                  padding: const EdgeInsets.all(30),
                  child: Card(
                    elevation: 6,
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(26)),
                    child: Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(40),
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(26),
                        gradient: const LinearGradient(
                          colors: [Color(0xFFE9EDF5), Color(0xFFDDE3F0)],
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                        ),
                      ),
                      child: _error != null
                          ? _buildError()
                          : Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                const SizedBox(
                                  width: 80,
                                  height: 80,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 8,
                                    color: Colors.blue,
                                  ),
                                ),
                                const SizedBox(height: 30),
                                Text(
                                  _statusText,
                                  style: const TextStyle(
                                      fontSize: 20,
                                      fontWeight: FontWeight.w600),
                                ),
                                const SizedBox(height: 10),
                                const Text(
                                  "Please wait while we process your data",
                                  style: TextStyle(
                                      fontSize: 14, color: Colors.grey),
                                  textAlign: TextAlign.center,
                                ),
                              ],
                            ),
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildError() {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        const Icon(Icons.error_outline, size: 60, color: Colors.red),
        const SizedBox(height: 20),
        Text(_error!,
            style: const TextStyle(color: Colors.red, fontSize: 16),
            textAlign: TextAlign.center),
        const SizedBox(height: 20),
        ElevatedButton(
          onPressed: () {
            setState(() => _error = null);
            _runAnalysis();
          },
          child: const Text('Retry'),
        ),
        const SizedBox(height: 10),
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Go Back'),
        ),
      ],
    );
  }
}
