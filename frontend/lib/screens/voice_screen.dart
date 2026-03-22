// Voice Screen – from abc/ frontend with backend upload
//
// abc/'s mic button UI with PlatformAudioRecorder for cross-platform
// recording and upload to POST /predict/audio.
import 'dart:async';
import 'package:flutter/material.dart';
import '../services/audio_recorder.dart';
import '../services/api_service.dart';
import '../models/health_data.dart';

class VoiceScreen extends StatefulWidget {
  final int userId;
  const VoiceScreen({super.key, required this.userId});

  @override
  State<VoiceScreen> createState() => _VoiceScreenState();
}

class _VoiceScreenState extends State<VoiceScreen>
    with SingleTickerProviderStateMixin {
  final PlatformAudioRecorder _recorder = PlatformAudioRecorder();
  bool _initializing = true;
  bool _recording = false;
  bool _uploading = false;
  int _seconds = 0;
  Timer? _timer;
  String? _error;
  Map<String, dynamic>? _result;
  int? _vitaScore;
  String? _riskLevel;

  late AnimationController _pulseCtrl;

  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
      lowerBound: 0.95,
      upperBound: 1.1,
    );
    _initRecorder();
  }

  Future<void> _initRecorder() async {
    final err = await _recorder.init();
    if (mounted) {
      setState(() {
        _initializing = false;
        _error = err;
      });
    }
  }

  void _startRecording() {
    if (_recording) return;
    _recorder.start();
    _pulseCtrl.repeat(reverse: true);
    setState(() {
      _recording = true;
      _result = null;
      _error = null;
      _seconds = 0;
    });
    _timer = Timer.periodic(const Duration(seconds: 1), (_) {
      setState(() => _seconds++);
    });
  }

  // Minimum valid recording duration in seconds.
  static const int _minRecordSeconds = 5;

  Future<void> _stopRecording() async {
    _timer?.cancel();
    _pulseCtrl.stop();
    _pulseCtrl.value = 1.0;

    // Enforce minimum duration to avoid uploading noise / silence.
    if (_seconds < _minRecordSeconds) {
      await _recorder.stop(); // discard bytes
      setState(() {
        _recording = false;
        _uploading = false;
        _error = 'Recording too short — please breathe normally for at least '
            '$_minRecordSeconds seconds and try again.';
      });
      return;
    }

    setState(() {
      _recording = false;
      _uploading = true;
      _error = null;
    });

    try {
      final bytes = await _recorder.stop();
      if (bytes.isEmpty) {
        throw Exception('No audio data recorded.');
      }

      final ext = _recorder.fileExtension;
      final fileName =
          'vita_breathing_${DateTime.now().millisecondsSinceEpoch}.$ext';
      debugPrint('[Voice] Uploading ${bytes.length} bytes as "$fileName"');

      final result = await ApiService.predictAudio(bytes, fileName);
      debugPrint('[Voice] Response: ${result.keys.toList()}');

      // Compute Vita score via /predict/final-score (all available modules)
      int? vitaScore;
      String riskStr = result['risk']?.toString() ?? 'unknown';
      // Store raw result before calling fusion so all modules are captured.
      HealthData.setModuleResult('breathing', result);
      Map<String, dynamic>? scoreResult;
      try {
        scoreResult = await ApiService.predictFinalScore(
          faceResult: HealthData.faceResult,
          audioResult: HealthData.audioResult,
          symptomResult: HealthData.symptomResult,
        );
        vitaScore = scoreResult['vita_health_score'] as int?;
        riskStr = scoreResult['overall_risk']?.toString() ?? riskStr;
        debugPrint('[Voice] Vita score: $vitaScore, risk: $riskStr');
      } catch (e) {
        debugPrint('[Voice] Final-score fallback: $e');
      }

      // Score engine returns null/unknown when it has no usable data
      if (vitaScore == null || vitaScore == 0 ||
          riskStr == 'error' || riskStr == 'unknown') {
        vitaScore = null;
        riskStr = result['risk']?.toString() ?? 'unknown';
      }

      if (!mounted) return;

      // If the backend flagged the recording as unusable, treat it as a
      // retake request rather than showing a spurious result.
      // Also reject when there is no breathing rate at all (matches face-scan
      // behaviour of refusing to estimate when signal is absent).
      final retakeRequired = result['retake_required'] == true;
      final reliability = result['reliability']?.toString().toLowerCase();
      final brValue = result['breathing_rate'];
      final noRate = brValue == null || brValue == 0;
      final isUnreliable = reliability == 'unreliable';

      final retakeReasons = result['retake_reasons'];
      final warningMsg = result['warning']?.toString() ?? '';
      final isEnvironmental = warningMsg.toLowerCase().contains('environmental');
      final retakeMsg = isEnvironmental
          ? warningMsg
          : retakeReasons is List && retakeReasons.isNotEmpty
              ? retakeReasons.first.toString()
              : noRate || isUnreliable
                  ? 'No measurable breathing pattern detected. Please breathe normally and try again.'
                  : 'Audio quality too low — please record again in a quiet environment.';

      if (retakeRequired || noRate || isUnreliable) {
        setState(() {
          _uploading = false;
          _error = retakeMsg;
        });
        return;
      }

      setState(() {
        _result = result;
        _vitaScore = vitaScore;
        _riskLevel = riskStr;
        _uploading = false;
      });

      // Update global health data
      if (vitaScore != null && scoreResult != null) {
        HealthData.applyFusionScore(
          scoreResult,
          module: 'breathing',
          moduleScore: vitaScore,
          moduleRisk: riskStr,
        );
      } else {
        HealthData.riskLevel = riskStr;
        final brVal = _toDouble(result['breathing_rate']);
        final brStr = brVal != null && brVal > 0
            ? 'BR ${brVal.toStringAsFixed(0)} br/min'
            : 'breathing scan';
        HealthData.addHistoryEntry(risk: riskStr, status: brStr, module: 'breathing');
      }

      // Save enriched scan to backend history
      if (widget.userId > 0) {
        try {
          final toSave = Map<String, dynamic>.from(result);
          if (vitaScore != null) toSave['vita_health_score'] = vitaScore;
          toSave['overall_risk'] = riskStr;
          await ApiService.saveScan(widget.userId, 'audio', toSave);
          debugPrint('[Voice] Scan saved to backend.');
        } catch (e) {
          debugPrint('[Voice] Failed to save scan: $e');
        }
      }
    } on ApiException catch (e) {
      if (mounted) {
        setState(() {
          _error = 'Backend error (${e.statusCode}): ${e.body}';
          _uploading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = 'Upload failed: $e';
          _uploading = false;
        });
      }
    }
  }

  String _formatTime(int secs) {
    final m = (secs ~/ 60).toString().padLeft(2, '0');
    final s = (secs % 60).toString().padLeft(2, '0');
    return '$m:$s';
  }

  double? _toDouble(dynamic v) {
    if (v == null) return null;
    if (v is double) return v;
    if (v is int) return v.toDouble();
    if (v is String) return double.tryParse(v);
    return null;
  }

  @override
  void dispose() {
    _timer?.cancel();
    _pulseCtrl.dispose();
    _recorder.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFEDEFF3),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(18),
          child: Column(
            children: [
              // ── Header ──
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: const [
                  Row(children: [
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
                    child: Icon(Icons.person, color: Colors.black),
                  ),
                ],
              ),

              const SizedBox(height: 20),

              // ── Content ──
              Expanded(
                child: _initializing
                    ? const Center(child: CircularProgressIndicator())
                    : _result != null
                        ? _buildResultView()
                        : _buildRecordingView(),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildRecordingView() {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(26)),
      child: Container(
        width: double.infinity,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(26),
          gradient: const LinearGradient(
            colors: [Color(0xFFE9EDF5), Color(0xFFDDE3F0)],
          ),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (_uploading) ...[
              const CircularProgressIndicator(),
              const SizedBox(height: 20),
              const Text("Analyzing your voice...",
                  style: TextStyle(fontSize: 18)),
            ] else ...[
              const Text("Voice Breathing Test",
                  style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
              const SizedBox(height: 10),
              Text(
                _recording
                    ? "Breathe normally into the microphone"
                    : "Tap the mic to start recording",
                style: const TextStyle(fontSize: 16, color: Colors.grey),
              ),

              const SizedBox(height: 30),

              // ── Mic Button ──
              // Disable the stop tap during the first 5 seconds so the user
              // cannot accidentally submit noise before any breathing is detected.
              GestureDetector(
                onTap: _recording && _seconds < _minRecordSeconds
                    ? null  // locked: must breathe for at least 5 s
                    : _recording
                        ? _stopRecording
                        : _startRecording,
                child: ScaleTransition(
                  scale: _pulseCtrl,
                  child: Opacity(
                    opacity: _recording && _seconds < _minRecordSeconds
                        ? 0.55
                        : 1.0,
                    child: Container(
                      width: 150,
                      height: 150,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: _recording ? Colors.red : Colors.green,
                        boxShadow: [
                          BoxShadow(
                            color: (_recording ? Colors.red : Colors.green)
                                .withValues(alpha: 0.4),
                            blurRadius: 20,
                            spreadRadius: 4,
                          ),
                        ],
                      ),
                      child: Icon(
                        _recording ? Icons.stop : Icons.mic,
                        color: Colors.white,
                        size: 60,
                      ),
                    ),
                  ),
                ),
              ),

              const SizedBox(height: 20),

              // Timer / progress hint
              if (_recording) ...[
                Text(_formatTime(_seconds),
                    style: const TextStyle(
                        fontSize: 36, fontWeight: FontWeight.w600)),
                if (_seconds < _minRecordSeconds)
                  Padding(
                    padding: const EdgeInsets.only(top: 6),
                    child: Text(
                      'Hold for ${_minRecordSeconds - _seconds}s more\u2026',
                      style: const TextStyle(
                          fontSize: 13, color: Colors.deepOrangeAccent),
                    ),
                  ),
              ],

              if (!_recording && _seconds > 0)
                const Text("Recording stopped",
                    style: TextStyle(fontSize: 16, color: Colors.grey)),

              if (_error != null)
                Padding(
                  padding: const EdgeInsets.only(top: 12),
                  child: Text(_error!,
                      style: const TextStyle(color: Colors.red),
                      textAlign: TextAlign.center),
                ),

              const SizedBox(height: 15),

              if (!_recording)
                const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 30),
                  child: Text(
                    "Record at least 5 seconds of normal breathing "
                    "for accurate analysis.",
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 13, color: Colors.grey),
                  ),
                ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildResultView() {
    final risk = _result!['risk']?.toString() ?? 'unknown';
    final br = _toDouble(_result!['breathing_rate']);
    final confidence = _toDouble(_result!['confidence']) ?? 0.0;
    final reliability = _result!['reliability']?.toString().toLowerCase();
    final message = _result!['message']?.toString() ?? '';
    final warning = _result!['warning']?.toString();
    final brNormalLow = _toDouble(_result!['breathing_rate_normal_low']);
    final brNormalHigh = _toDouble(_result!['breathing_rate_normal_high']);
    final retake = _result!['retake_required'] == true;
    final retakeReasons = _result!['retake_reasons'];

    final bool hasBr = br != null && br > 0;
    final bool lowConf = confidence < 0.33;
    final bool hasWarnings =
        lowConf || retake || (warning != null && warning.isNotEmpty);

    final content = Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // ── Vita score badge ──────────────────────────────────────────────
        if (_vitaScore != null)
          Center(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
              decoration: BoxDecoration(
                color: _scoreColor(_vitaScore!),
                borderRadius: BorderRadius.circular(24),
              ),
              child: Text(
                'Vita Score: $_vitaScore%  •  ${_riskLevel!.toUpperCase()}',
                style: const TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                  letterSpacing: 0.4,
                ),
              ),
            ),
          )
        else
          Center(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.grey.shade200,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                hasBr
                    ? 'Breathing Analysis Complete'
                    : 'Breathing Analysis — retake for a result',
                style: TextStyle(fontSize: 13, color: Colors.grey.shade600),
              ),
            ),
          ),
        const SizedBox(height: 16),

        // ── Prominent breathing rate ─────────────────────────────────────
        Center(
          child: Column(children: [
            Text(
              hasBr ? br.toStringAsFixed(0) : '—',
              style: TextStyle(
                fontSize: 80,
                fontWeight: FontWeight.w700,
                color: hasBr ? const Color(0xFF1A2340) : Colors.grey,
                height: 1.0,
              ),
            ),
            const SizedBox(height: 2),
            Text(
              'breaths / min',
              style: TextStyle(
                  fontSize: 17,
                  color: Colors.grey.shade600,
                  fontWeight: FontWeight.w500),
            ),
            const SizedBox(height: 4),
            Text(
              brNormalLow != null && brNormalHigh != null
                  ? 'Normal: ${brNormalLow.toStringAsFixed(0)}–${brNormalHigh.toStringAsFixed(0)} breaths/min'
                  : hasBr
                      ? _breathingLabel(risk)
                      : 'Unable to estimate breathing rate',
              style:
                  TextStyle(fontSize: 12, color: Colors.grey.shade500),
            ),
          ]),
        ),
        const SizedBox(height: 22),
        const Divider(height: 1),
        const SizedBox(height: 14),

        // ── Metrics ───────────────────────────────────────────────────────
        _metricRow(Icons.air, 'Assessment', _breathingLabel(risk),
            _riskColor(risk)),
        _metricRow(
            Icons.show_chart,
            'Confidence',
            '${(confidence * 100).toStringAsFixed(1)}%',
            _confidenceColor(confidence)),
        if (reliability != null)
          _metricRow(Icons.verified_outlined, 'Reliability',
              reliability.toUpperCase(), _reliabilityColor(reliability)),

        if (message.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Text(message,
                style: const TextStyle(
                    fontSize: 13, color: Color(0xFF6B7280))),
          ),

        // ── Warnings ─────────────────────────────────────────────────────
        if (hasWarnings) ...[
          const SizedBox(height: 18),
          Container(
            decoration: BoxDecoration(
              color: Colors.amber.shade50,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: Colors.amber.shade300),
            ),
            padding:
                const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (lowConf)
                  _warningLine(Icons.info_outline, Colors.amber.shade800,
                      'Low confidence — weak breathing signal detected'),
                if (retake)
                  _warningLine(
                      Icons.refresh,
                      Colors.orange.shade800,
                      retakeReasons is List && retakeReasons.isNotEmpty
                          ? retakeReasons.first.toString()
                          : 'Retake recommended for better accuracy'),
                if (!retake && warning != null && warning.isNotEmpty)
                  _warningLine(Icons.warning_amber_outlined,
                      Colors.orange.shade800, warning),
              ],
            ),
          ),
        ],

        const SizedBox(height: 24),

        // ── Actions ──────────────────────────────────────────────────────
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton.icon(
              icon: const Icon(Icons.replay),
              label: const Text('Record Again'),
              onPressed: () {
                setState(() {
                  _result = null;
                  _vitaScore = null;
                  _riskLevel = null;
                  _error = null;
                  _seconds = 0;
                  _uploading = false;
                });
              },
            ),
            const SizedBox(width: 16),
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Back to Home'),
            ),
          ],
        ),
        const SizedBox(height: 8),
      ],
    );

    final card = Card(
      elevation: 4,
      shape:
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(26)),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(24, 24, 24, 16),
        child: content,
      ),
    );

    final bool isDesktop = MediaQuery.of(context).size.width >= 720;
    if (isDesktop) {
      return Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 560),
          child: SingleChildScrollView(child: card),
        ),
      );
    }
    return SingleChildScrollView(child: card);
  }

  // ── Helpers ────────────────────────────────────────────────────────────────

  Color _scoreColor(int score) {
    if (score >= 70) return Colors.green.shade700;
    if (score >= 40) return Colors.orange.shade700;
    return Colors.red.shade600;
  }

  String _breathingLabel(String risk) {
    switch (risk.toLowerCase()) {
      case 'normal':
        return 'Normal Range';
      case 'elevated':
        return 'Above Resting Range';
      case 'high':
        return 'Exercise Range';
      case 'very_high':
        return 'Very High Rate';
      case 'low_rate':
        return 'Below Normal Rate';
      // legacy labels (backward compat)
      case 'low':
        return 'Normal Range';
      case 'moderate':
        return 'Above Resting Range';
      case 'unreliable':
        return 'No Signal';
      default:
        return risk.toUpperCase();
    }
  }

  Widget _metricRow(
      IconData icon, String label, String value, Color color) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(children: [
        Icon(icon, size: 17, color: color),
        const SizedBox(width: 10),
        Text(label,
            style: const TextStyle(
                fontSize: 14, color: Color(0xFF444A5A))),
        const Spacer(),
        Text(value,
            style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: color)),
      ]),
    );
  }

  Widget _warningLine(IconData icon, Color color, String text) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, color: color, size: 17),
            const SizedBox(width: 8),
            Expanded(
                child: Text(text,
                    style: TextStyle(fontSize: 13, color: color))),
          ]),
    );
  }

  Color _riskColor(String risk) {
    switch (risk.toLowerCase()) {
      case 'normal':
      case 'low':           // legacy
        return Colors.green.shade700;
      case 'elevated':
      case 'moderate':      // legacy
        return Colors.orange.shade600;
      case 'high':
        return Colors.deepOrange.shade600;
      case 'very_high':
        return Colors.red.shade600;
      case 'low_rate':
        return Colors.blue.shade700;
      default:
        return Colors.grey.shade600;
    }
  }

  Color _confidenceColor(double v) {
    if (v >= 0.7) return Colors.green.shade700;
    if (v >= 0.4) return Colors.orange.shade700;
    return Colors.red.shade600;
  }

  Color _reliabilityColor(String? r) {
    if (r == null) return Colors.grey;
    switch (r.toLowerCase()) {
      case 'high':
        return Colors.green.shade700;
      case 'medium':
      case 'moderate':
        return Colors.orange.shade700;
      case 'low':
        return Colors.red.shade600;
      default:
        return Colors.grey.shade600;
    }
  }
}
