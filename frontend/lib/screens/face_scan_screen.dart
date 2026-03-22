// Face Scan Screen — camera recording, upload to /predict/face, result display
//
// Uses an explicit _ScanPhase state machine to prevent mixed-state bugs.
// After receiving face results, calls /predict/final-score to compute
// a Vita Health Score before saving.
import 'dart:async';
import 'package:flutter/foundation.dart'
    show kIsWeb, defaultTargetPlatform, TargetPlatform;
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../services/api_service.dart';
import '../models/health_data.dart';

// ── State machine ────────────────────────────────────────────────────────────

enum _ScanPhase {
  initializing, // camera being set up
  ready, // camera ready, waiting for user
  recording, // actively recording video
  uploading, // upload + analysis in progress
  result, // result displayed
  error, // camera/hardware error (with retry)
}

// ── Responsive sizing ────────────────────────────────────────────────────────

class _ScanCircleSpec {
  final double outerDiameter;
  final double innerDiameter;
  final double previewDiameter;
  final double strokeWidth;
  const _ScanCircleSpec({
    required this.outerDiameter,
    required this.innerDiameter,
    required this.previewDiameter,
    required this.strokeWidth,
  });
}

// ── Widget ───────────────────────────────────────────────────────────────────

class FaceScanScreen extends StatefulWidget {
  final int userId;
  const FaceScanScreen({super.key, required this.userId});
  @override
  State<FaceScanScreen> createState() => _FaceScanScreenState();
}

class _FaceScanScreenState extends State<FaceScanScreen>
    with SingleTickerProviderStateMixin {
  // ── Core state ──
  _ScanPhase _phase = _ScanPhase.initializing;
  CameraController? _camera;
  int _previewEpoch = 0;

  // ── Recording ──
  int _timeLeft = 30;
  Timer? _timer;
  late AnimationController _progress;

  // ── Result ──
  Map<String, dynamic>? _faceResult;
  int? _vitaScore;
  String? _riskLevel;
  String? _error;

  // ── Desktop detection ──
  bool get _isDesktop =>
      kIsWeb || defaultTargetPlatform == TargetPlatform.windows;

  _ScanCircleSpec get _circleSpec {
    if (_isDesktop) {
      return const _ScanCircleSpec(
        outerDiameter: 360,
        innerDiameter: 300,
        previewDiameter: 210,
        strokeWidth: 20,
      );
    }
    return const _ScanCircleSpec(
      outerDiameter: 320,
      innerDiameter: 260,
      previewDiameter: 170,
      strokeWidth: 18,
    );
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Lifecycle
  // ──────────────────────────────────────────────────────────────────────────

  @override
  void initState() {
    super.initState();
    _progress = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 30),
    );
    _initCamera();
  }

  @override
  void dispose() {
    _timer?.cancel();
    _progress.dispose();
    _camera?.dispose();
    super.dispose();
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Camera
  // ──────────────────────────────────────────────────────────────────────────

  Future<void> _initCamera() async {
    if (mounted) setState(() => _phase = _ScanPhase.initializing);
    try {
      final cams = await availableCameras();
      if (cams.isEmpty) {
        _setCameraError('No cameras found on this device.');
        return;
      }
      final frontCam = cams.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cams.first,
      );
      final old = _camera;
      _camera = null;
      await old?.dispose();

      final controller = CameraController(frontCam, ResolutionPreset.medium);
      await controller.initialize();
      _camera = controller;

      if (mounted) {
        setState(() {
          _previewEpoch += 1;
          _phase = _ScanPhase.ready;
          _error = null;
        });
      }
    } catch (e) {
      _setCameraError('Camera error: $e');
    }
  }

  void _setCameraError(String msg) {
    if (mounted) {
      setState(() {
        _error = msg;
        _phase = _ScanPhase.error;
      });
    }
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Recording
  // ──────────────────────────────────────────────────────────────────────────

  Future<void> _startScan() async {
    if (_phase != _ScanPhase.ready) return;
    final c = _camera;
    if (c == null || !c.value.isInitialized) {
      await _initCamera();
      return;
    }
    try {
      await c.startVideoRecording();
    } catch (e) {
      if (mounted) setState(() => _error = 'Recording error: $e');
      return;
    }
    _progress.forward(from: 0);
    if (mounted) {
      setState(() {
        _phase = _ScanPhase.recording;
        _timeLeft = 30;
        _faceResult = null;
        _vitaScore = null;
        _riskLevel = null;
        _error = null;
      });
    }
    _timer = Timer.periodic(const Duration(seconds: 1), (t) {
      if (_timeLeft > 1) {
        if (mounted) setState(() => _timeLeft--);
      } else {
        _stopScan();
      }
    });
  }

  Future<void> _stopScan() async {
    _timer?.cancel();
    final c = _camera;
    if (c == null || _phase != _ScanPhase.recording) return;
    if (mounted) setState(() => _phase = _ScanPhase.uploading);

    try {
      final xfile = await c.stopVideoRecording();
      debugPrint('[FaceScan] Recording stopped. path=${xfile.path}');

      final bytes = await xfile.readAsBytes();
      debugPrint('[FaceScan] Bytes read: ${bytes.length}');
      if (bytes.isEmpty) {
        throw Exception('Recorded video is empty — nothing to upload.');
      }

      final fileName = kIsWeb
          ? 'face_scan.webm'
          : (xfile.name.isNotEmpty ? xfile.name : 'face_scan.mp4');

      // 1. Upload face video → get face metrics
      debugPrint('[FaceScan] Uploading to /predict/face as "$fileName"...');
      final faceResult = await ApiService.predictFace(bytes, fileName);
      debugPrint('[FaceScan] Response keys: ${faceResult.keys.toList()}');

      // 2. Compute Vita score via /predict/final-score
      int? vitaScore;
      String riskStr = faceResult['risk']?.toString() ?? 'unknown';
      // Store raw face result so fusion can be called with all modules.
      HealthData.setModuleResult('face', faceResult);

      // Call fusion with ALL available module results (face + any cached).
      Map<String, dynamic>? scoreResult;
      try {
        scoreResult = await ApiService.predictFinalScore(
          faceResult: HealthData.faceResult,
          audioResult: HealthData.audioResult,
          symptomResult: HealthData.symptomResult,
        );
        vitaScore = scoreResult['vita_health_score'] as int?;
        riskStr = scoreResult['overall_risk']?.toString() ?? riskStr;
        debugPrint('[FaceScan] Vita score: $vitaScore, risk: $riskStr');
      } catch (e) {
        debugPrint('[FaceScan] Final-score fallback: $e');
        // Score endpoint failed — leave vitaScore as null so the UI
        // shows "score unavailable" rather than a fabricated number.
      }

      // If score engine had no usable HR data it returns {vita_health_score:null,
      // overall_risk:'unknown'}.  Clear to null so the UI shows "unavailable"
      // rather than a misleading badge.
      if (vitaScore == null || vitaScore == 0 || riskStr == 'error' || riskStr == 'unknown') {
        if (vitaScore == null || vitaScore == 0) {
          vitaScore = null;
          riskStr = faceResult['risk']?.toString() ?? 'unreliable';
        }
      }

      // 3. Update global health data — always add a history entry so the
      // dashboard shows this scan even when no Vita score was computed.
      if (vitaScore != null && scoreResult != null) {
        HealthData.applyFusionScore(
          scoreResult,
          module: 'face',
          moduleScore: vitaScore,
          moduleRisk: riskStr,
        );
      } else {
        // Weak / unreliable scan — record HR + status without touching score.
        final hr = _toDouble(faceResult['heart_rate']);
        HealthData.addHistoryEntry(
          heartRate: hr?.toStringAsFixed(0),
          risk: riskStr,
          status: 'score unavailable',
          module: 'face',
        );
      }

      // 4. Save scan to backend (include vita score so history is useful)
      if (widget.userId > 0) {
        try {
          final toSave = Map<String, dynamic>.from(faceResult);
          if (vitaScore != null) toSave['vita_health_score'] = vitaScore;
          toSave['overall_risk'] = riskStr;
          await ApiService.saveScan(widget.userId, 'face', toSave);
          debugPrint('[FaceScan] Scan saved to backend.');
        } catch (e) {
          debugPrint('[FaceScan] Failed to save scan: $e');
        }
      }

      if (!mounted) return;
      setState(() {
        _faceResult = faceResult;
        _vitaScore = vitaScore;
        _riskLevel = riskStr;
        _phase = _ScanPhase.result;
      });
    } on ApiException catch (e) {
      debugPrint('[FaceScan] API error: ${e.statusCode} ${e.body}');
      if (mounted) {
        setState(() {
          _error = 'Backend error (${e.statusCode}): ${e.body}';
          _phase = _ScanPhase.ready;
        });
      }
    } catch (e) {
      debugPrint('[FaceScan] Upload error: $e');
      if (mounted) {
        setState(() {
          _error = 'Upload failed: $e';
          _phase = _ScanPhase.ready;
        });
      }
    }
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Scan Again
  // ──────────────────────────────────────────────────────────────────────────

  Future<void> _scanAgain() async {
    if (_phase == _ScanPhase.initializing || _phase == _ScanPhase.uploading) {
      return;
    }
    _timer?.cancel();
    _progress.stop();
    _progress.reset();

    final c = _camera;
    if (c != null && c.value.isRecordingVideo) {
      try {
        await c.stopVideoRecording();
      } catch (_) {}
    }

    if (!mounted) return;
    setState(() {
      _faceResult = null;
      _vitaScore = null;
      _riskLevel = null;
      _error = null;
      _timeLeft = 30;
    });
    await _initCamera();
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Helpers
  // ──────────────────────────────────────────────────────────────────────────

  double? _toDouble(dynamic v) {
    if (v == null) return null;
    if (v is double) return v;
    if (v is int) return v.toDouble();
    if (v is String) return double.tryParse(v);
    return null;
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Build
  // ──────────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFEDEFF3),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(18),
          child: Column(
            children: [
              _buildHeader(),
              const SizedBox(height: 20),
              Expanded(child: _buildBody()),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: const [
        Row(children: [
          Icon(Icons.health_and_safety, color: Colors.green, size: 38),
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
    );
  }

  Widget _buildBody() {
    switch (_phase) {
      case _ScanPhase.result:
        return _buildResultView();
      case _ScanPhase.error:
        return _buildErrorView();
      case _ScanPhase.initializing:
      case _ScanPhase.ready:
      case _ScanPhase.recording:
      case _ScanPhase.uploading:
        return _buildScanView();
    }
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Scan view (preview + ring + buttons)
  // ──────────────────────────────────────────────────────────────────────────

  Widget _buildScanView() {
    final spec = _circleSpec;
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
            if (_phase == _ScanPhase.uploading) ...[
              const CircularProgressIndicator(),
              const SizedBox(height: 20),
              const Text("Analyzing face scan...",
                  style: TextStyle(fontSize: 18)),
            ] else ...[
              // ── Ring + camera preview ──
              LayoutBuilder(
                builder: (context, constraints) {
                  final double safeMax = constraints.maxWidth.isFinite
                      ? (constraints.maxWidth - 28)
                      : spec.outerDiameter;
                  final double outer = safeMax < spec.outerDiameter
                      ? safeMax
                      : spec.outerDiameter;
                  final double ratio = outer / spec.outerDiameter;
                  final double inner = spec.innerDiameter * ratio;
                  final double preview = spec.previewDiameter * ratio;
                  final double stroke = spec.strokeWidth * ratio;

                  return SizedBox(
                    width: outer,
                    height: outer,
                    child: Stack(
                      alignment: Alignment.center,
                      children: [
                        // Inner static ring
                        SizedBox(
                          width: inner,
                          height: inner,
                          child: CircularProgressIndicator(
                            value: 1,
                            strokeWidth: stroke,
                            color: Colors.black87,
                          ),
                        ),
                        // Outer animated ring
                        SizedBox(
                          width: outer,
                          height: outer,
                          child: AnimatedBuilder(
                            animation: _progress,
                            builder: (_, __) => CircularProgressIndicator(
                              value: _progress.value,
                              strokeWidth: stroke,
                              color: Colors.greenAccent.shade400,
                            ),
                          ),
                        ),
                        // Camera preview / placeholder
                        _buildPreviewBubble(preview),
                      ],
                    ),
                  );
                },
              ),

              SizedBox(height: _isDesktop ? 34 : 25),

              // Timer
              if (_phase == _ScanPhase.recording)
                Text("Scanning... $_timeLeft s",
                    style: const TextStyle(fontSize: 18)),

              // Start button
              if (_phase == _ScanPhase.ready)
                Padding(
                  padding: EdgeInsets.only(top: _isDesktop ? 26 : 20),
                  child: ElevatedButton(
                    onPressed: _startScan,
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 30, vertical: 14),
                    ),
                    child: const Text("Start Face Scan",
                        style: TextStyle(fontSize: 18)),
                  ),
                ),

              // Initializing hint
              if (_phase == _ScanPhase.initializing) ...[
                const SizedBox(height: 12),
                const Text('Preparing camera...',
                    style: TextStyle(color: Colors.grey, fontSize: 14)),
              ],

              // Error text (for non-fatal errors like upload failure)
              if (_error != null)
                Padding(
                  padding: const EdgeInsets.only(top: 12),
                  child: Text(_error!,
                      style: const TextStyle(color: Colors.red),
                      textAlign: TextAlign.center),
                ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildPreviewBubble(double diameter) {
    final controller = _camera;
    final bool canShow = _phase != _ScanPhase.initializing &&
        controller != null &&
        controller.value.isInitialized;

    if (!canShow) return _buildPreviewPlaceholder(diameter);

    return ClipOval(
      child: SizedBox(
        width: diameter,
        height: diameter,
        child: CameraPreview(
          controller,
          key: ValueKey('preview_${_previewEpoch}_$diameter'),
        ),
      ),
    );
  }

  Widget _buildPreviewPlaceholder(double diameter) {
    return Container(
      width: diameter,
      height: diameter,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: const Color(0xFFF2F5FB),
        border: Border.all(color: const Color(0xFFD0D9EA), width: 2),
      ),
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: const [
            Icon(Icons.videocam_outlined, color: Color(0xFF7D889D), size: 30),
            SizedBox(height: 8),
            Text('Preview loading',
                style: TextStyle(color: Color(0xFF7D889D), fontSize: 12)),
          ],
        ),
      ),
    );
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Error view (camera init failure)
  // ──────────────────────────────────────────────────────────────────────────

  Widget _buildErrorView() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.videocam_off, size: 60, color: Colors.grey),
          const SizedBox(height: 16),
          Text(_error ?? 'An error occurred.',
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.red)),
          const SizedBox(height: 20),
          ElevatedButton.icon(
            icon: const Icon(Icons.refresh),
            label: const Text('Retry'),
            onPressed: _initCamera,
          ),
          const SizedBox(height: 12),
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Back to Home'),
          ),
        ],
      ),
    );
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Result view
  // ──────────────────────────────────────────────────────────────────────────

  Widget _buildResultView() {
    final r = _faceResult!;
    final stableHr = _toDouble(r['stable_heart_rate']);
    final hr = _toDouble(r['heart_rate']);
    final displayHr = stableHr ?? hr;

    final stability = r['stability']?.toString();
    final confidence = _toDouble(r['confidence']) ?? 0.0;
    final reliability = r['reliability']?.toString();
    final warning = r['warning']?.toString();
    final scanQuality = _toDouble(r['scan_quality']);
    final retake = r['retake_required'] == true;
    final retakeReasons = r['retake_reasons'];

    final bool hasHr = displayHr != null && displayHr > 0;
    final bool lowConf = confidence < 0.4;
    final bool hasWarnings =
        lowConf || retake || (warning != null && warning.isNotEmpty);

    final Widget content = Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // ── Vita Score badge ─────────────────────────────────────────────
        if (_vitaScore != null && _vitaScore! > 0 && _riskLevel != 'error') ...[
          Center(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
              decoration: BoxDecoration(
                color: _scoreColor(_vitaScore!).withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                'Vita Score: $_vitaScore%'
                '${_riskLevel != null && _riskLevel != "unknown" && _riskLevel != "unreliable" ? "  \u2022  ${_riskLevel!.toUpperCase()}" : ""}',
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                  color: _scoreColor(_vitaScore!),
                ),
              ),
            ),
          ),
          const SizedBox(height: 16),
        ] else ...[
          Center(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.grey.shade200,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                'Vita Score unavailable — retake for a complete result',
                style: TextStyle(
                  fontSize: 13,
                  color: Colors.grey.shade600,
                ),
              ),
            ),
          ),
          const SizedBox(height: 16),
        ],

        // ── Prominent HR value ───────────────────────────────────────────
        Center(
          child: Column(children: [
            Text(
              hasHr ? displayHr.toStringAsFixed(0) : '—',
              style: TextStyle(
                fontSize: 80,
                fontWeight: FontWeight.w700,
                color: hasHr ? const Color(0xFF1A2340) : Colors.grey,
                height: 1.0,
              ),
            ),
            const SizedBox(height: 2),
            Text('bpm',
                style: TextStyle(
                    fontSize: 17,
                    color: Colors.grey.shade600,
                    fontWeight: FontWeight.w500)),
            const SizedBox(height: 4),
            Text(
              hasHr ? 'Heart Rate' : 'Heart Rate — unable to estimate',
              style: TextStyle(fontSize: 12, color: Colors.grey.shade500),
            ),
          ]),
        ),
        const SizedBox(height: 22),
        const Divider(height: 1),
        const SizedBox(height: 14),

        // ── Metrics ──────────────────────────────────────────────────────
        _metricRow(Icons.show_chart, 'Confidence',
            '${(confidence * 100).toStringAsFixed(1)}%',
            _confidenceColor(confidence)),
        if (reliability != null)
          _metricRow(Icons.verified_outlined, 'Reliability',
              reliability.toUpperCase(), _reliabilityColor(reliability)),
        if (scanQuality != null)
          _metricRow(Icons.high_quality_outlined, 'Scan Quality',
              '${(scanQuality * 100).toStringAsFixed(0)}%',
              _qualityColor(scanQuality)),
        if (stability != null)
          _metricRow(Icons.timeline_outlined, 'Stability',
              stability.toUpperCase(), _stabilityColor(stability)),

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
                      'Low confidence — weak signal detected'),
                if (retake)
                  _warningLine(
                      Icons.refresh,
                      Colors.orange.shade800,
                      retakeReasons is List && retakeReasons.isNotEmpty
                          ? 'Retake recommended: ${retakeReasons.join(", ")}'
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
              label: const Text('Scan Again'),
              onPressed: _scanAgain,
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

    final Widget card = Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(26)),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(24, 24, 24, 16),
        child: content,
      ),
    );

    if (_isDesktop) {
      return Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 560),
          child: SingleChildScrollView(child: card),
        ),
      );
    }
    return SingleChildScrollView(child: card);
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Small reusable widgets
  // ──────────────────────────────────────────────────────────────────────────

  Widget _metricRow(IconData icon, String label, String value, Color color) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(children: [
        Icon(icon, size: 17, color: color),
        const SizedBox(width: 10),
        Text(label,
            style:
                const TextStyle(fontSize: 14, color: Color(0xFF444A5A))),
        const Spacer(),
        Text(value,
            style: TextStyle(
                fontSize: 14, fontWeight: FontWeight.w600, color: color)),
      ]),
    );
  }

  Widget _warningLine(IconData icon, Color color, String text) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Icon(icon, color: color, size: 17),
        const SizedBox(width: 8),
        Expanded(
            child: Text(text, style: TextStyle(fontSize: 13, color: color))),
      ]),
    );
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Colour helpers
  // ──────────────────────────────────────────────────────────────────────────

  Color _scoreColor(int score) {
    if (score >= 70) return Colors.green.shade700;
    if (score >= 40) return Colors.orange.shade700;
    return Colors.red.shade600;
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

  Color _qualityColor(double v) {
    if (v >= 0.6) return Colors.green.shade700;
    if (v >= 0.35) return Colors.orange.shade700;
    return Colors.red.shade600;
  }

  Color _stabilityColor(String? s) {
    if (s == null) return Colors.grey;
    switch (s.toLowerCase()) {
      case 'stable':
      case 'good':
        return Colors.green.shade700;
      case 'moderate':
        return Colors.orange.shade700;
      case 'unstable':
      case 'poor':
        return Colors.red.shade600;
      default:
        return Colors.grey.shade600;
    }
  }
}
