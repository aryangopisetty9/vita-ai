// Scan History Screen
//
// Root-cause fix: guest users (userId == 0) never save scans to the backend,
// so the history screen would always return empty while the dashboard showed
// in-memory HealthData.history entries.  This screen now:
//   1. For userId > 0 â†’ loads from backend API (newest first).
//   2. For userId == 0 (guest) â†’ falls back to HealthData.history (in-memory).
//   3. If backend returns empty AND in-memory history has entries â†’ shows both
//      sources merged so the current session is never invisible.
//   4. Supports per-scan deletion with confirmation dialog + snackbar.
import 'dart:convert';
import 'package:flutter/material.dart';
import '../models/health_data.dart';
import '../services/api_service.dart';

// â”€â”€ Unified scan item â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Both backend records and in-memory HealthData entries are normalised into
// this structure so the UI only has one code-path to render.
class _ScanItem {
  final int? id;          // null for in-memory / guest scans
  final String scanType;  // face | audio | breathing | symptom | final | session
  final int? vitaScore;
  final String? riskLevel;
  final DateTime? createdAt;
  final Map<String, dynamic>? resultJson;
  final bool isBackendRecord;

  const _ScanItem({
    this.id,
    required this.scanType,
    this.vitaScore,
    this.riskLevel,
    this.createdAt,
    this.resultJson,
    required this.isBackendRecord,
  });
}

class ScanHistoryScreen extends StatefulWidget {
  final int userId;
  const ScanHistoryScreen({super.key, required this.userId});

  @override
  State<ScanHistoryScreen> createState() => _ScanHistoryScreenState();
}

class _ScanHistoryScreenState extends State<ScanHistoryScreen> {
  List<_ScanItem> _items = [];
  bool _loading = true;
  bool _isResultDialogOpen = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  // â”€â”€ Data loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    // Guest path: no backend data, use in-memory session history.
    if (widget.userId == 0) {
      setState(() {
        _items = _buildFromMemory();
        _loading = false;
      });
      return;
    }

    // Logged-in path: fetch from backend.
    try {
      final raw = await ApiService.getScanHistory(widget.userId);
      final backendItems = raw
          .map((e) => _buildFromBackend(e as Map<String, dynamic>))
          .whereType<_ScanItem>()
          .toList();

      // Merge in-memory session entries that do NOT already appear in the
      // backend list (avoids duplicates when the scan was freshly saved).
      final memoryItems = _buildFromMemory()
          .where((m) => !backendItems.any((b) =>
              b.scanType == m.scanType &&
              b.createdAt != null &&
              m.createdAt != null &&
              b.createdAt!.difference(m.createdAt!).inMinutes.abs() < 2))
          .toList();

      final merged = [...backendItems, ...memoryItems];
      merged.sort((a, b) {
        final ta = a.createdAt ?? DateTime(0);
        final tb = b.createdAt ?? DateTime(0);
        return tb.compareTo(ta); // newest first
      });

      if (mounted) setState(() { _items = merged; _loading = false; });
    } catch (e) {
      // API failed: fall back to in-memory so data is never invisible.
      final fallback = _buildFromMemory();
      if (mounted) {
        setState(() {
          _items = fallback;
          _loading = false;
          // Only show the error banner when there is NO fallback data to display.
          if (fallback.isEmpty) _error = 'Could not load scan history: $e';
        });
      }
    }
  }

  // Converts a backend JSON record into a _ScanItem.
  _ScanItem? _buildFromBackend(Map<String, dynamic> record) {
    try {
      final id = record['id'] as int?;
      final scanType = record['scan_type']?.toString() ?? 'unknown';
      final vitaScore = record['vita_score'] as int?;
      final riskLevel = record['risk_level']?.toString();
      final createdAtStr = record['created_at']?.toString() ?? '';
      DateTime? createdAt;
      try { createdAt = DateTime.parse(createdAtStr); } catch (_) {}

      Map<String, dynamic>? resultJson;
      final rawJson = record['result_json'];
      if (rawJson is String && rawJson.isNotEmpty) {
        try { resultJson = jsonDecode(rawJson) as Map<String, dynamic>; } catch (_) {}
      } else if (rawJson is Map<String, dynamic>) {
        resultJson = rawJson;
      }

      return _ScanItem(
        id: id,
        scanType: scanType,
        vitaScore: vitaScore,
        riskLevel: riskLevel,
        createdAt: createdAt,
        resultJson: resultJson,
        isBackendRecord: true,
      );
    } catch (_) {
      return null;
    }
  }

  // Converts HealthData.history entries into _ScanItems (session/guest).
  List<_ScanItem> _buildFromMemory() {
    final now = DateTime.now();
    return HealthData.history.asMap().entries.map((entry) {
      final h = entry.value;
      final module = h['module'] ?? '';
      final scoreStr = h['score'] ?? '';
      int? vitaScore;
      if (scoreStr.endsWith('%')) {
        vitaScore = int.tryParse(scoreStr.replaceAll('%', ''));
      }
      // Reconstruct an approximate timestamp from the date/time fields.
      DateTime? dt;
      final dateStr = h['date'] ?? '';
      final timeStr = h['time'] ?? '';
      if (dateStr.isNotEmpty && timeStr.isNotEmpty) {
        try {
          final parts = dateStr.split('-');
          final timeParts = timeStr.split(':');
          if (parts.length == 3 && timeParts.length == 2) {
            dt = DateTime(
              int.parse(parts[2]),
              int.parse(parts[1]),
              int.parse(parts[0]),
              int.parse(timeParts[0]),
              int.parse(timeParts[1]),
            );
          }
        } catch (_) {}
      }
      // Offset by index so same-second entries don't collide.
      dt ??= now.subtract(Duration(minutes: entry.key));

      Map<String, dynamic>? resultJson;
      final rawJson = h['result_json'] ?? '';
      if (rawJson.isNotEmpty) {
        try {
          resultJson = jsonDecode(rawJson) as Map<String, dynamic>;
        } catch (_) {}
      }

      final scanType = module.isEmpty ? 'session' : module;
      return _ScanItem(
        id: null,
        scanType: scanType,
        vitaScore: vitaScore,
        riskLevel: h['risk'],
        createdAt: dt,
        resultJson: resultJson,
        isBackendRecord: false,
      );
    }).toList();
  }

  // â”€â”€ Delete â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

  Future<void> _confirmDelete(_ScanItem item, int index) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Delete this scan?'),
        content: Text(
          'This will permanently remove the ${_scanLabel(item.scanType)} '
          'recorded on ${_formatDate(item.createdAt)}.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Cancel'),
          ),
          TextButton(
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;
    await _doDelete(item, index);
  }

  Future<void> _doDelete(_ScanItem item, int index) async {
    // Optimistic removal from UI.
    setState(() { _items.removeAt(index); });

    // Also remove from in-memory history (dashboard will pick this up).
    if (!item.isBackendRecord) {
      // Match by scan type + approximate index in HealthData.history.
      final histIdx = HealthData.history.indexWhere((h) =>
          (h['module'] ?? '') == item.scanType &&
          (h['score'] ?? '') ==
              (item.vitaScore != null ? '${item.vitaScore}%' : '—'));
      if (histIdx >= 0) HealthData.history.removeAt(histIdx);
    }

    // If history is now empty, clear the cached Vita score so the dashboard
    // does not show a stale score that no longer has any backing scan.
    if (_items.isEmpty) {
      HealthData.score = null;
      HealthData.riskLevel = 'unknown';
      HealthData.faceResult = null;
      HealthData.audioResult = null;
      HealthData.symptomResult = null;
    } else if (item.scanType == 'final') {
      // A combined-score record was deleted — reset so it gets recomputed.
      HealthData.score = null;
      HealthData.riskLevel = 'unknown';
    }

    if (item.isBackendRecord && item.id != null && widget.userId > 0) {
      try {
        await ApiService.deleteScan(widget.userId, item.id!);
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Scan deleted.')),
          );
        }
      } catch (e) {
        // Restore the item in UI if backend delete failed.
        if (mounted) {
          setState(() { _items.insert(index, item); });
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Delete failed: $e'),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    } else {
      // In-memory delete (guest or session item) â€” nothing to call on backend.
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Scan removed from history.')),
        );
      }
    }
  }

  // â”€â”€ Build â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      appBar: AppBar(
        title: const Text('Scan History'),
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            tooltip: 'Refresh',
            onPressed: _load,
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null && _items.isEmpty
              ? _buildError()
              : _items.isEmpty
                  ? _buildEmptyState()
                  : RefreshIndicator(
                      onRefresh: _load,
                      child: ListView.builder(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 12, vertical: 10),
                        itemCount: _items.length,
                        itemBuilder: (ctx, i) => _buildCard(_items[i], i),
                      ),
                    ),
    );
  }

  Widget _buildError() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.cloud_off, size: 52, color: Colors.grey),
          const SizedBox(height: 12),
          Text(_error!, textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.red)),
          const SizedBox(height: 16),
          ElevatedButton.icon(
            icon: const Icon(Icons.refresh),
            label: const Text('Retry'),
            onPressed: _load,
          ),
        ],
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.history, size: 72, color: Colors.grey.shade300),
          const SizedBox(height: 16),
          const Text('No scans yet.',
              style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                  color: Colors.black54)),
          const SizedBox(height: 8),
          const Text(
            'Complete a face, breathing, or symptom scan\nto see your history here.',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.grey),
          ),
        ],
      ),
    );
  }

  Widget _buildCard(_ScanItem item, int index) {
    final riskColor = _riskColor(item.riskLevel);
    final typeLabel = _scanLabel(item.scanType);
    final dateLabel = _formatDate(item.createdAt);
    final timeLabel = _formatTime(item.createdAt);

    return Card(
      margin: const EdgeInsets.only(bottom: 10),
      shape:
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      elevation: 2,
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: () => _openResultDialog(item),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(12, 10, 8, 10),
          child: Row(
            children: [
              CircleAvatar(
                backgroundColor: riskColor.withValues(alpha: 0.15),
                child: Icon(_iconFor(item.scanType), color: riskColor, size: 22),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      typeLabel,
                      style: const TextStyle(
                          fontWeight: FontWeight.w700, fontSize: 15),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      '$dateLabel  $timeLabel'
                      '${item.isBackendRecord ? '' : '  •  session'}',
                      style:
                          const TextStyle(fontSize: 12, color: Colors.black54),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      if (item.vitaScore != null)
                        Padding(
                          padding: const EdgeInsets.only(right: 4),
                          child: Text(
                            '${item.vitaScore}%',
                            style: TextStyle(
                                fontWeight: FontWeight.bold, color: riskColor),
                          ),
                        )
                      else
                        const Padding(
                          padding: EdgeInsets.only(right: 4),
                          child: Text('—',
                              style:
                                  TextStyle(color: Colors.black38, fontSize: 13)),
                        ),
                      if (item.riskLevel != null && item.riskLevel!.isNotEmpty)
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 7, vertical: 3),
                          decoration: BoxDecoration(
                            color: riskColor.withValues(alpha: 0.12),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Text(
                            item.riskLevel!.toUpperCase(),
                            style: TextStyle(
                                fontSize: 10,
                                fontWeight: FontWeight.bold,
                                color: riskColor),
                          ),
                        ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  IconButton(
                    tooltip: 'Delete scan',
                    icon: const Icon(Icons.delete_outline, size: 20),
                    color: Colors.red.shade400,
                    splashRadius: 18,
                    constraints: const BoxConstraints(minWidth: 30, minHeight: 30),
                    onPressed: () => _confirmDelete(item, index),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _openResultDialog(_ScanItem item) async {
    if (_isResultDialogOpen || !mounted) return;
    _isResultDialogOpen = true;
    try {
      await showDialog<void>(
        context: context,
        barrierDismissible: true,
        builder: (ctx) {
          final payload = _resultPayload(item);
          final headerRisk = _effectiveRisk(item, payload);
          final headerColor = _riskColor(headerRisk);
          return Dialog(
            backgroundColor: Colors.transparent,
            insetPadding:
                const EdgeInsets.symmetric(horizontal: 16, vertical: 24),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 640),
              child: Stack(
                children: [
                  Container(
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(18),
                    ),
                    child: SingleChildScrollView(
                      padding: const EdgeInsets.fromLTRB(16, 16, 16, 16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              CircleAvatar(
                                backgroundColor: headerColor.withValues(alpha: 0.15),
                                child: Icon(_iconFor(item.scanType),
                                    color: headerColor, size: 22),
                              ),
                              const SizedBox(width: 10),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      _scanLabel(item.scanType),
                                      style: const TextStyle(
                                        fontSize: 16,
                                        fontWeight: FontWeight.w700,
                                      ),
                                    ),
                                    const SizedBox(height: 2),
                                    Text(
                                      '${_formatDate(item.createdAt)}  ${_formatTime(item.createdAt)}',
                                      style: const TextStyle(
                                          fontSize: 12,
                                          color: Colors.black54),
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 12),
                          _buildFullResultContent(item, payload),
                          if (!item.isBackendRecord) ...[
                            const SizedBox(height: 14),
                            _detailRow('Source', 'Session (not yet synced)',
                                muted: true),
                          ],
                        ],
                      ),
                    ),
                  ),
                  Positioned(
                    right: 6,
                    top: 6,
                    child: IconButton(
                      tooltip: 'Close',
                      icon: const Icon(Icons.close),
                      onPressed: () => Navigator.of(ctx).pop(),
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      );
    } finally {
      _isResultDialogOpen = false;
    }
  }

  Widget _buildFullResultContent(_ScanItem item, Map<String, dynamic> payload) {
    final type = _normalizedType(item.scanType);
    if (type == 'face') return _buildFaceResultContent(item, payload);
    if (type == 'breathing') return _buildBreathingResultContent(item, payload);
    if (type == 'symptom') return _buildSymptomResultContent(item, payload);

    final details = _buildGenericDetails(payload);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (item.vitaScore == null)
          _detailRow('Vita Score', 'Not available for this scan', muted: true),
        ...details.map((e) => _detailRow(e.key, e.value)),
      ],
    );
  }

  Widget _buildFaceResultContent(_ScanItem item, Map<String, dynamic> p) {
    final hr = _toDouble(p['heart_rate']);
    final displayHr = hr;
    final resultAvailable = p['result_available'] == true;
    final tier = (p['hr_result_tier']?.toString().toLowerCase() ?? '').trim();
    final estimatedWeak = p['estimated_from_weak_signal'] == true;
    final confidence = _toDouble(p['confidence']) ?? 0.0;
    final reliability = p['reliability']?.toString();
    final scanQuality = _toDouble(p['scan_quality']);
    final stability = p['stability']?.toString();
    final warning = p['warning']?.toString();
    final message = p['message']?.toString() ?? '';
    final retake = p['retake_required'] == true;
    final retakeReasons = p['retake_reasons'];
    final risk = _effectiveRisk(item, p);
    final score = _effectiveVitaScore(item, p);

    final isReliable = (reliability ?? '').toLowerCase() != 'unreliable';
    final fallbackValid =
      displayHr != null && displayHr > 0 && confidence >= 0.2 && isReliable;
    final isStrong = tier == 'strong_accept';
    final isWeak = tier == 'weak_accept' || estimatedWeak;
    final isReject = tier == 'reject' || tier == 'result_unavailable';
    final legacyAvailable = isStrong || isWeak || (tier.isEmpty && fallbackValid);
    final isValidHr =
      displayHr != null && displayHr > 0 && ((p.containsKey('result_available') ? resultAvailable : legacyAvailable)) && !isReject;
    debugPrint(
      'HR DEBUG => value=$displayHr confidence=$confidence reliability=$reliability tier=$tier weak=$isWeak isValid=$isValidHr',
    );

    final hasHr = isValidHr;
    final lowConf = confidence < 0.4 || isWeak;
    final hasWarnings =
      !isValidHr || lowConf || retake || (warning != null && warning.isNotEmpty);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _scoreBadge(score, risk),
        const SizedBox(height: 14),
        Center(
          child: Column(children: [
            Text(
              hasHr ? displayHr.toStringAsFixed(0) : '—',
              style: TextStyle(
                fontSize: 78,
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
              hasHr
                  ? 'Heart Rate'
                  : '-- bpm • No reliable pulse detected. Try again with better lighting and less movement.',
              style: TextStyle(fontSize: 12, color: Colors.grey.shade500),
              textAlign: TextAlign.center,
            ),
          ]),
        ),
        const SizedBox(height: 18),
        const Divider(height: 1),
        const SizedBox(height: 12),
        _resultMetricRow(
          Icons.show_chart,
          'Confidence',
          '${(confidence * 100).toStringAsFixed(1)}%',
          _confidenceColor(confidence),
        ),
        _resultMetricRow(
          Icons.flag_outlined,
          'Result Tier',
          isValidHr
            ? (isWeak ? 'RESULT AVAILABLE (ESTIMATED)' : 'RESULT AVAILABLE')
            : 'RESULT UNAVAILABLE',
          isStrong
              ? Colors.green.shade700
              : isWeak
                  ? Colors.orange.shade700
                  : Colors.red.shade600,
        ),
        if (reliability != null)
          _resultMetricRow(
            Icons.verified_outlined,
            'Reliability',
            reliability.toUpperCase(),
            _reliabilityColor(reliability),
          ),
        if (scanQuality != null)
          _resultMetricRow(
            Icons.high_quality_outlined,
            'Scan Quality',
            '${(scanQuality * 100).toStringAsFixed(0)}%',
            _qualityColor(scanQuality),
          ),
        if (stability != null)
          _resultMetricRow(
            Icons.timeline_outlined,
            'Stability',
            stability.toUpperCase(),
            _stabilityColor(stability),
          ),
        if (message.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Text(message,
                style: const TextStyle(fontSize: 13, color: Color(0xFF6B7280))),
          ),
        if (hasWarnings) ...[
          const SizedBox(height: 16),
          _warningBox([
            if (!isValidHr)
              _warningRow(
                Icons.error_outline,
                Colors.red.shade700,
                'No reliable pulse detected',
              ),
            if (!isValidHr)
              _warningRow(
                Icons.light_mode_outlined,
                Colors.orange.shade800,
                'Try again with better lighting and less movement',
              ),
            if (lowConf)
              _warningRow(
                Icons.info_outline,
                Colors.amber.shade800,
                isWeak
                    ? 'Estimated from weak signal'
                    : 'Low confidence — weak signal detected',
              ),
            if (isWeak)
              _warningRow(
                Icons.autorenew,
                Colors.orange.shade800,
                'Retake recommended',
              ),
            if (retake)
              _warningRow(
                Icons.refresh,
                Colors.orange.shade800,
                retakeReasons is List && retakeReasons.isNotEmpty
                    ? 'Retake recommended: ${retakeReasons.join(', ')}'
                    : 'Retake recommended for better accuracy',
              ),
            if (!retake && warning != null && warning.isNotEmpty)
              _warningRow(
                Icons.warning_amber_outlined,
                Colors.orange.shade800,
                warning,
              ),
          ]),
        ],
        _buildAdditionalDetails(p, const {
          'heart_rate',
          'hr_result_tier',
          'estimated_from_weak_signal',
          'confidence',
          'reliability',
          'scan_quality',
          'stability',
          'warning',
          'retake_required',
          'retake_reasons',
          'message',
          'risk',
          'overall_risk',
          'vita_health_score',
        }),
      ],
    );
  }

  Widget _buildBreathingResultContent(_ScanItem item, Map<String, dynamic> p) {
    final br = _toDouble(p['breathing_rate']);
    final confidence = _toDouble(p['confidence']) ?? 0.0;
    final reliability = p['reliability']?.toString().toLowerCase();
    final warning = p['warning']?.toString();
    final message = p['message']?.toString() ?? '';
    final retake = p['retake_required'] == true;
    final retakeReasons = p['retake_reasons'];
    final brNormalLow = _toDouble(p['breathing_rate_normal_low']);
    final brNormalHigh = _toDouble(p['breathing_rate_normal_high']);
    final risk = _effectiveRisk(item, p);
    final score = _effectiveVitaScore(item, p);

    final hasBr = br != null && br > 0;
    final lowConf = confidence < 0.33;
    final hasWarnings = lowConf || retake || (warning != null && warning.isNotEmpty);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _scoreBadge(score, risk),
        const SizedBox(height: 14),
        Center(
          child: Column(children: [
            Text(
              hasBr ? br.toStringAsFixed(0) : '—',
              style: TextStyle(
                fontSize: 78,
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
              style: TextStyle(fontSize: 12, color: Colors.grey.shade500),
            ),
          ]),
        ),
        const SizedBox(height: 18),
        const Divider(height: 1),
        const SizedBox(height: 12),
        _resultMetricRow(Icons.air, 'Assessment', _breathingLabel(risk), _voiceRiskColor(risk)),
        _resultMetricRow(
          Icons.show_chart,
          'Confidence',
          '${(confidence * 100).toStringAsFixed(1)}%',
          _confidenceColor(confidence),
        ),
        if (reliability != null)
          _resultMetricRow(
            Icons.verified_outlined,
            'Reliability',
            reliability.toUpperCase(),
            _reliabilityColor(reliability),
          ),
        if (message.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Text(message,
                style: const TextStyle(fontSize: 13, color: Color(0xFF6B7280))),
          ),
        if (hasWarnings) ...[
          const SizedBox(height: 16),
          _warningBox([
            if (lowConf)
              _warningRow(
                Icons.info_outline,
                Colors.amber.shade800,
                'Low confidence — weak breathing signal detected',
              ),
            if (retake)
              _warningRow(
                Icons.refresh,
                Colors.orange.shade800,
                retakeReasons is List && retakeReasons.isNotEmpty
                    ? retakeReasons.first.toString()
                    : 'Retake recommended for better accuracy',
              ),
            if (!retake && warning != null && warning.isNotEmpty)
              _warningRow(
                Icons.warning_amber_outlined,
                Colors.orange.shade800,
                warning,
              ),
          ]),
        ],
        _buildAdditionalDetails(p, const {
          'breathing_rate',
          'breathing_rate_normal_low',
          'breathing_rate_normal_high',
          'confidence',
          'reliability',
          'warning',
          'retake_required',
          'retake_reasons',
          'message',
          'risk',
          'overall_risk',
          'vita_health_score',
        }),
      ],
    );
  }

  Widget _buildSymptomResultContent(_ScanItem item, Map<String, dynamic> p) {
    final risk = _effectiveRisk(item, p);
    final score = _effectiveVitaScore(item, p);
    final statusPoints = _buildStatusPoints(p, risk);
    final recommendations = _buildRecommendations(p, risk);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _scoreBadge(score, risk),
        const SizedBox(height: 14),
        Card(
          elevation: 0,
          color: Colors.grey.shade50,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(children: [
                  Icon(Icons.monitor_heart, color: _riskColor(risk)),
                  const SizedBox(width: 8),
                  const Text(
                    'Health Status',
                    style: TextStyle(fontSize: 17, fontWeight: FontWeight.bold),
                  ),
                ]),
                const Divider(),
                ...statusPoints.map((p) => Padding(
                      padding: const EdgeInsets.symmetric(vertical: 4),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text('• ', style: TextStyle(fontSize: 16)),
                          Expanded(child: Text(p, style: const TextStyle(fontSize: 15))),
                        ],
                      ),
                    )),
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),
        Card(
          elevation: 0,
          color: Colors.grey.shade50,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(children: const [
                  Icon(Icons.tips_and_updates, color: Colors.amber),
                  SizedBox(width: 8),
                  Text(
                    'Recommendations',
                    style: TextStyle(fontSize: 17, fontWeight: FontWeight.bold),
                  ),
                ]),
                const Divider(),
                ...recommendations.map((r) => Padding(
                      padding: const EdgeInsets.symmetric(vertical: 4),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Icon(Icons.check_circle, size: 18, color: Colors.green),
                          const SizedBox(width: 8),
                          Expanded(child: Text(r, style: const TextStyle(fontSize: 15))),
                        ],
                      ),
                    )),
              ],
            ),
          ),
        ),
        _buildAdditionalDetails(p, const {
          'predicted_condition',
          'severity',
          'risk',
          'confidence',
          'reliability',
          'message',
          'recommendations',
          'overall_risk',
          'vita_health_score',
        }),
      ],
    );
  }

  Widget _scoreBadge(int? score, String risk) {
    if (score != null && score > 0) {
      final color = _scoreColor(score);
      return Center(
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.12),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Text(
            'Vita Score: $score%  •  ${risk.toUpperCase()}',
            style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: color),
          ),
        ),
      );
    }
    return Center(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
        decoration: BoxDecoration(
          color: Colors.grey.shade200,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Text(
          'Vita Score unavailable for this saved scan',
          style: TextStyle(fontSize: 13, color: Colors.grey.shade600),
        ),
      ),
    );
  }

  List<String> _buildStatusPoints(Map<String, dynamic> result, String risk) {
    final points = <String>[];
    final condition = result['predicted_condition']?.toString();
    if (condition != null && condition.isNotEmpty) {
      points.add('Predicted condition: $condition');
    }

    final severity = result['severity']?.toString();
    if (severity != null && severity.isNotEmpty) {
      points.add('Severity: $severity');
    }

    final hr = _toDouble(result['heart_rate']);
    final resultAvailable = result['result_available'] == true;
    final tier = (result['hr_result_tier']?.toString().toLowerCase() ?? '').trim();
    final estimatedWeak = result['estimated_from_weak_signal'] == true;
    final hrConfidence = _toDouble(result['confidence']) ?? 0.0;
    final hrReliability = result['reliability']?.toString().toLowerCase() ?? '';
    final fallbackValid = hr != null && hr > 0 && hrConfidence >= 0.2 && hrReliability != 'unreliable';
    final isStrong = tier == 'strong_accept';
    final isWeak = tier == 'weak_accept' || estimatedWeak;
    final isReject = tier == 'reject' || tier == 'result_unavailable';
    final legacyAvailable = isStrong || isWeak || (tier.isEmpty && fallbackValid);
    final isValidHr = hr != null && hr > 0 && ((result.containsKey('result_available') ? resultAvailable : legacyAvailable)) && !isReject;
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

    final stability = result['stability']?.toString();
    if (stability != null && stability.isNotEmpty) {
      points.add('Stability: $stability');
    }

    final br = result['breathing_rate'];
    if (br != null) points.add('Breathing Rate: $br breaths/min');

    points.add('Risk Level: ${risk.toUpperCase()}');

    final confidence = _toDouble(result['confidence']);
    if (confidence != null) {
      points.add('Confidence: ${(confidence * 100).round()}%');
    }

    final reliability = result['reliability']?.toString();
    if (reliability != null && reliability.isNotEmpty) {
      points.add('Reliability: ${reliability.toUpperCase()}');
    }

    final message = result['message']?.toString();
    if (message != null && message.isNotEmpty) points.add(message);
    return points;
  }

  List<String> _buildRecommendations(Map<String, dynamic> result, String risk) {
    if (result['recommendations'] is List) {
      return (result['recommendations'] as List)
          .map((e) => e.toString())
          .where((e) => e.trim().isNotEmpty)
          .toList();
    }

    final recs = <String>[];
    final r = risk.toLowerCase();
    if (r == 'high' || r == 'very_high') {
      recs.add('Consider consulting a healthcare professional soon.');
      recs.add('Monitor your symptoms closely.');
    } else if (r == 'moderate' || r == 'elevated') {
      recs.add('Keep tracking your symptoms over the next few days.');
      recs.add('Rest and stay hydrated.');
    } else {
      recs.add('Your results look stable. Keep healthy routines.');
    }
    recs.add('Stay hydrated and maintain a balanced diet.');
    recs.add('If symptoms persist, consult a healthcare provider.');
    return recs;
  }

  Widget _buildAdditionalDetails(Map<String, dynamic> p, Set<String> excludedKeys) {
    final rows = _buildGenericDetails(p, excludedKeys: excludedKeys);
    if (rows.isEmpty) return const SizedBox.shrink();

    return Padding(
      padding: const EdgeInsets.only(top: 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Divider(height: 1),
          const SizedBox(height: 10),
          const Text(
            'Additional Details',
            style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 6),
          ...rows.map((e) => _detailRow(e.key, e.value)),
        ],
      ),
    );
  }

  List<MapEntry<String, String>> _buildGenericDetails(
    Map<String, dynamic> payload, {
    Set<String> excludedKeys = const {},
  }) {
    final details = <MapEntry<String, String>>[];

    for (final entry in payload.entries) {
      final key = entry.key.toString();
      if (excludedKeys.contains(key)) continue;
      final value = entry.value;
      if (value == null || value is Map || value is List) continue;

      var text = value.toString().trim();
      if (text.isEmpty) continue;

      if (key.contains('confidence') || key.contains('quality')) {
        text = _fmtPct(value);
      }

      details.add(MapEntry(_titleCase(key.replaceAll('_', ' ')), text));
    }

    return details;
  }

  Widget _resultMetricRow(IconData icon, String label, String value, Color color) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(children: [
        Icon(icon, size: 17, color: color),
        const SizedBox(width: 10),
        Text(label,
            style: const TextStyle(fontSize: 14, color: Color(0xFF444A5A))),
        const Spacer(),
        Text(value,
            style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: color)),
      ]),
    );
  }

  Widget _warningBox(List<Widget> children) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.amber.shade50,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.amber.shade300),
      ),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: children),
    );
  }

  Widget _warningRow(IconData icon, Color color, String text) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Icon(icon, color: color, size: 17),
        const SizedBox(width: 8),
        Expanded(child: Text(text, style: TextStyle(fontSize: 13, color: color))),
      ]),
    );
  }

  Map<String, dynamic> _resultPayload(_ScanItem item) {
    final payload = <String, dynamic>{};
    if (item.resultJson != null) payload.addAll(item.resultJson!);
    if (item.vitaScore != null) payload.putIfAbsent('vita_health_score', () => item.vitaScore);
    if (item.riskLevel != null && item.riskLevel!.isNotEmpty) {
      payload.putIfAbsent('overall_risk', () => item.riskLevel);
    }
    return payload;
  }

  String _effectiveRisk(_ScanItem item, Map<String, dynamic> payload) {
    final risk = payload['overall_risk']?.toString() ??
        payload['risk']?.toString() ??
        item.riskLevel ??
        'unknown';
    return risk.toLowerCase();
  }

  int? _effectiveVitaScore(_ScanItem item, Map<String, dynamic> payload) {
    if (item.vitaScore != null) return item.vitaScore;
    final dynamic raw = payload['vita_health_score'];
    if (raw is int) return raw;
    if (raw is double) return raw.round();
    if (raw is String) return int.tryParse(raw);
    return null;
  }

  String _normalizedType(String type) {
    final t = type.trim().toLowerCase();
    if (t == 'audio' || t == 'voice') return 'breathing';
    return t;
  }

  double? _toDouble(dynamic v) {
    if (v == null) return null;
    if (v is double) return v;
    if (v is int) return v.toDouble();
    if (v is String) return double.tryParse(v);
    return null;
  }

  String _titleCase(String text) {
    return text
        .split(' ')
        .where((w) => w.isNotEmpty)
        .map((w) => '${w[0].toUpperCase()}${w.substring(1)}')
        .join(' ');
  }

  Widget _detailRow(String label, String value,
      {bool muted = false}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label,
              style: TextStyle(
                  fontSize: 13,
                  color: muted ? Colors.black38 : Colors.black54)),
          Flexible(
            child: Text(
              value,
              textAlign: TextAlign.end,
              style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w500,
                  color: muted ? Colors.black38 : Colors.black87),
            ),
          ),
        ],
      ),
    );
  }

  // â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

  String _scanLabel(String type) {
    switch (type.toLowerCase()) {
      case 'face':
        return 'Face Scan';
      case 'audio':
      case 'breathing':
        return 'Breathing Analysis';
      case 'symptom':
        return 'Symptom Check';
      case 'final':
        return 'Combined Score';
      default:
        return 'Health Scan';
    }
  }

  String _formatDate(DateTime? dt) {
    if (dt == null) return '—';
    return '${dt.day.toString().padLeft(2, '0')}-'
        '${dt.month.toString().padLeft(2, '0')}-'
        '${dt.year}';
  }

  String _formatTime(DateTime? dt) {
    if (dt == null) return '';
    return '${dt.hour.toString().padLeft(2, '0')}:'
        '${dt.minute.toString().padLeft(2, '0')}';
  }

  String _fmtPct(dynamic v) {
    double? d;
    if (v is double) d = v;
    if (v is int) d = v.toDouble();
    if (v is String) d = double.tryParse(v);
    if (d != null) {
      // Values already in 0-100 range
      if (d > 1.0) return '${d.toStringAsFixed(1)}%';
      return '${(d * 100).toStringAsFixed(1)}%';
    }
    return v.toString();
  }

  IconData _iconFor(String type) {
    switch (type.toLowerCase()) {
      case 'face':
        return Icons.face;
      case 'audio':
      case 'breathing':
        return Icons.mic;
      case 'symptom':
        return Icons.description;
      case 'final':
        return Icons.assessment;
      default:
        return Icons.health_and_safety;
    }
  }

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
    if (r == null) return Colors.grey.shade600;
    switch (r.toLowerCase()) {
      case 'high':
        return Colors.green.shade700;
      case 'medium':
      case 'moderate':
        return Colors.orange.shade700;
      case 'low':
      case 'unreliable':
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
    if (s == null) return Colors.grey.shade600;
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

  String _breathingLabel(String risk) {
    switch (risk.toLowerCase()) {
      case 'normal':
      case 'low':
        return 'Normal Range';
      case 'elevated':
      case 'moderate':
        return 'Above Resting Range';
      case 'high':
        return 'Exercise Range';
      case 'very_high':
        return 'Very High Rate';
      case 'low_rate':
        return 'Below Normal Rate';
      case 'unreliable':
        return 'No Signal';
      default:
        return risk.toUpperCase();
    }
  }

  Color _voiceRiskColor(String risk) {
    switch (risk.toLowerCase()) {
      case 'normal':
      case 'low':
        return Colors.green.shade700;
      case 'elevated':
      case 'moderate':
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

  Color _riskColor(String? risk) {
    switch (risk?.toLowerCase()) {
      case 'high':
        return Colors.red;
      case 'moderate':
        return Colors.orange;
      case 'low':
        return Colors.green;
      default:
        return Colors.blueGrey;
    }
  }
}
