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

      final scanType = module.isEmpty ? 'session' : module;
      return _ScanItem(
        id: null,
        scanType: scanType,
        vitaScore: vitaScore,
        riskLevel: h['risk'],
        createdAt: dt,
        resultJson: null,
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
      backgroundColor: const Color(0xFFEDEFF3),
      appBar: AppBar(
        title: const Text('Scan History'),
        backgroundColor: Colors.white,
        foregroundColor: Colors.black87,
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

    // Detail rows from result_json
    final details = <MapEntry<String, String>>[];
    if (item.resultJson != null) {
      final r = item.resultJson!;
      if (r['heart_rate'] != null) {
        details.add(MapEntry('Heart Rate', '${_fmt(r['heart_rate'])} bpm'));
      }
      if (r['heart_rate_confidence'] != null) {
        details.add(MapEntry(
            'HR Confidence', _fmtPct(r['heart_rate_confidence'])));
      }
      if (r['scan_quality'] != null) {
        details.add(MapEntry('Scan Quality', _fmtPct(r['scan_quality'])));
      }
      if (r['confidence'] != null) {
        details.add(MapEntry('Confidence', _fmtPct(r['confidence'])));
      }
      if (r['risk'] != null) {
        details.add(MapEntry('Risk (module)', r['risk'].toString()));
      }
      if (r['message'] != null && r['message'].toString().isNotEmpty) {
        details.add(MapEntry('Message', r['message'].toString()));
      }
    }

    return Card(
      margin: const EdgeInsets.only(bottom: 10),
      shape:
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      elevation: 2,
      child: ExpansionTile(
        leading: CircleAvatar(
          backgroundColor: riskColor.withValues(alpha: 0.15),
          child: Icon(_iconFor(item.scanType), color: riskColor, size: 22),
        ),
        title: Text(
          typeLabel,
          style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 15),
        ),
        subtitle: Text(
          '$dateLabel  $timeLabel'
          '${item.isBackendRecord ? '' : '  â€¢  session'}',
          style: const TextStyle(fontSize: 12, color: Colors.black54),
        ),
        trailing: Row(
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
                child: Text('â€”',
                    style: TextStyle(color: Colors.black38, fontSize: 13)),
              ),
            if (item.riskLevel != null && item.riskLevel!.isNotEmpty)
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
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
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 4, 16, 12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Divider(height: 1),
                const SizedBox(height: 8),
                // Score unavailable note
                if (item.vitaScore == null)
                  _detailRow(
                    'Vita Score',
                    'Not available for this scan',
                    muted: true,
                  ),
                ...details.map(
                    (e) => _detailRow(e.key, e.value)),
                if (!item.isBackendRecord)
                  _detailRow('Source', 'Session (not yet synced)',
                      muted: true),
                const SizedBox(height: 8),
                // Delete button
                Align(
                  alignment: Alignment.centerRight,
                  child: TextButton.icon(
                    style: TextButton.styleFrom(
                        foregroundColor: Colors.red.shade400),
                    icon: const Icon(Icons.delete_outline, size: 18),
                    label: const Text('Delete scan'),
                    onPressed: () => _confirmDelete(item, index),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
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

  String _fmt(dynamic v) {
    if (v is double) return v.toStringAsFixed(1);
    return v.toString();
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
