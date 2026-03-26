// Web-only face scan implementation using MediaDevices + MediaRecorder.
// ignore_for_file: avoid_web_libraries_in_flutter, deprecated_member_use

import 'dart:async';
import 'dart:convert';
import 'dart:html' as html;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../services/api_service.dart';

class FaceScanWebScreen extends StatefulWidget {
  final int userId;
  final String userName;
  const FaceScanWebScreen({super.key, required this.userId, required this.userName});

  @override
  State<FaceScanWebScreen> createState() => _FaceScanWebScreenState();
}

class _FaceScanWebScreenState extends State<FaceScanWebScreen> {
  static const int _recordSeconds = 30;
  static const int _maxProcessSeconds = 300;

  html.MediaStream? _stream;
  html.MediaRecorder? _recorder;
  final List<html.Blob> _chunks = [];
  html.VideoElement? _video;
  String? _viewId;

  Timer? _timer;
  int _timeLeft = _recordSeconds;
  bool _recording = false;
  bool _uploading = false;
  String _status = 'Ready to start face scan';
  String? _error;
  Map<String, dynamic>? _result;

  @override
  void initState() {
    super.initState();
    _setupPreview();
  }

  Future<void> _setupPreview() async {
    try {
      final stream = await html.window.navigator.mediaDevices?.getUserMedia({
        'video': {'facingMode': 'user'},
        'audio': false,
      });
      _stream = stream;
      _video = html.VideoElement()
        ..autoplay = true
        ..muted = true
        ..srcObject = stream
        ..style.objectFit = 'cover';
      final viewId = 'face-scan-web-${DateTime.now().millisecondsSinceEpoch}';
      // Register view for HtmlElementView
      // ignore: undefined_prefixed_name
      ui.platformViewRegistry.registerViewFactory(viewId, (int _) => _video!);
      if (mounted) {
        setState(() {
          _viewId = viewId;
          _status = 'Camera ready';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _error = 'Camera access denied or unavailable: $e');
      }
    }
  }

  Future<void> _startRecording() async {
    if (_recording || _uploading || _stream == null) return;
    setState(() {
      _error = null;
      _result = null;
      _status = 'Recording...';
      _timeLeft = _recordSeconds;
    });
    _chunks.clear();
    try {
      _recorder = html.MediaRecorder(_stream!, {'mimeType': 'video/webm'});
      _recorder!.addEventListener('dataavailable', (event) {
        final blobEvent = event as html.BlobEvent;
        if (blobEvent.data != null) {
          _chunks.add(blobEvent.data!);
        }
      });
      _recorder!.addEventListener('stop', (_) => _onRecorderStop());
      _recorder!.start();
      _recording = true;
      _timer?.cancel();
      _timer = Timer.periodic(const Duration(seconds: 1), (t) {
        if (!mounted) return;
        if (_timeLeft <= 1) {
          _stopRecording();
          return;
        }
        setState(() => _timeLeft--);
      });
      setState(() {});
    } catch (e) {
      setState(() => _error = 'Recording failed: $e');
    }
  }

  void _stopRecording() {
    if (!_recording) return;
    _recording = false;
    _timer?.cancel();
    try {
      _recorder?.stop();
    } catch (e) {
      setState(() => _error = 'Could not stop recording: $e');
    }
  }

  Future<void> _onRecorderStop() async {
    if (!mounted) return;
    setState(() {
      _status = 'Preparing upload...';
      _uploading = true;
    });
    try {
      final blob = html.Blob(_chunks, 'video/webm');
      final reader = html.FileReader();
      final completer = Completer<Uint8List>();
      reader.onLoadEnd.listen((_) {
        completer.complete(reader.result as Uint8List);
      });
      reader.onError.listen((_) {
        completer.completeError('Failed to read recorded video');
      });
      reader.readAsArrayBuffer(blob);
      final bytes = await completer.future;

      setState(() => _status = 'Uploading (this may take a while)...');
      final resp = await ApiService.predictVideo(bytes, 'face_scan.webm')
          .timeout(const Duration(seconds: _maxProcessSeconds));

      setState(() {
        _status = 'Analyzing with AI...';
        _result = resp;
        _uploading = false;
      });
    } on TimeoutException {
      setState(() {
        _error = 'Processing timed out. Please retry.';
        _uploading = false;
      });
    } catch (e) {
      setState(() {
        _error = 'Upload/analysis failed: $e';
        _uploading = false;
      });
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    _recorder?.stop();
    _stream?.getTracks().forEach((t) => t.stop());
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Face Scan (Web)'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            if (_viewId != null)
              SizedBox(
                height: 260,
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: HtmlElementView(viewType: _viewId!),
                ),
              )
            else
              Container(
                height: 260,
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  color: Colors.black12,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Text('Camera preview unavailable'),
              ),
            const SizedBox(height: 16),
            Text(
              _status,
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
            ),
            if (_recording) ...[
              const SizedBox(height: 8),
              Text('Time left: $_timeLeft s'),
            ],
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _uploading ? null : _startRecording,
                    icon: Icon(_recording ? Icons.videocam_off : Icons.videocam),
                    label: Text(_recording ? 'Recording...' : 'Start Face Scan'),
                  ),
                ),
                const SizedBox(width: 12),
                if (_recording)
                  ElevatedButton(
                    onPressed: _stopRecording,
                    child: const Text('Stop'),
                  ),
              ],
            ),
            const SizedBox(height: 16),
            if (_uploading) const LinearProgressIndicator(),
            if (_error != null) ...[
              const SizedBox(height: 12),
              Text(_error!, style: const TextStyle(color: Colors.red)),
            ],
            if (_result != null) ...[
              const SizedBox(height: 20),
              const Text('Result', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
              const SizedBox(height: 8),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.grey.shade100,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(_prettyJson(_result!), style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
              ),
            ],
          ],
        ),
      ),
    );
  }

  String _prettyJson(Map<String, dynamic> data) {
    try {
      return const JsonEncoder.withIndent('  ').convert(data);
    } catch (_) {
      return data.toString();
    }
  }
}
