// Web audio recorder using browser MediaRecorder API.
//
// Uses dart:js_interop + package:web to access:
//   navigator.mediaDevices.getUserMedia({ audio: true })
//   new MediaRecorder(stream, { mimeType: ... })
//
// Records audio and returns the result as raw bytes (Uint8List).

import 'dart:async';
import 'dart:js_interop';
import 'dart:typed_data';

import 'package:web/web.dart' as web;

/// Lightweight wrapper around the browser MediaRecorder API.
class PlatformAudioRecorder {
  web.MediaRecorder? _recorder;
  web.MediaStream? _stream;
  final List<web.Blob> _chunks = [];
  bool _recording = false;
  String _mimeType = 'audio/webm';

  bool get isRecording => _recording;

  /// Request microphone permission and prepare the recorder.
  /// Returns a user-facing error string on failure, or null on success.
  Future<String?> init() async {
    try {
      final constraints = web.MediaStreamConstraints(audio: true.toJS);
      _stream = await web.window.navigator.mediaDevices
          .getUserMedia(constraints)
          .toDart;

      // Pick a supported MIME type.
      for (final mime in ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg']) {
        if (web.MediaRecorder.isTypeSupported(mime)) {
          _mimeType = mime;
          break;
        }
      }
      return null; // success
    } catch (e) {
      final msg = e.toString();
      if (msg.contains('NotAllowedError') || msg.contains('Permission')) {
        return 'Microphone permission denied.';
      }
      if (msg.contains('NotFoundError')) {
        return 'No microphone found on this device.';
      }
      return 'Could not access microphone: $msg';
    }
  }

  /// Start recording.
  void start() {
    if (_stream == null) return;
    _chunks.clear();

    _recorder = web.MediaRecorder(
      _stream!,
      web.MediaRecorderOptions(mimeType: _mimeType),
    );

    _recorder!.ondataavailable = (web.BlobEvent event) {
      if (event.data.size > 0) {
        _chunks.add(event.data);
      }
    }.toJS;

    _recorder!.start(1000); // collect data every 1 s
    _recording = true;
  }

  /// Stop recording and return the audio bytes.
  Future<Uint8List> stop() async {
    final completer = Completer<Uint8List>();

    if (_recorder == null || !_recording) {
      return Uint8List(0);
    }

    _recorder!.onstop = (web.Event _) {
      _recording = false;
      // Merge all chunks into one Blob, then read as bytes.
      final blob = web.Blob(
        _chunks.map((c) => c as JSAny).toList().toJS,
        web.BlobPropertyBag(type: _mimeType),
      );
      final reader = web.FileReader();
      reader.onloadend = (web.Event _) {
        final result = reader.result;
        if (result != null) {
          final arrayBuffer = result as JSArrayBuffer;
          completer.complete(arrayBuffer.toDart.asUint8List());
        } else {
          completer.complete(Uint8List(0));
        }
      }.toJS;
      reader.readAsArrayBuffer(blob);
    }.toJS;

    _recorder!.stop();
    return completer.future;
  }

  /// File extension for the recorded audio.
  String get fileExtension {
    if (_mimeType.contains('ogg')) return 'ogg';
    return 'webm';
  }

  /// Release microphone tracks.
  void dispose() {
    _recording = false;
    try {
      _recorder?.stop();
    } catch (_) {}
    _stream?.getTracks().toDart.forEach((t) => t.stop());
    _stream = null;
    _recorder = null;
  }
}
