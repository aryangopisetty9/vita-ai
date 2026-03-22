// Native audio recorder using flutter_sound.
//
// This file is used on iOS, Android, and desktop platforms.
// On web, audio_recorder_web.dart is used instead.

import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_sound/flutter_sound.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';

/// Wrapper around FlutterSoundRecorder for native platforms.
class PlatformAudioRecorder {
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  bool _ready = false;
  bool _recording = false;
  String? _filePath;

  bool get isRecording => _recording;

  /// Request microphone permission and open the recorder.
  /// Returns a user-facing error string on failure, or null on success.
  Future<String?> init() async {
    final status = await Permission.microphone.request();
    if (!status.isGranted) {
      return 'Microphone permission denied.';
    }
    await _recorder.openRecorder();
    _ready = true;
    return null;
  }

  /// Start recording to a temp WAV file.
  void start() {
    if (!_ready) return;
    _recording = true;
    getTemporaryDirectory().then((dir) {
      _filePath =
          '${dir.path}/vita_breathing_${DateTime.now().millisecondsSinceEpoch}.wav';
      _recorder.startRecorder(toFile: _filePath, codec: Codec.pcm16WAV);
    });
  }

  /// Stop recording and return the audio bytes.
  Future<Uint8List> stop() async {
    await _recorder.stopRecorder();
    _recording = false;
    if (_filePath == null) return Uint8List(0);
    final file = File(_filePath!);
    if (!await file.exists()) return Uint8List(0);
    return file.readAsBytes();
  }

  /// File extension for the recorded audio.
  String get fileExtension => 'wav';

  /// Release recorder resources.
  void dispose() {
    _recording = false;
    _recorder.closeRecorder();
  }
}
