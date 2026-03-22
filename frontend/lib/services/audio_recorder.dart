// Platform dispatcher for audio recording.
//
// Web  → audio_recorder_web.dart  (browser MediaRecorder API)
// Native → audio_recorder_io.dart  (FlutterSoundRecorder)
export 'audio_recorder_io.dart'
    if (dart.library.js_interop) 'audio_recorder_web.dart';
