import 'package:flutter/foundation.dart' show kIsWeb;

/// App-wide runtime/build-time configuration.
///
/// Set at build time for real devices:
/// flutter run --dart-define=VITA_BASE_URL=http://100.95.34.73:8000
class AppConfig {
  static const String _baseUrlFromDefine =
      String.fromEnvironment('VITA_BASE_URL', defaultValue: '');

  // Safe default for Android phone + backend on same WiFi.
  static const String _defaultLanBaseUrl = 'http://100.95.34.73:8000';

  static String get baseUrl {
    final fromDefine = _baseUrlFromDefine.trim();
    if (fromDefine.isNotEmpty) {
      return fromDefine.endsWith('/')
          ? fromDefine.substring(0, fromDefine.length - 1)
          : fromDefine;
    }

    // For same-origin web deployments, keep relative URLs.
    if (kIsWeb) return '';

    return _defaultLanBaseUrl;
  }
}
