import 'dart:convert';

/// Thrown when a backend HTTP call returns a non-2xx status code.
class ApiException implements Exception {
  final int statusCode;
  final String body;
  ApiException(this.statusCode, this.body);

  @override
  String toString() => 'ApiException($statusCode): $body';

  /// Extract a clean, user-readable message from a FastAPI error response.
  ///
  /// Handles:
  ///   - `{"detail": "some string"}`            → returns the string
  ///   - `{"detail": [{"msg": "..."}]}`         → Pydantic validation list
  ///   - Unknown / non-JSON bodies              → fallback by status code
  static String parseMessage(ApiException e) {
    try {
      final decoded = jsonDecode(e.body);
      if (decoded is Map<String, dynamic>) {
        final detail = decoded['detail'];
        if (detail is String && detail.isNotEmpty) {
          return _humanise(detail);
        }
        if (detail is List && detail.isNotEmpty) {
          final first = detail.first;
          if (first is Map<String, dynamic>) {
            final msg = first['msg'] as String? ?? '';
            final loc = first['loc'];
            final field = (loc is List && loc.isNotEmpty) ? loc.last.toString() : '';
            return _humaniseField(field, msg);
          }
        }
      }
    } catch (_) {}
    return _fallbackByStatus(e.statusCode);
  }

  static String _humanise(String msg) {
    final m = msg.toLowerCase();
    if (m.contains('already registered') || m.contains('already exists')) {
      return 'An account with this email already exists.';
    }
    if (m.contains('invalid email or password') || m.contains('incorrect')) {
      return 'Invalid email or password.';
    }
    if (m.contains('not found')) return 'Account not found.';
    if (m.contains('expired') || m.contains('invalid') && m.contains('token')) {
      return 'Your session has expired. Please sign in again.';
    }
    return msg;
  }

  static String _humaniseField(String field, String msg) {
    final m = msg.toLowerCase();
    if (field == 'password' || (field.isEmpty && m.contains('password'))) {
      if (m.contains('at least') || m.contains('short') || m.contains('min_length')) {
        return 'Password must be at least 6 characters.';
      }
    }
    if (field == 'email') {
      if (m.contains('valid') || m.contains('at least') || m.contains('short')) {
        return 'Please enter a valid email address.';
      }
    }
    if (field == 'name') {
      if (m.contains('at least') || m.contains('short')) {
        return 'Name must not be empty.';
      }
    }
    if (m.contains('at least') && m.contains('character')) {
      return '${_fieldLabel(field)} is too short.';
    }
    return msg.isNotEmpty ? msg : _fallbackByStatus(422);
  }

  static String _fieldLabel(String field) {
    if (field.isEmpty) return 'Field';
    return field[0].toUpperCase() + field.substring(1);
  }

  static String _fallbackByStatus(int status) {
    switch (status) {
      case 400:
        return 'Invalid request. Please check your details.';
      case 401:
        return 'Invalid email or password.';
      case 403:
        return 'Access denied.';
      case 404:
        return 'Account not found.';
      case 409:
        return 'An account with this email already exists.';
      case 422:
        return 'Please check your input and try again.';
      case 500:
        return 'Server error. Please try again later.';
      default:
        return 'Something went wrong. Please try again.';
    }
  }
}
