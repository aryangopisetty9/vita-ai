import 'dart:convert';
import 'dart:io' show Platform;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:http/http.dart' as http;
import 'api_exception.dart';
import 'upload_helper.dart' as uploader;
export 'api_exception.dart';

/// Central API service that connects Flutter to the Vita AI FastAPI backend.
///
/// The base URL is chosen automatically:
/// - Web (Chrome): http://localhost:8000
/// - Android emulator: http://10.0.2.2:8000
/// - Desktop / other: http://127.0.0.1:8000
class ApiService {
  static String get baseUrl {
    // When served from the backend (same origin), use relative URLs → no CORS.
    if (kIsWeb) return '';
    // Android emulator uses 10.0.2.2 to reach host machine's localhost
    if (Platform.isAndroid) return 'http://10.0.2.2:8000';
    return 'http://127.0.0.1:8000';
  }

  // ── Token storage (in-memory; persists for the app session) ─────────────────

  static String? _token;

  /// Store the JWT access token returned from /auth/login.
  static void setToken(String token) => _token = token;

  /// Retrieve the current JWT token (null if not signed in).
  static String? getToken() => _token;

  /// Clear the stored token (call on logout).
  static void clearToken() => _token = null;

  /// JSON headers, optionally including [Authorization] if a token is stored.
  static Map<String, String> get _jsonHeaders => {
        'Content-Type': 'application/json',
        if (_token != null) 'Authorization': 'Bearer $_token',
      };

  // ── Auth ─────────────────────────────────────────────────────────────

  static Future<Map<String, dynamic>> signup(
      String name, String email, String password) async {
    final resp = await http.post(
      Uri.parse('$baseUrl/auth/signup'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'name': name, 'email': email, 'password': password}),
    );
    if (resp.statusCode != 200 && resp.statusCode != 201) {
      throw ApiException(resp.statusCode, resp.body);
    }
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  static Future<Map<String, dynamic>> login(
      String email, String password) async {
    final resp = await http.post(
      Uri.parse('$baseUrl/auth/login'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'email': email, 'password': password}),
    );
    if (resp.statusCode != 200) throw ApiException(resp.statusCode, resp.body);
    final data = jsonDecode(resp.body) as Map<String, dynamic>;
    // Store JWT for use in subsequent authenticated requests
    final token = data['access_token'] as String?;
    if (token != null) setToken(token);
    return data;
  }

  // ── Health check ─────────────────────────────────────────────────────

  static Future<bool> healthCheck() async {
    try {
      final resp = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 5));
      return resp.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  // ── Face scan ────────────────────────────────────────────────────────

  /// Upload a video file for face analysis.
  /// Uses dart:html FormData on web, http.MultipartRequest on native.
  static Future<Map<String, dynamic>> predictFace(
      List<int> fileBytes, String fileName) async {
    return uploader.postFileMultipart(
      '$baseUrl/predict/face',
      fileBytes,
      'file',
      fileName,
      mimeType: 'video/webm',
    );
  }

  // ── Audio scan ───────────────────────────────────────────────────────

  /// Upload an audio file for breathing/audio analysis.
  /// Uses dart:html FormData on web, http.MultipartRequest on native.
  static Future<Map<String, dynamic>> predictAudio(
      List<int> fileBytes, String fileName) async {
    return uploader.postFileMultipart(
      '$baseUrl/predict/audio',
      fileBytes,
      'file',
      fileName,
      mimeType: 'audio/wav',
    );
  }

  // ── Symptom analysis ─────────────────────────────────────────────────

  /// POST /predict/symptom with structured form payload.
  ///
  /// [payload] should contain at minimum ``major_symptom`` (or ``text``)
  /// plus any structured fields collected from the symptom form:
  /// ``age``, ``gender``, ``minor_symptoms``, ``days_suffering``,
  /// ``symptom_category``, ``fever``, ``pain``, ``difficulty_breathing``,
  /// ``severity``.
  static Future<Map<String, dynamic>> predictSymptom(
      Map<String, dynamic> payload) async {
    final resp = await http.post(
      Uri.parse('$baseUrl/predict/symptom'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(payload),
    );
    if (resp.statusCode != 200) throw ApiException(resp.statusCode, resp.body);
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  // ── Final score ──────────────────────────────────────────────────────

  static Future<Map<String, dynamic>> predictFinalScore({
    Map<String, dynamic>? faceResult,
    Map<String, dynamic>? audioResult,
    Map<String, dynamic>? symptomResult,
  }) async {
    final resp = await http.post(
      Uri.parse('$baseUrl/predict/final-score'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'face_result': faceResult,
        'audio_result': audioResult,
        'symptom_result': symptomResult,
      }),
    );
    if (resp.statusCode != 200) throw ApiException(resp.statusCode, resp.body);
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  // ── Health Data ──────────────────────────────────────────────────────

  static Future<Map<String, dynamic>> saveHealthData(
      int userId, Map<String, dynamic> data) async {
    final resp = await http.post(
      Uri.parse('$baseUrl/user/$userId/health-data'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(data),
    );
    if (resp.statusCode != 200) throw ApiException(resp.statusCode, resp.body);
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  static Future<Map<String, dynamic>?> getHealthData(int userId) async {
    final resp =
        await http.get(Uri.parse('$baseUrl/user/$userId/health-data'));
    if (resp.statusCode == 404) return null;
    if (resp.statusCode != 200) throw ApiException(resp.statusCode, resp.body);
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  // ── Scan History ─────────────────────────────────────────────────────

  /// Fetch all scan history entries for [userId], newest first.
  /// Includes the auth token in headers if one is stored.
  static Future<List<dynamic>> getScanHistory(int userId) async {
    final resp = await http.get(
      Uri.parse('$baseUrl/user/$userId/scans'),
      headers: _jsonHeaders,
    );
    if (resp.statusCode != 200) throw ApiException(resp.statusCode, resp.body);
    return jsonDecode(resp.body) as List<dynamic>;
  }

  /// Delete a single scan result.
  /// DELETE /user/{userId}/scan/{scanId}
  /// Throws [ApiException] on failure (e.g. 404 if scan not found).
  static Future<void> deleteScan(int userId, int scanId) async {
    final resp = await http.delete(
      Uri.parse('$baseUrl/user/$userId/scan/$scanId'),
      headers: _jsonHeaders,
    );
    if (resp.statusCode != 200) throw ApiException(resp.statusCode, resp.body);
  }

  // ── Save Scan Result ─────────────────────────────────────────────────

  /// Persist a scan result to the backend for scan history.
  /// POST /user/{userId}/scan?scan_type={scanType}  body = result JSON
  static Future<Map<String, dynamic>> saveScan(
      int userId, String scanType, Map<String, dynamic> result) async {
    final resp = await http.post(
      Uri.parse('$baseUrl/user/$userId/scan?scan_type=$scanType'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(result),
    );
    if (resp.statusCode != 200) throw ApiException(resp.statusCode, resp.body);
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  // ── Combined Analyze (Code contributed by Manogna) ───────────────────

  /// Run all available analysis modules and return combined Vita Health Score.
  static Future<Map<String, dynamic>> analyze({
    String? symptomText,
    Map<String, dynamic>? faceResult,
    Map<String, dynamic>? audioResult,
  }) async {
    final resp = await http.post(
      Uri.parse('$baseUrl/analyze'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'symptom_text': symptomText,
        'face_result': faceResult,
        'audio_result': audioResult,
      }),
    );
    if (resp.statusCode != 200) throw ApiException(resp.statusCode, resp.body);
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  // ── SOS / Emergency Contacts (SOS feature integrated from Manogna) ───

  static Future<List<dynamic>> getSosContacts(int userId) async {
    final resp =
        await http.get(Uri.parse('$baseUrl/sos/contacts/$userId'));
    if (resp.statusCode != 200) throw ApiException(resp.statusCode, resp.body);
    return jsonDecode(resp.body) as List<dynamic>;
  }

  static Future<Map<String, dynamic>> addSosContact(
      int userId, String name, String phone, String? relationship) async {
    final resp = await http.post(
      Uri.parse('$baseUrl/sos/contacts/$userId'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'name': name,
        'phone': phone,
        'relationship': relationship,
      }),
    );
    if (resp.statusCode != 200) throw ApiException(resp.statusCode, resp.body);
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  static Future<void> deleteSosContact(int userId, int contactId) async {
    final resp = await http
        .delete(Uri.parse('$baseUrl/sos/contacts/$userId/$contactId'));
    if (resp.statusCode != 200) throw ApiException(resp.statusCode, resp.body);
  }

  static Future<Map<String, dynamic>> triggerSos(
      int userId, {String? message, double? latitude, double? longitude}) async {
    final resp = await http.post(
      Uri.parse('$baseUrl/sos/trigger/$userId'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'message': message,
        'latitude': latitude,
        'longitude': longitude,
      }),
    );
    if (resp.statusCode != 200) throw ApiException(resp.statusCode, resp.body);
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }
}
