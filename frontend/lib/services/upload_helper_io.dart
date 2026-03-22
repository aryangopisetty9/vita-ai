// Native/Desktop multipart upload via http.MultipartRequest.
// This file is selected when dart:html is NOT available (iOS, Android, Desktop).
import 'dart:convert';

import 'package:http/http.dart' as http;

import 'api_exception.dart';

Future<Map<String, dynamic>> postFileMultipart(
  String url,
  List<int> bytes,
  String fieldName,
  String fileName, {
  String mimeType = 'application/octet-stream',
}) async {
  final request = http.MultipartRequest('POST', Uri.parse(url));
  request.files.add(
    http.MultipartFile.fromBytes(fieldName, bytes, filename: fileName),
  );
  final streamed =
      await request.send().timeout(const Duration(seconds: 120));
  final body = await streamed.stream.bytesToString();
  if (streamed.statusCode != 200) {
    throw ApiException(streamed.statusCode, body);
  }
  return jsonDecode(body) as Map<String, dynamic>;
}
