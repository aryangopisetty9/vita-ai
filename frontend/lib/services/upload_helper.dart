// Platform dispatcher for multipart file uploads.
//
// Flutter web  → upload_helper_web.dart  (dart:html FormData + XHR)
// Native / Desktop → upload_helper_io.dart   (http.MultipartRequest)
//
// Both files expose the same function signature:
//   Future<Map<String, dynamic>> postFileMultipart(
//       String url, List<int> bytes, String fieldName, String fileName,
//       {String mimeType})
export 'upload_helper_io.dart'
    if (dart.library.js_interop) 'upload_helper_web.dart';
