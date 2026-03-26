import 'package:shared_preferences/shared_preferences.dart';

class SettingsStorageService {
  SettingsStorageService._();

  static const String _guestScope = 'guest';

  static String _scopeForUser(int userId) {
    return userId == 0 ? _guestScope : 'user_$userId';
  }

  static Future<String> getName(int userId, {required String fallback}) async {
    final prefs = await SharedPreferences.getInstance();
    final key = 'settings_name_${_scopeForUser(userId)}';
    return prefs.getString(key) ?? fallback;
  }

  static Future<void> setName(int userId, String value) async {
    final prefs = await SharedPreferences.getInstance();
    final key = 'settings_name_${_scopeForUser(userId)}';
    await prefs.setString(key, value);
  }

  static Future<String> getBio(int userId) async {
    final prefs = await SharedPreferences.getInstance();
    final key = 'settings_bio_${_scopeForUser(userId)}';
    return prefs.getString(key) ?? '';
  }

  static Future<void> setBio(int userId, String value) async {
    final prefs = await SharedPreferences.getInstance();
    final key = 'settings_bio_${_scopeForUser(userId)}';
    await prefs.setString(key, value);
  }

  static Future<bool> getNotificationsEnabled(int userId) async {
    final prefs = await SharedPreferences.getInstance();
    final key = 'settings_notifications_${_scopeForUser(userId)}';
    return prefs.getBool(key) ?? false;
  }

  static Future<void> setNotificationsEnabled(int userId, bool enabled) async {
    final prefs = await SharedPreferences.getInstance();
    final key = 'settings_notifications_${_scopeForUser(userId)}';
    await prefs.setBool(key, enabled);
  }

  static Future<bool> getDataSharingEnabled(int userId) async {
    final prefs = await SharedPreferences.getInstance();
    final key = 'settings_data_sharing_${_scopeForUser(userId)}';
    return prefs.getBool(key) ?? false;
  }

  static Future<void> setDataSharingEnabled(int userId, bool enabled) async {
    final prefs = await SharedPreferences.getInstance();
    final key = 'settings_data_sharing_${_scopeForUser(userId)}';
    await prefs.setBool(key, enabled);
  }
}
