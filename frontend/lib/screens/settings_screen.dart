import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../services/app_theme_controller.dart';
import '../services/settings_storage_service.dart';
import 'welcome_screen.dart';

class SettingsScreen extends StatefulWidget {
  final int userId;
  final String userName;

  const SettingsScreen({super.key, required this.userId, required this.userName});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  bool _loading = false;
  String _name = '';
  String _bio = '';
  String? _email;
  bool _notificationsEnabled = false;
  bool _dataSharingEnabled = false;

  bool get _isGuest => widget.userId == 0;

  @override
  void initState() {
    super.initState();
    _loadSettings();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    debugPrint('SettingsScreen didChangeDependencies (userId=${widget.userId})');
  }

  Future<void> _loadSettings() async {
    setState(() => _loading = true);

    try {
      final fallbackName = widget.userName.isNotEmpty
          ? widget.userName
          : (_isGuest ? 'Guest' : 'User');

      final localName = await SettingsStorageService.getName(
        widget.userId,
        fallback: fallbackName,
      );
      final localBio = await SettingsStorageService.getBio(widget.userId);
      final localNotifications =
          await SettingsStorageService.getNotificationsEnabled(widget.userId);
      final localDataSharing =
          await SettingsStorageService.getDataSharingEnabled(widget.userId);

      String loadedName = localName;
      String? loadedEmail;

      if (!_isGuest) {
        try {
          final profile = await ApiService.getProfile();
          final remoteName = (profile['name'] as String?)?.trim();
          if (remoteName != null && remoteName.isNotEmpty) {
            loadedName = remoteName;
            await SettingsStorageService.setName(widget.userId, loadedName);
          }
          loadedEmail = profile['email'] as String?;
        } catch (_) {
          // Keep local settings when backend profile fetch fails.
        }
      }

      if (!mounted) return;
      setState(() {
        _name = loadedName;
        _bio = localBio;
        _notificationsEnabled = localNotifications;
        _dataSharingEnabled = localDataSharing;
        _email = loadedEmail;
      });
    } catch (_) {
      // Keep current state.
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _setName(String nextName) async {
    final trimmed = nextName.trim();
    if (trimmed.length < 2) {
      _showSnack('Name must be at least 2 characters', isError: true);
      return;
    }

    if (!_isGuest) {
      try {
        final updated = await ApiService.updateProfile(name: trimmed);
        final remoteName = (updated['name'] as String?)?.trim();
        _name = (remoteName != null && remoteName.isNotEmpty) ? remoteName : trimmed;
      } on ApiException catch (e) {
        _showSnack(ApiException.parseMessage(e), isError: true);
        return;
      } catch (_) {
        _showSnack('Could not update name right now.', isError: true);
        return;
      }
    } else {
      _name = trimmed;
    }

    await SettingsStorageService.setName(widget.userId, _name);
    if (!mounted) return;
    setState(() {});
  }

  Future<void> _setBio(String nextBio) async {
    _bio = nextBio.trim();
    await SettingsStorageService.setBio(widget.userId, _bio);
    if (!mounted) return;
    setState(() {});
  }

  Future<void> _editName() async {
    final value = await _showTextEditDialog(
      title: 'Edit Name',
      label: 'Name',
      initialValue: _name,
      maxLength: 100,
      validator: (v) {
        final t = (v ?? '').trim();
        if (t.isEmpty) return 'Name is required';
        if (t.length < 2) return 'Name must be at least 2 characters';
        return null;
      },
    );
    if (value == null) return;
    await _setName(value);
  }

  Future<void> _editBio() async {
    final value = await _showTextEditDialog(
      title: 'Edit Bio',
      label: 'Bio',
      initialValue: _bio,
      maxLength: 280,
      maxLines: 4,
    );
    if (value == null) return;
    await _setBio(value);
  }

  Future<void> _editAccountSection() async {
    final nextName = await _showTextEditDialog(
      title: 'Edit Account Name',
      label: 'Name',
      initialValue: _name,
      maxLength: 100,
      validator: (v) {
        final t = (v ?? '').trim();
        if (t.isEmpty) return 'Name is required';
        if (t.length < 2) return 'Name must be at least 2 characters';
        return null;
      },
    );
    if (nextName != null) {
      await _setName(nextName);
    }
  }

  Future<void> _setNotifications(bool enabled) async {
    await SettingsStorageService.setNotificationsEnabled(widget.userId, enabled);
    if (!mounted) return;
    setState(() {
      _notificationsEnabled = enabled;
    });
  }

  Future<void> _manageDataSharing() async {
    bool localValue = _dataSharingEnabled;
    final saved = await showDialog<bool>(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setLocal) => AlertDialog(
          title: const Text('Data Sharing'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('Allow anonymized data sharing for research and model improvement.'),
              const SizedBox(height: 12),
              SwitchListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text('Share anonymized data'),
                value: localValue,
                onChanged: (v) => setLocal(() => localValue = v),
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx, null),
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () => Navigator.pop(ctx, localValue),
              child: const Text('Save'),
            ),
          ],
        ),
      ),
    );

    if (saved == null) return;
    await SettingsStorageService.setDataSharingEnabled(widget.userId, saved);
    if (!mounted) return;
    setState(() {
      _dataSharingEnabled = saved;
    });
  }

  Future<void> _changePassword() async {
    if (_isGuest) {
      _showSnack('Password change is not available for guest mode.', isError: true);
      return;
    }

    final formKey = GlobalKey<FormState>();
    final currentCtrl = TextEditingController();
    final newCtrl = TextEditingController();
    final confirmCtrl = TextEditingController();
    String? errorText;
    bool submitting = false;

    await showDialog<void>(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setLocal) => AlertDialog(
          title: const Text('Change Password'),
          content: Form(
            key: formKey,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextFormField(
                  controller: currentCtrl,
                  obscureText: true,
                  decoration: const InputDecoration(labelText: 'Current password'),
                  validator: (v) => (v == null || v.isEmpty) ? 'Current password is required' : null,
                ),
                const SizedBox(height: 10),
                TextFormField(
                  controller: newCtrl,
                  obscureText: true,
                  decoration: const InputDecoration(labelText: 'New password'),
                  validator: (v) {
                    if (v == null || v.isEmpty) return 'New password is required';
                    if (v.length < 6) return 'Password must be at least 6 characters';
                    return null;
                  },
                ),
                const SizedBox(height: 10),
                TextFormField(
                  controller: confirmCtrl,
                  obscureText: true,
                  decoration: const InputDecoration(labelText: 'Confirm new password'),
                  validator: (v) {
                    if (v == null || v.isEmpty) return 'Confirm your new password';
                    if (v != newCtrl.text) return 'Passwords do not match';
                    return null;
                  },
                ),
                if (errorText != null) ...[
                  const SizedBox(height: 10),
                  Text(errorText!, style: const TextStyle(color: Colors.red)),
                ],
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: submitting ? null : () => Navigator.pop(ctx),
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: submitting
                  ? null
                  : () async {
                      if (!formKey.currentState!.validate()) return;
                      setLocal(() {
                        submitting = true;
                        errorText = null;
                      });
                      try {
                        await ApiService.changePassword(
                          currentPassword: currentCtrl.text,
                          newPassword: newCtrl.text,
                          confirmPassword: confirmCtrl.text,
                        );
                        if (!mounted || !ctx.mounted) return;
                        Navigator.pop(ctx);
                        _showSnack('Password updated successfully.');
                      } on ApiException catch (e) {
                        setLocal(() {
                          errorText = ApiException.parseMessage(e);
                          submitting = false;
                        });
                      } catch (_) {
                        setLocal(() {
                          errorText = 'Could not update password right now.';
                          submitting = false;
                        });
                      }
                    },
              child: submitting
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text('Update'),
            ),
          ],
        ),
      ),
    );
  }

  Future<String?> _showTextEditDialog({
    required String title,
    required String label,
    required String initialValue,
    int maxLength = 100,
    int maxLines = 1,
    String? Function(String?)? validator,
  }) async {
    final controller = TextEditingController(text: initialValue);
    final formKey = GlobalKey<FormState>();

    return showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(title),
        content: Form(
          key: formKey,
          child: TextFormField(
            controller: controller,
            maxLength: maxLength,
            maxLines: maxLines,
            decoration: InputDecoration(labelText: label),
            validator: validator,
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              if (!formKey.currentState!.validate()) return;
              Navigator.pop(ctx, controller.text.trim());
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }

  void _showSnack(String message, {bool isError = false}) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: isError ? Colors.red : Colors.green,
      ),
    );
  }

  void _logout() {
    // Clear token and return to welcome screen. Preserve safe state.
    ApiService.clearToken();
    debugPrint('Logging out / exiting guest mode (userId=${widget.userId})');
    Navigator.pushAndRemoveUntil(
      context,
      MaterialPageRoute(builder: (_) => const WelcomeScreen()),
      (r) => false,
    );
  }

  @override
  Widget build(BuildContext context) {
    final themeController = AppThemeController.instance;
    final displayName = _name.isNotEmpty
        ? _name
        : (widget.userName.isNotEmpty ? widget.userName : (_isGuest ? 'Guest' : 'User'));
    final errorContainer = Theme.of(context).colorScheme.errorContainer;
    final onErrorContainer = Theme.of(context).colorScheme.onErrorContainer;

    return AnimatedBuilder(
      animation: themeController,
      builder: (context, _) => Scaffold(
        appBar: AppBar(
          title: const Text('Settings'),
          leading: IconButton(
            icon: const Icon(Icons.arrow_back),
            onPressed: () => Navigator.pop(context),
          ),
        ),
        body: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
              if (_loading) const LinearProgressIndicator(),
              // Account section
              const SizedBox(height: 8),
              const Text('Account', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              const SizedBox(height: 8),
              Card(
                child: ListTile(
                  leading: const Icon(Icons.person),
                  title: Text(displayName),
                  subtitle: Text(_email ?? (_isGuest ? 'Guest Account' : 'Email not available')),
                  trailing: TextButton(
                    onPressed: _editAccountSection,
                    child: const Text('Edit'),
                  ),
                ),
              ),

              const SizedBox(height: 16),
              const Text('Profile', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              const SizedBox(height: 8),
              Card(
                child: Column(children: [
                  ListTile(
                    title: const Text('Name'),
                    subtitle: Text(displayName),
                    trailing: IconButton(
                      icon: const Icon(Icons.edit),
                      onPressed: _editName,
                    ),
                  ),
                  const Divider(height: 1),
                  ListTile(
                    title: const Text('Bio'),
                    subtitle: Text(_bio.isEmpty ? 'Not set' : _bio),
                    trailing: IconButton(
                      icon: const Icon(Icons.edit),
                      onPressed: _editBio,
                    ),
                  ),
                ]),
              ),

              const SizedBox(height: 16),
              const Text('Notifications', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              const SizedBox(height: 8),
              Card(
                child: SwitchListTile(
                  title: const Text('Enable notifications'),
                  subtitle: const Text('Preference only. OS-level push is not configured yet.'),
                  value: _notificationsEnabled,
                  onChanged: _setNotifications,
                ),
              ),

              const SizedBox(height: 16),
              const Text('Privacy', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              const SizedBox(height: 8),
              Card(
                child: Column(children: [
                  ListTile(
                    leading: const Icon(Icons.shield),
                    title: const Text('Data sharing'),
                    subtitle: Text(_dataSharingEnabled
                        ? 'Anonymized data sharing is enabled'
                        : 'Anonymized data sharing is disabled'),
                    trailing: TextButton(onPressed: _manageDataSharing, child: const Text('Manage')),
                  ),
                  const Divider(height: 1),
                  ListTile(
                    leading: const Icon(Icons.lock),
                    title: const Text('Change password'),
                    subtitle: _isGuest ? const Text('Not available for guest') : const Text('Update your password'),
                    trailing: TextButton(onPressed: _isGuest ? null : _changePassword, child: const Text('Change')),
                  ),
                ]),
              ),

              const SizedBox(height: 16),
              const Text('Appearance', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              const SizedBox(height: 8),
              Card(
                child: SwitchListTile(
                  title: const Text('Dark theme'),
                  value: themeController.isDarkTheme,
                  onChanged: (v) => themeController.setDarkTheme(v),
                ),
              ),

              const SizedBox(height: 16),
              const Text('About', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              const SizedBox(height: 8),
              Card(
                child: ListTile(
                  leading: const Icon(Icons.info_outline),
                  title: const Text('About Vita AI'),
                  subtitle: const Text('Version: 1.0.0 — Demo build'),
                  onTap: () {},
                ),
              ),

              const SizedBox(height: 24),
              Card(
                color: errorContainer,
                child: ListTile(
                  leading: Icon(Icons.exit_to_app, color: onErrorContainer),
                  title: Text(
                    _isGuest ? 'Exit guest mode' : 'Logout',
                    style: TextStyle(color: onErrorContainer),
                  ),
                  onTap: _logout,
                ),
              ),
              const SizedBox(height: 40),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
