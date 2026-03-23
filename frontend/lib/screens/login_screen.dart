import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import '../services/api_service.dart';
import '../models/health_data.dart';
import 'home_screen.dart';
import 'signup_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _formKey = GlobalKey<FormState>();
  final _emailCtrl = TextEditingController();
  final _passCtrl = TextEditingController();

  bool _loading = false;
  bool _obscurePass = true;
  String? _errorMessage;

  @override
  void dispose() {
    _emailCtrl.dispose();
    _passCtrl.dispose();
    super.dispose();
  }

  @override
  void initState() {
    super.initState();
    // If OAuth redirected back with a token in the query (web), apply it.
    if (kIsWeb) {
      final token = Uri.base.queryParameters['token'];
      if (token != null && token.isNotEmpty) {
        ApiService.setToken(token);
        // Fetch profile and navigate to home
        ApiService.getProfile().then((profile) {
          if (!mounted) return;
          final user = profile;
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(
              builder: (_) => HomeScreen(
                userId: user['id'] as int? ?? 0,
                userName: user['name'] as String? ?? 'User',
              ),
            ),
          );
        }).catchError((_) {});
      }
    }
  }

  Future<void> _login() async {
    setState(() => _errorMessage = null);
    if (!_formKey.currentState!.validate()) return;

    setState(() => _loading = true);
    try {
      final resp =
          await ApiService.login(_emailCtrl.text.trim(), _passCtrl.text);
      if (!mounted) return;

      final user =
          (resp['user'] as Map<String, dynamic>?) ?? resp;
      HealthData.clear();
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => HomeScreen(
            userId: user['id'] as int? ?? 0,
            userName: user['name'] as String? ?? 'User',
          ),
        ),
      );
    } on ApiException catch (e) {
      setState(() => _errorMessage = ApiException.parseMessage(e));
    } catch (_) {
      setState(() =>
          _errorMessage = 'Cannot reach server. Is the backend running?');
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _continueAsGuest() {
    HealthData.clear();
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
          builder: (_) => const HomeScreen(userId: 0, userName: 'Guest')),
    );
  }

  Future<void> _openUrl(String url) async {
    final uri = Uri.parse(url);
    if (!await launchUrl(uri, mode: LaunchMode.externalApplication)) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not open $url')),
        );
      }
    }
  }

  Widget _socialButton(
      Color color, IconData icon, String label, VoidCallback onTap) {
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton.icon(
        icon: Icon(icon, color: Colors.white),
        label: Text(label),
        style: ElevatedButton.styleFrom(
          backgroundColor: color,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(vertical: 13),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        ),
        onPressed: onTap,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Background
          SizedBox.expand(
              child: Image.asset('assets/bg.jpg', fit: BoxFit.cover)),
          Container(color: Colors.black.withValues(alpha: 0.55)),

          Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(24),
              child: Card(
                elevation: 12,
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(20)),
                child: Padding(
                  padding: const EdgeInsets.all(30),
                  child: SizedBox(
                    width: 360,
                    child: Form(
                      key: _formKey,
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          // ── Logo ──────────────────────────────────────
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: const [
                              Icon(Icons.health_and_safety,
                                  color: Colors.blue, size: 40),
                              SizedBox(width: 10),
                              Text('Vita AI',
                                  style: TextStyle(
                                      fontSize: 28,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.blue)),
                            ],
                          ),
                          const SizedBox(height: 6),
                          Text('Sign in to your account',
                              style: TextStyle(
                                  color: Colors.grey.shade600, fontSize: 14)),
                          const SizedBox(height: 24),

                          // ── Social login ──────────────────────────────
                          _socialButton(
                            const Color(0xFF1877F2),
                            Icons.facebook,
                            'Continue with Facebook',
                            () => _openUrl('${ApiService.baseUrl}/auth/oauth/facebook/login'),
                          ),
                          const SizedBox(height: 10),
                          _socialButton(
                            Colors.red,
                            Icons.g_mobiledata,
                            'Continue with Google',
                            () => _openUrl('${ApiService.baseUrl}/auth/oauth/google/login'),
                          ),
                          const SizedBox(height: 10),
                          _socialButton(
                            Colors.black,
                            Icons.apple,
                            'Continue with Apple',
                            () => _openUrl('${ApiService.baseUrl}/auth/oauth/apple/login'),
                          ),
                          const SizedBox(height: 20),

                          // ── Divider ───────────────────────────────────
                          Row(children: [
                            const Expanded(child: Divider()),
                            Padding(
                              padding:
                                  const EdgeInsets.symmetric(horizontal: 12),
                              child: Text('or',
                                  style: TextStyle(
                                      color: Colors.grey.shade500)),
                            ),
                            const Expanded(child: Divider()),
                          ]),
                          const SizedBox(height: 20),

                          // ── Email ─────────────────────────────────────
                          TextFormField(
                            controller: _emailCtrl,
                            keyboardType: TextInputType.emailAddress,
                            decoration: const InputDecoration(
                              labelText: 'Email address',
                              prefixIcon: Icon(Icons.email_outlined),
                              border: OutlineInputBorder(),
                            ),
                            validator: (v) {
                              if (v == null || v.trim().isEmpty) {
                                return 'Email is required';
                              }
                              final ok = RegExp(
                                r'^[\w.+\-]+@[\w\-]+\.[a-zA-Z]{2,}$',
                              ).hasMatch(v.trim());
                              if (!ok) return 'Enter a valid email address';
                              return null;
                            },
                          ),
                          const SizedBox(height: 16),

                          // ── Password ──────────────────────────────────
                          TextFormField(
                            controller: _passCtrl,
                            obscureText: _obscurePass,
                            decoration: InputDecoration(
                              labelText: 'Password',
                              prefixIcon: const Icon(Icons.lock_outline),
                              border: const OutlineInputBorder(),
                              suffixIcon: IconButton(
                                icon: Icon(_obscurePass
                                    ? Icons.visibility_off_outlined
                                    : Icons.visibility_outlined),
                                onPressed: () => setState(
                                    () => _obscurePass = !_obscurePass),
                              ),
                            ),
                            validator: (v) {
                              if (v == null || v.isEmpty) {
                                return 'Password is required';
                              }
                              return null;
                            },
                          ),

                          // ── Error banner ──────────────────────────────
                          if (_errorMessage != null) ...[
                            const SizedBox(height: 12),
                            _ErrorBanner(message: _errorMessage!),
                          ],
                          const SizedBox(height: 24),

                          // ── Sign In button ────────────────────────────
                          SizedBox(
                            width: double.infinity,
                            height: 48,
                            child: ElevatedButton(
                              onPressed: _loading ? null : _login,
                              style: ElevatedButton.styleFrom(
                                shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(10)),
                              ),
                              child: _loading
                                  ? const SizedBox(
                                      height: 20,
                                      width: 20,
                                      child: CircularProgressIndicator(
                                          strokeWidth: 2,
                                          color: Colors.white))
                                  : const Text('Sign In',
                                      style: TextStyle(fontSize: 16)),
                            ),
                          ),
                          const SizedBox(height: 10),

                          // ── Guest mode ────────────────────────────────
                          SizedBox(
                            width: double.infinity,
                            child: OutlinedButton(
                              onPressed: _continueAsGuest,
                              child: const Text('Continue as Guest'),
                            ),
                          ),
                          const SizedBox(height: 12),

                          // ── Link to signup ────────────────────────────
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Text("Don't have an account?",
                                  style:
                                      TextStyle(color: Colors.grey.shade600)),
                              TextButton(
                                onPressed: () => Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                      builder: (_) => const SignupScreen()),
                                ),
                                child: const Text('Create one'),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// Styled error banner shared by auth screens (defined in signup too,
/// but kept here so login_screen is self-contained).
class _ErrorBanner extends StatelessWidget {
  const _ErrorBanner({required this.message});
  final String message;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.red.shade50,
        border: Border.all(color: Colors.red.shade200),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.error_outline, color: Colors.red.shade700, size: 18),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              message,
              style:
                  TextStyle(color: Colors.red.shade800, fontSize: 13),
            ),
          ),
        ],
      ),
    );
  }
}
