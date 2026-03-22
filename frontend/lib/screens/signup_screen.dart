import 'package:flutter/material.dart';
import '../services/api_service.dart';

class SignupScreen extends StatefulWidget {
  const SignupScreen({super.key});

  @override
  State<SignupScreen> createState() => _SignupScreenState();
}

class _SignupScreenState extends State<SignupScreen> {
  final _formKey = GlobalKey<FormState>();
  final _nameCtrl = TextEditingController();
  final _emailCtrl = TextEditingController();
  final _passCtrl = TextEditingController();
  final _confirmCtrl = TextEditingController();

  bool _loading = false;
  bool _obscurePass = true;
  bool _obscureConfirm = true;
  String? _errorMessage;

  @override
  void dispose() {
    _nameCtrl.dispose();
    _emailCtrl.dispose();
    _passCtrl.dispose();
    _confirmCtrl.dispose();
    super.dispose();
  }

  double get _passwordStrength {
    final p = _passCtrl.text;
    if (p.isEmpty) return 0;
    double s = 0;
    if (p.length >= 6) s += 0.25;
    if (p.length >= 8) s += 0.25;
    if (RegExp(r'[A-Z]').hasMatch(p)) s += 0.25;
    if (RegExp(r'[0-9@#\$%^&+=!]').hasMatch(p)) s += 0.25;
    return s;
  }

  Color get _strengthColor {
    final v = _passwordStrength;
    if (v <= 0.25) return Colors.red;
    if (v <= 0.5) return Colors.orange;
    if (v <= 0.75) return Colors.amber.shade700;
    return Colors.green;
  }

  String get _strengthLabel {
    final v = _passwordStrength;
    if (v <= 0.25) return 'Weak';
    if (v <= 0.5) return 'Fair';
    if (v <= 0.75) return 'Good';
    return 'Strong';
  }

  Future<void> _signup() async {
    setState(() => _errorMessage = null);
    if (!_formKey.currentState!.validate()) return;

    setState(() => _loading = true);
    try {
      await ApiService.signup(
        _nameCtrl.text.trim(),
        _emailCtrl.text.trim(),
        _passCtrl.text,
      );
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Account created! Please sign in.'),
          backgroundColor: Colors.green,
          behavior: SnackBarBehavior.floating,
        ),
      );
      Navigator.pop(context); // back to login
    } on ApiException catch (e) {
      setState(() => _errorMessage = ApiException.parseMessage(e));
    } catch (_) {
      setState(() =>
          _errorMessage = 'Cannot reach server. Is the backend running?');
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
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
                          // ── Header ──────────────────────────────────
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
                          Text('Create your account',
                              style: TextStyle(
                                  color: Colors.grey.shade600, fontSize: 14)),
                          const SizedBox(height: 24),

                          // ── Full Name ────────────────────────────────
                          TextFormField(
                            controller: _nameCtrl,
                            textCapitalization: TextCapitalization.words,
                            decoration: const InputDecoration(
                              labelText: 'Full Name',
                              prefixIcon: Icon(Icons.person_outline),
                              border: OutlineInputBorder(),
                            ),
                            validator: (v) {
                              if (v == null || v.trim().isEmpty) {
                                return 'Name is required';
                              }
                              if (v.trim().length < 2) {
                                return 'Name must be at least 2 characters';
                              }
                              return null;
                            },
                          ),
                          const SizedBox(height: 16),

                          // ── Email ────────────────────────────────────
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

                          // ── Password ─────────────────────────────────
                          TextFormField(
                            controller: _passCtrl,
                            obscureText: _obscurePass,
                            onChanged: (_) => setState(() {}),
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
                              if (v.length < 6) {
                                return 'Password must be at least 6 characters';
                              }
                              return null;
                            },
                          ),

                          // ── Strength indicator ───────────────────────
                          if (_passCtrl.text.isNotEmpty) ...[
                            const SizedBox(height: 8),
                            Row(children: [
                              Expanded(
                                child: ClipRRect(
                                  borderRadius: BorderRadius.circular(4),
                                  child: LinearProgressIndicator(
                                    value: _passwordStrength,
                                    minHeight: 5,
                                    backgroundColor: Colors.grey.shade200,
                                    valueColor: AlwaysStoppedAnimation(
                                        _strengthColor),
                                  ),
                                ),
                              ),
                              const SizedBox(width: 8),
                              Text(
                                _strengthLabel,
                                style: TextStyle(
                                  fontSize: 12,
                                  color: _strengthColor,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ]),
                          ],
                          const SizedBox(height: 16),

                          // ── Confirm Password ─────────────────────────
                          TextFormField(
                            controller: _confirmCtrl,
                            obscureText: _obscureConfirm,
                            decoration: InputDecoration(
                              labelText: 'Confirm Password',
                              prefixIcon: const Icon(Icons.lock_outline),
                              border: const OutlineInputBorder(),
                              suffixIcon: IconButton(
                                icon: Icon(_obscureConfirm
                                    ? Icons.visibility_off_outlined
                                    : Icons.visibility_outlined),
                                onPressed: () => setState(() =>
                                    _obscureConfirm = !_obscureConfirm),
                              ),
                            ),
                            validator: (v) {
                              if (v == null || v.isEmpty) {
                                return 'Please confirm your password';
                              }
                              if (v != _passCtrl.text) {
                                return 'Passwords do not match';
                              }
                              return null;
                            },
                          ),

                          // ── Error banner ─────────────────────────────
                          if (_errorMessage != null) ...[
                            const SizedBox(height: 12),
                            _ErrorBanner(message: _errorMessage!),
                          ],
                          const SizedBox(height: 24),

                          // ── Submit button ────────────────────────────
                          SizedBox(
                            width: double.infinity,
                            height: 48,
                            child: ElevatedButton(
                              onPressed: _loading ? null : _signup,
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
                                  : const Text('Create Account',
                                      style: TextStyle(fontSize: 16)),
                            ),
                          ),
                          const SizedBox(height: 10),

                          // ── Back to login ────────────────────────────
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Text('Already have an account?',
                                  style:
                                      TextStyle(color: Colors.grey.shade600)),
                              TextButton(
                                onPressed: () => Navigator.pop(context),
                                child: const Text('Sign In'),
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

/// Reusable error banner widget used in both auth screens.
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
              style: TextStyle(color: Colors.red.shade800, fontSize: 13),
            ),
          ),
        ],
      ),
    );
  }
}
