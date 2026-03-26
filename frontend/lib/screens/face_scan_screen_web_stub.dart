import 'package:flutter/material.dart';

/// Stub implementation used on non-web platforms.
class FaceScanWebScreen extends StatelessWidget {
  final int userId;
  final String userName;
  const FaceScanWebScreen({super.key, required this.userId, required this.userName});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Face Scan')),
      body: const Center(
        child: Text('Web face scan is only available in the web build.'),
      ),
    );
  }
}
