import 'package:flutter/material.dart';
import 'screens/welcome_screen.dart';
import 'screens/home_screen.dart';
import 'screens/face_scan_screen.dart';
import 'screens/voice_screen.dart';
import 'services/app_theme_controller.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await AppThemeController.instance.load();
  runApp(const VitaAIApp());
}

class VitaAIApp extends StatelessWidget {
  const VitaAIApp({super.key});

  @override
  Widget build(BuildContext context) {
    final themeController = AppThemeController.instance;

    return AnimatedBuilder(
      animation: themeController,
      builder: (context, _) => MaterialApp(
        debugShowCheckedModeBanner: false,
        title: "Vita AI",
        themeMode: themeController.themeMode,
        theme: _buildLightTheme(),
        darkTheme: _buildDarkTheme(),
        home: const WelcomeScreen(),
        onGenerateRoute: (settings) {
          switch (settings.name) {
            case '/welcome':
              return MaterialPageRoute(
                settings: settings,
                builder: (_) => const WelcomeScreen(),
              );
            case '/home': {
              final args = settings.arguments as Map<String, dynamic>?;
              final userId = args != null && args['userId'] is int ? args['userId'] as int : 0;
              final userName = args != null && args['userName'] is String ? args['userName'] as String : 'Guest';
              return MaterialPageRoute(
                settings: settings,
                builder: (_) => HomeScreen(userId: userId, userName: userName),
              );
            }
            case '/face': {
              final args = settings.arguments as Map<String, dynamic>?;
              final userId = args != null && args['userId'] is int ? args['userId'] as int : 0;
              final userName = args != null && args['userName'] is String ? args['userName'] as String : 'Guest';
              return MaterialPageRoute(
                settings: settings,
                builder: (_) => FaceScanScreen(userId: userId, userName: userName),
              );
            }
            case '/voice': {
              final args = settings.arguments as Map<String, dynamic>?;
              final userId = args != null && args['userId'] is int ? args['userId'] as int : 0;
              final userName = args != null && args['userName'] is String ? args['userName'] as String : 'Guest';
              return MaterialPageRoute(
                settings: settings,
                builder: (_) => VoiceScreen(userId: userId, userName: userName),
              );
            }
            default:
              return MaterialPageRoute(
                settings: settings,
                builder: (_) => const WelcomeScreen(),
              );
          }
        },
      ),
    );
  }

  ThemeData _buildLightTheme() {
    final base = ThemeData(
      colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
      useMaterial3: true,
      brightness: Brightness.light,
    );
    return base.copyWith(
      scaffoldBackgroundColor: const Color(0xFFEDEFF3),
      cardTheme: base.cardTheme.copyWith(
        color: Colors.white,
      ),
      dialogTheme: base.dialogTheme.copyWith(
        backgroundColor: Colors.white,
      ),
    );
  }

  ThemeData _buildDarkTheme() {
    final base = ThemeData(
      colorScheme: ColorScheme.fromSeed(
        seedColor: Colors.blue,
        brightness: Brightness.dark,
      ),
      useMaterial3: true,
      brightness: Brightness.dark,
    );
    return base.copyWith(
      scaffoldBackgroundColor: const Color(0xFF11151C),
      cardTheme: base.cardTheme.copyWith(
        color: const Color(0xFF1B2330),
      ),
      dialogTheme: base.dialogTheme.copyWith(
        backgroundColor: const Color(0xFF1B2330),
      ),
      dividerTheme: base.dividerTheme.copyWith(
        color: Colors.white24,
      ),
    );
  }
}
