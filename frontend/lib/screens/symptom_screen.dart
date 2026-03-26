// Symptom Screen – from abc/ frontend with backend integration
//
// abc/'s rich symptom form (age, gender, major/minor symptoms, days,
// category, flags, severity) with submit to POST /predict/symptom.
import 'package:flutter/material.dart';
import 'analysis_screen.dart';

class SymptomScreen extends StatefulWidget {
  final int userId;
  const SymptomScreen({super.key, required this.userId});

  @override
  State<SymptomScreen> createState() => _SymptomScreenState();
}

class _SymptomScreenState extends State<SymptomScreen> {
  final ageCtrl = TextEditingController();
  final majorCtrl = TextEditingController();
  final minorCtrl = TextEditingController();
  final daysCtrl = TextEditingController();

  String gender = 'Male';
  String category = 'General';

  bool hasFever = false;
  bool hasPain = false;
  bool hasDifficultyBreathing = false;

  double severity = 5;

  void _submit() {
    final major = majorCtrl.text.trim();
    if (major.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter your major symptom.')),
      );
      return;
    }

    // Build a structured payload that the backend uses directly.
    // All fields are sent as explicit key-value pairs so the NLP layer
    // runs cleanly on symptom text (not on a "Age: 25. Gender: Male…" blob).
    final ageVal = int.tryParse(ageCtrl.text.trim());
    final daysVal = int.tryParse(daysCtrl.text.trim());

    final payload = <String, dynamic>{
      'major_symptom': major,
      if (minorCtrl.text.trim().isNotEmpty)
        'minor_symptoms': minorCtrl.text.trim(),
      if (ageVal != null) 'age': ageVal,
      'gender': gender,
      if (daysVal != null) 'days_suffering': daysVal,
      'symptom_category': category,
      'fever': hasFever,
      'pain': hasPain,
      'difficulty_breathing': hasDifficultyBreathing,
      'severity': severity.round(),
      'text': '',  // kept for schema compatibility
    };

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => AnalysisScreen(
          userId: widget.userId,
          symptomData: payload,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: SafeArea(
        child: Column(
          children: [
            // ── Header ──
            Padding(
              padding: const EdgeInsets.fromLTRB(18, 18, 18, 0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(children: const [
                    Icon(Icons.health_and_safety,
                        color: Colors.green, size: 38),
                    SizedBox(width: 8),
                    Text("Vita AI",
                        style: TextStyle(
                            fontSize: 28,
                            fontWeight: FontWeight.bold,
                            color: Colors.blue)),
                  ]),
                  IconButton(
                    onPressed: () => Navigator.pop(context),
                    icon: const Icon(Icons.close),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 10),

            // ── Form ──
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(18),
                child: Card(
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(20)),
                  child: Padding(
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text("Symptom Form",
                            style: TextStyle(
                                fontSize: 22, fontWeight: FontWeight.bold)),
                        const SizedBox(height: 20),

                        // Age
                        TextField(
                          controller: ageCtrl,
                          keyboardType: TextInputType.number,
                          decoration: const InputDecoration(
                            labelText: "Age",
                            border: OutlineInputBorder(),
                          ),
                        ),
                        const SizedBox(height: 15),

                        // Gender
                        DropdownButtonFormField<String>(
                          initialValue: gender,
                          decoration: const InputDecoration(
                            labelText: "Gender",
                            border: OutlineInputBorder(),
                          ),
                          items: ['Male', 'Female', 'Other']
                              .map((g) =>
                                  DropdownMenuItem(value: g, child: Text(g)))
                              .toList(),
                          onChanged: (v) => setState(() => gender = v!),
                        ),
                        const SizedBox(height: 15),

                        // Major symptom
                        TextField(
                          controller: majorCtrl,
                          decoration: const InputDecoration(
                            labelText: "Major Symptom",
                            hintText: "e.g., persistent cough",
                            border: OutlineInputBorder(),
                          ),
                        ),
                        const SizedBox(height: 15),

                        // Minor symptoms
                        TextField(
                          controller: minorCtrl,
                          decoration: const InputDecoration(
                            labelText: "Minor Symptoms (optional)",
                            hintText: "e.g., headache, fatigue",
                            border: OutlineInputBorder(),
                          ),
                        ),
                        const SizedBox(height: 15),

                        // Days suffering
                        TextField(
                          controller: daysCtrl,
                          keyboardType: TextInputType.number,
                          decoration: const InputDecoration(
                            labelText: "Days Suffering",
                            border: OutlineInputBorder(),
                          ),
                        ),
                        const SizedBox(height: 15),

                        // Symptom category
                        DropdownButtonFormField<String>(
                          initialValue: category,
                          decoration: const InputDecoration(
                            labelText: "Symptom Category",
                            border: OutlineInputBorder(),
                          ),
                          items: [
                            'General',
                            'Respiratory',
                            'Digestive',
                            'Neurological',
                            'Skin'
                          ]
                              .map((c) =>
                                  DropdownMenuItem(value: c, child: Text(c)))
                              .toList(),
                          onChanged: (v) => setState(() => category = v!),
                        ),
                        const SizedBox(height: 20),

                        // Boolean flags
                        SwitchListTile(
                          title: const Text("Fever"),
                          value: hasFever,
                          onChanged: (v) => setState(() => hasFever = v),
                        ),
                        SwitchListTile(
                          title: const Text("Pain"),
                          value: hasPain,
                          onChanged: (v) => setState(() => hasPain = v),
                        ),
                        SwitchListTile(
                          title: const Text("Difficulty Breathing"),
                          value: hasDifficultyBreathing,
                          onChanged: (v) =>
                              setState(() => hasDifficultyBreathing = v),
                        ),

                        const SizedBox(height: 15),

                        // Severity slider
                        Text("Severity: ${severity.round()} / 10",
                            style: const TextStyle(
                                fontSize: 16, fontWeight: FontWeight.w600)),
                        Slider(
                          value: severity,
                          min: 0,
                          max: 10,
                          divisions: 10,
                          label: severity.round().toString(),
                          onChanged: (v) => setState(() => severity = v),
                        ),

                        const SizedBox(height: 20),

                        // Submit button
                        SizedBox(
                          width: double.infinity,
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              padding: const EdgeInsets.symmetric(vertical: 16),
                              shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(12)),
                            ),
                            onPressed: _submit,
                            child: const Text("Submit Symptoms",
                                style: TextStyle(fontSize: 18)),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
