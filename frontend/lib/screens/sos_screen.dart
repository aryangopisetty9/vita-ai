// SOS / Emergency Contacts Screen — SOS feature integrated from Manogna
//
// Allows users to manage emergency contacts and trigger an SOS alert.
// Calls backend endpoints: GET/POST/DELETE /sos/contacts/{user_id},
// POST /sos/trigger/{user_id}.
import 'package:flutter/material.dart';
import '../services/api_service.dart';

class SosScreen extends StatefulWidget {
  final int userId;
  const SosScreen({super.key, required this.userId});
  @override
  State<SosScreen> createState() => _SosScreenState();
}

class _SosScreenState extends State<SosScreen> {
  List<Map<String, dynamic>> _contacts = [];
  bool _loading = true;
  String? _error;
  bool _triggering = false;

  @override
  void initState() {
    super.initState();
    _loadContacts();
  }

  Future<void> _loadContacts() async {
    if (widget.userId == 0) {
      setState(() {
        _loading = false;
        _error = 'Sign in to manage emergency contacts.';
      });
      return;
    }
    try {
      final raw = await ApiService.getSosContacts(widget.userId);
      setState(() {
        _contacts =
            raw.map((c) => Map<String, dynamic>.from(c as Map)).toList();
        _loading = false;
      });
    } on ApiException catch (e) {
      setState(() {
        _loading = false;
        _error = 'Failed to load contacts (${e.statusCode})';
      });
    } catch (e) {
      setState(() {
        _loading = false;
        _error = 'Failed to load contacts: $e';
      });
    }
  }

  Future<void> _addContact() async {
    final result = await showDialog<Map<String, String>>(
      context: context,
      builder: (_) => const _AddContactDialog(),
    );
    if (result == null) return;

    try {
      await ApiService.addSosContact(
        widget.userId,
        result['name']!,
        result['phone']!,
        result['relationship'],
      );
      _loadContacts();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to add contact: $e')),
        );
      }
    }
  }

  Future<void> _deleteContact(int contactId) async {
    try {
      await ApiService.deleteSosContact(widget.userId, contactId);
      _loadContacts();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to delete contact: $e')),
        );
      }
    }
  }

  Future<void> _triggerSos() async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Trigger SOS Alert?'),
        content: const Text(
          'This will log an emergency event and return your emergency '
          'contacts. In a production app, this would also send SMS/push '
          'notifications to your contacts.',
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Cancel')),
          ElevatedButton(
            style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
            onPressed: () => Navigator.pop(ctx, true),
            child:
                const Text('TRIGGER SOS', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
    if (confirm != true) return;

    setState(() => _triggering = true);
    try {
      final result = await ApiService.triggerSos(
        widget.userId,
        message: 'Emergency alert from Vita AI',
      );
      if (!mounted) return;
      final contacts = result['contacts'] as List<dynamic>? ?? [];
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'SOS triggered! ${contacts.length} emergency contact(s) notified.',
          ),
          backgroundColor: Colors.red,
        ),
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('SOS trigger failed: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _triggering = false);
    }
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
              padding: const EdgeInsets.all(18),
              child: Row(
                children: [
                  IconButton(
                    onPressed: () => Navigator.pop(context),
                    icon: const Icon(Icons.arrow_back),
                  ),
                  const Icon(Icons.sos, color: Colors.red, size: 32),
                  const SizedBox(width: 8),
                  const Text('Emergency SOS',
                      style:
                          TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
                ],
              ),
            ),

            // ── SOS Trigger Button ──
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: SizedBox(
                width: double.infinity,
                height: 56,
                child: ElevatedButton.icon(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red,
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16)),
                  ),
                  onPressed:
                      (widget.userId == 0 || _triggering) ? null : _triggerSos,
                  icon: _triggering
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                              strokeWidth: 2, color: Colors.white))
                      : const Icon(Icons.sos, color: Colors.white),
                  label: Text(
                    _triggering ? 'Sending...' : 'TRIGGER SOS ALERT',
                    style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Colors.white),
                  ),
                ),
              ),
            ),

            const SizedBox(height: 24),

            // ── Contacts Header ──
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text('Emergency Contacts',
                      style:
                          TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
                  if (widget.userId > 0)
                    IconButton(
                      onPressed: _addContact,
                      icon:
                          const Icon(Icons.person_add, color: Colors.blue),
                    ),
                ],
              ),
            ),

            const SizedBox(height: 8),

            // ── Contacts List ──
            Expanded(
              child: _loading
                  ? const Center(child: CircularProgressIndicator())
                  : _error != null
                      ? Center(
                          child: Padding(
                            padding: const EdgeInsets.all(24),
                            child: Text(_error!,
                                style: const TextStyle(
                                    color: Colors.grey, fontSize: 16),
                                textAlign: TextAlign.center),
                          ),
                        )
                      : _contacts.isEmpty
                          ? const Center(
                              child: Padding(
                                padding: EdgeInsets.all(24),
                                child: Text(
                                  'No emergency contacts yet.\nTap + to add one.',
                                  style: TextStyle(
                                      color: Colors.grey, fontSize: 16),
                                  textAlign: TextAlign.center,
                                ),
                              ),
                            )
                          : ListView.builder(
                              padding:
                                  const EdgeInsets.symmetric(horizontal: 18),
                              itemCount: _contacts.length,
                              itemBuilder: (_, i) {
                                final c = _contacts[i];
                                return Card(
                                  margin:
                                      const EdgeInsets.symmetric(vertical: 6),
                                  shape: RoundedRectangleBorder(
                                      borderRadius:
                                          BorderRadius.circular(14)),
                                  child: ListTile(
                                    leading: const CircleAvatar(
                                      backgroundColor: Colors.red,
                                      child: Icon(Icons.person,
                                          color: Colors.white),
                                    ),
                                    title: Text(c['name']?.toString() ?? ''),
                                    subtitle: Text(
                                      '${c['phone'] ?? ''}'
                                      '${c['relationship'] != null ? '  •  ${c['relationship']}' : ''}',
                                    ),
                                    trailing: IconButton(
                                      icon: const Icon(Icons.delete_outline,
                                          color: Colors.red),
                                      onPressed: () =>
                                          _deleteContact(c['id'] as int),
                                    ),
                                  ),
                                );
                              },
                            ),
            ),
          ],
        ),
      ),
    );
  }
}

// ── Add Contact Dialog ──────────────────────────────────────────────────

class _AddContactDialog extends StatefulWidget {
  const _AddContactDialog();
  @override
  State<_AddContactDialog> createState() => _AddContactDialogState();
}

class _AddContactDialogState extends State<_AddContactDialog> {
  final _nameCtrl = TextEditingController();
  final _phoneCtrl = TextEditingController();
  final _relCtrl = TextEditingController();

  @override
  void dispose() {
    _nameCtrl.dispose();
    _phoneCtrl.dispose();
    _relCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Add Emergency Contact'),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          TextField(
            controller: _nameCtrl,
            decoration: const InputDecoration(labelText: 'Name'),
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _phoneCtrl,
            decoration: const InputDecoration(labelText: 'Phone'),
            keyboardType: TextInputType.phone,
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _relCtrl,
            decoration:
                const InputDecoration(labelText: 'Relationship (optional)'),
          ),
        ],
      ),
      actions: [
        TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel')),
        ElevatedButton(
          onPressed: () {
            final name = _nameCtrl.text.trim();
            final phone = _phoneCtrl.text.trim();
            if (name.isEmpty || phone.isEmpty) return;
            Navigator.pop(context, {
              'name': name,
              'phone': phone,
              'relationship':
                  _relCtrl.text.trim().isEmpty ? null : _relCtrl.text.trim(),
            });
          },
          child: const Text('Save'),
        ),
      ],
    );
  }
}
