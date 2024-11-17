import 'package:flutter/material.dart';

class ModelInfo extends StatelessWidget {
  final String message;
  final String loraMessage;
  final VoidCallback onInitialize;
  final VoidCallback onInitializeLora;

  const ModelInfo({
    super.key,
    required this.message,
    required this.loraMessage,
    required this.onInitialize,
    required this.onInitializeLora,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: ElevatedButton(
                onPressed: onInitialize,
                child: const Text('Initialize Model'),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: ElevatedButton(
                onPressed: onInitializeLora,
                child: const Text('Load LORA'),
              ),
            ),
          ],
        ),
        const SizedBox(height: 20),
        Text(
          'Status: $message',
          style: const TextStyle(fontSize: 16),
        ),
        Text(
          'LORA: $loraMessage',
          style: const TextStyle(fontSize: 16),
        ),
      ],
    );
  }
}
