import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'dart:async';
import 'stable_diffusion_service.dart';

void main() {
  runApp(const MaterialApp(
    home: MyApp(),
  ));
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final TextEditingController _promptController = TextEditingController();
  final TextEditingController _negativePromptController =
      TextEditingController();
  Timer? _errorMessageTimer;

  String _message = '';
  String _loraMessage = '';
  String _taesdMessage = '';
  String _taesdError = '';
  int _cores = 0;
  double _cfgScale = 7.0;
  int _steps = 20;
  int _width = 512;
  int _height = 512;
  int _seed = 42;
  Image? _generatedImage;
  bool _useTinyAutoencoder = false;

  void _showTemporaryError(String error) {
    _errorMessageTimer?.cancel();
    setState(() {
      _taesdError = error;
    });
    _errorMessageTimer = Timer(const Duration(seconds: 10), () {
      setState(() {
        _taesdError = '';
      });
    });
  }

  @override
  void initState() {
    super.initState();
    _cores = StableDiffusionService.getCores();
  }

  @override
  void dispose() {
    _promptController.dispose();
    _negativePromptController.dispose();
    _errorMessageTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('Local Diffusion'),
          backgroundColor: Colors.blue,
        ),
        body: SingleChildScrollView(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Available CPU Cores: $_cores',
                style: const TextStyle(fontSize: 18),
              ),
              const SizedBox(height: 20),
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton(
                      onPressed: () async {
                        final bool? useFlashAttention = await showDialog<bool>(
                          context: context,
                          builder: (BuildContext context) {
                            return AlertDialog(
                              title: const Text('Model Initialization Options'),
                              content: const Text(
                                  'How would you like to load the model?'),
                              actions: <Widget>[
                                TextButton(
                                  onPressed: () => Navigator.pop(context, true),
                                  child: const Text('With Flash Attention'),
                                ),
                                TextButton(
                                  onPressed: () =>
                                      Navigator.pop(context, false),
                                  child: const Text('Without Flash Attention'),
                                ),
                              ],
                            );
                          },
                        );

                        if (useFlashAttention != null) {
                          StableDiffusionService.setFlashAttention(
                              useFlashAttention);
                          final result = await StableDiffusionService
                              .pickAndInitializeModel();
                          setState(() {
                            _message = result;
                            _taesdError = '';
                          });
                        }
                      },
                      child: const Text('Initialize Model'),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: ElevatedButton(
                      onPressed: () async {
                        final result = await StableDiffusionService
                            .pickAndInitializeLora();
                        setState(() {
                          _loraMessage = result;
                        });
                      },
                      child: const Text('Load LORA'),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Checkbox(
                        value: _useTinyAutoencoder,
                        onChanged: (bool? value) {
                          if (StableDiffusionService.taesdPath == null) {
                            _showTemporaryError(
                                'Please load TAESD model first');
                            return;
                          }
                          setState(() {
                            if (StableDiffusionService.setTinyAutoencoder(
                                value ?? false)) {
                              _useTinyAutoencoder = value ?? false;
                              StableDiffusionService.initializeModel();
                              _taesdError = '';
                            }
                          });
                        },
                      ),
                      const Text('Use Tiny AutoEncoder'),
                      const Spacer(),
                      ElevatedButton(
                        onPressed: () async {
                          final result = await StableDiffusionService
                              .pickAndInitializeTAESD();
                          setState(() {
                            _taesdMessage = result;
                            _taesdError = '';
                          });
                        },
                        child: const Text('Load TAESD'),
                      ),
                    ],
                  ),
                  if (_taesdError.isNotEmpty)
                    Padding(
                      padding: const EdgeInsets.only(left: 8.0),
                      child: Text(
                        _taesdError,
                        style: const TextStyle(color: Colors.red, fontSize: 12),
                      ),
                    ),
                  if (_taesdMessage.isNotEmpty)
                    Text(
                      'TAESD: $_taesdMessage',
                      style: const TextStyle(fontSize: 16),
                    ),
                ],
              ),
              const SizedBox(height: 20),
              Text(
                'Status: $_message',
                style: const TextStyle(fontSize: 16),
              ),
              Text(
                'LORA: $_loraMessage',
                style: const TextStyle(fontSize: 16),
              ),
              const SizedBox(height: 20),
              TextField(
                controller: _promptController,
                decoration: const InputDecoration(labelText: 'Prompt'),
                maxLines: 3,
              ),
              TextField(
                controller: _negativePromptController,
                decoration: const InputDecoration(labelText: 'Negative Prompt'),
                maxLines: 2,
              ),
              Row(
                children: [
                  Expanded(
                    child: Slider(
                      value: _cfgScale,
                      min: 1.0,
                      max: 20.0,
                      divisions: 38,
                      label: _cfgScale.toString(),
                      onChanged: (value) => setState(() => _cfgScale = value),
                    ),
                  ),
                  Text('CFG: ${_cfgScale.toStringAsFixed(1)}'),
                ],
              ),
              Row(
                children: [
                  Expanded(
                    child: Slider(
                      value: _steps.toDouble(),
                      min: 1,
                      max: 50,
                      divisions: 49,
                      label: _steps.toString(),
                      onChanged: (value) =>
                          setState(() => _steps = value.toInt()),
                    ),
                  ),
                  Text('Steps: $_steps'),
                ],
              ),
              Row(
                children: [
                  Expanded(
                    child: TextFormField(
                      initialValue: _width.toString(),
                      decoration: const InputDecoration(labelText: 'Width'),
                      keyboardType: TextInputType.number,
                      onChanged: (value) =>
                          setState(() => _width = int.tryParse(value) ?? 512),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: TextFormField(
                      initialValue: _height.toString(),
                      decoration: const InputDecoration(labelText: 'Height'),
                      keyboardType: TextInputType.number,
                      onChanged: (value) =>
                          setState(() => _height = int.tryParse(value) ?? 512),
                    ),
                  ),
                ],
              ),
              TextFormField(
                initialValue: _seed.toString(),
                decoration:
                    const InputDecoration(labelText: 'Seed (-1 for random)'),
                keyboardType: TextInputType.number,
                onChanged: (value) =>
                    setState(() => _seed = int.tryParse(value) ?? -1),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () async {
                  setState(() => _message = 'Generating image...');
                  final image = await StableDiffusionService.generateImage(
                    prompt: _promptController.text,
                    negativePrompt: _negativePromptController.text,
                    cfgScale: _cfgScale,
                    sampleSteps: _steps,
                    width: _width,
                    height: _height,
                    seed: _seed,
                  );

                  if (image != null) {
                    final bytes =
                        await image.toByteData(format: ui.ImageByteFormat.png);
                    setState(() {
                      _generatedImage =
                          Image.memory(bytes!.buffer.asUint8List());
                      _message = 'Generation complete';
                    });
                  }
                },
                child: const Text('Generate Image'),
              ),
              if (_generatedImage != null) ...[
                const SizedBox(height: 20),
                _generatedImage!,
              ],
            ],
          ),
        ),
      ),
    );
  }
}
