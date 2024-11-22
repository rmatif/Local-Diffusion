import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'dart:async';
import 'dart:developer' as developer;
import 'stable_diffusion_service.dart';
import 'ffi_bindings.dart';

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
  String _ramUsage = '';
  String _progressMessage = '';
  String _totalTime = '';
  int _cores = 0;
  double _cfgScale = 7.0;
  int _steps = 20;
  int _width = 512;
  int _height = 512;
  int _seed = 42;
  Image? _generatedImage;
  bool _useTinyAutoencoder = false;
  SDType selectedType = SDType.NONE;
  SampleMethod _selectedSampleMethod = SampleMethod.EULER_A;
  Schedule _selectedSchedule = Schedule.DISCRETE;

  StreamSubscription? _logSubscription;
  StreamSubscription? _progressSubscription;

  @override
  void initState() {
    super.initState();
    _cores = StableDiffusionService.getCores();
    _setupSubscriptions();
  }

  void _setupSubscriptions() {
    _logSubscription = StableDiffusionService.logStream.listen((logMessage) {
      if (logMessage.message.contains('total params memory size')) {
        final regex = RegExp(r'total params memory size = ([\d.]+)MB');
        final match = regex.firstMatch(logMessage.message);
        if (match != null) {
          setState(() {
            _ramUsage = 'Total RAM Usage: ${match.group(1)}MB';
          });
        }
      } else if (logMessage.message.contains('completed in')) {
        final regex = RegExp(r'completed in ([\d.]+)s');
        final match = regex.firstMatch(logMessage.message);
        if (match != null) {
          setState(() {
            _totalTime = 'Generation completed in ${match.group(1)}s';
          });
        }
      }
    });

    _progressSubscription =
        StableDiffusionService.progressStream.listen((progress) {
      setState(() {
        _progressMessage =
            'Progress: ${(progress.progress * 100).toInt()}% (Step ${progress.step}/${progress.totalSteps}, ${progress.time.toStringAsFixed(1)}s)';
      });
    });
  }

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
  void dispose() {
    _promptController.dispose();
    _negativePromptController.dispose();
    _errorMessageTimer?.cancel();
    _logSubscription?.cancel();
    _progressSubscription?.cancel();
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
              if (_ramUsage.isNotEmpty)
                Text(
                  _ramUsage,
                  style: const TextStyle(fontSize: 18),
                ),
              const SizedBox(height: 20),
              Row(
                children: [
                  Expanded(
                    child: StreamBuilder<bool>(
                        stream: StableDiffusionService.loadingStream,
                        builder: (context, snapshot) {
                          return ElevatedButton(
                            onPressed: snapshot.data == true
                                ? null
                                : () async {
                                    if (StableDiffusionService
                                        .isModelLoaded()) {
                                      StableDiffusionService.freeCurrentModel();
                                    }

                                    final result = await showDialog<
                                        (bool, SDType, Schedule)>(
                                      context: context,
                                      builder: (BuildContext context) {
                                        Schedule dialogSchedule =
                                            Schedule.DISCRETE;
                                        SDType dialogType = SDType.NONE;

                                        return StatefulBuilder(
                                          builder: (context, setState) {
                                            return AlertDialog(
                                              title: const Text(
                                                  'Model Initialization Options'),
                                              content: Column(
                                                mainAxisSize: MainAxisSize.min,
                                                children: [
                                                  const Text('Model Type'),
                                                  DropdownButton<SDType>(
                                                    value: dialogType,
                                                    isExpanded: true,
                                                    items: SDType.values
                                                        .map((SDType type) {
                                                      return DropdownMenuItem<
                                                          SDType>(
                                                        value: type,
                                                        child: Text(
                                                            type.displayName),
                                                      );
                                                    }).toList(),
                                                    onChanged:
                                                        (SDType? newValue) {
                                                      if (newValue != null) {
                                                        setState(() {
                                                          dialogType = newValue;
                                                        });
                                                      }
                                                    },
                                                  ),
                                                  const SizedBox(height: 10),
                                                  const Text('Schedule'),
                                                  DropdownButton<Schedule>(
                                                    value: dialogSchedule,
                                                    isExpanded: true,
                                                    items: Schedule.values
                                                        .map((schedule) {
                                                      return DropdownMenuItem<
                                                          Schedule>(
                                                        value: schedule,
                                                        child: Text(schedule
                                                            .displayName),
                                                      );
                                                    }).toList(),
                                                    onChanged:
                                                        (Schedule? newValue) {
                                                      setState(() {
                                                        dialogSchedule =
                                                            newValue!;
                                                      });
                                                    },
                                                  ),
                                                ],
                                              ),
                                              actions: <Widget>[
                                                TextButton(
                                                  onPressed: () =>
                                                      Navigator.pop(context),
                                                  child: const Text('Cancel'),
                                                ),
                                                TextButton(
                                                  onPressed: () {
                                                    Navigator.pop(context, (
                                                      false,
                                                      dialogType,
                                                      dialogSchedule
                                                    ));
                                                  },
                                                  child: const Text(
                                                      'Without Flash Attention'),
                                                ),
                                                TextButton(
                                                  onPressed: () =>
                                                      Navigator.pop(context, (
                                                    true,
                                                    dialogType,
                                                    dialogSchedule
                                                  )),
                                                  child: const Text(
                                                      'With Flash Attention'),
                                                ),
                                              ],
                                            );
                                          },
                                        );
                                      },
                                    );
                                    if (result != null) {
                                      final (
                                        useFlashAttention,
                                        selectedType,
                                        selectedSchedule
                                      ) = result;
                                      StableDiffusionService.setModelConfig(
                                          useFlashAttention,
                                          selectedType,
                                          selectedSchedule);
                                      final initResult =
                                          await StableDiffusionService
                                              .pickAndInitializeModel();
                                      setState(() {
                                        _message = initResult;
                                        _taesdError = '';
                                      });
                                    }
                                  },
                            child: snapshot.data == true
                                ? const SizedBox(
                                    width: 20,
                                    height: 20,
                                    child: CircularProgressIndicator(
                                        color: Colors.white))
                                : const Text('Initialize Model'),
                          );
                        }),
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
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('Sampling Method'),
                        DropdownButton<SampleMethod>(
                          value: _selectedSampleMethod,
                          isExpanded: true,
                          items: SampleMethod.values.map((method) {
                            return DropdownMenuItem<SampleMethod>(
                              value: method,
                              child: Text(method.displayName),
                            );
                          }).toList(),
                          onChanged: (SampleMethod? newValue) {
                            setState(() {
                              _selectedSampleMethod = newValue!;
                            });
                          },
                        ),
                      ],
                    ),
                  ),
                ],
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
              if (_progressMessage.isNotEmpty)
                Text(
                  _progressMessage,
                  style: const TextStyle(fontSize: 16),
                ),
              if (_totalTime.isNotEmpty)
                Text(
                  _totalTime,
                  style: const TextStyle(fontSize: 16),
                ),
              const SizedBox(height: 20),
              StreamBuilder<bool>(
                  stream: StableDiffusionService.loadingStream,
                  builder: (context, snapshot) {
                    return ElevatedButton(
                      onPressed: snapshot.data == true
                          ? null
                          : () async {
                              setState(() {
                                _message = 'Generating image...';
                                _progressMessage = '';
                                _totalTime = '';
                              });
                              final image =
                                  await StableDiffusionService.generateImage(
                                prompt: _promptController.text,
                                negativePrompt: _negativePromptController.text,
                                cfgScale: _cfgScale,
                                sampleSteps: _steps,
                                width: _width,
                                height: _height,
                                seed: _seed,
                                sampleMethod: _selectedSampleMethod.index,
                              );

                              if (image != null) {
                                final bytes = await image.toByteData(
                                    format: ui.ImageByteFormat.png);
                                setState(() {
                                  _generatedImage =
                                      Image.memory(bytes!.buffer.asUint8List());
                                  _message = 'Generation complete';
                                });
                              }
                            },
                      child: snapshot.data == true
                          ? const SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(
                                  color: Colors.white))
                          : const Text('Generate Image'),
                    );
                  }),
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
