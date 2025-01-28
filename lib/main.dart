import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'dart:async';
import 'dart:developer' as developer;
import 'stable_diffusion_service.dart';
import 'ffi_bindings.dart';
import 'stable_diffusion_processor.dart';
import 'package:file_picker/file_picker.dart';

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
  bool _isModelInitializing = false;
  bool _isGeneratingImage = false;
  bool _waitForRamUsage = false;
  bool get _isBusy =>
      _isModelInitializing || _isGeneratingImage || _waitForRamUsage;

  final TextEditingController _promptController = TextEditingController();
  final TextEditingController _negativePromptController =
      TextEditingController();
  Timer? _errorMessageTimer;
  StableDiffusionProcessor? _processor;

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
  String? _taesdPath;
  String? _loraPath;
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
    _cores = FFIBindings.getCores() * 2;
  }

  @override
  void dispose() {
    _promptController.dispose();
    _negativePromptController.dispose();
    _errorMessageTimer?.cancel();
    _processor?.dispose();
    super.dispose();
  }

  void _initializeProcessor(String modelPath, bool useFlashAttention,
      SDType modelType, Schedule schedule) {
    setState(() {
      _isModelInitializing = true;
      _waitForRamUsage = true;
      _message = 'Initializing model...';
    });

    _processor?.dispose();
    _processor = StableDiffusionProcessor(
      modelPath: modelPath,
      useFlashAttention: useFlashAttention,
      modelType: modelType,
      schedule: schedule,
      loraPath: _loraPath,
      taesdPath: _taesdPath,
      useTinyAutoencoder: _useTinyAutoencoder,
      onModelLoaded: () {
        setState(() {
          _message = 'Model initialized';
        });
      },
      onLog: (log) {
        if (log.message.contains('total params memory size')) {
          final regex = RegExp(r'total params memory size = ([\d.]+)MB');
          final match = regex.firstMatch(log.message);
          if (match != null) {
            setState(() {
              _ramUsage = 'Total RAM Usage: ${match.group(1)}MB';
              _waitForRamUsage = false;
              _isModelInitializing = false;
              _message = 'Model ready';
            });
          }
        }
        developer.log(log.message);
      },
      onProgress: (progress) {
        setState(() {
          _progressMessage =
              'Progress: ${(progress.progress * 100).toInt()}% (Step ${progress.step}/${progress.totalSteps}, ${progress.time.toStringAsFixed(1)}s)';
        });
      },
    );

    _processor!.imageStream.listen((image) async {
      final bytes = await image.toByteData(format: ui.ImageByteFormat.png);

      setState(() {
        _generatedImage = Image.memory(bytes!.buffer.asUint8List());
        _message = 'Generation complete';
      });

      final saveResult = await _processor!.saveGeneratedImage(
        image,
        _promptController.text,
        _width,
        _height,
        _selectedSampleMethod,
      );

      setState(() {
        _message = saveResult;
        _isGeneratingImage = false;
      });
    });
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
                    child: ElevatedButton(
                      onPressed: _isBusy
                          ? null
                          : () async {
                              setState(() => _isModelInitializing = true);
                              try {
                                final result =
                                    await showDialog<(bool, SDType, Schedule)>(
                                  context: context,
                                  builder: (BuildContext context) {
                                    Schedule dialogSchedule = Schedule.DISCRETE;
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
                                                    child:
                                                        Text(type.displayName),
                                                  );
                                                }).toList(),
                                                onChanged: (SDType? newValue) {
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
                                                    child: Text(
                                                        schedule.displayName),
                                                  );
                                                }).toList(),
                                                onChanged:
                                                    (Schedule? newValue) {
                                                  setState(() {
                                                    dialogSchedule = newValue!;
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
                                              onPressed: () => Navigator.pop(
                                                  context, (
                                                false,
                                                dialogType,
                                                dialogSchedule
                                              )),
                                              child: const Text(
                                                  'Without Flash Attention'),
                                            ),
                                            TextButton(
                                              onPressed: () => Navigator.pop(
                                                  context, (
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
                                  final pickedFile = await FilePicker.platform
                                      .pickFiles(
                                          type: FileType.any,
                                          allowMultiple: false);

                                  if (pickedFile != null) {
                                    final modelPath =
                                        pickedFile.files.single.path!;
                                    _initializeProcessor(
                                        modelPath,
                                        useFlashAttention,
                                        selectedType,
                                        selectedSchedule);
                                  }
                                }
                              } finally {
                                setState(() => _isModelInitializing = false);
                              }
                            },
                      child: _isModelInitializing
                          ? const Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                SizedBox(
                                  width: 16,
                                  height: 16,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    valueColor: AlwaysStoppedAnimation<Color>(
                                        Colors.white),
                                  ),
                                ),
                                SizedBox(width: 8),
                                Text('Initializing...'),
                              ],
                            )
                          : const Text('Initialize Model'),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: ElevatedButton(
                      onPressed: _isBusy
                          ? null
                          : () async {
                              final result =
                                  await FilePicker.platform.getDirectoryPath();
                              if (result != null) {
                                setState(() {
                                  _loraPath = result;
                                  _loraMessage =
                                      "LORA directory loaded: ${result.split('/').last}";
                                  if (_processor != null) {
                                    String currentModelPath =
                                        _processor!.modelPath;
                                    bool currentFlashAttention =
                                        _processor!.useFlashAttention;
                                    SDType currentModelType =
                                        _processor!.modelType;
                                    Schedule currentSchedule =
                                        _processor!.schedule;

                                    _initializeProcessor(
                                        currentModelPath,
                                        currentFlashAttention,
                                        currentModelType,
                                        currentSchedule);
                                  }
                                });
                              }
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
                          if (_taesdPath == null) {
                            _showTemporaryError(
                                'Please load TAESD model first');
                            return;
                          }
                          setState(() {
                            _useTinyAutoencoder = value ?? false;
                            _isModelInitializing = true;
                            _waitForRamUsage = true;
                          });

                          if (_processor != null) {
                            _initializeProcessor(
                              _processor!.modelPath,
                              _processor!.useFlashAttention,
                              _processor!.modelType,
                              _processor!.schedule,
                            );
                          } else {
                            setState(() {
                              _waitForRamUsage = false;
                              _isModelInitializing = false;
                            });
                          }
                        },
                      ),
                      const Text('Use Tiny AutoEncoder'),
                      const Spacer(),
                      ElevatedButton(
                        onPressed: _isBusy
                            ? null
                            : () async {
                                setState(() {
                                  _isModelInitializing = true;
                                  _waitForRamUsage = true;
                                });

                                try {
                                  final result = await FilePicker.platform
                                      .pickFiles(
                                          type: FileType.any,
                                          allowMultiple: false);

                                  if (result != null) {
                                    setState(() {
                                      _taesdPath = result.files.single.path!;
                                      _taesdMessage =
                                          "TAESD loaded: ${result.files.single.name}";
                                      _taesdError = '';
                                    });

                                    if (_processor != null) {
                                      _initializeProcessor(
                                        _processor!.modelPath,
                                        _processor!.useFlashAttention,
                                        _processor!.modelType,
                                        _processor!.schedule,
                                      );
                                    }
                                  }
                                } finally {
                                  if (_processor == null) {
                                    setState(() {
                                      _waitForRamUsage = false;
                                      _isModelInitializing = false;
                                    });
                                  }
                                }
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
              Text('Status: $_message', style: const TextStyle(fontSize: 16)),
              Text('LORA: $_loraMessage', style: const TextStyle(fontSize: 16)),
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
              ElevatedButton(
                onPressed: _isBusy
                    ? null
                    : () {
                        setState(() => _isGeneratingImage = true);
                        if (_processor != null) {
                          print("Sending generation request to processor");
                          setState(() => _message = 'Generating image...');
                          _processor!.generateImage(
                            prompt: _promptController.text,
                            negativePrompt: _negativePromptController.text,
                            cfgScale: _cfgScale,
                            sampleSteps: _steps,
                            width: _width,
                            height: _height,
                            seed: _seed,
                            sampleMethod: _selectedSampleMethod.index,
                          );
                        } else {
                          setState(() => _isGeneratingImage = false);
                          print("Processor is null!");
                        }
                      },
                child: _isGeneratingImage
                    ? const Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              valueColor:
                                  AlwaysStoppedAnimation<Color>(Colors.white),
                            ),
                          ),
                          SizedBox(width: 8),
                          Text('Generating...'),
                        ],
                      )
                    : const Text('Generate Image'),
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
