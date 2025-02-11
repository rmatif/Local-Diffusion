import 'dart:io';
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
  String? _clipLPath;
  String? _clipGPath;
  String? _t5xxlPath;
  String? _vaePath;
  String? _embedDirPath;
  int _clipSkip = 1;
  bool _vaeTiling = false;

  Future<String> getModelDirectory() async {
    final directory = Directory('/storage/emulated/0/Local Diffusion/Models');
    if (!await directory.exists()) {
      await directory.create(recursive: true);
    }
    return directory.path;
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
    _processor?.dispose();
    _processor = StableDiffusionProcessor(
      modelPath: modelPath,
      useFlashAttention: useFlashAttention,
      modelType: modelType,
      schedule: schedule,
      loraPath: _loraPath,
      taesdPath: _taesdPath,
      useTinyAutoencoder: _useTinyAutoencoder,
      clipLPath: _clipLPath,
      clipGPath: _clipGPath,
      t5xxlPath: _t5xxlPath,
      vaePath: _vaePath,
      embedDirPath: _embedDirPath,
      clipSkip: _clipSkip,
      vaeTiling: _vaeTiling,
      onModelLoaded: () {
        setState(() {
          _message = 'Model initialized successfully';
        });
      },
      onLog: (log) {
        if (log.message.contains('total params memory size')) {
          final regex = RegExp(r'total params memory size = ([\d.]+)MB');
          final match = regex.firstMatch(log.message);
          if (match != null) {
            setState(() {
              _ramUsage = 'Total RAM Usage: ${match.group(1)}MB';
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
        developer.log(
            'Progress: ${(progress.progress * 100).toInt()}% (Step ${progress.step}/${progress.totalSteps}, ${progress.time.toStringAsFixed(1)}s)');
      },
    );

    _processor!.imageStream.listen((image) async {
      // Convert ui.Image to bytes using toByteData
      final bytes = await image.toByteData(format: ui.ImageByteFormat.png);

      setState(() {
        _generatedImage = Image.memory(bytes!.buffer.asUint8List());
        _message = 'Generation complete';
      });

      // Save the image
      final saveResult = await _processor!.saveGeneratedImage(
        image,
        _promptController.text,
        _width,
        _height,
        _selectedSampleMethod,
      );

      setState(() {
        _message = saveResult;
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
                      onPressed: () async {
                        if (_processor != null) {
                          _processor!.dispose();
                          _processor = null;
                        }
                        final modelDirPath = await getModelDirectory();
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
                                        items: SDType.values.map((SDType type) {
                                          return DropdownMenuItem<SDType>(
                                            value: type,
                                            child: Text(type.displayName),
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
                                        items: Schedule.values.map((schedule) {
                                          return DropdownMenuItem<Schedule>(
                                            value: schedule,
                                            child: Text(schedule.displayName),
                                          );
                                        }).toList(),
                                        onChanged: (Schedule? newValue) {
                                          setState(() {
                                            dialogSchedule = newValue!;
                                          });
                                        },
                                      ),
                                    ],
                                  ),
                                  actions: <Widget>[
                                    TextButton(
                                      onPressed: () => Navigator.pop(context),
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
                                      child:
                                          const Text('Without Flash Attention'),
                                    ),
                                    TextButton(
                                      onPressed: () => Navigator.pop(context,
                                          (true, dialogType, dialogSchedule)),
                                      child: const Text('With Flash Attention'),
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

                          // Use directory selection first
                          final selectedDir = await FilePicker.platform
                              .getDirectoryPath(initialDirectory: modelDirPath);

                          if (selectedDir != null) {
                            // Then show a dialog to select from available model files
                            final directory = Directory(selectedDir);
                            final files = directory.listSync();
                            final modelFiles = files
                                .whereType<File>()
                                .where((file) =>
                                    file.path.endsWith('.safetensors') ||
                                    file.path.endsWith('.ckpt') ||
                                    file.path.endsWith('.gguf'))
                                .toList();

                            if (modelFiles.isNotEmpty) {
                              final selectedModel = await showDialog<String>(
                                context: context,
                                builder: (BuildContext context) {
                                  return AlertDialog(
                                    title: const Text('Select Model'),
                                    content: SingleChildScrollView(
                                      child: Column(
                                        children: modelFiles
                                            .map((file) => ListTile(
                                                  title: Text(file.path
                                                      .split('/')
                                                      .last),
                                                  onTap: () => Navigator.pop(
                                                      context, file.path),
                                                ))
                                            .toList(),
                                      ),
                                    ),
                                  );
                                },
                              );

                              if (selectedModel != null) {
                                _initializeProcessor(
                                    selectedModel,
                                    useFlashAttention,
                                    selectedType,
                                    selectedSchedule);
                              }
                            }
                          }
                        }
                      },
                      child: const Text('Initialize Model'),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: ElevatedButton(
                      onPressed: () async {
                        final result =
                            await FilePicker.platform.getDirectoryPath();
                        if (result != null) {
                          setState(() {
                            _loraPath = result;
                            _loraMessage =
                                "LORA directory loaded: ${result.split('/').last}";
                            if (_processor != null) {
                              String currentModelPath = _processor!.modelPath;
                              bool currentFlashAttention =
                                  _processor!.useFlashAttention;
                              SDType currentModelType = _processor!.modelType;
                              Schedule currentSchedule = _processor!.schedule;

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
              ExpansionTile(
                title: const Text('Advanced Options'),
                children: <Widget>[
                  const SizedBox(height: 10),
                  Row(
                    children: [
                      Checkbox(
                        value: _vaeTiling,
                        onChanged: (bool? value) {
                          setState(() {
                            _vaeTiling = value ?? false;
                            if (_processor != null) {
                              String currentModelPath = _processor!.modelPath;
                              bool currentFlashAttention =
                                  _processor!.useFlashAttention;
                              SDType currentModelType = _processor!.modelType;
                              Schedule currentSchedule = _processor!.schedule;

                              _initializeProcessor(
                                  currentModelPath,
                                  currentFlashAttention,
                                  currentModelType,
                                  currentSchedule);
                            }
                          });
                        },
                      ),
                      const Text('VAE Tiling'),
                    ],
                  ),
                  Row(
                    children: [
                      const Text('Clip Skip:'),
                      Expanded(
                        child: Slider(
                          value: _clipSkip.toDouble(),
                          min: 0,
                          max: 2,
                          divisions: 2,
                          label: _clipSkip.toString(),
                          onChanged: (value) {
                            setState(() {
                              _clipSkip = value.toInt();
                              if (_processor != null) {
                                String currentModelPath = _processor!.modelPath;
                                bool currentFlashAttention =
                                    _processor!.useFlashAttention;
                                SDType currentModelType = _processor!.modelType;
                                Schedule currentSchedule = _processor!.schedule;

                                _initializeProcessor(
                                    currentModelPath,
                                    currentFlashAttention,
                                    currentModelType,
                                    currentSchedule);
                              }
                            });
                          },
                        ),
                      ),
                      Text('$_clipSkip'),
                    ],
                  ),
                  const SizedBox(height: 10),
                  ElevatedButton(
                    onPressed: () async {
                      final modelDirPath = await getModelDirectory();
                      final selectedFile =
                          await pickModelFile(context, modelDirPath, 'CLIP_L');
                      if (selectedFile != null) {
                        setState(() {
                          _clipLPath = selectedFile;
                          if (_processor != null) {
                            String currentModelPath = _processor!.modelPath;
                            bool currentFlashAttention =
                                _processor!.useFlashAttention;
                            SDType currentModelType = _processor!.modelType;
                            Schedule currentSchedule = _processor!.schedule;

                            _initializeProcessor(
                                currentModelPath,
                                currentFlashAttention,
                                currentModelType,
                                currentSchedule);
                          }
                        });
                      }
                    },
                    child: Text(_clipLPath == null
                        ? 'Load CLIP_L'
                        : 'CLIP_L: ${_clipLPath!.split('/').last}'),
                  ),
                  ElevatedButton(
                    onPressed: () async {
                      final modelDirPath = await getModelDirectory();
                      final selectedFile =
                          await pickModelFile(context, modelDirPath, 'CLIP_G');
                      if (selectedFile != null) {
                        setState(() {
                          _clipGPath = selectedFile;
                          if (_processor != null) {
                            String currentModelPath = _processor!.modelPath;
                            bool currentFlashAttention =
                                _processor!.useFlashAttention;
                            SDType currentModelType = _processor!.modelType;
                            Schedule currentSchedule = _processor!.schedule;

                            _initializeProcessor(
                                currentModelPath,
                                currentFlashAttention,
                                currentModelType,
                                currentSchedule);
                          }
                        });
                      }
                    },
                    child: Text(_clipGPath == null
                        ? 'Load CLIP_G'
                        : 'CLIP_G: ${_clipGPath!.split('/').last}'),
                  ),
                  ElevatedButton(
                    onPressed: () async {
                      final modelDirPath = await getModelDirectory();
                      final selectedFile =
                          await pickModelFile(context, modelDirPath, 'T5XXL');
                      if (selectedFile != null) {
                        setState(() {
                          _t5xxlPath = selectedFile;
                          if (_processor != null) {
                            String currentModelPath = _processor!.modelPath;
                            bool currentFlashAttention =
                                _processor!.useFlashAttention;
                            SDType currentModelType = _processor!.modelType;
                            Schedule currentSchedule = _processor!.schedule;

                            _initializeProcessor(
                                currentModelPath,
                                currentFlashAttention,
                                currentModelType,
                                currentSchedule);
                          }
                        });
                      }
                    },
                    child: Text(_t5xxlPath == null
                        ? 'Load T5XXL'
                        : 'T5XXL: ${_t5xxlPath!.split('/').last}'),
                  ),
                  ElevatedButton(
                    onPressed: () async {
                      final modelDirPath = await getModelDirectory();
                      final selectedFile =
                          await pickModelFile(context, modelDirPath, 'VAE');
                      if (selectedFile != null) {
                        setState(() {
                          _vaePath = selectedFile;
                          if (_processor != null) {
                            String currentModelPath = _processor!.modelPath;
                            bool currentFlashAttention =
                                _processor!.useFlashAttention;
                            SDType currentModelType = _processor!.modelType;
                            Schedule currentSchedule = _processor!.schedule;

                            _initializeProcessor(
                                currentModelPath,
                                currentFlashAttention,
                                currentModelType,
                                currentSchedule);
                          }
                        });
                      }
                    },
                    child: Text(_vaePath == null
                        ? 'Load VAE'
                        : 'VAE: ${_vaePath!.split('/').last}'),
                  ),
                  ElevatedButton(
                    onPressed: () async {
                      final result =
                          await FilePicker.platform.getDirectoryPath();
                      if (result != null) {
                        setState(() {
                          _embedDirPath = result;
                          if (_processor != null) {
                            String currentModelPath = _processor!.modelPath;
                            bool currentFlashAttention =
                                _processor!.useFlashAttention;
                            SDType currentModelType = _processor!.modelType;
                            Schedule currentSchedule = _processor!.schedule;

                            _initializeProcessor(
                                currentModelPath,
                                currentFlashAttention,
                                currentModelType,
                                currentSchedule);
                          }
                        });
                      }
                    },
                    child: Text(_embedDirPath == null
                        ? 'Load Embeddings Dir'
                        : 'Embeddings Dir: ${_embedDirPath!.split('/').last}'),
                  ),
                  const SizedBox(height: 10),
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
                            if (_processor != null) {
                              String currentModelPath = _processor!.modelPath;
                              bool currentFlashAttention =
                                  _processor!.useFlashAttention;
                              SDType currentModelType = _processor!.modelType;
                              Schedule currentSchedule = _processor!.schedule;

                              _initializeProcessor(
                                  currentModelPath,
                                  currentFlashAttention,
                                  currentModelType,
                                  currentSchedule);
                            }
                          });
                        },
                      ),
                      const Text('Use Tiny AutoEncoder'),
                      const Spacer(),
                      ElevatedButton(
                        onPressed: () async {
                          final modelDirPath = await getModelDirectory();
                          final selectedDir = await FilePicker.platform
                              .getDirectoryPath(initialDirectory: modelDirPath);

                          if (selectedDir != null) {
                            final directory = Directory(selectedDir);
                            final files = directory.listSync();
                            final taesdFiles = files
                                .whereType<File>()
                                .where((file) =>
                                    file.path.endsWith('.safetensors') ||
                                    file.path.endsWith('.bin'))
                                .toList();

                            if (taesdFiles.isNotEmpty) {
                              final selectedTaesd = await showDialog<String>(
                                context: context,
                                builder: (BuildContext context) {
                                  return AlertDialog(
                                    title: const Text('Select TAESD Model'),
                                    content: SingleChildScrollView(
                                      child: Column(
                                        children: taesdFiles
                                            .map((file) => ListTile(
                                                  title: Text(file.path
                                                      .split('/')
                                                      .last),
                                                  onTap: () => Navigator.pop(
                                                      context, file.path),
                                                ))
                                            .toList(),
                                      ),
                                    ),
                                  );
                                },
                              );

                              if (selectedTaesd != null) {
                                setState(() {
                                  _taesdPath = selectedTaesd;
                                  _taesdMessage =
                                      "TAESD loaded: ${selectedTaesd.split('/').last}";
                                  _taesdError = '';
                                });
                              }
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
                onPressed: () {
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
                    print("Processor is null!");
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

  Future<String?> pickModelFile(
      BuildContext context, String modelDirPath, String modelType) async {
    final selectedDir = await FilePicker.platform
        .getDirectoryPath(initialDirectory: modelDirPath);
    if (selectedDir != null) {
      final directory = Directory(selectedDir);
      final files = directory.listSync();
      final modelFiles = files
          .whereType<File>()
          .where((file) =>
              file.path.endsWith('.safetensors') ||
              file.path.endsWith('.ckpt') ||
              file.path.endsWith('.gguf'))
          .toList();

      if (modelFiles.isNotEmpty) {
        return await showDialog<String>(
          context: context,
          builder: (BuildContext context) {
            return AlertDialog(
              title: Text('Select $modelType Model'),
              content: SingleChildScrollView(
                child: Column(
                  children: modelFiles
                      .map((file) => ListTile(
                            title: Text(file.path.split('/').last),
                            onTap: () => Navigator.pop(context, file.path),
                          ))
                      .toList(),
                ),
              ),
            );
          },
        );
      }
    }
    return null;
  }
}
