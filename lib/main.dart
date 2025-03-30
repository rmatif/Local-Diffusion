import 'dart:io';
import 'dart:typed_data';
import 'package:dotted_border/dotted_border.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:ui' as ui;
import 'dart:async';
import 'dart:developer' as developer;
import 'package:shadcn_ui/shadcn_ui.dart';
import 'package:file_picker/file_picker.dart';
import 'ffi_bindings.dart';
import 'inpainting_page.dart';
import 'stable_diffusion_processor.dart';
import 'upscaler_page.dart';
import 'img2img_page.dart';
import 'photomaker_page.dart';
import 'package:image/image.dart' as img;
import 'canny_processor.dart';
import 'scribble2img_page.dart';
import 'outpainting_page.dart';
import 'image_processing_utils.dart'; // Import the new utility

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return ShadApp(
      darkTheme: ShadThemeData(
        brightness: Brightness.dark,
        colorScheme: const ShadSlateColorScheme.dark(),
      ),
      home: const StableDiffusionApp(),
    );
  }
}

class StableDiffusionApp extends StatefulWidget {
  const StableDiffusionApp({super.key});
  @override
  State<StableDiffusionApp> createState() => _StableDiffusionAppState();
}

class _StableDiffusionAppState extends State<StableDiffusionApp>
    with SingleTickerProviderStateMixin {
  Timer? _modelErrorTimer;
  Timer? _errorMessageTimer;
  StableDiffusionProcessor? _processor;
  Image? _generatedImage;
  bool isModelLoading = false;
  bool isGenerating = false;

  // Status messages
  String _message = '';
  String _loraMessage = '';
  String _taesdMessage = '';
  String _taesdError = '';
  String _ramUsage = '';
  String _progressMessage = '';
  String _totalTime = '';
  int _cores = 0;
  List<String> _loraNames = [];
  final TextEditingController _promptController = TextEditingController();
  final Map<String, OverlayEntry?> _overlayEntries = {};
  final GlobalKey _promptFieldKey = GlobalKey();
  final Map<String, GlobalKey> _loraKeys = {};
  // UI State variables
  bool useTAESD = false;
  bool useVAETiling = false;
  double clipSkip = 0;
  bool useVAE = false;
  String samplingMethod = 'euler_a';
  double cfg = 7;
  int steps = 25;
  int width = 512;
  int height = 512;
  String seed = "-1";
  String prompt = '';
  String negativePrompt = '';
  double progress = 0;
  String status = '';
  Map<String, bool> loadedComponents = {};
  String loadingText = '';
  String _modelError = '';
  List<String> _generationLogs = []; // To store logs for the last generation
  bool _showLogsButton = false; // To control visibility of the log button

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

  // Path variables
  String? _taesdPath;
  String? _loraPath;
  String? _clipLPath;
  String? _clipGPath;
  String? _t5xxlPath;
  String? _vaePath;
  String? _embedDirPath;
  String? _controlNetPath; // Add this
  File? _controlImage; // Add this
  Uint8List? _controlRgbBytes; // Add this
  int? _controlWidth; // Add this
  int? _controlHeight; // Add this
  bool useControlNet = false; // Add this
  bool useControlImage = false; // Add this
  bool useCanny = false; // Add this
  double controlStrength = 0.9;
  CannyProcessor? _cannyProcessor;
  bool isCannyProcessing = false;
  Image? _cannyImage;
  String _controlImageProcessingMode = 'Resize'; // 'Resize' or 'Crop'

  final List<String> samplingMethods = const [
    'euler_a',
    'euler',
    'heun',
    'dpm2',
    'dmp ++2s_a',
    'dmp++2m',
    'dpm++2mv2',
    'ipndm',
    'ipndm_v',
    'lcm',
    'ddim_trailing', // New sampler
    'tcd' // New sampler
  ];

  List<int> getWidthOptions() {
    List<int> opts = [];
    for (int i = 128; i <= 512; i += 64) {
      opts.add(i);
    }
    return opts;
  }

  List<int> getHeightOptions() {
    return getWidthOptions();
  }

  @override
  void initState() {
    super.initState();
    _cores = FFIBindings.getCores() * 2;
    _cannyProcessor = CannyProcessor();
    _cannyProcessor!.init();

    _cannyProcessor!.loadingStream.listen((loading) {
      setState(() {
        isCannyProcessing = loading;
      });
    });

    _cannyProcessor!.imageStream.listen((image) async {
      final bytes = await image.toByteData(format: ui.ImageByteFormat.png);

      setState(() {
        _cannyImage = Image.memory(bytes!.buffer.asUint8List());
      });
    });
  }

  @override
  void dispose() {
    _errorMessageTimer?.cancel();
    _modelErrorTimer?.cancel();
    _processor?.dispose();
    _cannyProcessor?.dispose();
    _promptController.dispose(); // Dispose the text controller
    super.dispose();
  }

  Future<String> getModelDirectory() async {
    final directory = Directory('/storage/emulated/0/Local Diffusion/Models');
    if (!await directory.exists()) {
      await directory.create(recursive: true);
    }
    return directory.path;
  }

  Uint8List _ensureRgbFormat(Uint8List bytes, int width, int height) {
    // If the bytes length matches a 3-channel image, assume it's already in RGB format
    if (bytes.length == width * height * 3) {
      return bytes;
    }

    // If it's a grayscale image (1 channel)
    if (bytes.length == width * height) {
      final rgbBytes = Uint8List(width * height * 3);
      for (int i = 0; i < width * height; i++) {
        // Convert grayscale to RGB by duplicating the value to all channels
        rgbBytes[i * 3] = bytes[i];
        rgbBytes[i * 3 + 1] = bytes[i];
        rgbBytes[i * 3 + 2] = bytes[i];
      }
      return rgbBytes;
    }

    // If it's already in a different format, log the issue
    print(
        "Warning: Unexpected image format. Expected 1 or 3 channels, got: ${bytes.length / (width * height)} channels");

    // As a fallback, try to interpret it as is
    return bytes;
  }

  Future<void> _processCannyImage() async {
    if (_controlImage == null) return;

    final bytes = await _controlImage!.readAsBytes();
    final decodedImage = img.decodeImage(bytes);

    if (decodedImage == null) {
      _showTemporaryError('Failed to decode image');
      return;
    }

    // Convert to RGB format (3 channels)
    final rgbBytes = Uint8List(decodedImage.width * decodedImage.height * 3);
    int rgbIndex = 0;

    for (int y = 0; y < decodedImage.height; y++) {
      for (int x = 0; x < decodedImage.width; x++) {
        final pixel = decodedImage.getPixel(x, y);
        rgbBytes[rgbIndex] = pixel.r.toInt();
        rgbBytes[rgbIndex + 1] = pixel.g.toInt();
        rgbBytes[rgbIndex + 2] = pixel.b.toInt();
        rgbIndex += 3;
      }
    }

    // Process with Canny
    await _cannyProcessor!.processImage(
      rgbBytes,
      decodedImage.width,
      decodedImage.height,
      CannyParameters(
        highThreshold: 100.0,
        lowThreshold: 50.0,
        weak: 1.0,
        strong: 255.0,
        inverse: false,
      ),
    );
  }

  void _initializeProcessor(String modelPath, bool useFlashAttention,
      SDType modelType, Schedule schedule) {
    setState(() {
      isModelLoading = true;
      loadingText = 'Loading Model...'; // Set loading text immediately
    });
    _processor?.dispose();
    _processor = StableDiffusionProcessor(
      modelPath: modelPath,
      useFlashAttention: useFlashAttention,
      modelType: modelType,
      schedule: schedule,
      loraPath: _loraPath,
      taesdPath: _taesdPath,
      useTinyAutoencoder: useTAESD,
      clipLPath: _clipLPath,
      clipGPath: _clipGPath,
      t5xxlPath: _t5xxlPath,
      vaePath: _vaePath,
      embedDirPath: _embedDirPath,
      clipSkip: clipSkip.toInt(),
      vaeTiling: useVAETiling,
      controlNetPath: _controlNetPath, // Add this
      controlImageData: _controlRgbBytes, // Add this
      controlImageWidth: _controlWidth, // Add this
      controlImageHeight: _controlHeight, // Add this // Add this
      controlStrength: controlStrength, // Add this
      onModelLoaded: () {
        setState(() {
          isModelLoading = false;
          _message = 'Model initialized successfully';
          loadedComponents['Model'] = true;
          loadingText = '';
        });
      },
      onLog: (log) {
        if (log.message.contains('total params memory size')) {
          final regex = RegExp(r'total params memory size = ([\d.]+)MB');
          final match = regex.firstMatch(log.message);
          if (match != null) {
            setState(() {
              _ramUsage = 'Total RAM: ${match.group(1)}MB';
            });
          }
        }
        developer.log(log.message);
      },
      onProgress: (progress) {
        setState(() {
          this.progress = progress.progress;
          status =
              'Generating image... ${(progress.progress * 100).toInt()}% • Step ${progress.step}/${progress.totalSteps} • ${progress.time.toStringAsFixed(1)}s';
        });
      },
    );

    // Listen for the collected logs after generation
    _processor!.logListStream.listen((logs) {
      setState(() {
        _generationLogs = logs;
      });
    });

    _processor!.imageStream.listen((image) async {
      final bytes = await image.toByteData(format: ui.ImageByteFormat.png);

      setState(() {
        isGenerating = false;
        _generatedImage = Image.memory(bytes!.buffer.asUint8List());
        status = 'Generation complete';
        _showLogsButton = true; // Show the log button
      });

      await _processor!.saveGeneratedImage(
        image,
        prompt,
        width,
        height,
        SampleMethod.values.firstWhere(
          (method) =>
              method.displayName.toLowerCase() == samplingMethod.toLowerCase(),
          orElse: () => SampleMethod.EULER_A,
        ),
      );

      setState(() {
        status = 'Generation complete';
      });
    });
  }

  void simulateLoading(String component) {
    setState(() {
      loadingText = 'Loading $component...';
    });

    if (component == 'Model') {
      // Model loading is handled separately in showModelLoadDialog
      return;
    }

    // For other components, simulate loading
    Future.delayed(const Duration(seconds: 2), () {
      setState(() {
        loadedComponents[component] = true;
        loadingText = '';

        // Set appropriate paths based on component
        switch (component) {
          case 'TAESD':
            _taesdPath = 'simulated/path/to/taesd';
            _taesdMessage = "TAESD loaded successfully";
            break;
          case 'Lora':
            _loraPath = 'simulated/path/to/lora';
            _loraMessage = "LORA loaded successfully";
            break;
          case 'Clip_L':
            _clipLPath = 'simulated/path/to/clip_l';
            break;
          case 'Clip_G':
            _clipGPath = 'simulated/path/to/clip_g';
            break;
          case 'T5XXL':
            _t5xxlPath = 'simulated/path/to/t5xxl';
            break;
          case 'VAE':
            _vaePath = 'simulated/path/to/vae';
            break;
          case 'Embeddings':
            _embedDirPath = 'simulated/path/to/embeddings';
            break;
        }
      });
    });
  }

  void showModelLoadDialog() {
    String selectedQuantization = 'NONE';
    String selectedSchedule = 'DEFAULT';
    bool useFlashAttention = false;

    final List<String> quantizationOptions = [
      'NONE',
      'Q8_0',
      'Q8_1',
      'Q8_K',
      'Q6_K',
      'Q5_0',
      'Q5_1',
      'Q5_K',
      'Q4_0',
      'Q4_1',
      'Q4_K',
      'Q3_K',
      'Q2_K'
    ];

    final List<String> scheduleOptions = [
      'DEFAULT',
      'DISCRETE',
      'KARRAS',
      'EXPONENTIAL',
      'AYS'
    ];

    showShadDialog(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setState) => ShadDialog.alert(
          constraints: const BoxConstraints(maxWidth: 300),
          title: const Text('Load Model Settings'),
          description: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Text('Quantization Type:'),
                  const SizedBox(width: 8),
                  Expanded(
                    child: ShadSelect<String>(
                      placeholder: Text(selectedQuantization),
                      onChanged: (value) => setState(
                          () => selectedQuantization = value ?? 'NONE'),
                      options: quantizationOptions
                          .map((type) => ShadOption(
                                value: type,
                                child: Text(type),
                              ))
                          .toList(),
                      selectedOptionBuilder: (context, value) => Text(value),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              Row(
                children: [
                  const Text('Schedule:'),
                  const SizedBox(width: 8),
                  Expanded(
                    child: ShadSelect<String>(
                      placeholder: Text(selectedSchedule),
                      onChanged: (value) =>
                          setState(() => selectedSchedule = value ?? 'DEFAULT'),
                      options: scheduleOptions
                          .map((schedule) => ShadOption(
                                value: schedule,
                                child: Text(schedule),
                              ))
                          .toList(),
                      selectedOptionBuilder: (context, value) => Text(value),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              ShadSwitch(
                value: useFlashAttention,
                onChanged: (v) => setState(() => useFlashAttention = v),
                label: const Text('Use Flash Attention'),
              ),
            ],
          ),
          actions: [
            ShadButton.outline(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Cancel'),
            ),
            ShadButton(
              enabled: !(isModelLoading || isGenerating),
              onPressed: () async {
                final modelDirPath = await getModelDirectory();
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
                    final selectedModel = await showShadDialog<String>(
                      context: context,
                      builder: (BuildContext context) {
                        return ShadDialog.alert(
                          constraints: const BoxConstraints(
                              maxWidth: 400), // Increased dialog width
                          title: const Text('Select Model'),
                          description: SizedBox(
                            height: 300,
                            child: Material(
                              color: Colors.transparent,
                              child: ShadTable.list(
                                header: const [
                                  ShadTableCell.header(
                                    child: Text('Model',
                                        style: TextStyle(fontSize: 16)),
                                  ),
                                  ShadTableCell.header(
                                    alignment: Alignment.centerRight,
                                    child: Text('Size',
                                        style: TextStyle(fontSize: 16)),
                                  ),
                                ],
                                columnSpanExtent: (index) {
                                  if (index == 0) {
                                    return const FixedTableSpanExtent(
                                        250); // Wider model name column
                                  }
                                  if (index == 1) {
                                    return const FixedTableSpanExtent(
                                        80); // Size column
                                  }
                                  return null;
                                },
                                children: modelFiles
                                    .asMap()
                                    .entries
                                    .map(
                                      (entry) => [
                                        ShadTableCell(
                                          child: GestureDetector(
                                            onTap: () => Navigator.pop(
                                                context, entry.value.path),
                                            child: Padding(
                                              padding:
                                                  const EdgeInsets.symmetric(
                                                      vertical:
                                                          12.0), // Taller rows
                                              child: Text(
                                                entry.value.path
                                                    .split('/')
                                                    .last,
                                                style: const TextStyle(
                                                  fontWeight: FontWeight.w500,
                                                  fontSize: 14,
                                                ),
                                              ),
                                            ),
                                          ),
                                        ),
                                        ShadTableCell(
                                          alignment: Alignment.centerRight,
                                          child: GestureDetector(
                                            onTap: () => Navigator.pop(
                                                context, entry.value.path),
                                            child: Text(
                                              '${(entry.value.lengthSync() / (1024 * 1024)).toStringAsFixed(1)} MB',
                                              style:
                                                  const TextStyle(fontSize: 12),
                                            ),
                                          ),
                                        ),
                                      ],
                                    )
                                    .toList(),
                              ),
                            ),
                          ),
                          actions: [
                            ShadButton.outline(
                              onPressed: () => Navigator.pop(context),
                              child: const Text('Cancel'),
                            ),
                          ],
                        );
                      },
                    );

                    if (selectedModel != null) {
                      setState(() => loadingText = 'Loading Model...');
                      _initializeProcessor(
                        selectedModel,
                        useFlashAttention,
                        SDType.values.firstWhere(
                          (type) => type.displayName == selectedQuantization,
                          orElse: () => SDType.NONE,
                        ),
                        Schedule.values.firstWhere(
                          (s) => s.displayName == selectedSchedule,
                          orElse: () => Schedule.DISCRETE,
                        ),
                      );
                    }
                  }
                }
                Navigator.of(context).pop();
              },
              child: const Text('Load Model'),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = ShadTheme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Local Diffusion',
            style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: theme.colorScheme.background,
        elevation: 0,
      ),
      drawer: Drawer(
        width: 240, // Reduced width from default 304
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.horizontal(
              right: Radius.circular(4)), // Less round borders
        ),
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            const DrawerHeader(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [
                    Color.fromRGBO(24, 89, 38, 1), // Teal
                    Color.fromARGB(255, 59, 128, 160), // Blue
                    Color(0xFF0a2335), // Dark Blue
                  ],
                ),
              ),
              child: Text(
                'Menu',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            ListTile(
              leading: const Icon(LucideIcons.type, size: 32),
              title: const Text(
                'Text to Image',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              tileColor: theme.colorScheme.secondary.withOpacity(0.2),
              onTap: () {
                Navigator.pop(context);
              },
            ),
            ListTile(
              leading: const Icon(LucideIcons.images, size: 32),
              title: const Text(
                'Image to Image',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              onTap: () {
                if (_processor != null) {
                  _processor!.dispose();
                  _processor = null;
                }
                Navigator.pop(context);
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(builder: (context) => const Img2ImgPage()),
                );
              },
            ),
            ListTile(
              leading: const Icon(LucideIcons.imageUpscale, size: 32),
              title: const Text(
                'Upscaler',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              onTap: () {
                if (_processor != null) {
                  _processor!.dispose();
                  _processor = null;
                }
                Navigator.pop(context);
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(builder: (context) => const UpscalerPage()),
                );
              },
            ),
            ListTile(
              leading: const Icon(LucideIcons.aperture, size: 32),
              title: const Text('Photomaker',
                  style: TextStyle(fontWeight: FontWeight.bold)),
              onTap: () {
                if (_processor != null) {
                  _processor!.dispose();
                  _processor = null;
                }
                Navigator.pop(context);
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(
                      builder: (context) => const PhotomakerPage()),
                );
              },
            ),
            ListTile(
              leading: const Icon(Icons.draw, size: 32),
              title: const Text('Scribble to Image',
                  style: TextStyle(fontWeight: FontWeight.bold)),
              onTap: () {
                if (_processor != null) {
                  _processor!.dispose();
                  _processor = null;
                }
                Navigator.pop(context);
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(builder: (context) => const ScribblePage()),
                );
              },
            ),
            ListTile(
              leading: const Icon(LucideIcons.palette, size: 32),
              title: const Text('Inpainting',
                  style: TextStyle(fontWeight: FontWeight.bold)),
              onTap: () {
                if (_processor != null) {
                  _processor!.dispose();
                  _processor = null;
                }
                Navigator.pop(context);
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(
                      builder: (context) => const InpaintingPage()),
                );
              },
            ),
            ListTile(
              leading: const Icon(LucideIcons.expand, size: 32),
              title: const Text('Outpainting',
                  style: TextStyle(fontWeight: FontWeight.bold)),
              onTap: () {
                if (_processor != null) {
                  _processor!.dispose();
                  _processor = null;
                }
                Navigator.pop(context);
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(
                      builder: (context) => const OutpaintingPage()),
                );
              },
            ),
          ],
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (loadingText.isNotEmpty || loadedComponents.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(bottom: 16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: loadedComponents.entries
                          .map((entry) => Text.rich(
                                TextSpan(
                                  children: [
                                    TextSpan(
                                      text: '${entry.key} loaded ',
                                      style: theme.textTheme.p.copyWith(
                                        color: Colors.green,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                    const WidgetSpan(
                                      child: Icon(
                                        Icons.check,
                                        size: 20,
                                        color: Colors.green,
                                      ),
                                    ),
                                  ],
                                ),
                              )
                                  .animate()
                                  .fadeIn(
                                      duration:
                                          const Duration(milliseconds: 500))
                                  .slideY(begin: -0.2, end: 0))
                          .toList(),
                    ),
                    if (loadingText.isNotEmpty) const SizedBox(height: 8),
                    if (loadingText.isNotEmpty)
                      TweenAnimationBuilder(
                        duration: const Duration(milliseconds: 800),
                        tween: Tween(begin: 0.0, end: 1.0),
                        builder: (context, value, child) {
                          return Text(
                            '$loadingText${'.' * ((value * 5).floor())}',
                            style: theme.textTheme.p.copyWith(
                              color: Colors.orange,
                              fontWeight: FontWeight.bold,
                            ),
                          );
                        },
                      ).animate().fadeIn(),
                  ],
                ),
              ),
            Row(
              children: [
                ShadButton(
                  enabled: !(isModelLoading || isGenerating),
                  onPressed: showModelLoadDialog,
                  child: const Text('Load Model'),
                ),
                const SizedBox(width: 8),
                if (_ramUsage.isNotEmpty)
                  Text(
                    _ramUsage,
                    style: theme.textTheme.p,
                  ),
                const SizedBox(width: 8),
                /*
                IconButton(
                  icon: const Icon(LucideIcons.circleHelp,
                      size: 24, color: Colors.white),
                  onPressed: () {
                    showShadDialog(
                      context: context,
                      builder: (context) => ShadDialog.alert(
                        title: const Text('Model Information'),
                        constraints: const BoxConstraints(maxWidth: 400),
                        description: const Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('Load an SD or Flux model'),
                            SizedBox(height: 8),
                            Text('Supported models:',
                                style: TextStyle(fontWeight: FontWeight.bold)),
                            Text('\nSD 1.x, SD 2.x, SDXL, SDXL Turbo'),
                            Text('SD 3 Medium/Large, SD 3.5 Medium/Large'),
                            Text('Flux 1 Dev, Flux 1 Schnell, Flux Lite'),
                            SizedBox(height: 16),
                            Text('Supported formats:',
                                style: TextStyle(fontWeight: FontWeight.bold)),
                            Text('\nSafeTensors, CKPT, GGUF'),
                            Text('FP32/FP16 and quantized GGUF formats'),
                            Text(
                                'Distilled formats: Turbo, LCM, Lightning, Hyper'),
                            SizedBox(height: 16),
                            Text('Where to download models?',
                                style: TextStyle(fontWeight: FontWeight.bold)),
                            Text('\nRecommended websites:'),
                            Text('• civitai.com'),
                            Text('• huggingface.co'),
                          ],
                        ),
                        actions: [
                          ShadButton(
                            child: const Text('Close'),
                            onPressed: () => Navigator.of(context).pop(),
                          ),
                        ],
                      ),
                    );
                  },
                ), */
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                ShadButton(
                  enabled: !(isModelLoading || isGenerating),
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
                        final selectedTaesd = await showShadDialog<String>(
                          context: context,
                          builder: (BuildContext context) {
                            return ShadDialog.alert(
                              constraints: const BoxConstraints(maxWidth: 400),
                              title: const Text('Select TAESD Model'),
                              description: SizedBox(
                                height: 300,
                                child: Material(
                                  color: Colors.transparent,
                                  child: ShadTable.list(
                                    header: const [
                                      ShadTableCell.header(
                                        child: Text('Model',
                                            style: TextStyle(fontSize: 16)),
                                      ),
                                      ShadTableCell.header(
                                        alignment: Alignment.centerRight,
                                        child: Text('Size',
                                            style: TextStyle(fontSize: 16)),
                                      ),
                                    ],
                                    columnSpanExtent: (index) {
                                      if (index == 0)
                                        return const FixedTableSpanExtent(250);
                                      if (index == 1)
                                        return const FixedTableSpanExtent(80);
                                      return null;
                                    },
                                    children: taesdFiles
                                        .asMap()
                                        .entries
                                        .map(
                                          (entry) => [
                                            ShadTableCell(
                                              child: GestureDetector(
                                                onTap: () => Navigator.pop(
                                                    context, entry.value.path),
                                                child: Padding(
                                                  padding: const EdgeInsets
                                                      .symmetric(
                                                      vertical: 12.0),
                                                  child: Text(
                                                    entry.value.path
                                                        .split('/')
                                                        .last,
                                                    style: const TextStyle(
                                                      fontWeight:
                                                          FontWeight.w500,
                                                      fontSize: 14,
                                                    ),
                                                  ),
                                                ),
                                              ),
                                            ),
                                            ShadTableCell(
                                              alignment: Alignment.centerRight,
                                              child: GestureDetector(
                                                onTap: () => Navigator.pop(
                                                    context, entry.value.path),
                                                child: Text(
                                                  '${(entry.value.lengthSync() / (1024 * 1024)).toStringAsFixed(1)} MB',
                                                  style: const TextStyle(
                                                      fontSize: 12),
                                                ),
                                              ),
                                            ),
                                          ],
                                        )
                                        .toList(),
                                  ),
                                ),
                              ),
                              actions: [
                                ShadButton.outline(
                                  onPressed: () => Navigator.pop(context),
                                  child: const Text('Cancel'),
                                ),
                              ],
                            );
                          },
                        );

                        if (selectedTaesd != null) {
                          setState(() {
                            _taesdPath = selectedTaesd;
                            loadedComponents['TAESD'] = true;
                            _taesdError = '';

                            // Reinitialize processor if it exists
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
                                currentSchedule,
                              );
                            }
                          });
                        }
                      }
                    }
                  },
                  child: const Text('Load TAESD'),
                ),
                const SizedBox(width: 8),
                // Update the TAESD checkbox onChanged handler:
                ShadCheckbox(
                  value: useTAESD,
                  onChanged: (isModelLoading || isGenerating)
                      ? null
                      : (bool v) {
                          if (useVAETiling && v) {
                            _showTemporaryError(
                                'TAESD is incompatible with VAE Tiling');
                            return;
                          }
                          setState(() {
                            useTAESD = v;
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
                                currentSchedule,
                              );
                            }
                          });
                        },
                  label: const Text('Use TAESD'),
                ),
              ],
            ),
            if (_taesdError.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(left: 8.0, top: 4.0),
                child: Text(
                  _taesdError,
                  style: theme.textTheme.p.copyWith(
                    color: Colors.red,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            const SizedBox(height: 16),
            ShadInput(
              key: _promptFieldKey,
              placeholder: const Text('Prompt'),
              controller: _promptController,
              onChanged: (String? v) => setState(() => prompt = v ?? ''),
            ),
            const SizedBox(height: 16),
            ShadInput(
              placeholder: const Text('Negative Prompt'),
              onChanged: (String? v) =>
                  setState(() => negativePrompt = v ?? ''),
            ),
            const SizedBox(height: 16),
            ShadAccordion<Map<String, dynamic>>(
              children: [
                ShadAccordionItem<Map<String, dynamic>>(
                  value: const {},
                  title: const Text('Advanced Options'),
                  child: Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Column(
                      children: [
                        const SizedBox(height: 16),
                        Row(
                          children: [
                            SizedBox(
                              width: 120,
                              child: ShadButton(
                                enabled: !(isModelLoading || isGenerating),
                                onPressed: () async {
                                  final modelDirPath =
                                      await getModelDirectory();
                                  final selectedDir = await FilePicker.platform
                                      .getDirectoryPath(
                                          initialDirectory: modelDirPath);

                                  if (selectedDir != null) {
                                    final directory = Directory(selectedDir);
                                    final files = directory.listSync();
                                    final loraFiles = files
                                        .whereType<File>()
                                        .where((file) =>
                                            file.path
                                                .endsWith('.safetensors') ||
                                            file.path.endsWith('.pt') ||
                                            file.path.endsWith('.ckpt') ||
                                            file.path.endsWith('.bin') ||
                                            file.path.endsWith('.pth'))
                                        .toList();

                                    setState(() {
                                      _loraPath = selectedDir;
                                      loadedComponents['LORA'] = true;
                                      // Store full list of lora names
                                      _loraNames = loraFiles
                                          .map((file) => file.path
                                              .split('/')
                                              .last
                                              .split('.')
                                              .first)
                                          .toList();

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
                                          currentSchedule,
                                        );
                                      }
                                    });
                                  }
                                },
                                child: const Text('Load Lora'),
                              ),
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Wrap(
                                spacing: 8,
                                runSpacing: 4,
                                children: _loraNames.map((name) {
                                  // Create a key for each Lora name if it doesn't exist
                                  _loraKeys[name] ??= GlobalKey();

                                  return InkWell(
                                    key: _loraKeys[name],
                                    onTap: () {
                                      final loraTag = "<lora:$name:0.7>";

                                      final RenderBox clickedItem =
                                          _loraKeys[name]!
                                              .currentContext!
                                              .findRenderObject() as RenderBox;
                                      final Offset startPosition = clickedItem
                                          .localToGlobal(Offset.zero);

                                      final RenderBox promptField =
                                          _promptFieldKey.currentContext!
                                              .findRenderObject() as RenderBox;
                                      final Offset targetPosition = promptField
                                          .localToGlobal(Offset.zero);

                                      late final OverlayEntry entry;
                                      entry = OverlayEntry(
                                        builder: (context) => Stack(
                                          children: [
                                            TweenAnimationBuilder<double>(
                                              duration: const Duration(
                                                  milliseconds: 500),
                                              curve: Curves.easeInOut,
                                              tween:
                                                  Tween(begin: 0.0, end: 1.0),
                                              onEnd: () {
                                                setState(() {
                                                  prompt = prompt.isEmpty
                                                      ? loraTag
                                                      : "$prompt $loraTag";
                                                  _promptController.text =
                                                      prompt;
                                                  _promptController.selection =
                                                      TextSelection
                                                          .fromPosition(
                                                    TextPosition(
                                                        offset:
                                                            _promptController
                                                                .text.length),
                                                  );
                                                });
                                                entry.remove();
                                              },
                                              builder: (context, value, child) {
                                                return Positioned(
                                                  left: startPosition.dx,
                                                  top: startPosition.dy +
                                                      (targetPosition.dy -
                                                              startPosition
                                                                  .dy) *
                                                          value,
                                                  child: Opacity(
                                                    opacity: 1 - (value * 0.2),
                                                    child: Material(
                                                      color: Colors.transparent,
                                                      child: Text(
                                                        loraTag,
                                                        style:
                                                            theme.textTheme.p,
                                                      ),
                                                    ),
                                                  ),
                                                );
                                              },
                                            ),
                                          ],
                                        ),
                                      );

                                      Overlay.of(context).insert(entry);
                                    },
                                    child: Container(
                                      padding: const EdgeInsets.symmetric(
                                          horizontal: 6, vertical: 2),
                                      decoration: BoxDecoration(
                                        color: theme.colorScheme.primary
                                            .withOpacity(0.1),
                                        borderRadius: BorderRadius.circular(4),
                                      ),
                                      child: Text(
                                        name,
                                        style: theme.textTheme.p.copyWith(
                                          fontSize: 13, // Reduced font size
                                        ),
                                      ),
                                    ),
                                  );
                                }).toList(),
                              ),
                            )
                          ],
                        ),
                        const SizedBox(height: 16),
                        Row(
                          children: [
                            Expanded(
                              child: ShadButton(
                                enabled: !(isModelLoading || isGenerating),
                                onPressed: () async {
                                  final modelDirPath =
                                      await getModelDirectory();
                                  final selectedDir = await FilePicker.platform
                                      .getDirectoryPath(
                                          initialDirectory: modelDirPath);

                                  if (selectedDir != null) {
                                    final directory = Directory(selectedDir);
                                    final files = directory.listSync();
                                    final clipFiles = files
                                        .whereType<File>()
                                        .where((file) =>
                                            file.path
                                                .endsWith('.safetensors') ||
                                            file.path.endsWith('.bin'))
                                        .toList();

                                    if (clipFiles.isNotEmpty) {
                                      final selectedClip =
                                          await showShadDialog<String>(
                                        context: context,
                                        builder: (BuildContext context) {
                                          return ShadDialog.alert(
                                            constraints: const BoxConstraints(
                                                maxWidth: 400),
                                            title: const Text('Select Clip_L'),
                                            description: SizedBox(
                                              height: 300,
                                              child: Material(
                                                color: Colors.transparent,
                                                child: ShadTable.list(
                                                  header: const [
                                                    ShadTableCell.header(
                                                      child: Text('Model',
                                                          style: TextStyle(
                                                              fontSize: 16)),
                                                    ),
                                                    ShadTableCell.header(
                                                      alignment:
                                                          Alignment.centerRight,
                                                      child: Text('Size',
                                                          style: TextStyle(
                                                              fontSize: 16)),
                                                    ),
                                                  ],
                                                  columnSpanExtent: (index) {
                                                    if (index == 0) {
                                                      return const FixedTableSpanExtent(
                                                          250);
                                                    }
                                                    if (index == 1) {
                                                      return const FixedTableSpanExtent(
                                                          80);
                                                    }
                                                    return null;
                                                  },
                                                  children: clipFiles
                                                      .asMap()
                                                      .entries
                                                      .map(
                                                        (entry) => [
                                                          ShadTableCell(
                                                            child:
                                                                GestureDetector(
                                                              onTap: () =>
                                                                  Navigator.pop(
                                                                      context,
                                                                      entry
                                                                          .value
                                                                          .path),
                                                              child: Padding(
                                                                padding: const EdgeInsets
                                                                    .symmetric(
                                                                    vertical:
                                                                        12.0),
                                                                child: Text(
                                                                  entry.value
                                                                      .path
                                                                      .split(
                                                                          '/')
                                                                      .last,
                                                                  style:
                                                                      const TextStyle(
                                                                    fontWeight:
                                                                        FontWeight
                                                                            .w500,
                                                                    fontSize:
                                                                        14,
                                                                  ),
                                                                ),
                                                              ),
                                                            ),
                                                          ),
                                                          ShadTableCell(
                                                            alignment: Alignment
                                                                .centerRight,
                                                            child:
                                                                GestureDetector(
                                                              onTap: () =>
                                                                  Navigator.pop(
                                                                      context,
                                                                      entry
                                                                          .value
                                                                          .path),
                                                              child: Text(
                                                                '${(entry.value.lengthSync() / (1024 * 1024)).toStringAsFixed(1)} MB',
                                                                style:
                                                                    const TextStyle(
                                                                        fontSize:
                                                                            12),
                                                              ),
                                                            ),
                                                          ),
                                                        ],
                                                      )
                                                      .toList(),
                                                ),
                                              ),
                                            ),
                                            actions: [
                                              ShadButton.outline(
                                                onPressed: () =>
                                                    Navigator.pop(context),
                                                child: const Text('Cancel'),
                                              ),
                                            ],
                                          );
                                        },
                                      );

                                      if (selectedClip != null) {
                                        setState(() {
                                          _clipLPath = selectedClip;
                                          loadedComponents['Clip_L'] = true;

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
                                              currentSchedule,
                                            );
                                          }
                                        });
                                      }
                                    }
                                  }
                                },
                                child: const Text('Load Clip_L'),
                              ),
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: ShadButton(
                                enabled: !(isModelLoading || isGenerating),
                                onPressed: () async {
                                  final modelDirPath =
                                      await getModelDirectory();
                                  final selectedDir = await FilePicker.platform
                                      .getDirectoryPath(
                                          initialDirectory: modelDirPath);

                                  if (selectedDir != null) {
                                    final directory = Directory(selectedDir);
                                    final files = directory.listSync();
                                    final clipFiles = files
                                        .whereType<File>()
                                        .where((file) =>
                                            file.path
                                                .endsWith('.safetensors') ||
                                            file.path.endsWith('.bin'))
                                        .toList();

                                    if (clipFiles.isNotEmpty) {
                                      final selectedClip =
                                          await showShadDialog<String>(
                                        context: context,
                                        builder: (BuildContext context) {
                                          return ShadDialog.alert(
                                            constraints: const BoxConstraints(
                                                maxWidth: 400),
                                            title: const Text('Select Clip_G'),
                                            description: SizedBox(
                                              height: 300,
                                              child: Material(
                                                color: Colors.transparent,
                                                child: ShadTable.list(
                                                  header: const [
                                                    ShadTableCell.header(
                                                      child: Text('Model',
                                                          style: TextStyle(
                                                              fontSize: 16)),
                                                    ),
                                                    ShadTableCell.header(
                                                      alignment:
                                                          Alignment.centerRight,
                                                      child: Text('Size',
                                                          style: TextStyle(
                                                              fontSize: 16)),
                                                    ),
                                                  ],
                                                  columnSpanExtent: (index) {
                                                    if (index == 0) {
                                                      return const FixedTableSpanExtent(
                                                          250);
                                                    }
                                                    if (index == 1) {
                                                      return const FixedTableSpanExtent(
                                                          80);
                                                    }
                                                    return null;
                                                  },
                                                  children: clipFiles
                                                      .asMap()
                                                      .entries
                                                      .map(
                                                        (entry) => [
                                                          ShadTableCell(
                                                            child:
                                                                GestureDetector(
                                                              onTap: () =>
                                                                  Navigator.pop(
                                                                      context,
                                                                      entry
                                                                          .value
                                                                          .path),
                                                              child: Padding(
                                                                padding: const EdgeInsets
                                                                    .symmetric(
                                                                    vertical:
                                                                        12.0),
                                                                child: Text(
                                                                  entry.value
                                                                      .path
                                                                      .split(
                                                                          '/')
                                                                      .last,
                                                                  style:
                                                                      const TextStyle(
                                                                    fontWeight:
                                                                        FontWeight
                                                                            .w500,
                                                                    fontSize:
                                                                        14,
                                                                  ),
                                                                ),
                                                              ),
                                                            ),
                                                          ),
                                                          ShadTableCell(
                                                            alignment: Alignment
                                                                .centerRight,
                                                            child:
                                                                GestureDetector(
                                                              onTap: () =>
                                                                  Navigator.pop(
                                                                      context,
                                                                      entry
                                                                          .value
                                                                          .path),
                                                              child: Text(
                                                                '${(entry.value.lengthSync() / (1024 * 1024)).toStringAsFixed(1)} MB',
                                                                style:
                                                                    const TextStyle(
                                                                        fontSize:
                                                                            12),
                                                              ),
                                                            ),
                                                          ),
                                                        ],
                                                      )
                                                      .toList(),
                                                ),
                                              ),
                                            ),
                                            actions: [
                                              ShadButton.outline(
                                                onPressed: () =>
                                                    Navigator.pop(context),
                                                child: const Text('Cancel'),
                                              ),
                                            ],
                                          );
                                        },
                                      );

                                      if (selectedClip != null) {
                                        setState(() {
                                          _clipGPath = selectedClip;
                                          loadedComponents['Clip_G'] = true;

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
                                              currentSchedule,
                                            );
                                          }
                                        });
                                      }
                                    }
                                  }
                                },
                                child: const Text('Load Clip_G'),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        Row(
                          children: [
                            Expanded(
                              child: ShadButton(
                                enabled: !(isModelLoading || isGenerating),
                                onPressed: () async {
                                  final modelDirPath =
                                      await getModelDirectory();
                                  final selectedDir = await FilePicker.platform
                                      .getDirectoryPath(
                                          initialDirectory: modelDirPath);

                                  if (selectedDir != null) {
                                    final directory = Directory(selectedDir);
                                    final files = directory.listSync();
                                    final t5Files = files
                                        .whereType<File>()
                                        .where((file) =>
                                            file.path
                                                .endsWith('.safetensors') ||
                                            file.path.endsWith('.bin'))
                                        .toList();

                                    if (t5Files.isNotEmpty) {
                                      final selectedT5 =
                                          await showShadDialog<String>(
                                        context: context,
                                        builder: (BuildContext context) {
                                          return ShadDialog.alert(
                                            constraints: const BoxConstraints(
                                                maxWidth: 400),
                                            title: const Text('Select T5XXL'),
                                            description: SizedBox(
                                              height: 300,
                                              child: Material(
                                                color: Colors.transparent,
                                                child: ShadTable.list(
                                                  header: const [
                                                    ShadTableCell.header(
                                                      child: Text('Model',
                                                          style: TextStyle(
                                                              fontSize: 16)),
                                                    ),
                                                    ShadTableCell.header(
                                                      alignment:
                                                          Alignment.centerRight,
                                                      child: Text('Size',
                                                          style: TextStyle(
                                                              fontSize: 16)),
                                                    ),
                                                  ],
                                                  columnSpanExtent: (index) {
                                                    if (index == 0) {
                                                      return const FixedTableSpanExtent(
                                                          250);
                                                    }
                                                    if (index == 1) {
                                                      return const FixedTableSpanExtent(
                                                          80);
                                                    }
                                                    return null;
                                                  },
                                                  children: t5Files
                                                      .asMap()
                                                      .entries
                                                      .map(
                                                        (entry) => [
                                                          ShadTableCell(
                                                            child:
                                                                GestureDetector(
                                                              onTap: () =>
                                                                  Navigator.pop(
                                                                      context,
                                                                      entry
                                                                          .value
                                                                          .path),
                                                              child: Padding(
                                                                padding: const EdgeInsets
                                                                    .symmetric(
                                                                    vertical:
                                                                        12.0),
                                                                child: Text(
                                                                  entry.value
                                                                      .path
                                                                      .split(
                                                                          '/')
                                                                      .last,
                                                                  style:
                                                                      const TextStyle(
                                                                    fontWeight:
                                                                        FontWeight
                                                                            .w500,
                                                                    fontSize:
                                                                        14,
                                                                  ),
                                                                ),
                                                              ),
                                                            ),
                                                          ),
                                                          ShadTableCell(
                                                            alignment: Alignment
                                                                .centerRight,
                                                            child:
                                                                GestureDetector(
                                                              onTap: () =>
                                                                  Navigator.pop(
                                                                      context,
                                                                      entry
                                                                          .value
                                                                          .path),
                                                              child: Text(
                                                                '${(entry.value.lengthSync() / (1024 * 1024)).toStringAsFixed(1)} MB',
                                                                style:
                                                                    const TextStyle(
                                                                        fontSize:
                                                                            12),
                                                              ),
                                                            ),
                                                          ),
                                                        ],
                                                      )
                                                      .toList(),
                                                ),
                                              ),
                                            ),
                                            actions: [
                                              ShadButton.outline(
                                                onPressed: () =>
                                                    Navigator.pop(context),
                                                child: const Text('Cancel'),
                                              ),
                                            ],
                                          );
                                        },
                                      );

                                      if (selectedT5 != null) {
                                        setState(() {
                                          _t5xxlPath = selectedT5;
                                          loadedComponents['T5XXL'] = true;

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
                                              currentSchedule,
                                            );
                                          }
                                        });
                                      }
                                    }
                                  }
                                },
                                child: const Text('Load T5XXL'),
                              ),
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: ShadButton(
                                enabled: !(isModelLoading || isGenerating),
                                onPressed: () async {
                                  final modelDirPath =
                                      await getModelDirectory();
                                  final selectedDir = await FilePicker.platform
                                      .getDirectoryPath(
                                          initialDirectory: modelDirPath);

                                  if (selectedDir != null) {
                                    setState(() {
                                      _embedDirPath = selectedDir;
                                      loadedComponents['Embeddings'] = true;

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
                                          currentSchedule,
                                        );
                                      }
                                    });
                                  }
                                },
                                child: const Text('Load Embed'),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        Row(
                          children: [
                            Expanded(
                              child: ShadButton(
                                enabled: !(isModelLoading || isGenerating),
                                onPressed: () async {
                                  final modelDirPath =
                                      await getModelDirectory();
                                  final selectedDir = await FilePicker.platform
                                      .getDirectoryPath(
                                          initialDirectory: modelDirPath);

                                  if (selectedDir != null) {
                                    final directory = Directory(selectedDir);
                                    final files = directory.listSync();
                                    final vaeFiles = files
                                        .whereType<File>()
                                        .where((file) =>
                                            file.path
                                                .endsWith('.safetensors') ||
                                            file.path.endsWith('.bin'))
                                        .toList();

                                    if (vaeFiles.isNotEmpty) {
                                      final selectedVae =
                                          await showShadDialog<String>(
                                        context: context,
                                        builder: (BuildContext context) {
                                          return ShadDialog.alert(
                                            constraints: const BoxConstraints(
                                                maxWidth: 400),
                                            title: const Text('Select VAE'),
                                            description: SizedBox(
                                              height: 300,
                                              child: Material(
                                                color: Colors.transparent,
                                                child: ShadTable.list(
                                                  header: const [
                                                    ShadTableCell.header(
                                                      child: Text('Model',
                                                          style: TextStyle(
                                                              fontSize: 16)),
                                                    ),
                                                    ShadTableCell.header(
                                                      alignment:
                                                          Alignment.centerRight,
                                                      child: Text('Size',
                                                          style: TextStyle(
                                                              fontSize: 16)),
                                                    ),
                                                  ],
                                                  columnSpanExtent: (index) {
                                                    if (index == 0) {
                                                      return const FixedTableSpanExtent(
                                                          250);
                                                    }
                                                    if (index == 1) {
                                                      return const FixedTableSpanExtent(
                                                          80);
                                                    }
                                                    return null;
                                                  },
                                                  children: vaeFiles
                                                      .asMap()
                                                      .entries
                                                      .map(
                                                        (entry) => [
                                                          ShadTableCell(
                                                            child:
                                                                GestureDetector(
                                                              onTap: () =>
                                                                  Navigator.pop(
                                                                      context,
                                                                      entry
                                                                          .value
                                                                          .path),
                                                              child: Padding(
                                                                padding: const EdgeInsets
                                                                    .symmetric(
                                                                    vertical:
                                                                        12.0),
                                                                child: Text(
                                                                  entry.value
                                                                      .path
                                                                      .split(
                                                                          '/')
                                                                      .last,
                                                                  style:
                                                                      const TextStyle(
                                                                    fontWeight:
                                                                        FontWeight
                                                                            .w500,
                                                                    fontSize:
                                                                        14,
                                                                  ),
                                                                ),
                                                              ),
                                                            ),
                                                          ),
                                                          ShadTableCell(
                                                            alignment: Alignment
                                                                .centerRight,
                                                            child:
                                                                GestureDetector(
                                                              onTap: () =>
                                                                  Navigator.pop(
                                                                      context,
                                                                      entry
                                                                          .value
                                                                          .path),
                                                              child: Text(
                                                                '${(entry.value.lengthSync() / (1024 * 1024)).toStringAsFixed(1)} MB',
                                                                style:
                                                                    const TextStyle(
                                                                        fontSize:
                                                                            12),
                                                              ),
                                                            ),
                                                          ),
                                                        ],
                                                      )
                                                      .toList(),
                                                ),
                                              ),
                                            ),
                                            actions: [
                                              ShadButton.outline(
                                                onPressed: () =>
                                                    Navigator.pop(context),
                                                child: const Text('Cancel'),
                                              ),
                                            ],
                                          );
                                        },
                                      );

                                      if (selectedVae != null) {
                                        setState(() {
                                          _vaePath = selectedVae;
                                          loadedComponents['VAE'] = true;

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
                                              currentSchedule,
                                            );
                                          }
                                        });
                                      }
                                    }
                                  }
                                },
                                child: const Text('Load VAE'),
                              ),
                            ),
                            const SizedBox(width: 8),
                            ShadCheckbox(
                              value: useVAE,
                              onChanged: (isModelLoading || isGenerating)
                                  ? null
                                  : (bool v) {
                                      if (_vaePath == null) {
                                        _showTemporaryError(
                                            'Please load VAE model first');
                                        return;
                                      }
                                      setState(() {
                                        useVAE = v;
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
                                            currentSchedule,
                                          );
                                        }
                                      });
                                    },
                              label: const Text('Use VAE'),
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        ShadCheckbox(
                          value: useVAETiling,
                          onChanged: (isModelLoading || isGenerating)
                              ? null
                              : (bool v) {
                                  if (useTAESD) {
                                    _showTemporaryError(
                                        'VAE Tiling is incompatible with TAESD');
                                    return;
                                  }

                                  setState(() {
                                    useVAETiling = v;
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
                                        currentSchedule,
                                      );
                                    }
                                  });
                                },
                          label: const Text('VAE Tiling'),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                const Text('Use ControlNet'),
                const SizedBox(width: 8),
                ShadSwitch(
                  value: useControlNet,
                  onChanged: (bool v) {
                    setState(() {
                      useControlNet = v;
                      // Only reinitialize if processor is already loaded
                      if (_processor != null) {
                        String currentModelPath = _processor!.modelPath;
                        bool currentFlashAttention =
                            _processor!.useFlashAttention;
                        SDType currentModelType = _processor!.modelType;
                        Schedule currentSchedule = _processor!.schedule;

                        // If ControlNet is turned off, we'll set the controlNetPath to null temporarily
                        // before reinitializing the processor
                        String? originalControlNetPath = _controlNetPath;
                        if (!v) {
                          _controlNetPath = null;
                        }

                        _initializeProcessor(
                          currentModelPath,
                          currentFlashAttention,
                          currentModelType,
                          currentSchedule,
                        );

                        // Restore the original path after reinitialization
                        if (!v) {
                          _controlNetPath = originalControlNetPath;
                        }
                      }
                    });
                  },
                ),
              ],
            ),
            if (useControlNet) ...[
              const SizedBox(height: 16),
              ShadAccordion<Map<String, dynamic>>(
                children: [
                  // Replace the ControlNet options section with this improved version
                  ShadAccordionItem<Map<String, dynamic>>(
                    value: const {},
                    title: const Text('ControlNet Options'),
                    child: Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment
                            .start, // Align children to the left
                        children: [
                          Row(
                            children: [
                              Expanded(
                                // Made the button expand to full width to prevent overflow
                                child: ShadButton(
                                  enabled: !(isModelLoading || isGenerating),
                                  onPressed: () async {
                                    final modelDirPath =
                                        await getModelDirectory();
                                    final selectedDir = await FilePicker
                                        .platform
                                        .getDirectoryPath(
                                            initialDirectory: modelDirPath);

                                    if (selectedDir != null) {
                                      final directory = Directory(selectedDir);
                                      final files = directory.listSync();
                                      final controlNetFiles = files
                                          .whereType<File>()
                                          .where((file) =>
                                              file.path
                                                  .endsWith('.safetensors') ||
                                              file.path.endsWith('.bin') ||
                                              file.path.endsWith('.pth') ||
                                              file.path.endsWith('.ckpt'))
                                          .toList();

                                      if (controlNetFiles.isNotEmpty) {
                                        final selectedControlNet =
                                            await showShadDialog<String>(
                                          context: context,
                                          builder: (BuildContext context) {
                                            return ShadDialog.alert(
                                              constraints: const BoxConstraints(
                                                  maxWidth: 400),
                                              title: const Text(
                                                  'Select ControlNet Model'),
                                              description: SizedBox(
                                                height: 300,
                                                child: Material(
                                                  color: Colors.transparent,
                                                  child: ShadTable.list(
                                                    header: const [
                                                      ShadTableCell.header(
                                                          child: Text('Model',
                                                              style: TextStyle(
                                                                  fontSize:
                                                                      16))),
                                                      ShadTableCell.header(
                                                          alignment: Alignment
                                                              .centerRight,
                                                          child: Text('Size',
                                                              style: TextStyle(
                                                                  fontSize:
                                                                      16))),
                                                    ],
                                                    columnSpanExtent: (index) {
                                                      if (index == 0)
                                                        return const FixedTableSpanExtent(
                                                            250);
                                                      if (index == 1)
                                                        return const FixedTableSpanExtent(
                                                            80);
                                                      return null;
                                                    },
                                                    children: controlNetFiles
                                                        .asMap()
                                                        .entries
                                                        .map((entry) => [
                                                              ShadTableCell(
                                                                child:
                                                                    GestureDetector(
                                                                  onTap: () => Navigator.pop(
                                                                      context,
                                                                      entry
                                                                          .value
                                                                          .path),
                                                                  child:
                                                                      Padding(
                                                                    padding: const EdgeInsets
                                                                        .symmetric(
                                                                        vertical:
                                                                            12.0),
                                                                    child: Text(
                                                                      entry
                                                                          .value
                                                                          .path
                                                                          .split(
                                                                              '/')
                                                                          .last,
                                                                      style: const TextStyle(
                                                                          fontWeight: FontWeight
                                                                              .w500,
                                                                          fontSize:
                                                                              14),
                                                                    ),
                                                                  ),
                                                                ),
                                                              ),
                                                              ShadTableCell(
                                                                alignment: Alignment
                                                                    .centerRight,
                                                                child:
                                                                    GestureDetector(
                                                                  onTap: () => Navigator.pop(
                                                                      context,
                                                                      entry
                                                                          .value
                                                                          .path),
                                                                  child: Text(
                                                                    '${(entry.value.lengthSync() / (1024 * 1024)).toStringAsFixed(1)} MB',
                                                                    style: const TextStyle(
                                                                        fontSize:
                                                                            12),
                                                                  ),
                                                                ),
                                                              ),
                                                            ])
                                                        .toList(),
                                                  ),
                                                ),
                                              ),
                                              actions: [
                                                ShadButton.outline(
                                                  onPressed: () =>
                                                      Navigator.pop(context),
                                                  child: const Text('Cancel'),
                                                ),
                                              ],
                                            );
                                          },
                                        );

                                        if (selectedControlNet != null) {
                                          setState(() {
                                            _controlNetPath =
                                                selectedControlNet;
                                            loadedComponents['ControlNet'] =
                                                true;
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
                                                currentSchedule,
                                              );
                                            }
                                          });
                                        }
                                      }
                                    }
                                  },
                                  child: const Text('Load ControlNet'),
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 16),
                          // Left-aligned checkbox for "Use Image Reference"
                          Align(
                            alignment: Alignment.centerLeft,
                            child: ShadCheckbox(
                              value: useControlImage,
                              onChanged: (bool v) {
                                setState(() {
                                  useControlImage = v;
                                  if (!v) {
                                    _controlImage = null;
                                    _controlRgbBytes = null;
                                    _controlWidth = null;
                                    _controlHeight = null;
                                  }
                                });
                              },
                              label: const Text('Use Image Reference'),
                            ),
                          ),
                          if (useControlImage) ...[
                            const SizedBox(height: 16),
                            GestureDetector(
                              onTap: (isModelLoading || isGenerating)
                                  ? null
                                  : () async {
                                      final picker = ImagePicker();
                                      final pickedFile = await picker.pickImage(
                                          source: ImageSource.gallery);

                                      if (pickedFile != null) {
                                        final bytes =
                                            await pickedFile.readAsBytes();
                                        final decodedImage =
                                            img.decodeImage(bytes);

                                        if (decodedImage == null) {
                                          _showTemporaryError(
                                              'Failed to decode image');
                                          return;
                                        }

                                        final rgbBytes = Uint8List(
                                            decodedImage.width *
                                                decodedImage.height *
                                                3);
                                        int rgbIndex = 0;

                                        for (int y = 0;
                                            y < decodedImage.height;
                                            y++) {
                                          for (int x = 0;
                                              x < decodedImage.width;
                                              x++) {
                                            final pixel =
                                                decodedImage.getPixel(x, y);
                                            rgbBytes[rgbIndex] =
                                                pixel.r.toInt();
                                            rgbBytes[rgbIndex + 1] =
                                                pixel.g.toInt();
                                            rgbBytes[rgbIndex + 2] =
                                                pixel.b.toInt();
                                            rgbIndex += 3;
                                          }
                                        }

                                        setState(() {
                                          _controlImage = File(pickedFile.path);
                                          _controlRgbBytes = rgbBytes;
                                          _controlWidth = decodedImage.width;
                                          _controlHeight = decodedImage.height;
                                        });
                                      }
                                    },
                              child: DottedBorder(
                                borderType: BorderType.RRect,
                                radius: const Radius.circular(8),
                                color:
                                    theme.colorScheme.primary.withOpacity(0.5),
                                strokeWidth: 2,
                                dashPattern: const [8, 4],
                                child: Container(
                                  height: 200,
                                  width: double.infinity,
                                  child: Center(
                                    child: _controlImage == null
                                        ? Column(
                                            mainAxisAlignment:
                                                MainAxisAlignment.center,
                                            children: [
                                              Icon(
                                                Icons.add_photo_alternate,
                                                size: 64,
                                                color: theme.colorScheme.primary
                                                    .withOpacity(0.5),
                                              ),
                                              const SizedBox(height: 12),
                                              Text(
                                                'Load control image',
                                                style: TextStyle(
                                                    color: theme
                                                        .colorScheme.primary
                                                        .withOpacity(0.5),
                                                    fontSize: 16),
                                              ),
                                            ],
                                          )
                                        : Image.file(_controlImage!,
                                            fit: BoxFit.contain),
                                  ),
                                ),
                              ),
                            ),
                            const SizedBox(height: 16),
                            // Dropdown for Crop/Resize
                            Row(
                              children: [
                                const Text('Reference Handling:'),
                                const SizedBox(width: 8),
                                Expanded(
                                  child: ShadSelect<String>(
                                    placeholder:
                                        Text(_controlImageProcessingMode),
                                    options: const [
                                      ShadOption(
                                          value: 'Resize',
                                          child: Text('Resize')),
                                      ShadOption(
                                          value: 'Crop', child: Text('Crop')),
                                    ],
                                    selectedOptionBuilder: (context, value) =>
                                        Text(value),
                                    onChanged: (String? value) {
                                      if (value != null) {
                                        setState(() =>
                                            _controlImageProcessingMode =
                                                value);
                                      }
                                    },
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 16),
                            // Left-aligned checkbox for "Use Canny"
                            Align(
                              alignment: Alignment.centerLeft,
                              child: Row(
                                children: [
                                  ShadCheckbox(
                                    value: useCanny,
                                    onChanged: (_controlImage == null ||
                                            isModelLoading ||
                                            isGenerating ||
                                            isCannyProcessing)
                                        ? null
                                        : (bool v) {
                                            setState(() {
                                              useCanny = v;
                                              if (v && _controlImage != null) {
                                                // Process the image with Canny
                                                _processCannyImage();
                                              }
                                            });
                                          },
                                    label: const Text('Use Canny'),
                                  ),
                                  if (isCannyProcessing)
                                    Padding(
                                      padding: const EdgeInsets.only(left: 8.0),
                                      child: SizedBox(
                                        width: 16,
                                        height: 16,
                                        child: CircularProgressIndicator(
                                          strokeWidth: 2,
                                          color: theme.colorScheme.primary,
                                        ),
                                      ),
                                    ),
                                ],
                              ),
                            ),
                            if (useCanny && _cannyImage != null) ...[
                              const SizedBox(height: 16),
                              const Text('Canny Edge Detection Result',
                                  style:
                                      TextStyle(fontWeight: FontWeight.bold)),
                              const SizedBox(height: 8),
                              Container(
                                decoration: BoxDecoration(
                                  border: Border.all(
                                      color: theme.colorScheme.primary
                                          .withOpacity(0.5)),
                                  borderRadius: BorderRadius.circular(8),
                                ),
                                height: 200,
                                width: double.infinity,
                                child: Center(
                                  child: _cannyImage!,
                                ),
                              ),
                            ],
                          ],
                          const SizedBox(height: 16),
                          Row(
                            children: [
                              const Text('Control Strength'),
                              const SizedBox(width: 8),
                              Expanded(
                                child: ShadSlider(
                                  initialValue: controlStrength,
                                  min: 0.0,
                                  max: 1.0,
                                  divisions: 20,
                                  onChanged: (v) =>
                                      setState(() => controlStrength = v),
                                ),
                              ),
                              Text(controlStrength.toStringAsFixed(2)),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ],
            Row(
              children: [
                const Text('Sampling Method'),
                const SizedBox(width: 8),
                Expanded(
                  child: ShadSelect<String>(
                    placeholder: const Text('euler_a'),
                    options: samplingMethods
                        .map((method) => ShadOption(
                              value: method,
                              child: Text(method),
                            ))
                        .toList(),
                    selectedOptionBuilder: (context, value) => Text(value),
                    onChanged: (String? value) =>
                        setState(() => samplingMethod = value ?? 'euler_a'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                const Text('CFG'),
                const SizedBox(width: 8),
                Expanded(
                  child: ShadSlider(
                    initialValue: cfg,
                    min: 1,
                    max: 20,
                    divisions: 38,
                    onChanged: (v) => setState(() => cfg = v),
                  ),
                ),
                Text(cfg.toStringAsFixed(1)),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                const Text('Steps'),
                const SizedBox(width: 8),
                Expanded(
                  child: ShadSlider(
                    initialValue: steps.toDouble(),
                    min: 1,
                    max: 50,
                    divisions: 49,
                    onChanged: (v) => setState(() => steps = v.toInt()),
                  ),
                ),
                Text(steps.toString()),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                const Text('Width'),
                const SizedBox(width: 8),
                Expanded(
                  child: ShadSelect<int>(
                    placeholder: const Text('512'),
                    options: getWidthOptions()
                        .map((w) => ShadOption(
                              value: w,
                              child: Text(w.toString()),
                            ))
                        .toList(),
                    selectedOptionBuilder: (context, value) =>
                        Text(value.toString()),
                    onChanged: (int? value) {
                      if (value != null) setState(() => width = value);
                    },
                  ),
                ),
                const SizedBox(width: 16),
                const Text('Height'),
                const SizedBox(width: 8),
                Expanded(
                  child: ShadSelect<int>(
                    placeholder: const Text('512'),
                    options: getHeightOptions()
                        .map((h) => ShadOption(
                              value: h,
                              child: Text(h.toString()),
                            ))
                        .toList(),
                    selectedOptionBuilder: (context, value) =>
                        Text(value.toString()),
                    onChanged: (int? value) {
                      if (value != null) setState(() => height = value);
                    },
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            const Text('Seed (-1 for random)'),
            const SizedBox(height: 8),
            ShadInput(
              placeholder: const Text('Seed'),
              keyboardType: TextInputType.number,
              onChanged: (String? v) => setState(() => seed = v ?? "-1"),
              initialValue: seed,
            ),
            const SizedBox(height: 16),
            Row(
              // Wrap buttons in a Row
              children: [
                ShadButton(
                  enabled: !(isModelLoading || isGenerating),
                  onPressed: () {
                    if (_processor == null) {
                      _modelErrorTimer?.cancel();
                      setState(() {
                        _modelError = 'Please Load a model first';
                      });
                      _modelErrorTimer = Timer(const Duration(seconds: 10), () {
                        setState(() {
                          _modelError = '';
                        });
                      });
                      return;
                    }
                    setState(() {
                      isGenerating = true;
                      _modelError = '';
                      status = 'Generating image...';
                      progress = 0;
                      _generationLogs = []; // Clear previous logs
                      _showLogsButton =
                          false; // Hide log button until generation finishes
                    });

                    // --- Control Image Processing Logic ---
                    Uint8List? finalControlBytes = _controlRgbBytes;
                    int? finalControlWidth = _controlWidth;
                    int? finalControlHeight = _controlHeight;

                    // Determine the source image data (Canny or original control image)
                    Uint8List? sourceBytes;
                    int? sourceWidth;
                    int? sourceHeight;

                    if (useControlNet && useControlImage) {
                      if (useCanny && _cannyProcessor?.resultRgbBytes != null) {
                        // Use Canny result if available and Canny is enabled
                        sourceBytes = _cannyProcessor!.resultRgbBytes!;
                        sourceWidth = _cannyProcessor!.resultWidth!;
                        sourceHeight = _cannyProcessor!.resultHeight!;
                        // Ensure Canny result is RGB
                        sourceBytes = _ensureRgbFormat(
                            sourceBytes, sourceWidth, sourceHeight);
                      } else if (_controlRgbBytes != null) {
                        // Use original control image if Canny is not used or failed
                        sourceBytes = _controlRgbBytes;
                        sourceWidth = _controlWidth;
                        sourceHeight = _controlHeight;
                        // Ensure original control image is RGB (already done on load, but double-check)
                        if (sourceBytes != null &&
                            sourceWidth != null &&
                            sourceHeight != null) {
                          sourceBytes = _ensureRgbFormat(
                              sourceBytes, sourceWidth, sourceHeight);
                        }
                      }
                    }

                    if (sourceBytes != null &&
                        sourceWidth != null &&
                        sourceHeight != null) {
                      // Check if dimensions match the target output size
                      if (sourceWidth != width || sourceHeight != height) {
                        try {
                          print(
                              'Control image dimensions ($sourceWidth x $sourceHeight) differ from target ($width x $height). Processing using $_controlImageProcessingMode...');
                          ProcessedImageData processedData;
                          if (_controlImageProcessingMode == 'Crop') {
                            processedData = cropImage(sourceBytes, sourceWidth,
                                sourceHeight, width, height);
                          } else {
                            // Default to Resize
                            processedData = resizeImage(sourceBytes,
                                sourceWidth, sourceHeight, width, height);
                          }
                          finalControlBytes = processedData.bytes;
                          finalControlWidth = processedData.width;
                          finalControlHeight = processedData.height;
                          print(
                              'Control image processed to $finalControlWidth x $finalControlHeight.');
                        } catch (e) {
                          print("Error processing control image: $e");
                          // Optionally show an error to the user
                          _showTemporaryError(
                              'Error processing control image: $e');
                          setState(
                              () => isGenerating = false); // Stop generation
                          return; // Prevent calling generateImage
                        }
                      } else {
                        // Dimensions already match, use the source directly
                        finalControlBytes = sourceBytes;
                        finalControlWidth = sourceWidth;
                        finalControlHeight = sourceHeight;
                      }
                    } else {
                      // No valid source image, ensure control params are null
                      finalControlBytes = null;
                      finalControlWidth = null;
                      finalControlHeight = null;
                    }
                    // --- End Control Image Processing Logic ---

                    _processor!.generateImage(
                      prompt: prompt,
                      negativePrompt: negativePrompt,
                      cfgScale: cfg,
                      sampleSteps: steps,
                      width: width, // Target width
                      height: height, // Target height
                      seed: int.tryParse(seed) ?? -1,
                      sampleMethod: SampleMethod.values
                          .firstWhere(
                            (method) =>
                                method.displayName.toLowerCase() ==
                                samplingMethod.toLowerCase(),
                            orElse: () => SampleMethod.EULER_A,
                          )
                          .index,
                      controlImageData:
                          finalControlBytes, // Use processed bytes
                      controlImageWidth:
                          finalControlWidth, // Use processed width
                      controlImageHeight:
                          finalControlHeight, // Use processed height
                      controlStrength: controlStrength,
                    );
                  },
                  child: const Text('Generate'),
                ),
                const SizedBox(width: 8), // Add spacing between buttons
                if (_showLogsButton &&
                    _generationLogs
                        .isNotEmpty) // Conditionally show the log button only if logs exist
                  ShadButton.outline(
                    onPressed: _showLogsDialog,
                    child: const Text('Show Logs'),
                  ),
              ],
            ),
            if (_modelError.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 8.0),
                child: Text(
                  _modelError,
                  style: const TextStyle(
                    color: Colors.red,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            const SizedBox(height: 16),
            LinearProgressIndicator(
              value: progress,
              backgroundColor: theme.colorScheme.background,
              color: theme.colorScheme.primary,
            ),
            const SizedBox(height: 8),
            Text(status, style: theme.textTheme.p),
            if (_generatedImage != null) ...[
              const SizedBox(height: 20),
              _generatedImage!,
            ],
          ],
        ),
      ),
    );
  }

  // Method to show the logs dialog
  void _showLogsDialog() {
    showShadDialog(
      context: context,
      builder: (context) => ShadDialog.alert(
        constraints:
            const BoxConstraints(maxWidth: 600, maxHeight: 500), // Adjust size
        title: const Text('Generation Logs'),
        description: SizedBox(
          // Constrain the height of the scrollable area
          height: 300, // Adjust height as needed
          child: SingleChildScrollView(
            child: Padding(
              padding:
                  const EdgeInsets.symmetric(vertical: 8.0, horizontal: 4.0),
              child: SelectableText(
                _generationLogs.join('\n'), // Join logs with newlines
                style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
              ),
            ),
          ),
        ),
        actions: [
          ShadButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }
}
