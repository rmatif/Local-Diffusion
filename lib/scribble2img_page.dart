import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'dart:async';
import 'dart:developer' as developer;
import 'package:shadcn_ui/shadcn_ui.dart';
import 'package:file_picker/file_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:dotted_border/dotted_border.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'ffi_bindings.dart';
import 'inpainting_page.dart';
import 'photomaker_page.dart';
import 'stable_diffusion_processor.dart';
import 'utils.dart';
import 'main.dart';
import 'img2img_page.dart';
import 'upscaler_page.dart';
import 'package:flutter_drawing_board/flutter_drawing_board.dart';
import 'package:image/image.dart' as img;

class ScribblePage extends StatefulWidget {
  const ScribblePage({super.key});

  @override
  State<ScribblePage> createState() => _ScribblePageState();
}

class _ScribblePageState extends State<ScribblePage>
    with SingleTickerProviderStateMixin {
  Timer? _modelErrorTimer;
  Timer? _errorMessageTimer;
  StableDiffusionProcessor? _processor;
  Image? _generatedImage;
  bool isModelLoading = false;
  bool isGenerating = false;
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
  bool useTAESD = false;
  bool useVAETiling = false;
  double clipSkip = 1.0;
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
  String? _taesdPath;
  String? _loraPath;
  String? _vaePath;
  String? _embedDirPath;
  String? _controlNetPath;
  Uint8List? _controlImageData;
  int? _controlWidth;
  int? _controlHeight;
  double controlStrength = 0.9;
  final DrawingController _drawingController = DrawingController();
  Uint8List? _drawingImageData;
  bool _hasDrawing = false;

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
  }

  @override
  void dispose() {
    _errorMessageTimer?.cancel();
    _modelErrorTimer?.cancel();
    _processor?.dispose();
    super.dispose();
  }

  Future<String> getModelDirectory() async {
    final directory = Directory('/storage/emulated/0/Local Diffusion/Models');
    if (!await directory.exists()) {
      await directory.create(recursive: true);
    }
    return directory.path;
  }

  void _initializeProcessor(String modelPath, bool useFlashAttention,
      SDType modelType, Schedule schedule) {
    setState(() {
      isModelLoading = true;
      loadingText = 'Loading Model...';
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
      clipLPath: null,
      clipGPath: null,
      t5xxlPath: null,
      vaePath: useVAE ? _vaePath : null,
      embedDirPath: _embedDirPath,
      clipSkip: clipSkip.toInt(),
      vaeTiling: useVAETiling,
      controlNetPath: _controlNetPath,
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

    _processor!.imageStream.listen((image) async {
      final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
      setState(() {
        isGenerating = false;
        _generatedImage = Image.memory(bytes!.buffer.asUint8List());
        status = 'Generation complete';
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

  Future<void> _openDrawingBoard() async {
    final screenSize = MediaQuery.of(context).size;
    final dialogWidth = screenSize.width * 0.9;
    final dialogHeight = screenSize.height * 0.7;

    _drawingController.setStyle(color: Colors.black, strokeWidth: 3.0);

    final result = await showDialog<Uint8List>(
      context: context,
      builder: (context) => Dialog(
        backgroundColor: Colors.white,
        insetPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 24),
        child: Container(
          width: dialogWidth,
          height: dialogHeight,
          child: Column(
            children: [
              Padding(
                padding:
                    const EdgeInsets.symmetric(horizontal: 8.0, vertical: 4.0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text('Drawing Board',
                        style: TextStyle(
                            fontSize: 16, fontWeight: FontWeight.bold)),
                    Row(
                      children: [
                        TextButton(
                          onPressed: () => Navigator.of(context).pop(),
                          child: const Text('Cancel'),
                          style: TextButton.styleFrom(
                            padding: EdgeInsets.symmetric(horizontal: 8),
                            foregroundColor: Colors.black,
                          ),
                        ),
                        TextButton(
                          onPressed: () async {
                            final imageData =
                                await _drawingController.getImageData();
                            if (imageData != null) {
                              final bytes = imageData.buffer.asUint8List();
                              final originalImage = img.decodeImage(bytes);
                              if (originalImage != null) {
                                final resizedImage = img.copyResize(
                                  originalImage,
                                  width: 512,
                                  height: 512,
                                  interpolation: img.Interpolation.average,
                                );
                                final rgbBytes = Uint8List(512 * 512 * 3);
                                int index = 0;
                                for (int y = 0; y < 512; y++) {
                                  for (int x = 0; x < 512; x++) {
                                    final pixel = resizedImage.getPixel(x, y);
                                    rgbBytes[index] = pixel.r.toInt();
                                    rgbBytes[index + 1] = pixel.g.toInt();
                                    rgbBytes[index + 2] = pixel.b.toInt();
                                    index += 3;
                                  }
                                }
                                final resizedBytes =
                                    img.encodePng(resizedImage);
                                setState(() {
                                  _drawingImageData =
                                      Uint8List.fromList(resizedBytes);
                                  _controlImageData = rgbBytes;
                                  _controlWidth = 512;
                                  _controlHeight = 512;
                                  _hasDrawing = true;
                                });
                                Navigator.of(context).pop();
                              }
                            } else {
                              Navigator.of(context).pop();
                            }
                          },
                          child: const Text('Save'),
                          style: TextButton.styleFrom(
                            padding: EdgeInsets.symmetric(horizontal: 8),
                            foregroundColor: Colors.black,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 8.0),
                child: SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: Row(
                    children: [
                      ...[
                        Colors.black,
                        Colors.red,
                        Colors.blue,
                        Colors.green,
                        Colors.yellow,
                        Colors.orange,
                        Colors.purple,
                        Colors.teal,
                        Colors.pink,
                        Colors.brown,
                        Colors.grey
                      ].map(
                        (color) => GestureDetector(
                          onTap: () {
                            _drawingController.setStyle(color: color);
                          },
                          child: Container(
                            margin: EdgeInsets.only(right: 8),
                            width: 24,
                            height: 24,
                            decoration: BoxDecoration(
                              color: color,
                              shape: BoxShape.circle,
                              border: Border.all(color: Colors.grey.shade300),
                            ),
                          ),
                        ),
                      ),
                      ...[2.0, 5.0, 8.0, 12.0].map(
                        (thickness) => GestureDetector(
                          onTap: () {
                            _drawingController.setStyle(strokeWidth: thickness);
                          },
                          child: Container(
                            margin: EdgeInsets.only(right: 8),
                            width: 24,
                            height: 24,
                            decoration: BoxDecoration(
                              color: Colors.white,
                              shape: BoxShape.circle,
                              border: Border.all(color: Colors.grey.shade300),
                            ),
                            child: Center(
                              child: Container(
                                width: thickness,
                                height: thickness,
                                decoration: BoxDecoration(
                                  color: Colors.black,
                                  shape: BoxShape.circle,
                                ),
                              ),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              Expanded(
                child: DrawingBoard(
                  controller: _drawingController,
                  background: Container(
                    width: dialogWidth,
                    height: dialogHeight - 80,
                    color: Colors.white,
                  ),
                  showDefaultActions: true,
                  showDefaultTools: true,
                ),
              ),
            ],
          ),
        ),
      ),
    );
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
                          .map((type) =>
                              ShadOption(value: type, child: Text(type)))
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
                              value: schedule, child: Text(schedule)))
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
                          constraints: const BoxConstraints(maxWidth: 400),
                          title: const Text('Select Model'),
                          description: SizedBox(
                            height: 300,
                            child: Material(
                              color: Colors.transparent,
                              child: ShadTable.list(
                                header: const [
                                  ShadTableCell.header(
                                      child: Text('Model',
                                          style: TextStyle(fontSize: 16))),
                                  ShadTableCell.header(
                                      alignment: Alignment.centerRight,
                                      child: Text('Size',
                                          style: TextStyle(fontSize: 16))),
                                ],
                                columnSpanExtent: (index) {
                                  if (index == 0)
                                    return const FixedTableSpanExtent(250);
                                  if (index == 1)
                                    return const FixedTableSpanExtent(80);
                                  return null;
                                },
                                children: modelFiles
                                    .asMap()
                                    .entries
                                    .map((entry) => [
                                          ShadTableCell(
                                            child: GestureDetector(
                                              onTap: () => Navigator.pop(
                                                  context, entry.value.path),
                                              child: Padding(
                                                padding:
                                                    const EdgeInsets.symmetric(
                                                        vertical: 12.0),
                                                child: Text(
                                                  entry.value.path
                                                      .split('/')
                                                      .last,
                                                  style: const TextStyle(
                                                      fontWeight:
                                                          FontWeight.w500,
                                                      fontSize: 14),
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
                                        ])
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
        title: const Text('Scribble to Image',
            style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: theme.colorScheme.background,
        elevation: 0,
      ),
      drawer: Drawer(
        width: 240,
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.horizontal(right: Radius.circular(4)),
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
                    Color.fromRGBO(24, 89, 38, 1),
                    Color.fromARGB(255, 59, 128, 160),
                    Color(0xFF0a2335),
                  ],
                ),
              ),
              child: Text(
                'Menu',
                style: TextStyle(
                    color: Colors.white,
                    fontSize: 24,
                    fontWeight: FontWeight.bold),
              ),
            ),
            ListTile(
              leading: const Icon(LucideIcons.type, size: 32),
              title: const Text('Text to Image',
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
                      builder: (context) => const StableDiffusionApp()),
                );
              },
            ),
            ListTile(
              leading: const Icon(LucideIcons.images, size: 32),
              title: const Text('Image to Image',
                  style: TextStyle(fontWeight: FontWeight.bold)),
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
              title: const Text('Upscaler',
                  style: TextStyle(fontWeight: FontWeight.bold)),
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
                Navigator.pop(context);
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
                                      child: Icon(Icons.check,
                                          size: 20, color: Colors.green),
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
                  onPressed: () async {
                    final modelDirPath = await getModelDirectory();
                    final selectedDir = await FilePicker.platform
                        .getDirectoryPath(initialDirectory: modelDirPath);
                    if (selectedDir != null) {
                      final directory = Directory(selectedDir);
                      final files = directory.listSync();
                      final controlNetFiles = files
                          .whereType<File>()
                          .where((file) =>
                              file.path.endsWith('.safetensors') ||
                              file.path.endsWith('.bin') ||
                              file.path.endsWith('.pth') ||
                              file.path.endsWith('.ckpt'))
                          .toList();

                      if (controlNetFiles.isNotEmpty) {
                        final selectedControlNet = await showShadDialog<String>(
                          context: context,
                          builder: (BuildContext context) {
                            return ShadDialog.alert(
                              constraints: const BoxConstraints(maxWidth: 400),
                              title: const Text('Select ControlNet Model'),
                              description: SizedBox(
                                height: 300,
                                child: Material(
                                  color: Colors.transparent,
                                  child: ShadTable.list(
                                    header: const [
                                      ShadTableCell.header(
                                          child: Text('Model',
                                              style: TextStyle(fontSize: 16))),
                                      ShadTableCell.header(
                                          alignment: Alignment.centerRight,
                                          child: Text('Size',
                                              style: TextStyle(fontSize: 16))),
                                    ],
                                    columnSpanExtent: (index) {
                                      if (index == 0)
                                        return const FixedTableSpanExtent(250);
                                      if (index == 1)
                                        return const FixedTableSpanExtent(80);
                                      return null;
                                    },
                                    children: controlNetFiles
                                        .asMap()
                                        .entries
                                        .map((entry) => [
                                              ShadTableCell(
                                                child: GestureDetector(
                                                  onTap: () => Navigator.pop(
                                                      context,
                                                      entry.value.path),
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
                                                          fontSize: 14),
                                                    ),
                                                  ),
                                                ),
                                              ),
                                              ShadTableCell(
                                                alignment:
                                                    Alignment.centerRight,
                                                child: GestureDetector(
                                                  onTap: () => Navigator.pop(
                                                      context,
                                                      entry.value.path),
                                                  child: Text(
                                                    '${(entry.value.lengthSync() / (1024 * 1024)).toStringAsFixed(1)} MB',
                                                    style: const TextStyle(
                                                        fontSize: 12),
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
                                  onPressed: () => Navigator.pop(context),
                                  child: const Text('Cancel'),
                                ),
                              ],
                            );
                          },
                        );

                        if (selectedControlNet != null) {
                          setState(() {
                            _controlNetPath = selectedControlNet;
                            loadedComponents['ControlNet'] = true;
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
                  child: const Text('Load ControlNet'),
                ),
                const SizedBox(width: 8),
                ShadButton(
                  enabled: _controlNetPath != null &&
                      !(isModelLoading || isGenerating),
                  onPressed: showModelLoadDialog,
                  child: const Text('Load Model'),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
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
                                  constraints:
                                      const BoxConstraints(maxWidth: 400),
                                  title: const Text('Select TAESD Model'),
                                  description: SizedBox(
                                    height: 300,
                                    child: Material(
                                      color: Colors.transparent,
                                      child: ShadTable.list(
                                        header: const [
                                          ShadTableCell.header(
                                              child: Text('Model',
                                                  style:
                                                      TextStyle(fontSize: 16))),
                                          ShadTableCell.header(
                                              alignment: Alignment.centerRight,
                                              child: Text('Size',
                                                  style:
                                                      TextStyle(fontSize: 16))),
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
                                        children: taesdFiles
                                            .asMap()
                                            .entries
                                            .map((entry) => [
                                                  ShadTableCell(
                                                    child: GestureDetector(
                                                      onTap: () =>
                                                          Navigator.pop(context,
                                                              entry.value.path),
                                                      child: Padding(
                                                        padding:
                                                            const EdgeInsets
                                                                .symmetric(
                                                                vertical: 12.0),
                                                        child: Text(
                                                          entry.value.path
                                                              .split('/')
                                                              .last,
                                                          style:
                                                              const TextStyle(
                                                                  fontWeight:
                                                                      FontWeight
                                                                          .w500,
                                                                  fontSize: 14),
                                                        ),
                                                      ),
                                                    ),
                                                  ),
                                                  ShadTableCell(
                                                    alignment:
                                                        Alignment.centerRight,
                                                    child: GestureDetector(
                                                      onTap: () =>
                                                          Navigator.pop(context,
                                                              entry.value.path),
                                                      child: Text(
                                                        '${(entry.value.lengthSync() / (1024 * 1024)).toStringAsFixed(1)} MB',
                                                        style: const TextStyle(
                                                            fontSize: 12),
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
                      child: const Text('Load TAESD'),
                    ),
                    const SizedBox(width: 8),
                    ShadCheckbox(
                      value: useTAESD,
                      onChanged: (bool v) {
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
                if (_ramUsage.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 8.0),
                    child: Text(_ramUsage, style: theme.textTheme.p),
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
            GestureDetector(
              onTap: (isModelLoading || isGenerating)
                  ? null
                  : () async {
                      if (_hasDrawing) {
                        final shouldClear = await showShadDialog<bool>(
                          context: context,
                          builder: (context) => ShadDialog.alert(
                            title: const Text('Continue Drawing?'),
                            description: const Text(
                                'Would you like to continue with your current drawing or start a new one?'),
                            actions: [
                              ShadButton.outline(
                                onPressed: () =>
                                    Navigator.of(context).pop(true),
                                child: const Text('New Drawing'),
                              ),
                              ShadButton(
                                onPressed: () =>
                                    Navigator.of(context).pop(false),
                                child: const Text('Continue'),
                              ),
                            ],
                          ),
                        );

                        if (shouldClear == true) {
                          _drawingController.clear();
                        }
                      }
                      _openDrawingBoard();
                    },
              child: DottedBorder(
                borderType: BorderType.RRect,
                radius: const Radius.circular(8),
                color: theme.colorScheme.primary.withOpacity(0.5),
                strokeWidth: 2,
                dashPattern: const [8, 4],
                child: Container(
                  height: 200,
                  width: double.infinity,
                  child: _hasDrawing && _drawingImageData != null
                      ? Image.memory(_drawingImageData!, fit: BoxFit.contain)
                      : Center(
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(
                                Icons.brush,
                                size: 64,
                                color:
                                    theme.colorScheme.primary.withOpacity(0.5),
                              ),
                              const SizedBox(height: 12),
                              Text(
                                'Tap to draw',
                                style: TextStyle(
                                  color: theme.colorScheme.primary
                                      .withOpacity(0.5),
                                  fontSize: 16,
                                ),
                              ),
                            ],
                          ),
                        ),
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
                                        style: theme.textTheme.p
                                            .copyWith(fontSize: 13),
                                      ),
                                    ),
                                  );
                                }).toList(),
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
                                                                fontSize: 16))),
                                                    ShadTableCell.header(
                                                        alignment: Alignment
                                                            .centerRight,
                                                        child: Text('Size',
                                                            style: TextStyle(
                                                                fontSize: 16))),
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
                                                  children: vaeFiles
                                                      .asMap()
                                                      .entries
                                                      .map((entry) => [
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
                                                                    style: const TextStyle(
                                                                        fontWeight:
                                                                            FontWeight
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
                                                                onTap: () =>
                                                                    Navigator.pop(
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
                              onChanged: (bool v) {
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
                        const SizedBox(height: 8),
                        ShadCheckbox(
                          value: useVAETiling,
                          onChanged: (bool v) {
                            if (useTAESD) {
                              _showTemporaryError(
                                  'VAE Tiling is incompatible with TAESD');
                              return;
                            }
                            setState(() {
                              useVAETiling = v;
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
                const Text('Sampling Method'),
                const SizedBox(width: 8),
                Expanded(
                  child: ShadSelect<String>(
                    placeholder: const Text('euler_a'),
                    options: samplingMethods
                        .map((method) =>
                            ShadOption(value: method, child: Text(method)))
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
                const Text('Control Strength'),
                const SizedBox(width: 8),
                Expanded(
                  child: ShadSlider(
                    initialValue: controlStrength,
                    min: 0.0,
                    max: 1.0,
                    divisions: 20,
                    onChanged: (v) => setState(() => controlStrength = v),
                  ),
                ),
                Text(controlStrength.toStringAsFixed(2)),
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
                        .map((w) =>
                            ShadOption(value: w, child: Text(w.toString())))
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
                        .map((h) =>
                            ShadOption(value: h, child: Text(h.toString())))
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
                if (!_hasDrawing) {
                  _showTemporaryError('Please create a drawing first');
                  return;
                }
                setState(() {
                  isGenerating = true;
                  _modelError = '';
                  status = 'Generating image...';
                  progress = 0;
                });

                _processor!.generateImage(
                  prompt: prompt,
                  negativePrompt: negativePrompt,
                  cfgScale: cfg,
                  sampleSteps: steps,
                  width: width,
                  height: height,
                  seed: int.tryParse(seed) ?? -1,
                  sampleMethod: SampleMethod.values
                      .firstWhere(
                        (method) =>
                            method.displayName.toLowerCase() ==
                            samplingMethod.toLowerCase(),
                        orElse: () => SampleMethod.EULER_A,
                      )
                      .index,
                  controlImageData: _controlImageData,
                  controlImageWidth: _controlWidth,
                  controlImageHeight: _controlHeight,
                  controlStrength: controlStrength,
                );
              },
              child: const Text('Generate'),
            ),
            if (_modelError.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 8.0),
                child: Text(
                  _modelError,
                  style: const TextStyle(
                      color: Colors.red, fontWeight: FontWeight.bold),
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
}
