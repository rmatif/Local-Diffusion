import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'dart:async';
import 'dart:developer' as developer;
import 'dart:math' as math; // Import dart:math
import 'package:shadcn_ui/shadcn_ui.dart';
import 'package:file_picker/file_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:dotted_border/dotted_border.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'ffi_bindings.dart';
import 'inpainting_page.dart';
import 'outpainting_page.dart';
import 'photomaker_page.dart';
import 'stable_diffusion_processor.dart';
import 'utils.dart';
import 'main.dart';
import 'img2img_page.dart'; // Keep for reference if needed, but logic is being copied
import 'upscaler_page.dart';
import 'package:flutter_drawing_board/flutter_drawing_board.dart';
import 'package:image/image.dart' as img;
import 'image_processing_utils.dart'; // Import for CroppedImageData and InvertedRectClipper

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
  double clipSkip = 0.0; // Default from main.dart
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
  String _loadingError = ''; // Consolidated error message
  String _loadingErrorType = ''; // To track which component failed
  Timer? _loadingErrorTimer; // Timer to clear the loading error
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
  Uint8List? _drawingImageData; // PNG bytes for display
  bool _hasDrawing = false;
  int? _originalDrawingWidth; // Store original drawing dimensions
  int? _originalDrawingHeight;
  Uint8List?
      _originalDrawingRgbBytes; // Store original RGB bytes before cropping/resizing
  List<String> _generationLogs = []; // To store logs for the last generation
  bool _showLogsButton = false; // To control visibility of the log button
  // --- Added state variables from main.dart for Advanced Sampling Options ---
  double eta = 0.0;
  double guidance = 3.5;
  double slgScale = 0.0;
  String skipLayersText = '';
  double skipLayerStart = 0.01;
  double skipLayerEnd = 0.2;
  final TextEditingController _skipLayersController = TextEditingController();
  String? _skipLayersErrorText;
  // --- End added state variables ---

  // --- State for Cropping ---
  int _maxCropWidth = 512; // Default, will be updated
  int _maxCropHeight = 512; // Default, will be updated
  // Use 'width' and 'height' state variables for current slider values
  Rect _cropRect = Rect.zero; // Position relative to the displayed image widget
  Size _imageDisplaySize = Size.zero; // Size of the image widget as displayed
  final GlobalKey _imageKey = GlobalKey();
  Offset _dragStartOffset = Offset.zero;
  Rect _dragStartCropRect = Rect.zero;
  bool _showCropUI = false; // Flag to control visibility of crop UI
  // --- End State for Cropping ---
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
    // _modelErrorTimer?.cancel(); // Removed
    _loadingErrorTimer?.cancel(); // Cancel the general loading error timer
    _processor?.dispose();
    _drawingController.dispose(); // Dispose drawing controller
    _promptController.dispose(); // Dispose text controller
    _skipLayersController.dispose(); // Dispose new controller
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
      isDiffusionModelType: false, // Scribble likely doesn't use this path
      onModelLoaded: () {
        setState(() {
          isModelLoading = false;
          _message = 'Model initialized successfully';
          loadedComponents['Model'] = true;
          loadingText = '';
          _loadingError = ''; // Clear any previous loading errors on success
          _loadingErrorType = '';
          _loadingErrorTimer?.cancel();
        });
      },
      onLog: (log) {
        // Handle RAM usage log
        if (log.message.contains('total params memory size')) {
          final regex = RegExp(r'total params memory size = ([\d.]+)MB');
          final match = regex.firstMatch(log.message);
          if (match != null) {
            setState(() {
              _ramUsage = 'Total RAM: ${match.group(1)}MB';
            });
          }
        }

        // Check for error messages passed via the log stream
        if (log.level == -1 && log.message.startsWith("Error (")) {
          final errorMatch =
              RegExp(r'Error \((.*?)\): (.*)').firstMatch(log.message);
          if (errorMatch != null) {
            final errorType = errorMatch.group(1)!;
            final errorMessage = errorMatch.group(2)!;
            _handleLoadingError(errorType, errorMessage); // Use the new handler
          }
        } else {
          // Log other messages normally
          developer.log(log.message);
        }
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

    _processor!.generationResultStream.listen((result) async {
      // Use new stream
      final ui.Image image = result['image']; // Extract image from map
      final String? generationTime = result['generationTime']; // Extract time

      final bytes = await image.toByteData(format: ui.ImageByteFormat.png);

      setState(() {
        isGenerating = false;
        _generatedImage = Image.memory(bytes!.buffer.asUint8List());
        // Update status using the extracted time
        status = generationTime != null
            ? 'Generation completed in $generationTime'
            : 'Generation complete';
        _showLogsButton = true; // Show the log button
      });

      await _processor!.saveGeneratedImage(
        image, // Use extracted image
        prompt,
        width,
        height,
        SampleMethod.values.firstWhere(
          (method) =>
              method.displayName.toLowerCase() == samplingMethod.toLowerCase(),
          orElse: () => SampleMethod.EULER_A,
        ),
      );

      // No need for the redundant status update here
    });
  }

  // --- Copied Error Handling Logic from main.dart ---
  // New method to handle loading errors centrally - FULL RESET
  void _handleLoadingError(String errorType, String errorMessage) {
    _loadingErrorTimer?.cancel(); // Cancel previous timer if any

    // Dispose the processor and kill the isolate regardless of error type
    _processor?.dispose();

    setState(() {
      _processor = null; // Set processor to null
      isModelLoading = false; // Stop loading indicator
      loadingText = ''; // Clear loading text
      _loadingError = errorMessage; // Display the specific error
      _loadingErrorType = errorType; // Store error type if needed elsewhere

      // --- Full Reset ---
      // Clear all loaded component indicators
      loadedComponents.clear();
      // Reset all paths
      _taesdPath = null;
      _loraPath = null;
      // Scribble doesn't use Clip L/G/T5XXL directly in this setup
      _vaePath = null;
      _embedDirPath = null;
      _controlNetPath = null;
      // Reset related flags
      useTAESD = false;
      useVAE = false;
      useVAETiling = false;
      // Reset other related state specific to scribble
      _loraNames = [];
      _ramUsage = ''; // Clear RAM usage display
      _drawingImageData = null; // Reset drawing
      _hasDrawing = false;
      _originalDrawingWidth = null;
      _originalDrawingHeight = null;
      _originalDrawingRgbBytes = null;
      _showCropUI = false; // Hide crop UI
      _cropRect = Rect.zero;
      _imageDisplaySize = Size.zero;
      _controlImageData = null; // Reset control data derived from drawing
      _controlWidth = null;
      _controlHeight = null;
      _message = ''; // Clear success messages too
      _taesdMessage = '';
      _loraMessage = '';

      // Handle generation-specific errors separately if needed
      if (errorType == 'generationError') {
        status = 'Generation failed: $errorMessage';
        isGenerating = false; // Stop generation indicator
      } else if (errorType == 'inputError' || errorType == 'dimensionError') {
        status = '';
        progress = 0;
      } else {
        // For loading errors, clear status too
        status = '';
        progress = 0;
      }
      // --- End Full Reset ---

      // Clear the error message after a delay
      _loadingErrorTimer = Timer(const Duration(seconds: 10), () {
        if (mounted) {
          // Check if the widget is still in the tree
          setState(() {
            _loadingError = '';
            _loadingErrorType = '';
          });
        }
      });
    });
  }
  // --- End Copied Error Handling Logic ---

  // Method to show the logs dialog (Copied from main.dart)
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
                              final pngBytesForDisplay =
                                  imageData.buffer.asUint8List();
                              final originalImage =
                                  img.decodeImage(pngBytesForDisplay);

                              if (originalImage != null) {
                                // --- Store Original Data ---
                                _originalDrawingWidth = originalImage.width;
                                _originalDrawingHeight = originalImage.height;

                                // Convert original to RGB immediately for later cropping
                                final originalRgbBytes = Uint8List(
                                    _originalDrawingWidth! *
                                        _originalDrawingHeight! *
                                        3);
                                int rgbIndex = 0;
                                for (int y = 0;
                                    y < _originalDrawingHeight!;
                                    y++) {
                                  for (int x = 0;
                                      x < _originalDrawingWidth!;
                                      x++) {
                                    final pixel = originalImage.getPixel(x, y);
                                    originalRgbBytes[rgbIndex++] =
                                        pixel.r.toInt();
                                    originalRgbBytes[rgbIndex++] =
                                        pixel.g.toInt();
                                    originalRgbBytes[rgbIndex++] =
                                        pixel.b.toInt();
                                  }
                                }
                                _originalDrawingRgbBytes = originalRgbBytes;
                                // --- End Store Original Data ---

                                // --- Initialize Cropping State ---
                                _maxCropWidth =
                                    largestMultipleOf64(_originalDrawingWidth!);
                                _maxCropHeight = largestMultipleOf64(
                                    _originalDrawingHeight!);

                                // Reset width/height sliders to default/initial aspect ratio
                                double initialAspectRatio =
                                    _originalDrawingWidth! /
                                        _originalDrawingHeight!;
                                int initialCropW, initialCropH;
                                if (initialAspectRatio >= 1) {
                                  // Wider or square
                                  initialCropW = math.min(512, _maxCropWidth);
                                  initialCropH = largestMultipleOf64(
                                      (initialCropW / initialAspectRatio)
                                          .round());
                                  initialCropH =
                                      math.min(initialCropH, _maxCropHeight);
                                  initialCropW = largestMultipleOf64(
                                      (initialCropH * initialAspectRatio)
                                          .round());
                                  initialCropW =
                                      math.min(initialCropW, _maxCropWidth);
                                } else {
                                  // Taller
                                  initialCropH = math.min(512, _maxCropHeight);
                                  initialCropW = largestMultipleOf64(
                                      (initialCropH * initialAspectRatio)
                                          .round());
                                  initialCropW =
                                      math.min(initialCropW, _maxCropWidth);
                                  initialCropH = largestMultipleOf64(
                                      (initialCropW / initialAspectRatio)
                                          .round());
                                  initialCropH =
                                      math.min(initialCropH, _maxCropHeight);
                                }
                                // Ensure minimum size
                                int newWidth = math.max(64, initialCropW);
                                int newHeight = math.max(64, initialCropH);
                                // --- End Initialize Cropping State ---

                                setState(() {
                                  _drawingImageData =
                                      pngBytesForDisplay; // Use original PNG for display
                                  _hasDrawing = true;
                                  _showCropUI = true; // Enable cropping UI

                                  // Set initial slider values
                                  width = newWidth;
                                  height = newHeight;

                                  // Reset crop rect position and display size
                                  _cropRect = Rect.zero;
                                  _imageDisplaySize = Size.zero;

                                  // Set initial control data (will be updated by cropping before generation)
                                  _controlImageData = _originalDrawingRgbBytes;
                                  _controlWidth = _originalDrawingWidth;
                                  _controlHeight = _originalDrawingHeight;
                                });

                                // Calculate initial crop rect after the frame renders
                                WidgetsBinding.instance.addPostFrameCallback(
                                    _calculateInitialCropRect);

                                Navigator.of(context)
                                    .pop(); // Close the drawing dialog
                              } else {
                                Navigator.of(context)
                                    .pop(); // Close if image decoding failed
                              }
                            } else {
                              Navigator.of(context)
                                  .pop(); // Close if no image data
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

  // --- Cropping Logic (adapted from img2img_page.dart) ---

  void _calculateInitialCropRect(_) {
    if (!mounted) return;
    final RenderBox? imageBox =
        _imageKey.currentContext?.findRenderObject() as RenderBox?;
    if (imageBox != null && imageBox.hasSize) {
      // Check if size is valid before proceeding
      if (imageBox.size.width > 0 && imageBox.size.height > 0) {
        setState(() {
          _imageDisplaySize = imageBox.size;
          _updateCropRect(center: true); // Center the initial crop rect
        });
      } else {
        // Retry if the size wasn't valid yet
        WidgetsBinding.instance.addPostFrameCallback(_calculateInitialCropRect);
      }
    } else {
      // Retry if the size wasn't available yet
      WidgetsBinding.instance.addPostFrameCallback(_calculateInitialCropRect);
    }
  }

  // Updates the _cropRect based on current dimensions and optionally centers it
  void _updateCropRect({bool center = false}) {
    if (_imageDisplaySize == Size.zero ||
        _imageDisplaySize.width <= 0 ||
        _imageDisplaySize.height <= 0) return; // Not ready yet or invalid size
    if (width <= 0 || height <= 0) return; // Invalid crop dimensions

    // Use the current 'width' and 'height' state variables for crop dimensions
    double currentCropWidth = width.toDouble();
    double currentCropHeight = height.toDouble();

    double displayAspect = _imageDisplaySize.width / _imageDisplaySize.height;
    double cropAspect = currentCropWidth / currentCropHeight;

    double rectWidth, rectHeight;

    // Calculate the dimensions of the crop rectangle within the image display bounds
    if (displayAspect > cropAspect) {
      // Image display is wider than crop aspect ratio
      rectHeight = _imageDisplaySize.height;
      rectWidth = rectHeight * cropAspect;
    } else {
      // Image display is taller or same aspect ratio
      rectWidth = _imageDisplaySize.width;
      rectHeight = rectWidth / cropAspect;
    }

    // Ensure calculated dimensions are positive
    rectWidth = math.max(1.0, rectWidth);
    rectHeight = math.max(1.0, rectHeight);

    double left, top;

    if (center || _cropRect == Rect.zero) {
      // Center if first time or explicitly requested
      left = (_imageDisplaySize.width - rectWidth) / 2;
      top = (_imageDisplaySize.height - rectHeight) / 2;
    } else {
      // Keep the current center if possible, otherwise clamp to bounds
      double currentCenterX = _cropRect.left + _cropRect.width / 2;
      double currentCenterY = _cropRect.top + _cropRect.height / 2;

      left = currentCenterX - rectWidth / 2;
      top = currentCenterY - rectHeight / 2;
    }

    // Clamp the rectangle to the image bounds
    left = left.clamp(0.0, _imageDisplaySize.width - rectWidth);
    top = top.clamp(0.0, _imageDisplaySize.height - rectHeight);

    // Ensure left/top are not NaN
    left = left.isNaN ? 0.0 : left;
    top = top.isNaN ? 0.0 : top;

    // Only update state if the rect actually changes to avoid infinite loops
    Rect newRect = Rect.fromLTWH(left, top, rectWidth, rectHeight);
    if (newRect != _cropRect) {
      setState(() {
        _cropRect = newRect;
      });
    }
  }

  void _onPanStart(DragStartDetails details) {
    if (_cropRect.contains(details.localPosition)) {
      _dragStartOffset = details.localPosition;
      _dragStartCropRect = _cropRect;
    } else {
      _dragStartOffset = Offset.zero; // Indicate drag didn't start inside
    }
  }

  void _onPanUpdate(DragUpdateDetails details) {
    if (_dragStartOffset == Offset.zero) return; // Drag didn't start inside

    final Offset delta = details.localPosition - _dragStartOffset;
    double newLeft = _dragStartCropRect.left + delta.dx;
    double newTop = _dragStartCropRect.top + delta.dy;

    // Clamp movement within the image bounds
    newLeft = newLeft.clamp(0.0, _imageDisplaySize.width - _cropRect.width);
    newTop = newTop.clamp(0.0, _imageDisplaySize.height - _cropRect.height);

    // Ensure left/top are not NaN
    newLeft = newLeft.isNaN ? 0.0 : newLeft;
    newTop = newTop.isNaN ? 0.0 : newTop;

    setState(() {
      _cropRect =
          Rect.fromLTWH(newLeft, newTop, _cropRect.width, _cropRect.height);
    });
  }

  // Performs the actual cropping based on current state
  // Returns null if cropping is not active or fails
  Future<CroppedImageData?> _getCroppedDrawingData() async {
    // Only crop if the UI is shown and we have original drawing data
    if (!_showCropUI ||
        _originalDrawingRgbBytes == null ||
        _originalDrawingWidth == null ||
        _originalDrawingHeight == null) {
      return null;
    }
    // Ensure display size is calculated and valid
    if (_imageDisplaySize == Size.zero ||
        _imageDisplaySize.width <= 0 ||
        _imageDisplaySize.height <= 0) {
      print(
          "Warning: Image display size not calculated or invalid. Cannot crop.");
      // Attempt to calculate it now, might delay generation slightly
      final RenderBox? imageBox =
          _imageKey.currentContext?.findRenderObject() as RenderBox?;
      if (imageBox != null &&
          imageBox.hasSize &&
          imageBox.size.width > 0 &&
          imageBox.size.height > 0) {
        _imageDisplaySize = imageBox.size;
        _updateCropRect(); // Try updating rect again
      } else {
        _showTemporaryError('Cannot determine image size for cropping.');
        return null; // Still can't get size
      }
    }
    // Ensure crop rect is valid
    if (_cropRect == Rect.zero ||
        _cropRect.width <= 0 ||
        _cropRect.height <= 0) {
      print("Warning: Invalid crop rectangle: $_cropRect");
      // Attempt to recalculate if it seems invalid
      _updateCropRect(center: true);
      if (_cropRect == Rect.zero ||
          _cropRect.width <= 0 ||
          _cropRect.height <= 0) {
        _showTemporaryError('Invalid crop area selected.');
        return null;
      }
    }

    // 1. Use the stored original RGB bytes
    final Uint8List originalBytes = _originalDrawingRgbBytes!;
    final int originalWidth = _originalDrawingWidth!;
    final int originalHeight = _originalDrawingHeight!;

    // Create an img.Image from raw RGB bytes
    final img.Image? originalImage = img.Image.fromBytes(
        width: originalWidth,
        height: originalHeight,
        bytes: originalBytes.buffer,
        numChannels: 3); // Assuming 3 channels (RGB)

    if (originalImage == null) {
      print("Error: Could not create image from RGB bytes for cropping.");
      _showTemporaryError('Error processing drawing for cropping');
      return null;
    }

    // 2. Calculate the source crop rectangle in original image coordinates
    final double scaleX = originalWidth / _imageDisplaySize.width;
    final double scaleY = originalHeight / _imageDisplaySize.height;

    final int srcX = (_cropRect.left * scaleX).round().clamp(0, originalWidth);
    final int srcY = (_cropRect.top * scaleY).round().clamp(0, originalHeight);
    // Clamp width/height to ensure they don't exceed bounds from srcX/srcY
    final int srcWidth = (_cropRect.width * scaleX)
        .round()
        .clamp(1, originalWidth - srcX); // Ensure at least 1 pixel
    final int srcHeight = (_cropRect.height * scaleY)
        .round()
        .clamp(1, originalHeight - srcY); // Ensure at least 1 pixel

    // Double check dimensions after clamping
    if (srcWidth <= 0 || srcHeight <= 0) {
      print(
          "Error: Calculated crop dimensions are invalid after clamping ($srcWidth x $srcHeight). CropRect: $_cropRect, DisplaySize: $_imageDisplaySize, Scale: $scaleX, $scaleY");
      _showTemporaryError('Invalid crop area calculation.');
      return null;
    }

    // 3. Crop the image using img.copyCrop
    final img.Image croppedImage = img.copyCrop(
      originalImage,
      x: srcX,
      y: srcY,
      width: srcWidth,
      height: srcHeight,
    );

    // 4. Resize the cropped image to the target 'width' and 'height' (from sliders)
    final img.Image resizedImage = img.copyResize(
      croppedImage,
      width: width, // Use state variable 'width'
      height: height, // Use state variable 'height'
      interpolation: img
          .Interpolation.average, // Use average for potentially better quality
    );

    // 5. Encode the resized image back to Uint8List (RGB format)
    final Uint8List finalImageBytes = Uint8List(width * height * 3);
    int rgbIndex = 0;
    try {
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final pixel = resizedImage.getPixel(x, y);
          finalImageBytes[rgbIndex++] = pixel.r.toInt();
          finalImageBytes[rgbIndex++] = pixel.g.toInt();
          finalImageBytes[rgbIndex++] = pixel.b.toInt();
        }
      }
    } catch (e) {
      print("Error converting resized image to RGB bytes: $e");
      _showTemporaryError('Error processing final image.');
      return null;
    }

    // 6. Return the data
    return CroppedImageData(
      imageBytes: finalImageBytes, // Return raw RGB bytes
      maskBytes: null, // No mask in scribble cropping
      width: width, // Final width from slider
      height: height, // Final height from slider
    );
  }

  // --- End Cropping Logic ---
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
              tileColor: theme.colorScheme.secondary.withOpacity(0.2),
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
            // Display Loading Status / Success / Error Messages (Copied)
            Padding(
              padding: const EdgeInsets.only(bottom: 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Display Loading Error FIRST if present
                  if (_loadingError.isNotEmpty)
                    Text.rich(
                      TextSpan(
                        children: [
                          const WidgetSpan(
                            child: Icon(
                              Icons.error_outline, // Error Icon
                              size: 20,
                              color: Colors.red,
                            ),
                            alignment: PlaceholderAlignment.middle,
                          ),
                          const WidgetSpan(child: SizedBox(width: 6)),
                          TextSpan(
                            text: _loadingError, // Display the error message
                            style: theme.textTheme.p.copyWith(
                              color: Colors.red,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                    )
                        .animate()
                        .fadeIn(duration: const Duration(milliseconds: 300))
                        .shake(hz: 4, offset: const Offset(2, 0)), // Use offset
                  // Display Success Messages ONLY if NO loading error is active
                  if (_loadingError.isEmpty)
                    ...loadedComponents.entries.map((entry) => Text.rich(
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
                                  Icons.check_circle_outline, // Check Icon
                                  size: 20,
                                  color: Colors.green,
                                ),
                                alignment: PlaceholderAlignment.middle,
                              ),
                            ],
                          ),
                        )
                            .animate()
                            .fadeIn(duration: const Duration(milliseconds: 500))
                            .slideY(begin: -0.2, end: 0)),
                  // Display Loading Text if loading and NO error
                  if (loadingText.isNotEmpty && _loadingError.isEmpty)
                    const SizedBox(height: 8),
                  if (loadingText.isNotEmpty && _loadingError.isEmpty)
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
                child: AspectRatio(
                  // Maintain aspect ratio of the original drawing
                  aspectRatio: (_originalDrawingWidth != null &&
                          _originalDrawingHeight != null &&
                          _originalDrawingHeight! > 0)
                      ? _originalDrawingWidth! / _originalDrawingHeight!
                      : 1.0, // Default aspect ratio if no drawing
                  child: Container(
                    child: !_hasDrawing || _drawingImageData == null
                        ? Center(
                            // Placeholder when no drawing
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(
                                  Icons.brush,
                                  size: 64,
                                  color: theme.colorScheme.primary
                                      .withOpacity(0.5),
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
                          )
                        : LayoutBuilder(
                            // Use LayoutBuilder for display size
                            builder: (context, constraints) {
                              // Update display size post-frame if changed
                              WidgetsBinding.instance.addPostFrameCallback((_) {
                                if (mounted &&
                                    _imageDisplaySize != constraints.biggest) {
                                  // Check if size is valid before setting state
                                  if (constraints.biggest.width > 0 &&
                                      constraints.biggest.height > 0) {
                                    // Use setState only if size actually changed and is valid
                                    if (_imageDisplaySize !=
                                        constraints.biggest) {
                                      setState(() {
                                        _imageDisplaySize = constraints.biggest;
                                        _updateCropRect(); // Recalculate rect
                                      });
                                    }
                                  }
                                }
                              });

                              return GestureDetector(
                                onPanStart: _showCropUI ? _onPanStart : null,
                                onPanUpdate: _showCropUI ? _onPanUpdate : null,
                                child: Stack(
                                  fit: StackFit.expand,
                                  children: [
                                    // Drawing Preview
                                    Image.memory(
                                      _drawingImageData!,
                                      key: _imageKey, // Assign key here
                                      fit: BoxFit.contain,
                                      gaplessPlayback: true, // Smoother updates
                                    ),
                                    // Cropping Frame Overlay (only if cropping)
                                    if (_showCropUI && _cropRect != Rect.zero)
                                      Positioned.fromRect(
                                        rect: _cropRect,
                                        child: Container(
                                          decoration: BoxDecoration(
                                            border: Border.all(
                                              color:
                                                  Colors.white.withOpacity(0.8),
                                              width: 2,
                                            ),
                                          ),
                                        ),
                                      ),
                                    // Dimming outside the crop area (only if cropping)
                                    if (_showCropUI && _cropRect != Rect.zero)
                                      ClipPath(
                                        clipper: InvertedRectClipper(
                                            clipRect: _cropRect),
                                        child: Container(
                                          color: Colors.black.withOpacity(0.5),
                                        ),
                                      ),
                                  ],
                                ),
                              );
                            },
                          ),
                  ),
                ),
              ),
            ),
            // --- Sliders for Cropping ---
            if (_showCropUI) ...[
              const SizedBox(height: 16),
              Text('Crop Width: $width px'),
              ShadSlider(
                min: 64,
                max: _maxCropWidth.toDouble(),
                divisions:
                    (_maxCropWidth > 64) ? (_maxCropWidth - 64) ~/ 64 : 1,
                initialValue: width.toDouble().clamp(
                    64.0, _maxCropWidth.toDouble()), // Clamp initial value
                onChanged: (value) {
                  int newWidth = largestMultipleOf64(value.round());
                  if (newWidth != width) {
                    setState(() {
                      width = newWidth.clamp(64, _maxCropWidth); // Clamp result
                      // Maintain aspect ratio (optional, could allow free resize)
                      // double aspect = width / height.toDouble();
                      // height = largestMultipleOf64((width / aspect).round());
                      // height = height.clamp(64, _maxCropHeight);
                      // width = largestMultipleOf64((height * aspect).round());
                      // width = width.clamp(64, _maxCropWidth);
                      _updateCropRect();
                    });
                  }
                },
              ),
              const SizedBox(height: 10),
              Text('Crop Height: $height px'),
              ShadSlider(
                min: 64,
                max: _maxCropHeight.toDouble(),
                divisions:
                    (_maxCropHeight > 64) ? (_maxCropHeight - 64) ~/ 64 : 1,
                initialValue: height.toDouble().clamp(
                    64.0, _maxCropHeight.toDouble()), // Clamp initial value
                onChanged: (value) {
                  int newHeight = largestMultipleOf64(value.round());
                  if (newHeight != height) {
                    setState(() {
                      height =
                          newHeight.clamp(64, _maxCropHeight); // Clamp result
                      // Maintain aspect ratio (optional)
                      // double aspect = width.toDouble() / height;
                      // width = largestMultipleOf64((height * aspect).round());
                      // width = width.clamp(64, _maxCropWidth);
                      // height = largestMultipleOf64((width / aspect).round());
                      // height = height.clamp(64, _maxCropHeight);
                      _updateCropRect();
                    });
                  }
                },
              ),
              const SizedBox(height: 16), // Spacing after sliders
            ],
            // --- End Sliders ---
            const SizedBox(height: 16),
            // --- Moved Advanced Model Options Accordion ---
            ShadAccordion<Map<String, dynamic>>(
              children: [
                ShadAccordionItem<Map<String, dynamic>>(
                  value: const {},
                  title: const Text('Advanced Model Options'), // Renamed title
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
                        // VAE Tiling removed from here
                      ],
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            // --- End Moved Accordion ---
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
            // --- New Advanced Sampling Options Accordion ---
            ShadAccordion<Map<String, dynamic>>(
              children: [
                ShadAccordionItem<Map<String, dynamic>>(
                  value: const {}, // Unique value for this item
                  title: const Text('Advanced Sampling Options'),
                  child: Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // Moved VAE Tiling Checkbox
                        ShadCheckbox(
                          value: useVAETiling,
                          onChanged: (isModelLoading || isGenerating)
                              ? null
                              : (bool v) {
                                  if (useTAESD && v) {
                                    _showTemporaryError(
                                        'VAE Tiling is incompatible with TAESD');
                                    return;
                                  }
                                  setState(() {
                                    useVAETiling = v;
                                    // Reinitialize processor if needed
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
                        const SizedBox(height: 16),

                        // Clip Skip Slider
                        Row(
                          children: [
                            const Text('Clip Skip'),
                            const SizedBox(width: 8),
                            Expanded(
                              child: ShadSlider(
                                initialValue: clipSkip,
                                min: 0,
                                max: 2,
                                divisions: 2,
                                onChanged: (v) => setState(() => clipSkip = v),
                              ),
                            ),
                            Text(clipSkip.toInt().toString()),
                          ],
                        ),
                        const SizedBox(height: 16),

                        // Eta Slider
                        Row(
                          children: [
                            const Text('Eta'),
                            const SizedBox(width: 8),
                            Expanded(
                              child: ShadSlider(
                                initialValue: eta,
                                min: 0.0,
                                max: 1.0,
                                divisions: 20, // 1.0 / 0.05 = 20
                                onChanged: (v) => setState(() => eta = v),
                              ),
                            ),
                            Text(eta.toStringAsFixed(2)),
                          ],
                        ),
                        const SizedBox(height: 16),

                        // Guidance Slider
                        Row(
                          children: [
                            const Text('Guidance'),
                            const SizedBox(width: 8),
                            Expanded(
                              child: ShadSlider(
                                initialValue: guidance,
                                min: 0.0,
                                max: 40.0,
                                divisions: 800, // 40.0 / 0.05 = 800
                                onChanged: (v) => setState(() => guidance = v),
                              ),
                            ),
                            Text(guidance.toStringAsFixed(2)),
                          ],
                        ),
                        const SizedBox(height: 16),

                        // SLG Scale Slider
                        Row(
                          children: [
                            const Text('SLG Scale'),
                            const SizedBox(width: 8),
                            Expanded(
                              child: ShadSlider(
                                initialValue: slgScale,
                                min: 0.0,
                                max: 7.0,
                                divisions: 140, // 7.0 / 0.05 = 140
                                onChanged: (v) => setState(() => slgScale = v),
                              ),
                            ),
                            Text(slgScale.toStringAsFixed(2)),
                          ],
                        ),
                        const SizedBox(height: 16),

                        // Skip Layers Text Field
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            ShadInput(
                              controller: _skipLayersController,
                              placeholder:
                                  const Text('Skip Layers (e.g., 7,8,9)'),
                              keyboardType: TextInputType.text,
                              // Removed errorText, handled below
                              onChanged: (String? v) {
                                final text = v ?? '';
                                // Basic validation: allow empty, or numbers separated by commas
                                final regex = RegExp(r'^(?:\d+(?:,\s*\d+)*)?$');
                                if (text.isEmpty || regex.hasMatch(text)) {
                                  setState(() {
                                    skipLayersText = text;
                                    _skipLayersErrorText = null; // Clear error
                                  });
                                } else {
                                  setState(() {
                                    // Keep skipLayersText as is, but show error
                                    _skipLayersErrorText =
                                        'Invalid format (use numbers separated by commas)';
                                  });
                                }
                              },
                            ),
                            if (_skipLayersErrorText != null)
                              Padding(
                                padding:
                                    const EdgeInsets.only(top: 4.0, left: 2.0),
                                child: Text(
                                  _skipLayersErrorText!,
                                  style: theme.textTheme.p.copyWith(
                                    color: Colors.red,
                                    fontSize: 12,
                                  ),
                                ),
                              ),
                          ],
                        ),
                        const SizedBox(height: 16),

                        // Skip Layer Start Slider
                        Row(
                          children: [
                            const Text('Skip Layer Start'),
                            const SizedBox(width: 8),
                            Expanded(
                              child: ShadSlider(
                                initialValue: skipLayerStart,
                                min: 0.0,
                                max: 1.0,
                                divisions: 100, // 1.0 / 0.01 = 100
                                onChanged: (v) {
                                  if (v < skipLayerEnd) {
                                    setState(() => skipLayerStart = v);
                                  }
                                },
                              ),
                            ),
                            Text(skipLayerStart.toStringAsFixed(2)),
                          ],
                        ),
                        const SizedBox(height: 16),

                        // Skip Layer End Slider
                        Row(
                          children: [
                            const Text('Skip Layer End'),
                            const SizedBox(width: 8),
                            Expanded(
                              child: ShadSlider(
                                initialValue: skipLayerEnd,
                                min: 0.0,
                                max: 1.0,
                                divisions: 100, // 1.0 / 0.01 = 100
                                onChanged: (v) {
                                  if (v > skipLayerStart) {
                                    setState(() => skipLayerEnd = v);
                                  }
                                },
                              ),
                            ),
                            Text(skipLayerEnd.toStringAsFixed(2)),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
            // --- End New Accordion ---
            const SizedBox(height: 16),
            // This accordion was moved above the prompt field
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
                    enabled: !_showCropUI, // Disable dropdown when cropping
                    placeholder: Text(
                      // Display effective width from slider or default
                      (_showCropUI ? width : 512).toString(),
                    ),
                    options: _showCropUI
                        ? const []
                        : getWidthOptions() // No options needed when disabled
                            .map((w) =>
                                ShadOption(value: w, child: Text(w.toString())))
                            .toList(),
                    selectedOptionBuilder: (context, value) =>
                        Text(value.toString()),
                    onChanged: _showCropUI
                        ? null
                        : (int? value) {
                            // Disable onChanged when cropping
                            if (value != null) setState(() => width = value);
                          },
                  ),
                ),
                const SizedBox(width: 16),
                const Text('Height'),
                const SizedBox(width: 8),
                Expanded(
                  child: ShadSelect<int>(
                    enabled: !_showCropUI, // Disable dropdown when cropping
                    placeholder: Text(
                      // Display effective height from slider or default
                      (_showCropUI ? height : 512).toString(),
                    ),
                    options: _showCropUI
                        ? const []
                        : getHeightOptions() // No options needed when disabled
                            .map((h) =>
                                ShadOption(value: h, child: Text(h.toString())))
                            .toList(),
                    selectedOptionBuilder: (context, value) =>
                        Text(value.toString()),
                    onChanged: _showCropUI
                        ? null
                        : (int? value) {
                            // Disable onChanged when cropping
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
                  onPressed: () async {
                    // Make onPressed async
                    if (_processor == null) {
                      _handleLoadingError(
                          'modelError', 'Please load a model first.');
                      return;
                    }
                    if (!_hasDrawing || _originalDrawingRgbBytes == null) {
                      _handleLoadingError(
                          'inputError', 'Please create a drawing first.');
                      return;
                    }
                    // Clear any previous loading errors before generating
                    if (_loadingError.isNotEmpty) {
                      setState(() {
                        _loadingError = '';
                        _loadingErrorType = '';
                        _loadingErrorTimer?.cancel();
                      });
                    }

                    setState(() {
                      isGenerating = true;
                      // _modelError = ''; // Removed
                      status = 'Preparing drawing...'; // Update status
                      progress = 0;
                      _generationLogs = []; // Clear previous logs
                      _showLogsButton = false; // Hide log button
                    });

                    // Perform cropping if UI is active
                    CroppedImageData? croppedData;
                    Uint8List? finalControlBytes;
                    int? finalControlWidth;
                    int? finalControlHeight;

                    if (_showCropUI) {
                      croppedData = await _getCroppedDrawingData();
                      if (croppedData == null) {
                        // Cropping failed or was invalid, stop generation
                        setState(() {
                          isGenerating = false;
                          status = 'Drawing cropping failed.';
                        });
                        return; // Don't proceed if cropping failed
                      }
                      finalControlBytes = croppedData.imageBytes;
                      finalControlWidth = croppedData.width;
                      finalControlHeight = croppedData.height;
                    } else {
                      // Not cropping, use original (or last used if applicable)
                      // This case might need refinement depending on desired behavior when cropping is off
                      finalControlBytes =
                          _controlImageData; // Use existing control data
                      finalControlWidth = _controlWidth;
                      finalControlHeight = _controlHeight;
                      // Ensure width/height match the control data if not cropping
                      // width = _controlWidth ?? 512;
                      // height = _controlHeight ?? 512;
                    }

                    // Ensure we have valid dimensions for generation
                    final int effectiveWidth = finalControlWidth ??
                        width; // Use cropped width or slider width
                    final int effectiveHeight = finalControlHeight ??
                        height; // Use cropped height or slider height

                    if (finalControlBytes == null ||
                        finalControlWidth == null ||
                        finalControlHeight == null) {
                      _handleLoadingError(
                          'inputError', 'Failed to prepare drawing data.');
                      setState(() => isGenerating = false);
                      return;
                    }

                    setState(() {
                      status =
                          'Generating image...'; // Update status before calling processor
                      // Update control data state just before generation
                      _controlImageData = finalControlBytes;
                      _controlWidth = finalControlWidth;
                      _controlHeight = finalControlHeight;
                    });

                    // --- Skip Layers Formatting ---
                    String? formattedSkipLayers;
                    if (_skipLayersErrorText == null &&
                        skipLayersText.trim().isNotEmpty) {
                      // Remove whitespace and format as "[num1,num2,...]"
                      final numbers = skipLayersText
                          .split(',')
                          .map((s) => s.trim())
                          .where((s) => s.isNotEmpty)
                          .toList();
                      if (numbers.isNotEmpty) {
                        formattedSkipLayers = '[${numbers.join(',')}]';
                      }
                    }
                    // --- End Skip Layers Formatting ---

                    _processor!.generateImage(
                      prompt: prompt,
                      negativePrompt: negativePrompt,
                      cfgScale: cfg,
                      sampleSteps: steps,
                      width: effectiveWidth, // Use effective width for output
                      height:
                          effectiveHeight, // Use effective height for output
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
                          finalControlBytes, // Use final processed bytes
                      controlImageWidth:
                          finalControlWidth, // Use final processed width
                      controlImageHeight:
                          finalControlHeight, // Use final processed height
                      controlStrength: controlStrength,
                      // Pass new sampling parameters
                      clipSkip: clipSkip.toInt(),
                      eta: eta,
                      guidance: guidance,
                      slgScale: slgScale,
                      skipLayersText:
                          formattedSkipLayers, // Pass formatted string or null
                      skipLayerStart: skipLayerStart,
                      skipLayerEnd: skipLayerEnd,
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
            // Removed old _modelError display logic
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
