import 'dart:io';
import 'dart:ui' as ui;
import 'dart:async';
import 'dart:typed_data';
import 'dart:developer' as developer;
import 'dart:math' as math; // Import dart:math

import 'package:flutter/material.dart';
import 'package:shadcn_ui/shadcn_ui.dart';
import 'package:file_picker/file_picker.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:dotted_border/dotted_border.dart';
import 'package:flutter_animate/flutter_animate.dart';

import 'canny_processor.dart';
import 'ffi_bindings.dart';
import 'img2img_processor.dart';
import 'inpainting_page.dart';
import 'outpainting_page.dart';
import 'scribble2img_page.dart';
import 'utils.dart';
import 'main.dart';
import 'upscaler_page.dart';
import 'photomaker_page.dart';
import 'image_processing_utils.dart'; // Import the image processing utils

class Img2ImgPage extends StatefulWidget {
  const Img2ImgPage({super.key});

  @override
  State<Img2ImgPage> createState() => _Img2ImgPageState();
}

class _Img2ImgPageState extends State<Img2ImgPage>
    with SingleTickerProviderStateMixin {
  final ScrollController _scrollController =
      ScrollController(); // Add ScrollController
  Timer? _modelErrorTimer;
  Timer? _errorMessageTimer;
  Img2ImgProcessor? _processor;
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
  double clipSkip = 0; // Default to match main.dart
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
  File? _inputImage;
  Uint8List? _rgbBytes;
  int? _inputWidth;
  int? _inputHeight;
  double strength = 0.5;
  List<String> _generationLogs = []; // To store logs for the last generation
  bool _showLogsButton = false; // To control visibility of the log button
  bool _isDiffusionModelType = false; // Added state for the new switch

  // --- State for Advanced Sampling Options (copied from main.dart) ---
  double eta = 0.0; // New state for eta slider
  double guidance = 3.5; // Default to match main.dart
  double slgScale = 0.0; // New state for slg-scale slider
  String skipLayersText = ''; // New state for skip-layers text field
  double skipLayerStart = 0.01; // New state for skip-layer-start slider
  double skipLayerEnd = 0.2; // New state for skip-layer-end slider
  final TextEditingController _skipLayersController =
      TextEditingController(); // Controller for skip-layers input
  String? _skipLayersErrorText; // Error text for skip-layers validation
  // --- End State for Advanced Sampling Options ---
  // bool _isDiffusionModelType = false; // Removed duplicate definition

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
  String _controlImageProcessingMode =
      'Resize'; // 'Resize' or 'Crop' for ControlNet image
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

  // Restore getWidthOptions and getHeightOptions
  List<int> getWidthOptions() {
    List<int> opts = [];
    // Generate options dynamically or use a fixed list if preferred
    // Example: Multiples of 64 up to a reasonable limit like 1024
    for (int i = 128; i <= 1024; i += 64) {
      opts.add(i);
    }
    // Ensure the current _inputWidth is an option if available and cropping is off
    if (!_showCropUI && _inputWidth != null && !opts.contains(_inputWidth!)) {
      opts.add(_inputWidth!);
      opts.sort();
    }
    return opts;
  }

  List<int> getHeightOptions() {
    List<int> opts = [];
    // Example: Multiples of 64 up to a reasonable limit like 1024
    for (int i = 128; i <= 1024; i += 64) {
      opts.add(i);
    }
    // Ensure the current _inputHeight is an option if available and cropping is off
    if (!_showCropUI && _inputHeight != null && !opts.contains(_inputHeight!)) {
      opts.add(_inputHeight!);
      opts.sort();
    }
    return opts;
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
    // _modelErrorTimer?.cancel(); // Removed as _modelError is removed
    _loadingErrorTimer?.cancel(); // Cancel the general loading error timer
    _processor?.dispose();
    _processor = null;
    _cannyProcessor?.dispose();
    _promptController.dispose(); // Dispose text controller
    _skipLayersController.dispose(); // Dispose the new controller
    _scrollController.dispose(); // Dispose ScrollController
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
      loadingText = 'Loading Model...';
    });
    _processor?.dispose();
    _processor = Img2ImgProcessor(
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
      vaePath: useVAE ? _vaePath : null, // Only pass VAE path if useVAE is true
      embedDirPath: _embedDirPath,
      clipSkip: clipSkip.toInt(),
      vaeTiling: useVAETiling,
      controlNetPath: _controlNetPath, // Add this
      controlImageData: _controlRgbBytes, // Add this
      controlImageWidth: _controlWidth, // Add this
      controlImageHeight: _controlHeight, // Add this // Add this
      controlStrength: controlStrength,
      isDiffusionModelType: _isDiffusionModelType, // Pass the state variable
      // isDiffusionModelType: _isDiffusionModelType, // Removed duplicate argument
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
      final String? generationTime =
          result['generationTime']; // Extract time from map

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

      // Save the extracted image
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

      // No need for the redundant status update here
    });
  }

  // --- Copied Error Handling Logic from main.dart ---
  // New method to handle loading errors centrally
  void _handleLoadingError(String errorType, String errorMessage) {
    _loadingErrorTimer?.cancel(); // Cancel previous timer if any

    // Call the central reset function
    _resetState(); // Call the new reset function

    // Set the specific error message for loading failure
    setState(() {
      _loadingError = errorMessage; // Display the specific error
      _loadingErrorType = errorType; // Store error type if needed elsewhere

      // Handle generation-specific errors separately if needed
      if (errorType == 'generationError') {
        status = 'Generation failed: $errorMessage';
        isGenerating = false; // Stop generation indicator
      } else if (errorType == 'inputError') {
        // Keep status potentially, just show the error message
        status = ''; // Or clear status for input errors too? Let's clear it.
        progress = 0;
      } else {
        // For loading errors, clear status too
        status = '';
        progress = 0;
      }

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

  // Central function to reset the state (adapted for Img2ImgPage)
  void _resetState() {
    _processor?.dispose(); // Dispose the processor if it exists

    setState(() {
      _processor = null; // Set processor to null
      isModelLoading = false; // Stop loading indicator
      isGenerating = false; // Stop generation indicator
      loadingText = ''; // Clear loading text
      _loadingError = ''; // Clear loading error
      _loadingErrorType = '';
      _loadingErrorTimer?.cancel(); // Cancel timer if active

      // Clear all loaded component indicators
      loadedComponents.clear();
      // Reset all paths
      _taesdPath = null;
      _loraPath = null;
      _clipLPath = null;
      _clipGPath = null;
      _t5xxlPath = null;
      _vaePath = null;
      _embedDirPath = null;
      _controlNetPath = null;
      // Reset related flags
      useTAESD = false;
      useVAE = false;
      useVAETiling = false;
      useControlNet = false;
      useControlImage = false;
      useCanny = false;
      // Reset other related state specific to img2img
      _loraNames = [];
      _ramUsage = ''; // Clear RAM usage display
      _inputImage = null; // Reset input image
      _rgbBytes = null;
      _inputWidth = null;
      _inputHeight = null;
      _showCropUI = false; // Hide crop UI
      _cropRect = Rect.zero;
      _imageDisplaySize = Size.zero;
      _controlImage = null;
      _controlRgbBytes = null;
      _controlWidth = null;
      _controlHeight = null;
      _cannyImage = null;
      _message = ''; // Clear success messages too
      _taesdMessage = '';
      _loraMessage = '';
      _taesdError = ''; // Clear TAESD specific errors
      _errorMessageTimer?.cancel();
      status = ''; // Clear generation status
      progress = 0; // Reset progress
      _generatedImage = null; // Clear generated image
      _generationLogs = []; // Clear logs
      _showLogsButton = false; // Hide log button

      // Reset UI elements (optional, but good practice)
      _promptController.clear();
      prompt = '';
      negativePrompt = '';
      // Reset advanced options to defaults if needed
      // clipSkip = 0;
      // eta = 0.0;
      // guidance = 3.5;
      // slgScale = 0.0;
      // skipLayersText = '';
      // _skipLayersController.clear();
      // skipLayerStart = 0.01;
      // skipLayerEnd = 0.2;
      // samplingMethod = 'euler_a';
      // cfg = 7;
      // steps = 25;
      // width = 512; // Reset width/height if needed, or keep based on last input?
      // height = 512;
      // seed = "-1";
      // strength = 0.5;
      // controlStrength = 0.9;
      // _controlImageProcessingMode = 'Resize';
      // _isDiffusionModelType = false;
    });
  }
  // --- End Copied Error Handling Logic ---

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

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      final bytes = await pickedFile.readAsBytes();
      final decodedImage = img.decodeImage(bytes);

      if (decodedImage == null) {
        _showTemporaryError('Failed to decode image');
        return;
      }

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

      setState(() {
        _inputImage = File(pickedFile.path);
        _rgbBytes = rgbBytes;
        _inputWidth = decodedImage.width;
        _inputHeight = decodedImage.height;
        // width = _inputWidth!; // Don't set width/height directly here anymore
        // height = _inputHeight!; // Use cropping logic below

        // --- Initialize Cropping State ---
        _maxCropWidth = largestMultipleOf64(_inputWidth!);
        _maxCropHeight = largestMultipleOf64(_inputHeight!);

        // Reset width/height sliders to default/initial aspect ratio
        double initialAspectRatio = _inputWidth! / _inputHeight!;
        int initialCropW, initialCropH;
        if (initialAspectRatio >= 1) {
          // Wider or square
          initialCropW = math.min(512, _maxCropWidth);
          initialCropH =
              largestMultipleOf64((initialCropW / initialAspectRatio).round());
          initialCropH = math.min(initialCropH, _maxCropHeight);
          initialCropW =
              largestMultipleOf64((initialCropH * initialAspectRatio).round());
          initialCropW = math.min(initialCropW, _maxCropWidth);
        } else {
          // Taller
          initialCropH = math.min(512, _maxCropHeight);
          initialCropW =
              largestMultipleOf64((initialCropH * initialAspectRatio).round());
          initialCropW = math.min(initialCropW, _maxCropWidth);
          initialCropH =
              largestMultipleOf64((initialCropW / initialAspectRatio).round());
          initialCropH = math.min(initialCropH, _maxCropHeight);
        }
        // Ensure minimum size
        width = math.max(64, initialCropW); // Use 'width' state for slider
        height = math.max(64, initialCropH); // Use 'height' state for slider

        _showCropUI = true; // Show the cropping UI now
        _cropRect = Rect.zero; // Reset crop rect position
        _imageDisplaySize = Size.zero; // Reset display size

        // Calculate initial crop rect after the frame renders
        WidgetsBinding.instance.addPostFrameCallback(_calculateInitialCropRect);
        // --- End Initialize Cropping State ---
      });
    } else {
      // If user cancels image picking, hide crop UI
      setState(() {
        _showCropUI = false;
      });
    }
  }

  // --- Cropping Logic (copied from inpainting_page.dart) ---

  void _calculateInitialCropRect(_) {
    if (!mounted) return;
    final RenderBox? imageBox =
        _imageKey.currentContext?.findRenderObject() as RenderBox?;
    if (imageBox != null && imageBox.hasSize) {
      setState(() {
        _imageDisplaySize = imageBox.size;
        _updateCropRect(center: true); // Center the initial crop rect
      });
    } else {
      // Retry if the size wasn't available yet
      WidgetsBinding.instance.addPostFrameCallback(_calculateInitialCropRect);
    }
  }

  // Updates the _cropRect based on current dimensions and optionally centers it
  void _updateCropRect({bool center = false}) {
    if (_imageDisplaySize == Size.zero) return; // Not ready yet

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

    setState(() {
      _cropRect =
          Rect.fromLTWH(newLeft, newTop, _cropRect.width, _cropRect.height);
    });
  }

  // Performs the actual cropping based on current state
  // Returns null if cropping is not active or fails
  Future<CroppedImageData?> _getCroppedImageData() async {
    // Only crop if the UI is shown and we have an input image
    if (!_showCropUI ||
        _inputImage == null ||
        _inputWidth == null ||
        _inputHeight == null) {
      return null;
    }
    // Ensure display size is calculated
    if (_imageDisplaySize == Size.zero) {
      print("Warning: Image display size not calculated yet. Cannot crop.");
      // Attempt to calculate it now, might delay generation slightly
      final RenderBox? imageBox =
          _imageKey.currentContext?.findRenderObject() as RenderBox?;
      if (imageBox != null && imageBox.hasSize) {
        _imageDisplaySize = imageBox.size;
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
      _showTemporaryError('Invalid crop area selected.');
      return null;
    }

    // 1. Read original image bytes
    final Uint8List imageBytes = await _inputImage!.readAsBytes();
    final img.Image? originalImage = img.decodeImage(imageBytes);

    if (originalImage == null) {
      print("Error: Could not decode image for cropping.");
      _showTemporaryError('Error decoding image for cropping');
      return null;
    }

    // 2. Calculate the source crop rectangle in original image coordinates
    final double scaleX = originalImage.width / _imageDisplaySize.width;
    final double scaleY = originalImage.height / _imageDisplaySize.height;

    final int srcX =
        (_cropRect.left * scaleX).round().clamp(0, originalImage.width);
    final int srcY =
        (_cropRect.top * scaleY).round().clamp(0, originalImage.height);
    // Clamp width/height to ensure they don't exceed bounds from srcX/srcY
    final int srcWidth = (_cropRect.width * scaleX)
        .round()
        .clamp(1, originalImage.width - srcX); // Ensure at least 1 pixel
    final int srcHeight = (_cropRect.height * scaleY)
        .round()
        .clamp(1, originalImage.height - srcY); // Ensure at least 1 pixel

    // Double check dimensions after clamping
    if (srcWidth <= 0 || srcHeight <= 0) {
      print(
          "Error: Calculated crop dimensions are invalid after clamping ($srcWidth x $srcHeight).");
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
      interpolation: img.Interpolation.nearest,
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

    // 6. Return the data (maskBytes is null for img2img)
    return CroppedImageData(
      imageBytes: finalImageBytes, // Return raw RGB bytes
      maskBytes: null, // No mask in img2img cropping
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
        title: const Text('Image to Image',
            style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: theme.colorScheme.background,
        elevation: 0,
        actions: [
          // Add Unload button only if a model is loaded
          if (_processor != null)
            Padding(
              padding: const EdgeInsets.only(right: 8.0),
              child: Tooltip(
                message: 'Unload Model & Reset',
                child: ShadButton.ghost(
                  icon: const Icon(
                    LucideIcons.powerOff,
                    size: 20,
                  ),
                  onPressed: (isModelLoading || isGenerating)
                      ? null // Disable if loading or generating
                      : () {
                          // Show confirmation dialog
                          showShadDialog(
                            context: context,
                            builder: (context) => ShadDialog.alert(
                              title: const Text('Confirm Unload'),
                              description: const Text(
                                  'Are you sure you want to unload the current model and reset all settings?'),
                              actions: [
                                ShadButton.outline(
                                  onPressed: () => Navigator.of(context).pop(),
                                  child: const Text('Cancel'),
                                ),
                                ShadButton.destructive(
                                  onPressed: () {
                                    Navigator.of(context).pop(); // Close dialog
                                    _resetState(); // Call the reset function
                                  },
                                  child: const Text('Confirm Unload'),
                                ),
                              ],
                            ),
                          );
                        },
                ),
              ),
            ),
        ],
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
              tileColor: theme.colorScheme.secondary.withOpacity(0.2),
              onTap: () {
                Navigator.pop(context);
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
        controller: _scrollController, // Attach ScrollController
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Display Loading Status / Success / Error Messages (Copied from main.dart)
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
                    LoadingDotsAnimation(
                      // Use the custom widget from main.dart
                      loadingText: loadingText,
                      style: theme.textTheme.p.copyWith(
                        color: Colors.orange,
                        fontWeight: FontWeight.bold,
                      ),
                    ).animate().fadeIn(), // Keep the fade-in animation
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
                  Text(_ramUsage, style: theme.textTheme.p),
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
                                    children: taesdFiles
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

                        if (selectedTaesd != null) {
                          setState(() {
                            _taesdPath = selectedTaesd;
                            loadedComponents['TAESD'] = true;
                            _taesdError = '';
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
                ShadCheckbox(
                  value: useTAESD,
                  onChanged: (bool v) {
                    // Removed check: if (useVAETiling && v) { ... }
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
            GestureDetector(
              onTap: (isModelLoading || isGenerating) ? null : _pickImage,
              child: DottedBorder(
                borderType: BorderType.RRect,
                radius: const Radius.circular(8),
                color: theme.colorScheme.primary.withOpacity(0.5),
                strokeWidth: 2,
                dashPattern: const [8, 4],
                child: AspectRatio(
                  // Maintain aspect ratio of the original image
                  aspectRatio: (_inputWidth != null &&
                          _inputHeight != null &&
                          _inputHeight! > 0)
                      ? _inputWidth! / _inputHeight!
                      : 1.0, // Default aspect ratio if no image
                  child: Container(
                    // Removed color: Colors.grey[300]
                    child: _inputImage == null
                        ? Center(
                            // Center the placeholder
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(
                                  Icons.add_photo_alternate,
                                  size: 64,
                                  color: theme.colorScheme.primary
                                      .withOpacity(0.5),
                                ),
                                const SizedBox(height: 12),
                                Text(
                                  'Load image',
                                  style: TextStyle(
                                      color: theme.colorScheme.primary
                                          .withOpacity(0.5),
                                      fontSize: 16),
                                ),
                              ],
                            ),
                          )
                        : LayoutBuilder(
                            // Use LayoutBuilder for display size
                            builder: (context, constraints) {
                              // Update display size post-frame
                              WidgetsBinding.instance.addPostFrameCallback((_) {
                                if (mounted &&
                                    _imageDisplaySize != constraints.biggest) {
                                  // Use setState only if size actually changed
                                  if (_imageDisplaySize !=
                                      constraints.biggest) {
                                    setState(() {
                                      _imageDisplaySize = constraints.biggest;
                                      _updateCropRect(); // Recalculate rect
                                    });
                                  }
                                }
                              });

                              return GestureDetector(
                                onPanStart: _showCropUI ? _onPanStart : null,
                                onPanUpdate: _showCropUI ? _onPanUpdate : null,
                                child: Stack(
                                  fit: StackFit.expand,
                                  children: [
                                    // Image
                                    Image.file(
                                      _inputImage!,
                                      key: _imageKey, // Assign key here
                                      fit: BoxFit.contain,
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
                divisions: (_maxCropWidth > 64)
                    ? (_maxCropWidth - 64) ~/ 64
                    : 1, // Avoid division by zero
                initialValue: width.toDouble(),
                onChanged: (value) {
                  int newWidth = largestMultipleOf64(value.round());
                  if (newWidth != width) {
                    setState(() {
                      width = newWidth;
                      double aspect = width / height.toDouble();
                      height = largestMultipleOf64((width / aspect).round());
                      height = height.clamp(64, _maxCropHeight);
                      width = largestMultipleOf64((height * aspect).round());
                      width = width.clamp(64, _maxCropWidth);
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
                divisions: (_maxCropHeight > 64)
                    ? (_maxCropHeight - 64) ~/ 64
                    : 1, // Avoid division by zero
                initialValue: height.toDouble(),
                onChanged: (value) {
                  int newHeight = largestMultipleOf64(value.round());
                  if (newHeight != height) {
                    setState(() {
                      height = newHeight;
                      double aspect = width.toDouble() / height;
                      width = largestMultipleOf64((height * aspect).round());
                      width = width.clamp(64, _maxCropWidth);
                      height = largestMultipleOf64((width / aspect).round());
                      height = height.clamp(64, _maxCropHeight);
                      _updateCropRect();
                    });
                  }
                },
              ),
              const SizedBox(height: 16), // Spacing after sliders
            ],
            // --- End Sliders ---
            const SizedBox(height: 16),
            // --- Advanced Model Options Accordion ---
            ShadAccordion<Map<String, dynamic>>(
              children: [
                ShadAccordionItem<Map<String, dynamic>>(
                  value: const {}, // Unique value for this item
                  title: const Text('Advanced Model Options'),
                  child: Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // --- PASTED CONTENT START ---
                        // Standalone Model Switch (Cut from here)
                        // Padding(...)
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
                        // Standalone Model Switch (Pasted here)
                        Padding(
                          padding: const EdgeInsets.only(
                              top: 8.0,
                              bottom: 16.0), // Adjust padding as needed
                          child: ShadSwitch(
                            value: _isDiffusionModelType,
                            onChanged: (v) =>
                                setState(() => _isDiffusionModelType = v),
                            label: const Text('Standalone Model'),
                          ),
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
                                            file.path.endsWith('.bin') ||
                                            file.path.endsWith(
                                                '.gguf')) // Added .gguf
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
                                                  children: clipFiles
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
                                            file.path.endsWith('.bin') ||
                                            file.path.endsWith(
                                                '.gguf')) // Added .gguf
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
                                                  children: clipFiles
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
                                            file.path.endsWith('.bin') ||
                                            file.path.endsWith(
                                                '.gguf')) // Added .gguf
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
                                                  children: t5Files
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
                        // --- PASTED CONTENT END ---

                        // Removed duplicate Standalone Model Switch
                        // VAE Tiling checkbox was moved to the Advanced Sampling Options accordion
                      ],
                    ),
                  ),
                ),
              ],
            ),
            // --- End Advanced Model Options Accordion ---

            const SizedBox(height: 16), // Spacing after accordion
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

            // --- New Advanced Sampling Options Accordion (Copied from main.dart, VAE Tiling moved here) ---
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
                                  // Removed check: if (useTAESD && v) { ... }
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
                                max: 2, // Adjust max if needed for img2img
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
                                max: 40.0, // Adjust max if needed
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
                                max: 7.0, // Adjust max if needed
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
                              onChanged: (String? v) {
                                final text = v ?? '';
                                final regex = RegExp(r'^(?:\d+(?:,\s*\d+)*)?$');
                                if (text.isEmpty || regex.hasMatch(text)) {
                                  setState(() {
                                    skipLayersText = text;
                                    _skipLayersErrorText = null; // Clear error
                                  });
                                } else {
                                  setState(() {
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
            // Removed the old, empty "Advanced Options" accordion
            const SizedBox(height: 16),
            Row(
              children: [
                const Text('Use ControlNet'),
                const SizedBox(width: 8),
                ShadSwitch(
                  value: useControlNet,
                  onChanged: (isModelLoading || isGenerating)
                      ? null // Disable switch while loading or generating
                      : (bool v) {
                          setState(() {
                            useControlNet = v;
                            // Only reinitialize if a main model is already loaded
                            if (_processor != null) {
                              String currentModelPath = _processor!.modelPath;
                              bool currentFlashAttention =
                                  _processor!.useFlashAttention;
                              SDType currentModelType = _processor!.modelType;
                              Schedule currentSchedule = _processor!.schedule;

                              if (v) {
                                // Enabling ControlNet
                                // Reload ONLY if a ControlNet model path is actually set.
                                if (_controlNetPath != null) {
                                  print(
                                      "Re-initializing processor to ENABLE ControlNet with path: $_controlNetPath");
                                  _initializeProcessor(
                                    currentModelPath,
                                    currentFlashAttention,
                                    currentModelType,
                                    currentSchedule,
                                  );
                                } else {
                                  print(
                                      "Enabled ControlNet switch, but no ControlNet model loaded. No reload needed.");
                                }
                              } else {
                                // Disabling ControlNet
                                // Reload ONLY if a ControlNet model path was previously set.
                                if (_controlNetPath != null) {
                                  print(
                                      "Re-initializing processor to DISABLE ControlNet. Original path was: $_controlNetPath");
                                  String? originalControlNetPath =
                                      _controlNetPath; // Store original path
                                  _controlNetPath =
                                      null; // Temporarily remove path for reload

                                  _initializeProcessor(
                                    currentModelPath,
                                    currentFlashAttention,
                                    currentModelType,
                                    currentSchedule,
                                  );

                                  // DO NOT restore the path. Clear it permanently when disabling.
                                  // _controlNetPath = originalControlNetPath;
                                  // Remove the loaded indicator regardless of reload
                                  loadedComponents.remove('ControlNet');
                                } else {
                                  print(
                                      "Disabled ControlNet switch, but no ControlNet model was loaded anyway. No reload needed.");
                                  // Still remove the indicator if it somehow exists
                                  loadedComponents.remove('ControlNet');
                                }
                              }
                            } else {
                              print(
                                  "ControlNet switch toggled, but no main model loaded. No action taken.");
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
                          // Dropdown for Crop/Resize ControlNet Image
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
                                        value: 'Resize', child: Text('Resize')),
                                    ShadOption(
                                        value: 'Crop', child: Text('Crop')),
                                  ],
                                  selectedOptionBuilder: (context, value) =>
                                      Text(value),
                                  onChanged: (String? value) {
                                    if (value != null) {
                                      setState(() =>
                                          _controlImageProcessingMode = value);
                                    }
                                  },
                                ),
                              ),
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
                const Text('Strength'),
                const SizedBox(width: 8),
                Expanded(
                  child: ShadSlider(
                    initialValue: strength,
                    min: 0.0,
                    max: 1.0,
                    divisions: 20,
                    onChanged: (v) => setState(() => strength = v),
                  ),
                ),
                Text(strength.toStringAsFixed(2)),
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
                      // Display effective width
                      (_showCropUI ? width : (_inputWidth ?? 512))
                          .toString(), // Default to 512 if no input
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
                            // Remove onChanged when cropping
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
                      // Display effective height
                      (_showCropUI
                              ? height
                              : (_inputHeight ??
                                  512)) // Default to 512 if no input
                          .toString(),
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
                            // Remove onChanged when cropping
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
                      // Scroll to top to show the error
                      _scrollController.animateTo(
                        0.0,
                        duration: const Duration(milliseconds: 500),
                        curve: Curves.easeInOut,
                      );
                      return;
                    }
                    if (_inputImage == null) {
                      // Show temporary error without full reset
                      setState(() {
                        _loadingError = 'Please select an input image first.';
                        _loadingErrorType = 'inputError';
                      });
                      _loadingErrorTimer?.cancel(); // Cancel previous timer
                      _loadingErrorTimer =
                          Timer(const Duration(seconds: 10), () {
                        if (mounted) {
                          setState(() {
                            _loadingError = '';
                            _loadingErrorType = '';
                          });
                        }
                      });
                      // Scroll to top to show the error
                      _scrollController.animateTo(
                        0.0,
                        duration: const Duration(milliseconds: 500),
                        curve: Curves.easeInOut,
                      );
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
                      status = 'Preparing image...'; // Update status
                      progress = 0;
                      _generationLogs = []; // Clear previous logs
                      _showLogsButton =
                          false; // Hide log button until generation finishes
                    });

                    // Perform cropping if UI is active
                    CroppedImageData? croppedData;
                    if (_showCropUI) {
                      croppedData = await _getCroppedImageData();
                      if (croppedData == null && _showCropUI) {
                        // Cropping failed or was invalid, stop generation
                        setState(() {
                          isGenerating = false;
                          status = 'Image cropping failed.';
                        });
                        return; // Don't proceed if cropping failed but was expected
                      }
                    }

                    // Determine effective image data and dimensions
                    final Uint8List effectiveImageData =
                        croppedData?.imageBytes ?? _rgbBytes!;
                    final int effectiveWidth =
                        croppedData?.width ?? _inputWidth!;
                    final int effectiveHeight =
                        croppedData?.height ?? _inputHeight!;

                    // --- Control Image Processing Logic ---
                    Uint8List? finalControlBytes;
                    int? finalControlWidth;
                    int? finalControlHeight;

                    // Determine the source control image data (Canny or original control image)
                    Uint8List? sourceControlBytes;
                    int? sourceControlWidth;
                    int? sourceControlHeight;

                    if (useControlNet && useControlImage) {
                      if (useCanny && _cannyProcessor?.resultRgbBytes != null) {
                        // Use Canny result if available and Canny is enabled
                        sourceControlBytes = _cannyProcessor!.resultRgbBytes!;
                        sourceControlWidth = _cannyProcessor!.resultWidth!;
                        sourceControlHeight = _cannyProcessor!.resultHeight!;
                        // Ensure Canny result is RGB
                        sourceControlBytes = _ensureRgbFormat(
                            sourceControlBytes,
                            sourceControlWidth,
                            sourceControlHeight);
                      } else if (_controlRgbBytes != null) {
                        // Use original control image if Canny is not used or failed
                        sourceControlBytes = _controlRgbBytes;
                        sourceControlWidth = _controlWidth;
                        sourceControlHeight = _controlHeight;
                        // Ensure original control image is RGB (already done on load, but double-check)
                        if (sourceControlBytes != null &&
                            sourceControlWidth != null &&
                            sourceControlHeight != null) {
                          sourceControlBytes = _ensureRgbFormat(
                              sourceControlBytes,
                              sourceControlWidth,
                              sourceControlHeight);
                        }
                      }
                    }

                    if (sourceControlBytes != null &&
                        sourceControlWidth != null &&
                        sourceControlHeight != null) {
                      // Check if control image dimensions match the target (effective input) size
                      if (sourceControlWidth != effectiveWidth ||
                          sourceControlHeight != effectiveHeight) {
                        try {
                          print(
                              'Control image dimensions ($sourceControlWidth x $sourceControlHeight) differ from target ($effectiveWidth x $effectiveHeight). Processing using $_controlImageProcessingMode...');
                          ProcessedImageData processedData;
                          if (_controlImageProcessingMode == 'Crop') {
                            processedData = cropImage(
                                sourceControlBytes,
                                sourceControlWidth,
                                sourceControlHeight,
                                effectiveWidth,
                                effectiveHeight);
                          } else {
                            // Default to Resize
                            processedData = resizeImage(
                                sourceControlBytes,
                                sourceControlWidth,
                                sourceControlHeight,
                                effectiveWidth,
                                effectiveHeight);
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
                        finalControlBytes = sourceControlBytes;
                        finalControlWidth = sourceControlWidth;
                        finalControlHeight = sourceControlHeight;
                      }
                    } else {
                      // No valid source control image, ensure control params are null
                      finalControlBytes = null;
                      finalControlWidth = null;
                      finalControlHeight = null;
                    }
                    // --- End Control Image Processing Logic ---

                    // --- Skip Layers Formatting ---
                    String? formattedSkipLayers;
                    if (_skipLayersErrorText == null &&
                        skipLayersText.trim().isNotEmpty) {
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

                    _processor!.generateImg2Img(
                      inputImageData: effectiveImageData, // Use effective data
                      inputWidth: effectiveWidth, // Use effective width
                      inputHeight: effectiveHeight, // Use effective height
                      channel: 3, // Assuming RGB
                      outputWidth:
                          effectiveWidth, // Output should match effective input
                      outputHeight:
                          effectiveHeight, // Output should match effective input
                      prompt: prompt,
                      negativePrompt: negativePrompt,
                      clipSkip: clipSkip.toInt(), // Already passed
                      cfgScale: cfg, // Already passed
                      guidance: guidance, // New
                      eta: eta, // New
                      sampleMethod: SampleMethod.values
                          .firstWhere(
                            (method) =>
                                method.displayName.toLowerCase() ==
                                samplingMethod.toLowerCase(),
                            orElse: () => SampleMethod.EULER_A,
                          )
                          .index, // Already passed
                      sampleSteps: steps, // Already passed
                      strength: strength, // Already passed
                      seed: int.tryParse(seed) ?? -1, // Already passed
                      batchCount: 1, // Already passed
                      controlImageData:
                          finalControlBytes, // Use processed bytes (Already passed)
                      controlImageWidth:
                          finalControlWidth, // Use processed width (Already passed)
                      controlImageHeight:
                          finalControlHeight, // Use processed height (Already passed)
                      controlStrength: controlStrength, // Already passed
                      // New sampling parameters
                      slgScale: slgScale,
                      skipLayersText: formattedSkipLayers,
                      skipLayerStart: skipLayerStart,
                      skipLayerEnd: skipLayerEnd,
                      // Mask parameters (if needed in the future, add here)
                      maskImageData: null,
                      maskWidth: null,
                      maskHeight: null,
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

  // Method to show the logs dialog
  void _showLogsDialog() {
    final theme = ShadTheme.of(context); // Get theme for dialog styling
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
