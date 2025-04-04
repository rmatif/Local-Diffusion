import 'dart:io';
import 'dart:ui' as ui;
import 'dart:async';
import 'dart:typed_data';
import 'dart:developer' as developer;

import 'package:flutter/material.dart';
import 'package:shadcn_ui/shadcn_ui.dart';
import 'package:file_picker/file_picker.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:dotted_border/dotted_border.dart';
import 'package:flutter_animate/flutter_animate.dart';

import 'canny_processor.dart'; // Kept in case needed for controlnet later
import 'ffi_bindings.dart';
import 'img2img_page.dart';
import 'img2img_processor.dart';
import 'scribble2img_page.dart';
import 'utils.dart';
import 'main.dart';
import 'upscaler_page.dart';
import 'photomaker_page.dart';
import 'inpainting_page.dart';
import 'image_processing_utils.dart'; // Import the image processing utils
// import 'mask_editor.dart'; // Not directly needed for mask creation anymore

class OutpaintingPage extends StatefulWidget {
  const OutpaintingPage({super.key});

  @override
  State<OutpaintingPage> createState() => _OutpaintingPageState();
}

class _OutpaintingPageState extends State<OutpaintingPage>
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
  double clipSkip = 0.0; // Default set to 0.0
  bool useVAE = false;
  String samplingMethod = 'euler_a';
  double cfg = 7;
  int steps = 25;
  // Output width/height are now calculated based on input + padding
  int outputWidth = 512;
  int outputHeight = 512;
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
  double strength = 0.75; // Default strength for outpainting often higher
  Uint8List? _maskData; // Generated based on padding
  ui.Image? _maskImageUi; // To display the generated mask
  List<String> _generationLogs = []; // To store logs for the last generation
  bool _showLogsButton = false; // To control visibility of the log button
  bool _isDiffusionModelType =
      false; // Added state for the standalone model switch

  // --- State for Advanced Sampling Options (copied) ---
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

  // Padding state variables
  int paddingTop = 0;
  int paddingBottom = 0;
  int paddingLeft = 0;
  int paddingRight = 0;

  // Padding options (multiples of 64 up to 256)
  final List<int> paddingOptions =
      List.generate(256 ~/ 64 + 1, (index) => index * 64);

  String? _taesdPath;
  String? _loraPath;
  String? _clipLPath;
  String? _clipGPath;
  String? _t5xxlPath;
  String? _vaePath;
  String? _embedDirPath;
  String? _controlNetPath;
  File? _controlImage;
  Uint8List? _controlRgbBytes;
  int? _controlWidth;
  int? _controlHeight;
  bool useControlNet = false;
  bool useControlImage = false;
  bool useCanny = false;
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
    'ddim_trailing',
    'tcd'
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

  // Not used for selection anymore, but might be useful elsewhere
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

    // Calculate initial dimensions if input exists (though unlikely on init)
    if (_inputWidth != null && _inputHeight != null) {
      _calculateDimensionsAndGenerateMask();
    }
  }

  @override
  void dispose() {
    _errorMessageTimer?.cancel();
    // _modelErrorTimer?.cancel(); // Removed
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
    if (bytes.length == width * height * 3) {
      return bytes;
    }
    if (bytes.length == width * height) {
      final rgbBytes = Uint8List(width * height * 3);
      for (int i = 0; i < width * height; i++) {
        rgbBytes[i * 3] = bytes[i];
        rgbBytes[i * 3 + 1] = bytes[i];
        rgbBytes[i * 3 + 2] = bytes[i];
      }
      return rgbBytes;
    }
    print(
        "Warning: Unexpected image format. Expected 1 or 3 channels, got: ${bytes.length / (width * height)} channels");
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

  // No longer loads external mask, generates it based on padding
  Future<void> _generateMask() async {
    if (_inputWidth == null || _inputHeight == null) return;

    // Calculate output dimensions based on input and padding
    outputWidth = _inputWidth! + paddingLeft + paddingRight;
    outputHeight = _inputHeight! + paddingTop + paddingBottom;

    // Ensure dimensions are positive
    if (outputWidth <= 0 || outputHeight <= 0) {
      _showTemporaryError("Invalid dimensions after padding.");
      return;
    }

    // Create mask data (1 channel, 0=black, 255=white)
    final maskData = Uint8List(outputWidth * outputHeight);

    // Fill mask based on padding
    for (int y = 0; y < outputHeight; y++) {
      for (int x = 0; x < outputWidth; x++) {
        // Check if the pixel is within the original image area (after padding offset)
        bool isInsideOriginal = x >= paddingLeft &&
            x < paddingLeft + _inputWidth! &&
            y >= paddingTop &&
            y < paddingTop + _inputHeight!;

        // If inside original image area -> black (0)
        // If in padding area -> white (255)
        maskData[y * outputWidth + x] = isInsideOriginal ? 0 : 255;
      }
    }

    setState(() {
      _maskData = maskData;
      // Decode the generated mask for preview
      _decodeMaskImage(maskData, outputWidth, outputHeight);
    });
  }

  // Calculates dimensions and triggers mask generation
  void _calculateDimensionsAndGenerateMask() {
    if (_inputWidth == null || _inputHeight == null) {
      // Reset if no input image
      setState(() {
        outputWidth = 512; // Or some default
        outputHeight = 512;
        _maskData = null;
        _maskImageUi = null;
      });
      return;
    }

    // Calculate output dimensions
    int newWidth = _inputWidth! + paddingLeft + paddingRight;
    int newHeight = _inputHeight! + paddingTop + paddingBottom;

    // Ensure dimensions are valid (e.g., non-negative)
    if (newWidth <= 0 || newHeight <= 0) {
      _showTemporaryError("Invalid dimensions resulting from padding.");
      // Optionally reset padding or handle error differently
      return;
    }

    // Update state for UI display and processor input
    setState(() {
      outputWidth = newWidth;
      outputHeight = newHeight;
    });

    // Generate the mask based on the new dimensions and padding
    _generateMask();
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
      vaePath: useVAE ? _vaePath : null,
      embedDirPath: _embedDirPath,
      clipSkip: clipSkip.toInt(),
      vaeTiling: useVAETiling,
      controlNetPath: _controlNetPath,
      controlImageData: _controlRgbBytes,
      controlImageWidth: _controlWidth,
      controlImageHeight: _controlHeight,
      controlStrength: controlStrength,
      isDiffusionModelType: _isDiffusionModelType, // Use the state variable
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

      // Use the calculated output dimensions for saving metadata
      await _processor!.saveGeneratedImage(
        image, // Use extracted image
        prompt,
        outputWidth, // Use calculated output width
        outputHeight, // Use calculated output height
        SampleMethod.values.firstWhere(
          (method) =>
              method.displayName.toLowerCase() == samplingMethod.toLowerCase(),
          orElse: () => SampleMethod.EULER_A,
        ),
      );

      // No need for the redundant status update here
    });

    // Listen for the collected logs after generation
    _processor!.logListStream.listen((logs) {
      setState(() {
        _generationLogs = logs;
      });
    });
  }

  // --- Copied Error Handling Logic from img2img_page.dart ---
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
      } else if (errorType == 'inputError' ||
          errorType == 'maskError' ||
          errorType == 'dimensionError') {
        // For input/mask/dimension errors, just show the message, don't clear status necessarily
        status = ''; // Clear status for these errors too for consistency
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

  // Central function to reset the state (adapted for OutpaintingPage)
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
      _isDiffusionModelType = false;
      // Reset other related state specific to outpainting
      _loraNames = [];
      _ramUsage = ''; // Clear RAM usage display
      _inputImage = null; // Reset input image
      _rgbBytes = null;
      _inputWidth = null;
      _inputHeight = null;
      _maskData = null; // Reset mask data
      _maskImageUi = null;
      paddingTop = 0; // Reset padding
      paddingBottom = 0;
      paddingLeft = 0;
      paddingRight = 0;
      outputWidth = 512; // Reset output dimensions to default
      outputHeight = 512;
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
      // clipSkip = 0.0;
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
      // seed = "-1";
      // strength = 0.75;
      // controlStrength = 0.9;
      // _controlImageProcessingMode = 'Resize';
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
                                  if (index == 0) {
                                    return const FixedTableSpanExtent(250);
                                  }
                                  if (index == 1) {
                                    return const FixedTableSpanExtent(80);
                                  }
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

      // Ensure image is RGB
      img.Image imageToProcess;
      if (decodedImage.numChannels == 4) {
        // Manually create an RGB image from RGBA
        imageToProcess = img.Image(
            width: decodedImage.width,
            height: decodedImage.height,
            numChannels: 3);
        for (int y = 0; y < decodedImage.height; ++y) {
          for (int x = 0; x < decodedImage.width; ++x) {
            final pixel = decodedImage.getPixel(x, y);
            imageToProcess.setPixelRgb(x, y, pixel.r, pixel.g, pixel.b);
          }
        }
      } else if (decodedImage.numChannels == 3) {
        imageToProcess = decodedImage;
      } else {
        // Handle grayscale or other formats if necessary, or show error
        _showTemporaryError(
            'Image must be RGB or RGBA'); // Updated error message
        return;
      }
// Now imageToProcess is guaranteed to be an RGB image (numChannels == 3)
      final rgbBytesList = imageToProcess.toUint8List();

      // Store raw RGB bytes
      setState(() {
        _inputImage = File(pickedFile.path);
        _rgbBytes = rgbBytesList;
        _inputWidth = imageToProcess.width;
        _inputHeight = imageToProcess.height;
        // Reset padding when new image is picked
        paddingTop = 0;
        paddingBottom = 0;
        paddingLeft = 0;
        paddingRight = 0;
        // Recalculate dimensions and generate initial mask (which will be just black)
        _calculateDimensionsAndGenerateMask();
      });
    }
  }

  // Not used, replaced by _generateMask
  // Future<void> _createMask() async { ... }

  // Takes dimensions as parameters now
  Future<void> _decodeMaskImage(
      Uint8List maskData, int width, int height) async {
    // Convert 1-channel mask to 4-channel RGBA for display
    final rgbaBytes = Uint8List(width * height * 4);
    for (int i = 0; i < maskData.length; i++) {
      final value = maskData[i]; // 0 or 255
      rgbaBytes[i * 4] = value; // R
      rgbaBytes[i * 4 + 1] = value; // G
      rgbaBytes[i * 4 + 2] = value; // B
      rgbaBytes[i * 4 + 3] = 255; // A (fully opaque)
    }
    final completer = Completer<ui.Image>();
    ui.decodeImageFromPixels(
      rgbaBytes,
      width,
      height,
      ui.PixelFormat.rgba8888,
      completer.complete,
    );
    final maskImage = await completer.future;
    // Check if mounted before setting state if async operation might outlive widget
    if (mounted) {
      setState(() {
        _maskImageUi = maskImage;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = ShadTheme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Outpainting',
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
        // Keep the drawer for navigation consistency
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
                _processor?.dispose();
                Navigator.pop(context); // Close drawer
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
                _processor?.dispose();
                Navigator.pop(context); // Close drawer
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
                _processor?.dispose();
                Navigator.pop(context); // Close drawer
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
                _processor?.dispose();
                Navigator.pop(context); // Close drawer
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
                _processor?.dispose();
                Navigator.pop(context); // Close drawer
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
                _processor?.dispose();
                Navigator.pop(context); // Close drawer
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(
                      builder: (context) => const InpaintingPage()),
                );
              },
            ),
            // Current Page: Outpainting
            ListTile(
              leading: const Icon(LucideIcons.expand, size: 32), // Example icon
              title: const Text('Outpainting',
                  style: TextStyle(fontWeight: FontWeight.bold)),
              tileColor: theme.colorScheme.secondary
                  .withOpacity(0.2), // Indicate active
              onTap: () {
                Navigator.pop(context); // Close drawer, already here
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
            // --- Loading Indicators & Model Info ---
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
                                      if (index == 0) {
                                        return const FixedTableSpanExtent(250);
                                      }
                                      if (index == 1) {
                                        return const FixedTableSpanExtent(80);
                                      }
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
                  style: theme.textTheme.p
                      .copyWith(color: Colors.red, fontWeight: FontWeight.bold),
                ),
              ),
            const SizedBox(height: 16),

            // --- Input Image Picker ---
            GestureDetector(
              onTap: (isModelLoading || isGenerating) ? null : _pickImage,
              child: DottedBorder(
                borderType: BorderType.RRect,
                radius: const Radius.circular(8),
                color: theme.colorScheme.primary.withOpacity(0.5),
                strokeWidth: 2,
                dashPattern: const [8, 4],
                child: Container(
                  height: 300,
                  width: double.infinity, // Take full width
                  child: Center(
                    child: _inputImage == null
                        ? Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(
                                Icons.add_photo_alternate,
                                size: 64,
                                color:
                                    theme.colorScheme.primary.withOpacity(0.5),
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
                          )
                        : Image.file(_inputImage!, fit: BoxFit.contain),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 16),

            // --- Padding Controls ---
            if (_inputImage != null) ...[
              const Text('Padding (pixels to add):',
                  style: TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 8),
              Row(
                children: [
                  const SizedBox(width: 90, child: Text('Top:')),
                  Expanded(
                    child: ShadSelect<int>(
                      enabled: !(isModelLoading || isGenerating),
                      initialValue: paddingTop,
                      placeholder: Text(paddingTop.toString()),
                      options: paddingOptions
                          .map((p) =>
                              ShadOption(value: p, child: Text(p.toString())))
                          .toList(),
                      selectedOptionBuilder: (context, value) =>
                          Text(value.toString()),
                      onChanged: (int? value) {
                        if (value != null) {
                          setState(() {
                            paddingTop = value;
                            _calculateDimensionsAndGenerateMask();
                          });
                        }
                      },
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              Row(
                children: [
                  const SizedBox(width: 90, child: Text('Bottom:')),
                  Expanded(
                    child: ShadSelect<int>(
                      enabled: !(isModelLoading || isGenerating),
                      initialValue: paddingBottom,
                      placeholder: Text(paddingBottom.toString()),
                      options: paddingOptions
                          .map((p) =>
                              ShadOption(value: p, child: Text(p.toString())))
                          .toList(),
                      selectedOptionBuilder: (context, value) =>
                          Text(value.toString()),
                      onChanged: (int? value) {
                        if (value != null) {
                          setState(() {
                            paddingBottom = value;
                            _calculateDimensionsAndGenerateMask();
                          });
                        }
                      },
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              Row(
                children: [
                  const SizedBox(width: 90, child: Text('Left:')),
                  Expanded(
                    child: ShadSelect<int>(
                      enabled: !(isModelLoading || isGenerating),
                      initialValue: paddingLeft,
                      placeholder: Text(paddingLeft.toString()),
                      options: paddingOptions
                          .map((p) =>
                              ShadOption(value: p, child: Text(p.toString())))
                          .toList(),
                      selectedOptionBuilder: (context, value) =>
                          Text(value.toString()),
                      onChanged: (int? value) {
                        if (value != null) {
                          setState(() {
                            paddingLeft = value;
                            _calculateDimensionsAndGenerateMask();
                          });
                        }
                      },
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              Row(
                children: [
                  const SizedBox(width: 90, child: Text('Right:')),
                  Expanded(
                    child: ShadSelect<int>(
                      enabled: !(isModelLoading || isGenerating),
                      initialValue: paddingRight,
                      placeholder: Text(paddingRight.toString()),
                      options: paddingOptions
                          .map((p) =>
                              ShadOption(value: p, child: Text(p.toString())))
                          .toList(),
                      selectedOptionBuilder: (context, value) =>
                          Text(value.toString()),
                      onChanged: (int? value) {
                        if (value != null) {
                          setState(() {
                            paddingRight = value;
                            _calculateDimensionsAndGenerateMask();
                          });
                        }
                      },
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              // --- Display Generated Mask ---
              if (_maskImageUi != null) ...[
                const Text('Generated Mask Preview:',
                    style: TextStyle(fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                Container(
                  constraints: const BoxConstraints(
                      maxHeight: 200), // Limit preview height
                  decoration: BoxDecoration(
                    border: Border.all(color: theme.colorScheme.border),
                  ),
                  child: RawImage(image: _maskImageUi, fit: BoxFit.contain),
                ),
                const SizedBox(height: 16),
              ],
            ], // End of padding controls

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
                        // VAE Tiling checkbox removed from here, moved to Advanced Sampling Options
                        // --- PASTED CONTENT END ---

                        // Removed duplicate Standalone Model Switch
                      ],
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16), // Spacing after accordion

            // --- Prompts ---
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

            // --- New Advanced Sampling Options Accordion (Copied) ---
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
                                max: 2, // Adjust max if needed
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

            const SizedBox(height: 16),

            // --- ControlNet Options ---
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
                                // Also reset ControlNet specific UI/state when disabling
                                _controlImage = null;
                                _controlRgbBytes = null;
                                _cannyImage = null;
                                useCanny = false;
                                useControlImage = false;
                              }
                            } else {
                              print(
                                  "ControlNet switch toggled, but no main model loaded. No action taken.");
                              // Reset ControlNet specific UI/state if toggled off without model
                              if (!v) {
                                _controlImage = null;
                                _controlRgbBytes = null;
                                _cannyImage = null;
                                useCanny = false;
                                useControlImage = false;
                                loadedComponents.remove('ControlNet');
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
                // Keep consistent with inpainting
                children: [
                  ShadAccordionItem<Map<String, dynamic>>(
                    value: const {},
                    title: const Text('ControlNet Options'),
                    child: Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          // --- ControlNet Model Loading ---
                          Row(
                            children: [
                              Expanded(
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
                                            // ... [ControlNet selection dialog - same as inpainting] ...
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
                                                  currentSchedule);
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
                          // --- Control Image Reference ---
                          Align(
                            alignment: Alignment.centerLeft,
                            child: ShadCheckbox(
                              value: useControlImage,
                              onChanged: (_controlNetPath == null ||
                                      isModelLoading ||
                                      isGenerating)
                                  ? null
                                  : (bool v) {
                                      // Enable only if CN model loaded
                                      setState(() {
                                        useControlImage = v;
                                        if (!v) {
                                          _controlImage = null;
                                          _controlRgbBytes = null;
                                          _controlWidth = null;
                                          _controlHeight = null;
                                          _cannyImage =
                                              null; // Clear canny result if ref image removed
                                          useCanny =
                                              false; // Disable canny if ref image removed
                                        }
                                        // Re-initialize processor if CN image status changes? Maybe not needed unless image data itself changes.
                                      });
                                    },
                              label: const Text('Use Image Reference'),
                            ),
                          ),
                          if (useControlImage) ...[
                            const SizedBox(height: 16),
                            // --- Control Image Picker ---
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
                                              'Failed to decode control image');
                                          return;
                                        }

                                        // Ensure RGB format for control image
                                        img.Image imageToProcess;
                                        if (decodedImage.numChannels == 4) {
                                          // Manually create an RGB image from RGBA
                                          imageToProcess = img.Image(
                                              width: decodedImage.width,
                                              height: decodedImage.height,
                                              numChannels: 3);
                                          for (int y = 0;
                                              y < decodedImage.height;
                                              ++y) {
                                            for (int x = 0;
                                                x < decodedImage.width;
                                                ++x) {
                                              final pixel =
                                                  decodedImage.getPixel(x, y);
                                              imageToProcess.setPixelRgb(x, y,
                                                  pixel.r, pixel.g, pixel.b);
                                            }
                                          }
                                        } else if (decodedImage.numChannels ==
                                            3) {
                                          imageToProcess = decodedImage;
                                        } else {
                                          // Handle grayscale or other formats if necessary, or show error
                                          _showTemporaryError(
                                              'Image must be RGB or RGBA'); // Updated error message
                                          return;
                                        }
// Now imageToProcess is guaranteed to be an RGB image (numChannels == 3)
                                        final rgbBytesList =
                                            imageToProcess.toUint8List();

                                        setState(() {
                                          _controlImage = File(pickedFile.path);
                                          _controlRgbBytes = rgbBytesList;
                                          _controlWidth = imageToProcess.width;
                                          _controlHeight =
                                              imageToProcess.height;
                                          // Reset canny if new control image loaded
                                          useCanny = false;
                                          _cannyImage = null;
                                          // Need to re-initialize processor with new control image data
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
                                              Icon(Icons.add_photo_alternate,
                                                  size: 64,
                                                  color: theme
                                                      .colorScheme.primary
                                                      .withOpacity(0.5)),
                                              const SizedBox(height: 12),
                                              Text('Load control image',
                                                  style: TextStyle(
                                                      color: theme
                                                          .colorScheme.primary
                                                          .withOpacity(0.5),
                                                      fontSize: 16)),
                                            ],
                                          )
                                        : Image.file(_controlImage!,
                                            fit: BoxFit.contain),
                                  ),
                                ),
                              ),
                            ),
                            const SizedBox(height: 16),
                            // --- Canny Option ---
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
                                            // Enable only if control image exists
                                            setState(() {
                                              useCanny = v;
                                              if (v) {
                                                _processCannyImage(); // Generate canny edges
                                              } else {
                                                _cannyImage =
                                                    null; // Clear canny result
                                                // Re-initialize processor without canny data if needed (or handle in generate call)
                                              }
                                            });
                                          },
                                    label: const Text('Use Canny Preprocessor'),
                                  ),
                                  if (isCannyProcessing)
                                    Padding(
                                      padding: const EdgeInsets.only(left: 8.0),
                                      child: SizedBox(
                                          width: 16,
                                          height: 16,
                                          child: CircularProgressIndicator(
                                              strokeWidth: 2,
                                              color:
                                                  theme.colorScheme.primary)),
                                    ),
                                ],
                              ),
                            ),
                            if (useCanny && _cannyImage != null) ...[
                              const SizedBox(height: 16),
                              const Text('Canny Edge Preview:',
                                  style:
                                      TextStyle(fontWeight: FontWeight.bold)),
                              const SizedBox(height: 8),
                              Container(
                                decoration: BoxDecoration(
                                    border: Border.all(
                                        color: theme.colorScheme.primary
                                            .withOpacity(0.5)),
                                    borderRadius: BorderRadius.circular(8)),
                                constraints: const BoxConstraints(
                                    maxHeight: 200), // Limit preview height
                                width: double.infinity,
                                child: Center(child: _cannyImage!),
                              ),
                              const SizedBox(height: 16),
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
                          ], // End of if(useControlImage)
                          // --- Control Strength ---
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
                                  onChanged: (isModelLoading || isGenerating)
                                      ? null
                                      : (v) => setState(() {
                                            controlStrength = v;
                                            // Re-initialize processor with new strength
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
                                          }),
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
            ], // End of if(useControlNet)
            const SizedBox(height: 16),

            // --- Sampling Parameters ---
            Row(
              children: [
                const Text('Sampling Method'),
                const SizedBox(width: 8),
                Expanded(
                  child: ShadSelect<String>(
                    enabled: !(isModelLoading || isGenerating),
                    initialValue: samplingMethod,
                    placeholder: Text(samplingMethod),
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
                const Text('CFG Scale'),
                const SizedBox(width: 8),
                Expanded(
                  child: ShadSlider(
                    initialValue: cfg,
                    min: 1,
                    max: 20,
                    divisions: 38,
                    onChanged: (isModelLoading || isGenerating)
                        ? null
                        : (v) => setState(() => cfg = v),
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
                    max: 50, // Adjust max steps if needed
                    divisions: 49,
                    onChanged: (isModelLoading || isGenerating)
                        ? null
                        : (v) => setState(() => steps = v.toInt()),
                  ),
                ),
                Text(steps.toString()),
              ],
            ),
            const SizedBox(height: 16),
            // Strength (Denoising Strength for img2img)
            Row(
              children: [
                const Text('Denoising Strength'),
                const SizedBox(width: 8),
                Expanded(
                  child: ShadSlider(
                    initialValue: strength,
                    min: 0.0,
                    max: 1.0,
                    divisions: 20,
                    onChanged: (isModelLoading || isGenerating)
                        ? null
                        : (v) => setState(() => strength = v),
                  ),
                ),
                Text(strength.toStringAsFixed(2)),
              ],
            ),
            const SizedBox(height: 16),

            // --- Output Dimensions (Display Only) ---
            Row(
              children: [
                const Text('Width'), // Keep the label consistent
                const SizedBox(width: 8),
                Expanded(
                  child: ShadSelect<int>(
                    enabled: false, // Disable the dropdown
                    placeholder: Text(
                      // Use placeholder to display current value
                      _inputImage == null ? 'N/A' : outputWidth.toString(),
                      style: TextStyle(
                        color: Theme.of(context)
                            .disabledColor, // Indicate disabled state
                      ),
                    ),
                    options: const [], // No options needed as it's disabled
                    selectedOptionBuilder: (context, value) => Text(value
                        .toString()), // Builder (won't be used when disabled)
                    // No onChanged needed
                  ),
                ),
                const SizedBox(width: 16),
                const Text('Height'), // Keep the label consistent
                const SizedBox(width: 8),
                Expanded(
                  child: ShadSelect<int>(
                    enabled: false, // Disable the dropdown
                    placeholder: Text(
                      // Use placeholder to display current value
                      _inputImage == null ? 'N/A' : outputHeight.toString(),
                      style: TextStyle(
                        color: Theme.of(context)
                            .disabledColor, // Indicate disabled state
                      ),
                    ),
                    options: const [], // No options needed as it's disabled
                    selectedOptionBuilder: (context, value) => Text(value
                        .toString()), // Builder (won't be used when disabled)
                    // No onChanged needed
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),

            // --- Seed ---
            const Text('Seed (-1 for random)'),
            const SizedBox(height: 8),
            ShadInput(
              enabled: !(isModelLoading || isGenerating),
              placeholder: const Text('Seed'),
              keyboardType: TextInputType.number,
              onChanged: (String? v) => setState(() => seed = v ?? "-1"),
              initialValue: seed,
            ),
            const SizedBox(height: 16),

            // --- Generate Button ---
            Row(
              // Wrap buttons in a Row
              children: [
                ShadButton(
                  enabled: !(isModelLoading || isGenerating),
                  onPressed: () {
                    // --- Pre-generation Checks ---
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
                    if (_inputImage == null ||
                        _rgbBytes == null ||
                        _inputWidth == null ||
                        _inputHeight == null) {
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
                    // Ensure mask is generated and dimensions are calculated
                    _calculateDimensionsAndGenerateMask(); // Recalculate dimensions and mask
                    if (_maskData == null) {
                      _handleLoadingError('maskError',
                          'Failed to generate outpainting mask. Adjust padding.');
                      return;
                    }
                    // Use the dimensions calculated by _calculateDimensionsAndGenerateMask
                    final int finalOutputWidth = outputWidth;
                    final int finalOutputHeight = outputHeight;

                    if (finalOutputWidth <= 0 || finalOutputHeight <= 0) {
                      _handleLoadingError('dimensionError',
                          'Invalid output dimensions calculated.');
                      return;
                    }

                    // --- Create Padded Input Image Data ---
                    // Create a new buffer for the padded image, filled with grey (127, 127, 127)
                    final int paddedDataSize =
                        finalOutputWidth * finalOutputHeight * 3;
                    final paddedRgbBytes = Uint8List(paddedDataSize);
                    for (int i = 0; i < paddedDataSize; i++) {
                      paddedRgbBytes[i] = 127; // Fill with mid-grey
                    }

                    // Copy the original image (_rgbBytes) onto the grey padded buffer
                    for (int y = 0; y < _inputHeight!; y++) {
                      for (int x = 0; x < _inputWidth!; x++) {
                        // Calculate source index in original _rgbBytes
                        int srcIndex = (y * _inputWidth! + x) * 3;
                        // Calculate destination index in paddedRgbBytes (offset by padding)
                        int dstIndex = ((y + paddingTop) * finalOutputWidth +
                                (x + paddingLeft)) *
                            3;

                        // Ensure indices are within bounds (should be, but good practice)
                        if (srcIndex + 2 < _rgbBytes!.length &&
                            dstIndex + 2 < paddedRgbBytes.length) {
                          paddedRgbBytes[dstIndex] = _rgbBytes![srcIndex]; // R
                          paddedRgbBytes[dstIndex + 1] =
                              _rgbBytes![srcIndex + 1]; // G
                          paddedRgbBytes[dstIndex + 2] =
                              _rgbBytes![srcIndex + 2]; // B
                        }
                      }
                    }
                    // --- End of Padded Input Image Creation ---

                    // --- Control Image Processing Logic ---
                    Uint8List? finalControlImageData;
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
                      // Check if control image dimensions match the target OUTPUT size
                      if (sourceControlWidth != finalOutputWidth ||
                          sourceControlHeight != finalOutputHeight) {
                        try {
                          print(
                              'Control image dimensions ($sourceControlWidth x $sourceControlHeight) differ from target ($finalOutputWidth x $finalOutputHeight). Processing using $_controlImageProcessingMode...');
                          ProcessedImageData processedData;
                          if (_controlImageProcessingMode == 'Crop') {
                            processedData = cropImage(
                                sourceControlBytes,
                                sourceControlWidth,
                                sourceControlHeight,
                                finalOutputWidth,
                                finalOutputHeight);
                          } else {
                            // Default to Resize
                            processedData = resizeImage(
                                sourceControlBytes,
                                sourceControlWidth,
                                sourceControlHeight,
                                finalOutputWidth,
                                finalOutputHeight);
                          }
                          finalControlImageData = processedData.bytes;
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
                        finalControlImageData = sourceControlBytes;
                        finalControlWidth = sourceControlWidth;
                        finalControlHeight = sourceControlHeight;
                      }
                    } else {
                      // No valid source control image, ensure control params are null
                      finalControlImageData = null;
                      finalControlWidth = null;
                      finalControlHeight = null;
                    }
                    // --- End Control Image Processing Logic ---

                    // --- Start Generation ---
                    setState(() {
                      isGenerating = true;
                      // _modelError = ''; // Removed
                      status = 'Generating image...';
                      progress = 0;
                      _generatedImage = null; // Clear previous image
                      _generationLogs = []; // Clear previous logs
                      _showLogsButton =
                          false; // Hide log button until generation finishes
                    });

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
                      // Input Image (NOW THE PADDED VERSION)
                      inputImageData:
                          paddedRgbBytes, // Send the grey-padded buffer
                      inputWidth:
                          finalOutputWidth, // Width is the final output width
                      inputHeight:
                          finalOutputHeight, // Height is the final output height
                      channel: 3, // Input is RGB

                      // Output Dimensions (should match padded input and mask)
                      outputWidth: finalOutputWidth,
                      outputHeight: finalOutputHeight,

                      // Mask (Generated based on padding, dimensions match output)
                      maskImageData: _maskData!,
                      maskWidth:
                          finalOutputWidth, // Mask dimensions match output
                      maskHeight: finalOutputHeight,

                      // Other Parameters
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
                      strength: strength, // Denoising strength (Already passed)
                      seed: int.tryParse(seed) ?? -1, // Already passed
                      batchCount: 1, // Already passed

                      // ControlNet Parameters
                      controlImageData:
                          finalControlImageData, // Use processed bytes (Already passed)
                      controlImageWidth:
                          finalControlWidth, // Use processed width (Already passed)
                      controlImageHeight:
                          finalControlHeight, // Use processed height (Already passed)
                      controlStrength: useControlNet
                          ? controlStrength
                          : 0.0, // Already passed

                      // New sampling parameters
                      slgScale: slgScale,
                      skipLayersText: formattedSkipLayers,
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

            // --- Progress Indicator ---
            if (isGenerating || progress > 0) ...[
              LinearProgressIndicator(
                value: progress,
                backgroundColor: theme.colorScheme.background,
                color: theme.colorScheme.primary,
              ),
              const SizedBox(height: 8),
              Text(status, style: theme.textTheme.p),
            ],
            if (_generatedImage != null) ...[
              const SizedBox(height: 20),
              Center(child: _generatedImage!), // Center the generated image
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
