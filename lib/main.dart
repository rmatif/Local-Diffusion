import 'dart:io';
import 'dart:typed_data';
import 'package:dotted_border/dotted_border.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:ui' as ui;
import 'dart:async';
import 'dart:developer' as developer;
import 'package:permission_handler/permission_handler.dart'; // Added
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
  // Ensure Flutter bindings are initialized
  WidgetsFlutterBinding.ensureInitialized();
  // Initialize FFI bindings with the default backend BEFORE running the app
  FFIBindings.initializeBindings('CPU');
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

// Add WidgetsBindingObserver to listen for app lifecycle changes
class _MyAppState extends State<MyApp> with WidgetsBindingObserver {
  bool _hasPermission = false;
  bool _isLoading = true; // Track initial loading state

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _checkPermissionStatus();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  // Re-check permission when app resumes
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      _checkPermissionStatus();
    }
  }

  Future<void> _checkPermissionStatus() async {
    // Don't show loading indicator on subsequent checks (e.g., after resuming)
    // Only show it during the initial initState check.
    if (mounted && _isLoading) {
      setState(() {
        // Keep isLoading true until check is complete
      });
    }

    final status = await Permission.manageExternalStorage.status;
    if (mounted) {
      // Check if the widget is still mounted before calling setState
      setState(() {
        _hasPermission = status.isGranted;
        _isLoading = false; // Mark loading as complete
      });
    }
    // If permission is denied and we are not loading anymore, prompt the user.
    // Avoid prompting immediately on first load if denied, let the UI show first.
    if (!_hasPermission && !_isLoading) {
      // Optionally, you could automatically trigger the request here,
      // but it's often better UX to let the user click a button.
      // _requestPermission();
    }
  }

  // Removed unused _requestPermission function

  // Function to request the MANAGE_EXTERNAL_STORAGE permission
  // This usually opens the specific system screen for this permission.
  Future<void> _requestManageStoragePermission() async {
    await Permission.manageExternalStorage.request();
    // Re-check status after the user potentially interacts with the settings screen
    _checkPermissionStatus();
  }

  @override
  Widget build(BuildContext context) {
    return ShadApp(
      darkTheme: ShadThemeData(
        brightness: Brightness.dark,
        colorScheme: const ShadSlateColorScheme.dark(),
      ),
      home: _isLoading
          ? const Scaffold(
              body: Center(
                  child:
                      CircularProgressIndicator())) // Show loading indicator initially
          : _hasPermission
              ? const StableDiffusionApp()
              : PermissionRequiredScreen(
                  onRequestPermission:
                      _requestManageStoragePermission), // Pass the request function
    );
  }
}

// New Screen to show when permission is denied
class PermissionRequiredScreen extends StatelessWidget {
  final VoidCallback onRequestPermission;

  const PermissionRequiredScreen(
      {super.key, required this.onRequestPermission});

  @override
  Widget build(BuildContext context) {
    final theme = ShadTheme.of(context);
    return Scaffold(
      backgroundColor: theme.colorScheme.background,
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Icon(
                Icons.folder_off_outlined, // Or another relevant icon
                size: 80,
                color: theme.colorScheme.primary.withOpacity(0.7),
              ),
              const SizedBox(height: 24),
              Text(
                'Storage Permission Required',
                style: theme.textTheme.h2.copyWith(fontWeight: FontWeight.bold),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
              Text(
                'This app needs permission to read and write files (including models) in storage to function correctly. Please grant the "All files access" permission in the app settings.',
                style: theme.textTheme.p,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 32),
              ShadButton(
                onPressed: onRequestPermission, // Use the passed function
                child: const Text('Grant Permission'), // Changed button text
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class StableDiffusionApp extends StatefulWidget {
  const StableDiffusionApp({super.key});
  @override
  State<StableDiffusionApp> createState() => _StableDiffusionAppState();
}

// Custom widget for looping dot animation
class LoadingDotsAnimation extends StatefulWidget {
  final String loadingText;
  final TextStyle? style;
  final Duration duration;

  const LoadingDotsAnimation({
    super.key,
    required this.loadingText,
    this.style,
    this.duration = const Duration(milliseconds: 1200), // Total loop duration
  });

  @override
  State<LoadingDotsAnimation> createState() => _LoadingDotsAnimationState();
}

class _LoadingDotsAnimationState extends State<LoadingDotsAnimation>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: widget.duration,
    )..repeat(reverse: true); // Make the animation loop back and forth
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        // Calculate dots: 1 to 4 dots (value goes 0 -> 1 -> 0)
        // Map value (0.0 to 1.0) to dot count (1 to 4)
        final dotCount = (_controller.value * 3).floor() + 1;

        return Text(
          '${widget.loadingText}${'.' * dotCount}',
          style: widget.style,
        );
      },
    );
  }
}

class _StableDiffusionAppState extends State<StableDiffusionApp>
    with SingleTickerProviderStateMixin {
  final ScrollController _scrollController =
      ScrollController(); // Add ScrollController
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
  double clipSkip = 0; // Existing, used for new accordion
  double eta = 0.0; // New state for eta slider
  double guidance = 3.5; // New state for guidance slider
  double slgScale = 0.0; // New state for slg-scale slider
  String skipLayersText = ''; // New state for skip-layers text field
  double skipLayerStart = 0.01; // New state for skip-layer-start slider
  double skipLayerEnd = 0.2; // New state for skip-layer-end slider
  final TextEditingController _skipLayersController =
      TextEditingController(); // Controller for skip-layers input
  String? _skipLayersErrorText; // Error text for skip-layers validation

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
  String _loadingError =
      ''; // Consolidated error message for model/taesd/controlnet
  String _loadingErrorType =
      ''; // To track which component failed ('model', 'taesd', 'controlnet')
  Timer? _loadingErrorTimer; // Timer to clear the loading error
  List<String> _generationLogs = []; // To store logs for the last generation
  bool _showLogsButton = false; // To control visibility of the log button
  bool _isDiffusionModelType = false; // Added state for the new switch
  String _selectedBackend = 'CPU'; // State for the selected backend
  final List<String> _availableBackends = [
    'CPU',
    'Vulkan',
    'OpenCL'
  ]; // Available backends

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
    'dpm++2s_a',
    'dpm++2m',
    'dpm++2mv2',
    'ipndm',
    'ipndm_v',
    'lcm',
    'ddim_trailing', // New sampler
    'tcd' // New sampler
  ];

  List<int> getWidthOptions() {
    List<int> opts = [];
    // 128 to 512 (inclusive)
    for (int i = 128; i <= 512; i += 64) {
      opts.add(i);
    }
    // 576 to 1024 (inclusive)
    for (int i = 576; i <= 1024; i += 64) {
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
    // Ensure bindings are initialized before getting cores
    // FFIBindings.initializeBindings(_selectedBackend); // Moved to main()
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
    _errorMessageTimer?.cancel(); // Keep for non-loading TAESD errors if needed
    _loadingErrorTimer?.cancel(); // Cancel the general loading error timer
    _processor?.dispose();
    _cannyProcessor?.dispose();
    _promptController.dispose();
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
      loadingText = 'Loading Model...'; // Set loading text immediately
    });
    _processor?.dispose();
    _processor = StableDiffusionProcessor(
      // Removed backend: _selectedBackend,
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
      clipSkip: clipSkip.toInt(), // Passed from state
      vaeTiling: useVAETiling, // Passed from state
      controlNetPath: _controlNetPath,
      controlImageData: _controlRgbBytes,
      controlImageWidth: _controlWidth,
      controlImageHeight: _controlHeight,
      controlStrength: controlStrength,
      isDiffusionModelType: _isDiffusionModelType,
      // Note: Other new params (eta, guidance, etc.) are generation-time only
      // and don't need to be passed during initialization via newSdCtx.
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

        // Check for error messages passed via the log stream (as implemented in processor)
        if (log.level == -1 && log.message.startsWith("Error (")) {
          final errorMatch =
              RegExp(r'Error \((.*?)\): (.*)').firstMatch(log.message);
          if (errorMatch != null) {
            final errorType = errorMatch.group(1)!;
            final errorMessage = errorMatch.group(2)!;
            _handleLoadingError(errorType, errorMessage);
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

  // New method to handle loading errors centrally
  void _handleLoadingError(String errorType, String errorMessage) {
    _loadingErrorTimer?.cancel(); // Cancel previous timer if any

    // Call the central reset function
    _resetState(); // Call the new reset function

    // Set the specific error message for loading failure
    _processor?.dispose();

    setState(() {
      _loadingError = errorMessage; // Display the specific error
      _loadingErrorType = errorType; // Store error type if needed elsewhere

      // Handle generation-specific errors separately if needed
      if (errorType == 'generationError') {
        status = 'Generation failed: $errorMessage';
        isGenerating = false; // Stop generation indicator
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

  // Central function to reset the state
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
      // Reset other related state
      _loraNames = [];
      _ramUsage = ''; // Clear RAM usage display
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
      // width = 512;
      // height = 512;
      // seed = "-1";
      // controlStrength = 0.9;
      // _controlImageProcessingMode = 'Resize';
      // _isDiffusionModelType = false;
    });
  }

  // Removed simulateLoading as it's not relevant to the core task

  void showModelLoadDialog() {
    String selectedQuantization = 'NONE';
    String selectedSchedule = 'DEFAULT';
    bool useFlashAttention = false;
    String? flashAttentionError; // Added state for error message

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
                onChanged: (v) {
                  // Check backend and desired state
                  if (_selectedBackend != 'CPU' && v) {
                    // Trying to enable on non-CPU backend
                    setState(() {
                      flashAttentionError =
                          'Flash Attention is supported only on CPU';
                      // Do NOT set useFlashAttention = true
                    });
                  } else {
                    // Either CPU backend or turning the switch off
                    setState(() {
                      useFlashAttention = v;
                      flashAttentionError = null; // Clear error if any
                    });
                  }
                },
                label: const Text('Use Flash Attention'),
              ),
              // Display error message if it exists
              if (flashAttentionError != null)
                Padding(
                  padding: const EdgeInsets.only(top: 8.0),
                  child: Text(
                    flashAttentionError!,
                    style: TextStyle(
                      color: ShadTheme.of(context)
                          .colorScheme
                          .destructive, // Use theme's destructive color
                      fontSize: 12,
                    ),
                  ),
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
      // Disable drawer drag gesture when loading or generating
      drawerEnableOpenDragGesture: !(isModelLoading || isGenerating),
      appBar: AppBar(
        // Explicitly add the leading menu button to control its state
        leading: Builder(
          builder: (context) => IconButton(
            icon: const Icon(Icons.menu),
            // Disable the button when loading or generating
            onPressed: (isModelLoading || isGenerating)
                ? null
                : () => Scaffold.of(context).openDrawer(),
            tooltip: MaterialLocalizations.of(context).openAppDrawerTooltip,
          ),
        ),
        title: const Text('Local Diffusion',
            style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: theme.colorScheme.background,
        elevation: 0,
        actions: [
          // Add Unload button only if a model is loaded
          if (_processor != null)
            Padding(
              padding: const EdgeInsets.only(right: 8.0),
              // Wrap the button with a Tooltip widget
              child: Tooltip(
                message: 'Unload Model & Reset', // Use the message property
                child: ShadButton.ghost(
                  icon: const Icon(
                    LucideIcons
                        .powerOff, // Or LucideIcons.unload, LucideIcons.xCircle
                    size: 20,
                  ),
                  // Remove the invalid tooltip parameter from ShadButton.ghost
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
        controller: _scrollController, // Attach ScrollController
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Display Loading Status / Success / Error Messages
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
                        .shake(
                            hz: 4,
                            offset: const Offset(
                                2, 0)), // Use offset for shake amount
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
                      loadingText: loadingText,
                      style: theme.textTheme.p.copyWith(
                        color: Colors.orange,
                        fontWeight: FontWeight.bold,
                      ),
                    ).animate().fadeIn(), // Keep the fade-in animation
                ],
              ),
            ),
            // --- Backend Selection Row ---
            Row(
              children: [
                const Text('Backend:'),
                const SizedBox(width: 8),
                Expanded(
                  child: ShadSelect<String>(
                    placeholder: Text(_selectedBackend),
                    enabled: !(isModelLoading ||
                        isGenerating), // Disable during loading/generation
                    options: _availableBackends
                        .map((backend) => ShadOption(
                              value: backend,
                              child: Text(backend),
                            ))
                        .toList(),
                    selectedOptionBuilder: (context, value) => Text(value),
                    onChanged: (String? newBackend) {
                      if (newBackend != null &&
                          newBackend != _selectedBackend) {
                        // Check if a model is loaded
                        if (_processor != null) {
                          // Show confirmation dialog
                          showShadDialog(
                            context: context,
                            builder: (context) => ShadDialog.alert(
                              title: const Text('Change Backend?'),
                              description: const Text(
                                  'Changing the backend requires unloading the current model and resetting settings. Proceed?'),
                              actions: [
                                ShadButton.outline(
                                  onPressed: () => Navigator.of(context).pop(),
                                  child: const Text('Cancel'),
                                ),
                                ShadButton.destructive(
                                  onPressed: () {
                                    Navigator.of(context).pop(); // Close dialog
                                    print(
                                        "Backend changed with model loaded. Resetting state.");
                                    _resetState(); // Reset state first
                                    print(
                                        "Initializing FFI bindings for: $newBackend");
                                    FFIBindings.initializeBindings(
                                        newBackend); // Re-init FFI
                                    setState(() {
                                      _selectedBackend = newBackend;
                                      // Re-fetch cores in case it changes based on backend (though unlikely)
                                      _cores = FFIBindings.getCores() * 2;
                                    });
                                    print(
                                        "Backend changed to: $_selectedBackend");
                                  },
                                  child: const Text('Confirm Change'),
                                ),
                              ],
                            ),
                          );
                        } else {
                          // No model loaded, just change the backend
                          print("Initializing FFI bindings for: $newBackend");
                          FFIBindings.initializeBindings(
                              newBackend); // Re-init FFI
                          setState(() {
                            _selectedBackend = newBackend;
                            // Re-fetch cores
                            _cores = FFIBindings.getCores() * 2;
                          });
                          print("Backend changed to: $_selectedBackend");
                        }
                      }
                    },
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16), // Spacing after backend dropdown
            // --- Model Loading Row ---
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
            ShadAccordion<Map<String, dynamic>>(
              children: [
                ShadAccordionItem<Map<String, dynamic>>(
                  value: const {},
                  title: const Text('Advanced Model Options'), // Renamed title
                  child: Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Column(
                      crossAxisAlignment:
                          CrossAxisAlignment.start, // Align items left
                      children: [
                        // Original content starts here
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
                        // Added Switch for Diffusion Model Type (Moved)
                        Padding(
                          padding: const EdgeInsets.only(
                              top: 8.0, bottom: 16.0), // Adjusted padding
                          child: ShadSwitch(
                            value: _isDiffusionModelType,
                            onChanged: (v) =>
                                setState(() => _isDiffusionModelType = v),
                            label:
                                const Text('Standalone Model'), // Renamed label
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
                        // VAE Tiling checkbox moved to the new accordion below
                      ],
                    ),
                  ),
                ),
              ],
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
                                  // Removed check: if (useTAESD && v) { ... }
                                  setState(() {
                                    useVAETiling = v;
                                    // Reinitialize processor if needed (already handled in original code)
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
                                // If _controlNetPath is null, enabling the switch does nothing yet.
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
                          // Removed "Use Image Reference" checkbox
                          // Always show image input and Canny options when ControlNet is enabled
                          ...[
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
                          ], // End of always shown block
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
            const SizedBox(height: 16), // Added spacing
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
                    maxHeight: 200, // Added max height
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
                    maxHeight: 200, // Added max height
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
                      // Show error if no model is loaded
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
                      // _modelError = ''; // Replaced by _loadingError logic
                      status = 'Generating image...';
                      progress = 0;
                      _generationLogs = []; // Clear previous logs
                      _showLogsButton =
                          false; // Hide log button until generation finishes
                    });

                    // --- Control Image Processing Logic (Existing) ---
                    Uint8List? finalControlBytes = _controlRgbBytes;
                    int? finalControlWidth = _controlWidth;
                    int? finalControlHeight = _controlHeight;
                    Uint8List? sourceBytes;
                    int? sourceWidth;
                    int? sourceHeight;
                    if (useControlNet) {
                      // Removed check for useControlImage
                      if (useCanny && _cannyProcessor?.resultRgbBytes != null) {
                        sourceBytes = _cannyProcessor!.resultRgbBytes!;
                        sourceWidth = _cannyProcessor!.resultWidth!;
                        sourceHeight = _cannyProcessor!.resultHeight!;
                        sourceBytes = _ensureRgbFormat(
                            sourceBytes, sourceWidth, sourceHeight);
                      } else if (_controlRgbBytes != null) {
                        sourceBytes = _controlRgbBytes;
                        sourceWidth = _controlWidth;
                        sourceHeight = _controlHeight;
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
                      if (sourceWidth != width || sourceHeight != height) {
                        try {
                          print(
                              'Control image dimensions ($sourceWidth x $sourceHeight) differ from target ($width x $height). Processing using $_controlImageProcessingMode...');
                          ProcessedImageData processedData;
                          if (_controlImageProcessingMode == 'Crop') {
                            processedData = cropImage(sourceBytes, sourceWidth,
                                sourceHeight, width, height);
                          } else {
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
                          _showTemporaryError(
                              'Error processing control image: $e');
                          setState(() => isGenerating = false);
                          return;
                        }
                      } else {
                        finalControlBytes = sourceBytes;
                        finalControlWidth = sourceWidth;
                        finalControlHeight = sourceHeight;
                      }
                    } else {
                      finalControlBytes = null;
                      finalControlWidth = null;
                      finalControlHeight = null;
                    }
                    // --- End Control Image Processing Logic ---

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
                      // Pass new parameters
                      clipSkip: clipSkip.toInt(),
                      eta: eta,
                      guidance: guidance,
                      slgScale: slgScale,
                      skipLayersText:
                          formattedSkipLayers, // Pass formatted string or null
                      skipLayerStart: skipLayerStart,
                      skipLayerEnd: skipLayerEnd,
                      // Existing control parameters
                      controlImageData: finalControlBytes,
                      controlImageWidth: finalControlWidth,
                      controlImageHeight: finalControlHeight,
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
            // Removed the old _modelError display, it's handled above
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
