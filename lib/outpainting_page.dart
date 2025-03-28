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
// import 'mask_editor.dart'; // Not directly needed for mask creation anymore

class OutpaintingPage extends StatefulWidget {
  const OutpaintingPage({super.key});

  @override
  State<OutpaintingPage> createState() => _OutpaintingPageState();
}

class _OutpaintingPageState extends State<OutpaintingPage>
    with SingleTickerProviderStateMixin {
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
  double clipSkip = 1.0;
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
  String _modelError = '';
  File? _inputImage;
  Uint8List? _rgbBytes;
  int? _inputWidth;
  int? _inputHeight;
  double strength = 0.75; // Default strength for outpainting often higher
  Uint8List? _maskData; // Generated based on padding
  ui.Image? _maskImageUi; // To display the generated mask

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
    _modelErrorTimer?.cancel();
    _processor?.dispose();
    _processor = null;
    _cannyProcessor?.dispose();
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

      // Use the calculated output dimensions for saving metadata
      await _processor!.saveGeneratedImage(
        image,
        prompt,
        outputWidth, // Use calculated output width
        outputHeight, // Use calculated output height
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
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // --- Loading Indicators & Model Info ---
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
                                          fontWeight: FontWeight.bold),
                                    ),
                                    const WidgetSpan(
                                        child: Icon(Icons.check,
                                            size: 20, color: Colors.green)),
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
                                fontWeight: FontWeight.bold),
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
                                'Load image for Outpainting',
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

            // --- Advanced Options (LoRA, VAE, etc.) ---
            ShadAccordion<Map<String, dynamic>>(
              children: [
                ShadAccordionItem<Map<String, dynamic>>(
                  value: const {},
                  title: const Text('Advanced Options'),
                  child: Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Column(
                      children: [
                        // --- LoRA Loading and Selection ---
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
                        // --- Clip L/G Loading ---
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
                                          // ... [Clip_L selection dialog - same as inpainting] ...
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
                                                currentSchedule);
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
                                          // ... [Clip_G selection dialog - same as inpainting] ...
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
                                                currentSchedule);
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
                        // --- T5XXL & Embedding Loading ---
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
                                          // ... [T5XXL selection dialog - same as inpainting] ...
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
                                                currentSchedule);
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
                                            currentSchedule);
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
                        // --- VAE Loading & Options ---
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
                                          // ... [VAE selection dialog - same as inpainting] ...
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
                                                currentSchedule);
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
                                if (_vaePath == null && v) {
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
                                        currentSchedule);
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
                            if (useTAESD && v) {
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
                                    currentSchedule);
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

            // --- ControlNet Options ---
            Row(
              children: [
                const Text('Use ControlNet'),
                const SizedBox(width: 8),
                ShadSwitch(
                  value: useControlNet,
                  onChanged: (bool v) {
                    setState(() {
                      useControlNet = v;
                      if (_processor != null) {
                        String currentModelPath = _processor!.modelPath;
                        bool currentFlashAttention =
                            _processor!.useFlashAttention;
                        SDType currentModelType = _processor!.modelType;
                        Schedule currentSchedule = _processor!.schedule;
                        String? originalControlNetPath = _controlNetPath;
                        if (!v)
                          _controlNetPath =
                              null; // Disable CN path if toggling off
                        _initializeProcessor(
                          currentModelPath,
                          currentFlashAttention,
                          currentModelType,
                          currentSchedule,
                        );
                        if (!v)
                          _controlNetPath =
                              originalControlNetPath; // Restore for future toggle
                      }
                      // If turning off CN, maybe reset related states?
                      if (!v) {
                        _controlImage = null;
                        _controlRgbBytes = null;
                        _cannyImage = null;
                        useCanny = false;
                        useControlImage = false;
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
            ShadButton(
              enabled: !(isModelLoading || isGenerating),
              onPressed: () {
                // --- Pre-generation Checks ---
                if (_processor == null) {
                  _modelErrorTimer?.cancel();
                  setState(() => _modelError = 'Please Load a model first');
                  _modelErrorTimer = Timer(const Duration(seconds: 10),
                      () => setState(() => _modelError = ''));
                  return;
                }
                if (_inputImage == null ||
                    _rgbBytes == null ||
                    _inputWidth == null ||
                    _inputHeight == null) {
                  _modelErrorTimer?.cancel();
                  setState(
                      () => _modelError = 'Please select an input image first');
                  _modelErrorTimer = Timer(const Duration(seconds: 10),
                      () => setState(() => _modelError = ''));
                  return;
                }
                // Ensure mask is generated and dimensions are calculated
                _calculateDimensionsAndGenerateMask(); // Recalculate dimensions and mask
                if (_maskData == null) {
                  _modelErrorTimer?.cancel();
                  setState(() =>
                      _modelError = 'Failed to generate outpainting mask');
                  _modelErrorTimer = Timer(const Duration(seconds: 10),
                      () => setState(() => _modelError = ''));
                  return;
                }
                // Use the dimensions calculated by _calculateDimensionsAndGenerateMask
                final int finalOutputWidth = outputWidth;
                final int finalOutputHeight = outputHeight;

                if (finalOutputWidth <= 0 || finalOutputHeight <= 0) {
                  _modelErrorTimer?.cancel();
                  setState(() =>
                      _modelError = 'Invalid output dimensions calculated');
                  _modelErrorTimer = Timer(const Duration(seconds: 10),
                      () => setState(() => _modelError = ''));
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

                // Determine ControlNet input (logic remains the same)
                Uint8List? finalControlImageData = _controlRgbBytes;
                int? finalControlWidth = _controlWidth;
                int? finalControlHeight = _controlHeight;

                if (useControlNet &&
                    useControlImage &&
                    useCanny &&
                    _cannyProcessor?.resultRgbBytes != null) {
                  finalControlImageData = _ensureRgbFormat(
                      _cannyProcessor!.resultRgbBytes!,
                      _cannyProcessor!.resultWidth!,
                      _cannyProcessor!.resultHeight!);
                  finalControlWidth = _cannyProcessor!.resultWidth;
                  finalControlHeight = _cannyProcessor!.resultHeight;
                } else if (useControlNet && !useControlImage) {
                  finalControlImageData = null;
                  finalControlWidth = null;
                  finalControlHeight = null;
                } else if (!useControlNet) {
                  finalControlImageData = null;
                  finalControlWidth = null;
                  finalControlHeight = null;
                }

                // --- Start Generation ---
                setState(() {
                  isGenerating = true;
                  _modelError = '';
                  status = 'Generating image...';
                  progress = 0;
                  _generatedImage = null; // Clear previous image
                });

                _processor!.generateImg2Img(
                  // Input Image (NOW THE PADDED VERSION)
                  inputImageData: paddedRgbBytes, // Send the grey-padded buffer
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
                  maskWidth: finalOutputWidth, // Mask dimensions match output
                  maskHeight: finalOutputHeight,

                  // Other Parameters
                  prompt: prompt,
                  negativePrompt: negativePrompt,
                  clipSkip: clipSkip.toInt(),
                  cfgScale: cfg,
                  sampleSteps: steps,
                  sampleMethod: SampleMethod.values
                      .firstWhere(
                        (method) =>
                            method.displayName.toLowerCase() ==
                            samplingMethod.toLowerCase(),
                        orElse: () => SampleMethod.EULER_A,
                      )
                      .index,
                  strength: strength, // Denoising strength
                  seed: int.tryParse(seed) ?? -1,
                  batchCount: 1,

                  // ControlNet Parameters
                  controlImageData: finalControlImageData,
                  controlImageWidth: finalControlWidth,
                  controlImageHeight: finalControlHeight,
                  controlStrength: useControlNet ? controlStrength : 0.0,
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
}
