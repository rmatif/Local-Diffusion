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
  double clipSkip = 1.0; // Changed from 0 to 1.0 to align with typical default
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
  File? _inputImage;
  Uint8List? _rgbBytes;
  int? _inputWidth;
  int? _inputHeight;
  double strength = 0.5;

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
    _modelErrorTimer?.cancel();
    _processor?.dispose();
    _processor = null;
    _cannyProcessor?.dispose(); // Ensure processor is nullified
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
            ShadButton(
              enabled: !(isModelLoading || isGenerating),
              onPressed: () async {
                // Make onPressed async
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
                if (_inputImage == null) {
                  _modelErrorTimer?.cancel();
                  setState(() {
                    _modelError = 'Please select an image first';
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
                  status = 'Preparing image...'; // Update status
                  progress = 0;
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
                final int effectiveWidth = croppedData?.width ?? _inputWidth!;
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
                    sourceControlBytes = _ensureRgbFormat(sourceControlBytes,
                        sourceControlWidth, sourceControlHeight);
                  } else if (_controlRgbBytes != null) {
                    // Use original control image if Canny is not used or failed
                    sourceControlBytes = _controlRgbBytes;
                    sourceControlWidth = _controlWidth;
                    sourceControlHeight = _controlHeight;
                    // Ensure original control image is RGB (already done on load, but double-check)
                    if (sourceControlBytes != null &&
                        sourceControlWidth != null &&
                        sourceControlHeight != null) {
                      sourceControlBytes = _ensureRgbFormat(sourceControlBytes,
                          sourceControlWidth, sourceControlHeight);
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
                      _showTemporaryError('Error processing control image: $e');
                      setState(() => isGenerating = false); // Stop generation
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
                  strength: strength,
                  seed: int.tryParse(seed) ?? -1,
                  batchCount: 1,
                  controlImageData: finalControlBytes, // Use processed bytes
                  controlImageWidth: finalControlWidth, // Use processed width
                  controlImageHeight:
                      finalControlHeight, // Use processed height
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
