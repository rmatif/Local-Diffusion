import 'dart:io';
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:typed_data';
import 'package:file_picker/file_picker.dart';
import 'package:image/image.dart' as img;
import 'package:shadcn_ui/shadcn_ui.dart';
import 'package:gal/gal.dart';
import 'ffi_bindings.dart';
import 'inpainting_page.dart';
import 'scribble2img_page.dart';
import 'upscaler_processor.dart';
import 'package:path_provider/path_provider.dart';
import 'package:dotted_border/dotted_border.dart';
import 'package:before_after/before_after.dart';
import 'img2img_page.dart';
import 'main.dart';
import 'photomaker_page.dart';

class UpscalerPage extends StatefulWidget {
  const UpscalerPage({super.key});

  @override
  State<UpscalerPage> createState() => _UpscalerPageState();
}

class _UpscalerPageState extends State<UpscalerPage> {
  File? _inputImage;
  Image? _outputImage;
  bool _isProcessing = false;
  UpscalerProcessor? _processor;
  String _status = '';
  Timer? _errorTimer;
  Uint8List? _processedInputBytes;
  int? _imageWidth;
  int? _imageHeight;
  String? _originalFileName;
  Map<String, bool> loadedComponents = {};
  bool _isError = false;
  double _sliderValue = 0.5;

  @override
  void dispose() {
    _errorTimer?.cancel();
    _processor?.dispose();
    super.dispose();
  }

  void _showTemporaryError(String error) {
    _errorTimer?.cancel();
    setState(() {
      _status = error;
      _isError = true;
    });
    _errorTimer = Timer(const Duration(seconds: 5), () {
      setState(() {
        _status = '';
        _isError = false;
      });
    });
  }

  Future<String> getModelDirectory() async {
    final appDir = await getApplicationDocumentsDirectory();
    final modelDir = Directory('${appDir.path}/models');
    if (!await modelDir.exists()) {
      await modelDir.create(recursive: true);
    }
    return modelDir.path;
  }

  Future<void> _initializeProcessor() async {
    final modelDirPath = await getModelDirectory();
    final selectedDir = await FilePicker.platform
        .getDirectoryPath(initialDirectory: modelDirPath);

    if (selectedDir == null) {
      _showTemporaryError('No directory selected');
      return;
    }

    final modelFiles = Directory(selectedDir)
        .listSync()
        .whereType<File>()
        .where((file) => file.path.endsWith('.pth'))
        .toList();

    if (modelFiles.isEmpty) {
      _showTemporaryError('No upscaler model found in selected directory');
      return;
    }

    final selectedModel = await showShadDialog<String>(
      context: context,
      builder: (BuildContext context) {
        return ShadDialog.alert(
          constraints: const BoxConstraints(maxWidth: 400),
          title: const Text('Select Upscaler Model'),
          description: SizedBox(
            height: 300,
            child: Material(
              color: Colors.transparent,
              child: ShadTable.list(
                header: const [
                  ShadTableCell.header(
                    child: Text('Model'),
                  ),
                  ShadTableCell.header(
                    alignment: Alignment.centerRight,
                    child: Text('Size'),
                  ),
                ],
                columnSpanExtent: (index) {
                  if (index == 0) return const FixedTableSpanExtent(250);
                  if (index == 1) return const FixedTableSpanExtent(80);
                  return null;
                },
                children: modelFiles
                    .map(
                      (file) => [
                        ShadTableCell(
                          child: GestureDetector(
                            onTap: () => Navigator.pop(context, file.path),
                            child: Padding(
                              padding:
                                  const EdgeInsets.symmetric(vertical: 12.0),
                              child: Text(file.path.split('/').last),
                            ),
                          ),
                        ),
                        ShadTableCell(
                          alignment: Alignment.centerRight,
                          child: GestureDetector(
                            onTap: () => Navigator.pop(context, file.path),
                            child: Text(
                              '${(file.lengthSync() / (1024 * 1024)).toStringAsFixed(1)} MB',
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

    if (selectedModel == null) {
      _showTemporaryError('No model selected');
      return;
    }

    _processor = UpscalerProcessor(
      modelPath: selectedModel,
      nThreads: FFIBindings.getCores() * 2,
      wtype: SDType.NONE,
    );

    setState(() {
      loadedComponents['Model'] = true;
    });

    _processor?.imageStream.listen(
      (upscaledImageBytes) async {
        final fileName =
            _originalFileName?.replaceAll(RegExp(r'\.[^\.]+$'), '_x4.png');
        if (fileName != null) {
          await Gal.putImageBytes(upscaledImageBytes, name: fileName);
        }
        setState(() {
          _outputImage = Image.memory(upscaledImageBytes);
          _isProcessing = false;
          _status = 'Upscaling complete';
        });
      },
      onError: (error) {
        setState(() {
          _isProcessing = false;
          _showTemporaryError('Error: $error');
        });
      },
    );
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      _originalFileName = pickedFile.path.split('/').last;
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
        _outputImage = null;
        _processedInputBytes = rgbBytes;
        _imageWidth = decodedImage.width;
        _imageHeight = decodedImage.height;
      });
    }
  }

  Future<void> _upscaleImage() async {
    if (_inputImage == null || _processedInputBytes == null) {
      _showTemporaryError('Please select an image first');
      return;
    }

    if (_processor == null) {
      _showTemporaryError('Please load an upscaler model first');
      return;
    }

    setState(() {
      _isProcessing = true;
      _status = 'Processing...';
    });

    try {
      _processor!.upscaleImage(
        inputData: _processedInputBytes!,
        width: _imageWidth!,
        height: _imageHeight!,
        channel: 3,
        upscaleFactor: 4,
      );
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _showTemporaryError('Error processing image: $e');
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = ShadTheme.of(context);
    return WillPopScope(
      onWillPop: () async => !_isProcessing,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Image Upscaler',
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
                    MaterialPageRoute(
                        builder: (context) => const Img2ImgPage()),
                  );
                },
              ),
              ListTile(
                leading: const Icon(LucideIcons.imageUpscale, size: 32),
                title: const Text(
                  'Upscaler',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                onTap: () => Navigator.pop(context),
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
                    MaterialPageRoute(
                        builder: (context) => const ScribblePage()),
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
            ],
          ),
        ),
        body: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              if (loadedComponents.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.only(bottom: 16),
                  child: Column(
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
                                    duration: const Duration(milliseconds: 500))
                                .slideY(begin: -0.2, end: 0))
                        .toList(),
                  ),
                ),
              const SizedBox(height: 16),
              GestureDetector(
                onTap: _isProcessing ? null : _pickImage,
                child: DottedBorder(
                  borderType: BorderType.RRect,
                  radius: const Radius.circular(8),
                  color: theme.colorScheme.primary.withOpacity(0.5),
                  strokeWidth: 2,
                  dashPattern: const [8, 4],
                  child: Center(
                    // Added Center widget here
                    child: Container(
                      height: 300,
                      width: double
                          .infinity, // Changed from fixed width to infinity
                      margin: const EdgeInsets.symmetric(horizontal: 32),
                      child: _inputImage == null
                          ? Center(
                              // Added Center widget for the column
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                crossAxisAlignment:
                                    CrossAxisAlignment.center, // Added this
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
                                    textAlign: TextAlign
                                        .center, // Added text alignment
                                    style: TextStyle(
                                      color: theme.colorScheme.primary
                                          .withOpacity(0.5),
                                      fontSize: 16,
                                    ),
                                  ),
                                ],
                              ),
                            )
                          : ClipRRect(
                              borderRadius: BorderRadius.circular(8),
                              child: Image.file(
                                _inputImage!,
                                fit: BoxFit.cover,
                              ),
                            ),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Row(
                children: [
                  Expanded(
                    child: ShadButton(
                      enabled: !_isProcessing,
                      onPressed: _initializeProcessor,
                      child: const Text('Load Model'),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: ShadButton(
                      enabled: !_isProcessing,
                      onPressed: _upscaleImage,
                      child: const Text('Upscale'),
                    ),
                  ),
                ],
              ),
              if (_status.isNotEmpty) ...[
                const SizedBox(height: 8),
                Text(
                  _status,
                  style: _isError
                      ? theme.textTheme.p.copyWith(
                          color: Colors.red,
                          fontWeight: FontWeight.bold,
                        )
                      : theme.textTheme.p.copyWith(
                          color: theme.colorScheme.primary,
                        ),
                ),
              ],
              if (_outputImage != null) ...[
                const SizedBox(height: 16),
                Container(
                  height: 500,
                  child: BeforeAfter(
                    value: _sliderValue,
                    before: Image.file(
                      _inputImage!,
                      fit: BoxFit.contain,
                      width:
                          _imageWidth! * 4.0, // Scale up to match upscaled size
                      height: _imageHeight! * 4.0,
                      filterQuality:
                          FilterQuality.none, // Shows original pixels clearly
                    ),
                    after: _outputImage!,
                    onValueChanged: (value) {
                      setState(() => _sliderValue = value);
                    },
                  ),
                ),
                const SizedBox(height: 8),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
