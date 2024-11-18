import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:ui' as ui;
import 'dart:async';
import 'dart:developer' as developer;
import 'ffi_bindings.dart';
import 'sd_image.dart';

class BufferPool {
  static final Map<int, Uint8List> _rgbaBuffers = {};
  static final Map<String, Pointer<Utf8>> _stringBuffers = {};

  static Uint8List getRgbaBuffer(int size) {
    return _rgbaBuffers.putIfAbsent(size, () => Uint8List(size));
  }

  static Pointer<Utf8> getStringBuffer(String key, String value) {
    if (!_stringBuffers.containsKey(key)) {
      _stringBuffers[key] = value.toNativeUtf8();
    }
    return _stringBuffers[key]!;
  }

  static void dispose() {
    _rgbaBuffers.clear();
    for (var ptr in _stringBuffers.values) {
      calloc.free(ptr);
    }
    _stringBuffers.clear();
  }
}

class StableDiffusionService {
  static String? modelPath;
  static String? loraPath;
  static Pointer<Void>? _ctx;
  static final _progressController =
      StreamController<ProgressUpdate>.broadcast();
  static final _logController = StreamController<LogMessage>.broadcast();
  static bool _isInitialized = false;
  static bool _useFlashAttention = false;
  static late int _numCores;

  static Stream<ProgressUpdate> get progressStream =>
      _progressController.stream;
  static Stream<LogMessage> get logStream => _logController.stream;

  static void setFlashAttention(bool value) {
    _useFlashAttention = value;
    if (_ctx != null) {
      initializeModel();
    }
  }

  static void _logCallback(int level, Pointer<Utf8> text, Pointer<Void> data) {
    final message = text.toDartString();
    _logController.add(LogMessage(level, message));
    developer.log(message);
  }

  static void _progressCallback(
      int step, int steps, double time, Pointer<Void> data) {
    final update = ProgressUpdate(step, steps, time);
    _progressController.add(update);
    developer.log(
        'Progress: ${(update.progress * 100).toInt()}% (Step $step/$steps, ${time.toStringAsFixed(1)}s)');
  }

  static void _initializeOnce() {
    if (!_isInitialized) {
      _numCores = getCores();
      developer.log("Initializing with $_numCores cores for computation");

      final logCallbackPointer =
          Pointer.fromFunction<LogCallbackNative>(_logCallback);
      final progressCallbackPointer =
          Pointer.fromFunction<ProgressCallbackNative>(_progressCallback);

      FFIBindings.setLogCallback(logCallbackPointer, nullptr);
      FFIBindings.setProgressCallback(progressCallbackPointer, nullptr);
      _isInitialized = true;
    }
  }

  static Future<String> pickAndInitializeModel() async {
    try {
      final result = await FilePicker.platform
          .pickFiles(type: FileType.any, allowMultiple: false);

      if (result == null) return "No file selected";

      modelPath = result.files.single.path!;
      final filename = result.files.single.name;

      if (!filename.endsWith('.ckpt') && !filename.endsWith('.safetensors')) {
        return "Please select a .ckpt or .safetensors file";
      }

      return initializeModel();
    } catch (e) {
      return "Error picking file: $e";
    }
  }

  static Future<String> pickAndInitializeLora() async {
    try {
      final result = await FilePicker.platform
          .pickFiles(type: FileType.any, allowMultiple: false);

      if (result == null) return "No LORA file selected";

      loraPath = result.files.single.path!;
      final filename = result.files.single.name;

      if (!filename.endsWith('.safetensors')) {
        return "Please select a .safetensors file for LORA";
      }

      return "LORA model selected: ${loraPath!.split('/').last}";
    } catch (e) {
      return "Error picking LORA file: $e";
    }
  }

  static String initializeModel() {
    if (modelPath == null || modelPath!.isEmpty) {
      return "Model path not set";
    }

    _initializeOnce();

    final modelPathPtr = modelPath!.toNativeUtf8();
    final emptyPtr = BufferPool.getStringBuffer('empty', "");
    final loraDirPtr =
        "/data/user/0/com.example.sd_test_app/cache/file_picker".toNativeUtf8();

    try {
      developer.log(
          "LORA directory set to: /data/user/0/com.example.sd_test_app/cache/file_picker");
      developer.log(
          "Flash Attention is ${_useFlashAttention ? 'enabled' : 'disabled'}");

      _ctx = FFIBindings.newSdCtx(
          modelPathPtr,
          emptyPtr,
          emptyPtr,
          emptyPtr,
          emptyPtr,
          emptyPtr,
          emptyPtr,
          emptyPtr,
          loraDirPtr,
          emptyPtr,
          emptyPtr,
          false,
          false,
          false,
          _numCores,
          _useFlashAttention
              ? SDType.SD_TYPE_F16.index
              : SDType.SD_TYPE_F32.index,
          0,
          0,
          false,
          false,
          false);

      if (_ctx!.address == 0) {
        return "Failed to initialize model";
      }
      return "Model initialized successfully: ${modelPath!.split('/').last}";
    } finally {
      calloc.free(modelPathPtr);
      calloc.free(loraDirPtr);
    }
  }

  static Future<ui.Image?> generateImage({
    required String prompt,
    String negativePrompt = "",
    int clipSkip = 1,
    double cfgScale = 7.0,
    double guidance = 1.0,
    int width = 512,
    int height = 512,
    int sampleMethod = 0,
    int sampleSteps = 20,
    int seed = 42,
    int batchCount = 1,
  }) async {
    if (_ctx == null) return null;

    final promptPtr = prompt.toNativeUtf8();
    final negPromptPtr = negativePrompt.toNativeUtf8();
    final emptyPtr = BufferPool.getStringBuffer('empty', "");

    try {
      final result = FFIBindings.txt2img(
          _ctx!,
          promptPtr,
          negPromptPtr,
          clipSkip,
          cfgScale,
          guidance,
          width,
          height,
          sampleMethod,
          sampleSteps,
          seed,
          batchCount,
          nullptr,
          1.0,
          1.0,
          false,
          emptyPtr);

      if (result.address == 0) return null;

      final image = result.cast<SDImage>().ref;
      final bytes = image.data.asTypedList(width * height * image.channel);

      final rgbaBytes = BufferPool.getRgbaBuffer(width * height * 4);
      for (var i = 0; i < width * height; i++) {
        rgbaBytes[i * 4] = bytes[i * 3];
        rgbaBytes[i * 4 + 1] = bytes[i * 3 + 1];
        rgbaBytes[i * 4 + 2] = bytes[i * 3 + 2];
        rgbaBytes[i * 4 + 3] = 255;
      }

      final completer = Completer<ui.Image>();
      ui.decodeImageFromPixels(
        rgbaBytes,
        width,
        height,
        ui.PixelFormat.rgba8888,
        completer.complete,
      );

      return completer.future;
    } finally {
      calloc.free(promptPtr);
      calloc.free(negPromptPtr);
    }
  }

  static int getCores() => FFIBindings.getCores();

  static void dispose() {
    BufferPool.dispose();
    _progressController.close();
    _logController.close();
    _isInitialized = false;
  }
}

class ProgressUpdate {
  final int step;
  final int totalSteps;
  final double time;
  final double progress;

  ProgressUpdate(this.step, this.totalSteps, this.time)
      : progress = step / totalSteps;
}

class LogMessage {
  final int level;
  final String message;

  LogMessage(this.level, this.message);
}
