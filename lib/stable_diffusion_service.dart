import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:ui' as ui;
import 'dart:async';
import 'dart:developer' as developer;
import 'dart:isolate';
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

class IsolateMessage {
  final String type;
  final dynamic data;

  IsolateMessage(this.type, this.data);
}

class IsolateData {
  final SendPort sendPort;
  final String modelPath;
  final String? loraPath;
  final String? taesdPath;
  final bool useFlashAttention;
  final bool useTinyAutoencoder;
  final SDType modelType;
  final Schedule schedule;
  final int numCores;

  IsolateData({
    required this.sendPort,
    required this.modelPath,
    this.loraPath,
    this.taesdPath,
    required this.useFlashAttention,
    required this.useTinyAutoencoder,
    required this.modelType,
    required this.schedule,
    required this.numCores,
  });
}

class GenerationData {
  final SendPort sendPort;
  final Pointer<Void> ctx;
  final String prompt;
  final String negativePrompt;
  final int clipSkip;
  final double cfgScale;
  final double guidance;
  final int width;
  final int height;
  final int sampleMethod;
  final int sampleSteps;
  final int seed;
  final int batchCount;

  GenerationData({
    required this.sendPort,
    required this.ctx,
    required this.prompt,
    required this.negativePrompt,
    required this.clipSkip,
    required this.cfgScale,
    required this.guidance,
    required this.width,
    required this.height,
    required this.sampleMethod,
    required this.sampleSteps,
    required this.seed,
    required this.batchCount,
  });
}

class StableDiffusionService {
  static String? modelPath;
  static String? loraPath;
  static String? taesdPath;
  static Pointer<Void>? _ctx;
  static final _progressController =
      StreamController<ProgressUpdate>.broadcast();
  static final _logController = StreamController<LogMessage>.broadcast();
  static bool _useFlashAttention = false;
  static bool _useTinyAutoencoder = false;
  static SDType _modelType = SDType.NONE;
  static Schedule _schedule = Schedule.DISCRETE;
  static late int _numCores;
  static SendPort? _currentSendPort;
  static final _loadingController = StreamController<bool>.broadcast();
  static Stream<bool> get loadingStream => _loadingController.stream;

  static Stream<ProgressUpdate> get progressStream =>
      _progressController.stream;
  static Stream<LogMessage> get logStream => _logController.stream;

  static void setModelConfig(
      bool useFlashAttention, SDType modelType, Schedule schedule) {
    developer.log(
        "Setting config - Flash: $useFlashAttention, Type: $modelType, Schedule: $schedule");
    _useFlashAttention = useFlashAttention;
    _modelType = modelType;
    _schedule = schedule;
  }

  static bool setTinyAutoencoder(bool value) {
    if (value && taesdPath == null) {
      return false;
    }
    if (_ctx != null) {
      freeCurrentModel();
    }
    _useTinyAutoencoder = value;
    return true;
  }

  static void _handleIsolateMessage(IsolateMessage message) {
    switch (message.type) {
      case 'log':
        final logData = message.data as (int, String);
        _logController.add(LogMessage(logData.$1, logData.$2));
        developer.log(logData.$2);
        break;
      case 'progress':
        final progressData = message.data as (int, int, double);
        final update =
            ProgressUpdate(progressData.$1, progressData.$2, progressData.$3);
        _progressController.add(update);
        developer.log(
            'Progress: ${(update.progress * 100).toInt()}% (Step ${progressData.$1}/${progressData.$2}, ${progressData.$3.toStringAsFixed(1)}s)');
        break;
    }
  }

  static void _isolateLogCallback(
      int level, Pointer<Utf8> text, Pointer<Void> data, SendPort sendPort) {
    final message = text.toDartString();
    sendPort.send(IsolateMessage('log', (level, message)));
  }

  static void _isolateProgressCallback(
      int step, int steps, double time, Pointer<Void> data, SendPort sendPort) {
    sendPort.send(IsolateMessage('progress', (step, steps, time)));
  }

  static void _staticLogCallback(
      int level, Pointer<Utf8> text, Pointer<Void> ptr) {
    if (_currentSendPort != null) {
      _isolateLogCallback(level, text, ptr, _currentSendPort!);
    }
  }

  static void _staticProgressCallback(
      int step, int steps, double time, Pointer<Void> ptr) {
    if (_currentSendPort != null) {
      _isolateProgressCallback(step, steps, time, ptr, _currentSendPort!);
    }
  }

  static Future<String> pickAndInitializeModel() async {
    try {
      final result = await FilePicker.platform
          .pickFiles(type: FileType.any, allowMultiple: false);

      if (result == null) return "No file selected";

      modelPath = result.files.single.path!;
      final filename = result.files.single.name;

      if (!filename.endsWith('.ckpt') &&
          !filename.endsWith('.safetensors') &&
          !filename.endsWith('.gguf')) {
        return "Please select a .ckpt or .safetensors or .gguf file";
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

  static Future<String> pickAndInitializeTAESD() async {
    try {
      final result = await FilePicker.platform
          .pickFiles(type: FileType.any, allowMultiple: false);

      if (result == null) return "No TAESD file selected";

      final filename = result.files.single.name;
      if (!filename.endsWith('.safetensors')) {
        return "Please select a .safetensors file for TAESD";
      }

      taesdPath = result.files.single.path!;
      if (modelPath != null && _useTinyAutoencoder) {
        initializeModel();
      }

      return "TAESD model selected: ${taesdPath!.split('/').last}";
    } catch (e) {
      return "Error picking TAESD file: $e";
    }
  }

  static bool isModelLoaded() {
    return _ctx != null;
  }

  static void freeCurrentModel() {
    if (_ctx != null) {
      FFIBindings.freeSdCtx(_ctx!);
      _ctx = null;
    }
  }

  static Future<String> initializeModel() async {
    if (modelPath == null || modelPath!.isEmpty) {
      return "Model path not set";
    }

    _loadingController.add(true);
    _numCores = getCores();
    developer.log("Initializing with $_numCores cores for computation");

    if (_ctx != null) {
      freeCurrentModel();
    }

    final completer = Completer<String>();
    final receivePort = ReceivePort();

    final isolateData = IsolateData(
      sendPort: receivePort.sendPort,
      modelPath: modelPath!,
      loraPath: loraPath,
      taesdPath: taesdPath,
      useFlashAttention: _useFlashAttention,
      useTinyAutoencoder: _useTinyAutoencoder,
      modelType: _modelType,
      schedule: _schedule,
      numCores: _numCores,
    );

    receivePort.listen((message) {
      if (message is IsolateMessage) {
        _handleIsolateMessage(message);
      } else if (message is Pointer<Void>) {
        _ctx = message;
        completer.complete(
            "Model initialized successfully: ${modelPath!.split('/').last}");
      } else if (message is String) {
        completer.complete(message);
      }
    });

    await Isolate.spawn(_initializeModelIsolate, isolateData);

    final result = await completer.future;
    _loadingController.add(false);
    return result;
  }

  static void _initializeModelIsolate(IsolateData data) {
    _currentSendPort = data.sendPort;

    final logCallbackPointer =
        Pointer.fromFunction<LogCallbackNative>(_staticLogCallback);
    final progressCallbackPointer =
        Pointer.fromFunction<ProgressCallbackNative>(_staticProgressCallback);

    FFIBindings.setLogCallback(logCallbackPointer, nullptr);
    FFIBindings.setProgressCallback(progressCallbackPointer, nullptr);

    int mappedTypeIndex;
    switch (data.modelType) {
      case SDType.NONE:
        mappedTypeIndex = 34;
        break;
      case SDType.SD_TYPE_Q8_0:
        mappedTypeIndex = 8;
        break;
      case SDType.SD_TYPE_Q8_1:
        mappedTypeIndex = 9;
        break;
      case SDType.SD_TYPE_Q8_K:
        mappedTypeIndex = 15;
        break;
      case SDType.SD_TYPE_Q6_K:
        mappedTypeIndex = 14;
        break;
      case SDType.SD_TYPE_Q5_0:
        mappedTypeIndex = 6;
        break;
      case SDType.SD_TYPE_Q5_1:
        mappedTypeIndex = 7;
        break;
      case SDType.SD_TYPE_Q5_K:
        mappedTypeIndex = 13;
        break;
      case SDType.SD_TYPE_Q4_0:
        mappedTypeIndex = 2;
        break;
      case SDType.SD_TYPE_Q4_1:
        mappedTypeIndex = 3;
        break;
      case SDType.SD_TYPE_Q4_K:
        mappedTypeIndex = 12;
        break;
      case SDType.SD_TYPE_Q3_K:
        mappedTypeIndex = 11;
        break;
      case SDType.SD_TYPE_Q2_K:
        mappedTypeIndex = 10;
        break;
      default:
        mappedTypeIndex = data.modelType.index;
    }

    final modelPathPtr = data.modelPath.toNativeUtf8();
    final emptyPtr = "".toNativeUtf8();
    final loraDirPtr =
        "/data/user/0/com.example.sd_test_app/cache/file_picker".toNativeUtf8();
    final taesdPathPtr = (data.useTinyAutoencoder && data.taesdPath != null)
        ? data.taesdPath!.toNativeUtf8()
        : emptyPtr;

    try {
      final ctx = FFIBindings.newSdCtx(
          modelPathPtr,
          emptyPtr,
          emptyPtr,
          emptyPtr,
          emptyPtr,
          emptyPtr,
          taesdPathPtr,
          emptyPtr,
          loraDirPtr,
          emptyPtr,
          emptyPtr,
          data.useFlashAttention,
          false,
          false,
          data.numCores,
          mappedTypeIndex,
          0,
          data.schedule.index,
          false,
          false,
          false);

      data.sendPort.send(ctx);
    } catch (e) {
      data.sendPort.send("Failed to initialize model: $e");
    } finally {
      calloc.free(modelPathPtr);
      calloc.free(loraDirPtr);
      if (data.useTinyAutoencoder &&
          data.taesdPath != null &&
          taesdPathPtr != emptyPtr) {
        calloc.free(taesdPathPtr);
      }
      calloc.free(emptyPtr);
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

    _loadingController.add(true);
    final completer = Completer<ui.Image?>();
    final receivePort = ReceivePort();
    Isolate? isolate;

    final generationData = GenerationData(
      sendPort: receivePort.sendPort,
      ctx: _ctx!,
      prompt: prompt,
      negativePrompt: negativePrompt,
      clipSkip: clipSkip,
      cfgScale: cfgScale,
      guidance: guidance,
      width: width,
      height: height,
      sampleMethod: sampleMethod,
      sampleSteps: sampleSteps,
      seed: seed,
      batchCount: batchCount,
    );

    receivePort.listen((message) async {
      if (message is IsolateMessage) {
        _handleIsolateMessage(message);
      } else if (message is Uint8List) {
        final completer2 = Completer<ui.Image>();
        ui.decodeImageFromPixels(
          message,
          width,
          height,
          ui.PixelFormat.rgba8888,
          completer2.complete,
        );
        final image = await completer2.future;
        completer.complete(image);
      } else {
        completer.complete(null);
      }
    });

    isolate = await Isolate.spawn(_generateImageIsolate, generationData);

    try {
      final result = await completer.future;
      isolate.kill();
      receivePort.close();
      return result;
    } finally {
      _loadingController.add(false);
    }
  }

  static void _generateImageIsolate(GenerationData data) {
    _currentSendPort = data.sendPort;

    final logCallbackPointer =
        Pointer.fromFunction<LogCallbackNative>(_staticLogCallback);
    final progressCallbackPointer =
        Pointer.fromFunction<ProgressCallbackNative>(_staticProgressCallback);

    FFIBindings.setLogCallback(logCallbackPointer, nullptr);
    FFIBindings.setProgressCallback(progressCallbackPointer, nullptr);

    final promptPtr = data.prompt.toNativeUtf8();
    final negPromptPtr = data.negativePrompt.toNativeUtf8();
    final emptyPtr = "".toNativeUtf8();

    try {
      final result = FFIBindings.txt2img(
          data.ctx,
          promptPtr,
          negPromptPtr,
          data.clipSkip,
          data.cfgScale,
          data.guidance,
          data.width,
          data.height,
          data.sampleMethod,
          data.sampleSteps,
          data.seed,
          data.batchCount,
          nullptr,
          1.0,
          1.0,
          false,
          emptyPtr);

      if (result.address == 0) {
        data.sendPort.send(null);
        return;
      }

      final image = result.cast<SDImage>().ref;
      final bytes =
          image.data.asTypedList(data.width * data.height * image.channel);

      final rgbaBytes = Uint8List(data.width * data.height * 4);
      for (var i = 0; i < data.width * data.height; i++) {
        rgbaBytes[i * 4] = bytes[i * 3];
        rgbaBytes[i * 4 + 1] = bytes[i * 3 + 1];
        rgbaBytes[i * 4 + 2] = bytes[i * 3 + 2];
        rgbaBytes[i * 4 + 3] = 255;
      }

      data.sendPort.send(rgbaBytes);
    } finally {
      calloc.free(promptPtr);
      calloc.free(negPromptPtr);
      calloc.free(emptyPtr);
    }
  }

  static int getCores() => FFIBindings.getCores();

  static void dispose() {
    if (_ctx != null) {
      FFIBindings.freeSdCtx(_ctx!);
      _ctx = null;
    }
    BufferPool.dispose();
    _progressController.close();
    _logController.close();
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
