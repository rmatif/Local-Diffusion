import 'dart:async';
import 'dart:isolate';
import 'dart:ui' as ui;
import 'dart:typed_data';
import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:gal/gal.dart';
import 'utils.dart';
import 'ffi_bindings.dart';
import 'stable_diffusion_service.dart';
import 'sd_image.dart';

SendPort? _globalSendPort;

late final Pointer<NativeFunction<LogCallbackNative>> _logCallbackPtr;
late final Pointer<NativeFunction<ProgressCallbackNative>> _progressCallbackPtr;

void _staticLogCallback(int level, Pointer<Utf8> text, Pointer<Void> data) {
  final message = text.toDartString();

  if (message.contains("generating image")) {
    final seedMatch = RegExp(r'seed (\d+)').firstMatch(message);
    if (seedMatch != null) {
      final extractedSeed = int.parse(seedMatch.group(1)!);
      _globalSendPort?.send({
        'type': 'log',
        'level': level,
        'message': message,
        'seed': extractedSeed
      });
      return;
    }
  }

  _globalSendPort?.send({
    'type': 'log',
    'level': level,
    'message': message,
  });
}

void _staticProgressCallback(
    int step, int steps, double time, Pointer<Void> data) {
  print("SD Progress: $step/$steps - ${time}s");
  _globalSendPort?.send({
    'type': 'progress',
    'step': step,
    'steps': steps,
    'time': time,
  });
}

class StableDiffusionProcessor {
  final String modelPath;
  final bool useFlashAttention;
  final SDType modelType;
  final Schedule schedule;
  final String? loraPath;
  final String? taesdPath;
  final bool useTinyAutoencoder;
  final void Function()? onModelLoaded;
  final void Function(LogMessage)? onLog;
  final void Function(ProgressUpdate)? onProgress;
  final String? clipLPath;
  final String? clipGPath;
  final String? t5xxlPath;
  final String? vaePath;
  final String? embedDirPath;
  final String? stackedIdEmbedDir;
  final int clipSkip;
  final bool vaeTiling;
  final String? controlNetPath; // Add this
  final Uint8List? controlImageData; // Add this
  final int? controlImageWidth; // Add this
  final int? controlImageHeight; // Add this // Add this
  final double controlStrength;
  late Isolate _sdIsolate;
  late SendPort _sdSendPort;
  final Completer _uninitialized = Completer();
  final ReceivePort _receivePort = ReceivePort();
  final StreamController<ui.Image> _imageController =
      StreamController<ui.Image>.broadcast();
  final _loadingController = StreamController<bool>.broadcast();

  Stream<bool> get loadingStream => _loadingController.stream;
  Stream<ui.Image> get imageStream => _imageController.stream;

  StableDiffusionProcessor({
    required this.modelPath,
    required this.useFlashAttention,
    required this.modelType,
    required this.schedule,
    this.loraPath,
    this.taesdPath,
    this.useTinyAutoencoder = false,
    this.onModelLoaded,
    this.onLog,
    this.onProgress,
    this.clipLPath,
    this.clipGPath,
    this.t5xxlPath,
    this.vaePath,
    this.embedDirPath,
    this.stackedIdEmbedDir,
    this.clipSkip = 1,
    this.vaeTiling = false,
    this.controlNetPath, // Add this
    this.controlImageData, // Add this
    this.controlImageWidth, // Add this
    this.controlImageHeight, // Add this
    this.controlStrength = 0.9, // Add this
  }) {
    _initializeIsolate();
  }

  static int mapModelTypeToIndex(SDType modelType) {
    switch (modelType) {
      case SDType.NONE:
        return 39;
      case SDType.SD_TYPE_Q8_0:
        return 8;
      case SDType.SD_TYPE_Q8_1:
        return 9;
      case SDType.SD_TYPE_Q8_K:
        return 15;
      case SDType.SD_TYPE_Q6_K:
        return 14;
      case SDType.SD_TYPE_Q5_0:
        return 6;
      case SDType.SD_TYPE_Q5_1:
        return 7;
      case SDType.SD_TYPE_Q5_K:
        return 13;
      case SDType.SD_TYPE_Q4_0:
        return 2;
      case SDType.SD_TYPE_Q4_1:
        return 3;
      case SDType.SD_TYPE_Q4_K:
        return 12;
      case SDType.SD_TYPE_Q3_K:
        return 11;
      case SDType.SD_TYPE_Q2_K:
        return 10;
      default:
        return modelType.index;
    }
  }

  Future<void> _initializeIsolate() async {
    _loadingController.add(true);
    try {
      _sdIsolate = await Isolate.spawn(
        _isolateEntryPoint,
        {'port': _receivePort.sendPort},
      );

      _receivePort.listen((message) async {
        if (message is SendPort) {
          _sdSendPort = message;
          _sdSendPort.send({
            'command': 'initialize',
            'modelPath': modelPath,
            'loraPath': loraPath,
            'taesdPath': taesdPath,
            'useTinyAutoencoder': useTinyAutoencoder,
            'useFlashAttention': useFlashAttention,
            'modelType': modelType.index,
            'schedule': schedule.index,
            'clipLPath': clipLPath,
            'clipGPath': clipGPath,
            't5xxlPath': t5xxlPath,
            'vaePath': vaePath,
            'embedDirPath': embedDirPath,
            'stackedIdEmbedDir': stackedIdEmbedDir,
            'clipSkip': clipSkip,
            'vaeTiling': vaeTiling,
            'controlNetPath': controlNetPath, // Add this
          });
        } else if (message is Map) {
          if (message['type'] == 'modelLoaded') {
            _uninitialized.complete();
            if (onModelLoaded != null) onModelLoaded!();
          } else if (message['type'] == 'log') {
            if (message.containsKey('seed')) {
              StableDiffusionService.lastUsedSeed = message['seed'];
            }
            if (onLog != null) {
              onLog!(LogMessage(message['level'], message['message']));
            }
          } else if (message['type'] == 'progress') {
            if (onProgress != null) {
              onProgress!(ProgressUpdate(
                message['step'],
                message['steps'],
                message['time'],
              ));
            }
          } else if (message['type'] == 'image') {
            final completer = Completer<ui.Image>();
            ui.decodeImageFromPixels(
              message['bytes'],
              message['width'],
              message['height'],
              ui.PixelFormat.rgba8888,
              completer.complete,
            );
            final image = await completer.future;
            _imageController.add(image);
          }
        }
      });
    } finally {
      _loadingController.add(false);
    }
  }

  static void _isolateEntryPoint(Map<String, dynamic> args) {
    final SendPort mainSendPort = args['port'];
    _globalSendPort = mainSendPort;
    _logCallbackPtr =
        Pointer.fromFunction<LogCallbackNative>(_staticLogCallback);
    _progressCallbackPtr =
        Pointer.fromFunction<ProgressCallbackNative>(_staticProgressCallback);
    final ReceivePort isolateReceivePort = ReceivePort();
    mainSendPort.send(isolateReceivePort.sendPort);

    Pointer<Void>? ctx;
    print("Isolate started");

    isolateReceivePort.listen((message) {
      print("Received message in isolate: ${message['command']}");
      if (message is Map) {
        switch (message['command']) {
          case 'initialize':
            print("Initializing SD model...");
            FFIBindings.setLogCallback(_logCallbackPtr, nullptr);
            FFIBindings.setProgressCallback(_progressCallbackPtr, nullptr);

            try {
              final modelPathUtf8 =
                  message['modelPath'].toString().toNativeUtf8();
              final clipLPathUtf8 = message['clipLPath'] != null &&
                      message['clipLPath'].toString().isNotEmpty
                  ? message['clipLPath'].toString().toNativeUtf8()
                  : "".toNativeUtf8();
              final clipGPathUtf8 = message['clipGPath'] != null &&
                      message['clipGPath'].toString().isNotEmpty
                  ? message['clipGPath'].toString().toNativeUtf8()
                  : "".toNativeUtf8();
              final t5xxlPathUtf8 = message['t5xxlPath'] != null &&
                      message['t5xxlPath'].toString().isNotEmpty
                  ? message['t5xxlPath'].toString().toNativeUtf8()
                  : "".toNativeUtf8();
              final vaePathUtf8 = message['vaePath'] != null &&
                      message['vaePath'].toString().isNotEmpty
                  ? message['vaePath'].toString().toNativeUtf8()
                  : "".toNativeUtf8();
              final emptyUtf8 = "".toNativeUtf8();
              final loraDirUtf8 = message['loraPath'] != null &&
                      message['loraPath'].toString().isNotEmpty
                  ? message['loraPath'].toString().toNativeUtf8()
                  : "/"
                      .toNativeUtf8(); // Provide a valid default path instead of empty string
              final taesdPathUtf8 = (message['useTinyAutoencoder'] &&
                      message['taesdPath'] != null)
                  ? message['taesdPath'].toString().toNativeUtf8()
                  : emptyUtf8;
              final stackedIdEmbedDirUtf8 =
                  message['stackedIdEmbedDir'] != null &&
                          message['stackedIdEmbedDir'].toString().isNotEmpty
                      ? message['stackedIdEmbedDir'].toString().toNativeUtf8()
                      : "".toNativeUtf8();
              final embedDirUtf8 = message['embedDirPath'] != null &&
                      message['embedDirPath'].toString().isNotEmpty
                  ? message['embedDirPath'].toString().toNativeUtf8()
                  : "".toNativeUtf8();
              final controlNetPathUtf8 = message['controlNetPath'] != null &&
                      message['controlNetPath'].toString().isNotEmpty
                  ? message['controlNetPath'].toString().toNativeUtf8()
                  : "".toNativeUtf8();

              FFIBindings.setLogCallback(_logCallbackPtr, nullptr);
              FFIBindings.setProgressCallback(_progressCallbackPtr, nullptr);

              ctx = FFIBindings.newSdCtx(
                modelPathUtf8,
                clipLPathUtf8,
                clipGPathUtf8,
                t5xxlPathUtf8,
                emptyUtf8,
                vaePathUtf8,
                taesdPathUtf8,
                controlNetPathUtf8,
                loraDirUtf8,
                embedDirUtf8,
                stackedIdEmbedDirUtf8,
                message['useFlashAttention'],
                message['vaeTiling'],
                false,
                FFIBindings.getCores() * 2,
                mapModelTypeToIndex(SDType.values[message['modelType']]),
                0,
                message['schedule'],
                false,
                false,
                false,
                message['clipSkip'],
                message[
                    'useFlashAttention'], // Added diffusion_flash_attn parameter (using same value as useFlashAttention)
              );

              calloc.free(modelPathUtf8);
              calloc.free(loraDirUtf8);
              calloc.free(clipLPathUtf8);
              calloc.free(clipGPathUtf8);
              calloc.free(t5xxlPathUtf8);
              calloc.free(vaePathUtf8);
              calloc.free(embedDirUtf8);
              calloc.free(controlNetPathUtf8); // Add this
              if (stackedIdEmbedDirUtf8 != emptyUtf8) {
                calloc.free(stackedIdEmbedDirUtf8);
              }
              if (message['useTinyAutoencoder'] &&
                  message['taesdPath'] != null &&
                  taesdPathUtf8 != emptyUtf8) {
                calloc.free(taesdPathUtf8);
              }
              calloc.free(emptyUtf8);

              if (ctx != null && ctx!.address != 0) {
                print("Model initialized successfully");
                mainSendPort.send({'type': 'modelLoaded'});
              } else {
                print("Failed to initialize model");
                mainSendPort.send(
                    {'type': 'error', 'message': 'Failed to initialize model'});
              }
            } catch (e) {
              print("Error initializing model: $e");
              mainSendPort.send({'type': 'error', 'message': e.toString()});
            }
            break;

          case 'generate':
            if (ctx != null) {
              print("Starting image generation with context: ${ctx!.address}");
              try {
                final promptUtf8 = message['prompt'].toString().toNativeUtf8();
                final negPromptUtf8 =
                    message['negativePrompt'].toString().toNativeUtf8();
                final inputIdImagesPathUtf8 =
                    message['inputIdImagesPath'].toString().toNativeUtf8();
                final styleStrength = message['styleStrength'];
                final emptyUtf8 = "".toNativeUtf8();

                Pointer<SDImage>? controlCondPtr;
                if (message['controlImageData'] != null) {
                  final controlImageData =
                      message['controlImageData'] as Uint8List;
                  final controlWidth = message['controlImageWidth'] as int;
                  final controlHeight = message['controlImageHeight'] as int;
                  final controlDataPtr = malloc<Uint8>(controlImageData.length);
                  controlDataPtr
                      .asTypedList(controlImageData.length)
                      .setAll(0, controlImageData);

                  // In the _isolateEntryPoint function, modify the code for Canny processing:

                  controlCondPtr = malloc<SDImage>();
                  controlCondPtr.ref
                    ..width = controlWidth
                    ..height = controlHeight
                    ..channel = 3
                    ..data = controlDataPtr;
                }

                final result = FFIBindings.txt2img(
                  ctx!,
                  promptUtf8,
                  negPromptUtf8,
                  message['clipSkip'],
                  message['cfgScale'],
                  message['guidance'],
                  0.0, // eta (new parameter)
                  message['width'],
                  message['height'],
                  message['sampleMethod'],
                  message['sampleSteps'],
                  message['seed'],
                  message['batchCount'],
                  controlCondPtr ?? nullptr,
                  message['controlStrength'] ?? 0.0,
                  styleStrength,
                  false,
                  inputIdImagesPathUtf8,
                  nullptr, // skip_layers
                  0, // skip_layers_count
                  0.0, // slg_scale
                  0.0, // skip_layer_start
                  0.0, // skip_layer_end
                );

                if (controlCondPtr != null) {
                  calloc.free(controlCondPtr.ref.data);

                  malloc.free(controlCondPtr);
                }

                calloc.free(promptUtf8);
                calloc.free(negPromptUtf8);
                calloc.free(inputIdImagesPathUtf8);
                calloc.free(emptyUtf8);

                print("Generation result address: ${result.address}");

                if (result.address != 0) {
                  final image = result.cast<SDImage>().ref;
                  final bytes = image.data.asTypedList(
                      message['width'] * message['height'] * image.channel);
                  final rgbaBytes =
                      Uint8List(message['width'] * message['height'] * 4);

                  for (var i = 0;
                      i < message['width'] * message['height'];
                      i++) {
                    rgbaBytes[i * 4] = bytes[i * 3];
                    rgbaBytes[i * 4 + 1] = bytes[i * 3 + 1];
                    rgbaBytes[i * 4 + 2] = bytes[i * 3 + 2];
                    rgbaBytes[i * 4 + 3] = 255;
                  }

                  mainSendPort.send({
                    'type': 'image',
                    'bytes': rgbaBytes,
                    'width': message['width'],
                    'height': message['height'],
                  });

                  calloc.free(image.data);
                  calloc.free(result.cast<Void>());
                }
              } catch (e) {
                print("Error generating image: $e");
                mainSendPort.send({'type': 'error', 'message': e.toString()});
              }
            } else {
              print("Context is null, cannot generate image");
              mainSendPort
                  .send({'type': 'error', 'message': 'Model not initialized'});
            }
            break;
          case 'dispose':
            if (ctx != null && ctx?.address != 0) {
              FFIBindings.freeSdCtx(ctx!);
              ctx = null;
              print("Model context freed.");
            } else {
              print("No model context to free.");
            }
            mainSendPort.send({'type': 'disposed'});
            break;
        }
      }
    });
  }

  Future<void> generateImage({
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
    String? inputIdImagesPath,
    double styleStrength = 1.0,
    Uint8List? controlImageData, // Add this
    int? controlImageWidth, // Add this
    int? controlImageHeight, // Add this
    double controlStrength = 0.9, // Add this
  }) async {
    _loadingController.add(true);
    try {
      await _uninitialized.future;
      await Future.delayed(const Duration(milliseconds: 500));
      _sdSendPort.send({
        'command': 'generate',
        'prompt': prompt,
        'negativePrompt': negativePrompt,
        'clipSkip': clipSkip,
        'cfgScale': cfgScale,
        'guidance': guidance,
        'width': width,
        'height': height,
        'sampleMethod': sampleMethod,
        'sampleSteps': sampleSteps,
        'seed': seed,
        'batchCount': batchCount,
        'inputIdImagesPath': inputIdImagesPath ?? '',
        'styleStrength': styleStrength,
        'controlImageData': controlImageData, // Add this
        'controlImageWidth': controlImageWidth, // Add this
        'controlImageHeight': controlImageHeight, // Add this // Add this
        'controlStrength': controlStrength, // Add this
      });
    } finally {
      _loadingController.add(false);
    }
  }

  Future<String> saveGeneratedImage(ui.Image image, String prompt, int width,
      int height, SampleMethod sampleMethod) async {
    final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
    if (bytes == null) return 'Failed to encode image';

    print(
        "Current seed value: ${StableDiffusionService.lastUsedSeed}"); // Debug print

    final seedString = StableDiffusionService.lastUsedSeed != null
        ? '_seed${StableDiffusionService.lastUsedSeed}'
        : '';

    print("Seed string: $seedString"); // Debug print

    final fileName =
        '${sanitizePrompt(prompt)}_${width}x${height}_${sampleMethod.displayName}${seedString}_${generateRandomSequence(5)}';

    try {
      await Gal.putImageBytes(bytes.buffer.asUint8List(), name: fileName);
      return 'Image saved as $fileName';
    } catch (e) {
      return 'Failed to save image: $e';
    }
  }

  void dispose() {
    _loadingController.close();
    if (_sdSendPort != null) {
      _sdSendPort.send({'command': 'dispose'});
    }

    Future.delayed(Duration(milliseconds: 100), () {
      _sdIsolate.kill();
    });

    _receivePort.close();
    _imageController.close();
  }
}
