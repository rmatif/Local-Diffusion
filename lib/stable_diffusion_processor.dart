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

void _staticLogCallback(int level, Pointer<Utf8> text, Pointer<Void> data) {
  final message = text.toDartString();
  print("SD Log: $message");
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
  }) {
    _initializeIsolate();
  }

  static int mapModelTypeToIndex(SDType modelType) {
    switch (modelType) {
      case SDType.NONE:
        return 34; // Maps to FP32
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
        {
          'port': _receivePort.sendPort,
        },
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
          });
        } else if (message is Map) {
          if (message['type'] == 'modelLoaded') {
            _uninitialized.complete();
            if (onModelLoaded != null) onModelLoaded!();
          } else if (message['type'] == 'log') {
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
            final logCallbackPointer =
                Pointer.fromFunction<LogCallbackNative>(_staticLogCallback);
            final progressCallbackPointer =
                Pointer.fromFunction<ProgressCallbackNative>(
                    _staticProgressCallback);

            try {
              final modelPathUtf8 =
                  message['modelPath'].toString().toNativeUtf8();
              final emptyUtf8 = "".toNativeUtf8();
              final loraDirUtf8 =
                  "/data/user/0/com.example.sd_test_app/cache/file_picker"
                      .toNativeUtf8();
              final taesdPathUtf8 = (message['useTinyAutoencoder'] &&
                      message['taesdPath'] != null)
                  ? message['taesdPath'].toString().toNativeUtf8()
                  : emptyUtf8;

              FFIBindings.setLogCallback(logCallbackPointer, nullptr);
              FFIBindings.setProgressCallback(progressCallbackPointer, nullptr);

              ctx = FFIBindings.newSdCtx(
                modelPathUtf8,
                emptyUtf8,
                emptyUtf8,
                emptyUtf8,
                emptyUtf8,
                emptyUtf8,
                taesdPathUtf8,
                emptyUtf8,
                loraDirUtf8,
                emptyUtf8,
                emptyUtf8,
                message['useFlashAttention'],
                false,
                false,
                FFIBindings.getCores() * 2,
                mapModelTypeToIndex(SDType.values[message['modelType']]),
                0,
                message['schedule'],
                false,
                false,
                false,
              );

              calloc.free(modelPathUtf8);
              calloc.free(loraDirUtf8);
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
                final emptyUtf8 = "".toNativeUtf8();

                final result = FFIBindings.txt2img(
                  ctx!,
                  promptUtf8,
                  negPromptUtf8,
                  message['clipSkip'],
                  message['cfgScale'],
                  message['guidance'],
                  message['width'],
                  message['height'],
                  message['sampleMethod'],
                  message['sampleSteps'],
                  message['seed'],
                  message['batchCount'],
                  nullptr,
                  1.0,
                  1.0,
                  false,
                  emptyUtf8,
                );

                calloc.free(promptUtf8);
                calloc.free(negPromptUtf8);
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
  }) async {
    _loadingController.add(true);
    try {
      await _uninitialized.future;

      // Add delay between generations to allow memory cleanup
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
      });
    } finally {
      _loadingController.add(false);
    }
  }

  Future<String> saveGeneratedImage(ui.Image image, String prompt, int width,
      int height, SampleMethod sampleMethod) async {
    final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
    if (bytes == null) return 'Failed to encode image';

    final fileName =
        '${sanitizePrompt(prompt)}_${width}x${height}_${sampleMethod.displayName}_${generateRandomSequence(5)}';

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
