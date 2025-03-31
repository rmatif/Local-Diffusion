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
List<String> _collectedLogs = []; // Added to collect logs within the isolate

late final Pointer<NativeFunction<LogCallbackNative>> _logCallbackPtr;
late final Pointer<NativeFunction<ProgressCallbackNative>> _progressCallbackPtr;

// Static FFI log callback (Updated with error detection)
void _staticLogCallback(int level, Pointer<Utf8> text, Pointer<Void> data) {
  final message = text.toDartString();
  final logEntry = '[Log L$level] $message';
  _collectedLogs.add(logEntry); // Add to collected logs first

  // Check for specific error messages during loading
  if (message.contains("get sd version from file failed") ||
      message.contains("new_sd_ctx_t failed") ||
      message.contains("load tensors from model loader failed")) {
    _globalSendPort?.send({
      'type': 'error',
      'errorType': 'modelError',
      'message': 'Unsupported model format or corrupted file.',
    });
    // Optionally return here if you don't want to send the raw log for these errors
    // return;
  } else if (message.contains("load tae tensors from model loader failed")) {
    _globalSendPort?.send({
      'type': 'error',
      'errorType': 'taesdError',
      'message': 'Unsupported TAESD model format or corrupted file.',
    });
    // return;
  } else if (message
      .contains("load control net tensors from model loader failed")) {
    _globalSendPort?.send({
      'type': 'error',
      'errorType': 'controlNetError',
      'message': 'Unsupported ControlNet model format or corrupted file.',
    });
    // return;
  }

  // Handle seed extraction for logs (if needed)
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
      // Don't return, allow the general log message below if needed
    }
  }

  // Send the general log message for other cases or if not returned above
  _globalSendPort?.send({
    'type': 'log',
    'level': level,
    'message': message,
  });
}

void _staticProgressCallback(
    int step, int steps, double time, Pointer<Void> data) {
  final progressEntry =
      '[Progress] Step $step/$steps (${time.toStringAsFixed(1)}s)';
  _collectedLogs.add(progressEntry);
  print(
      "SD Progress: $step/$steps - ${time.toStringAsFixed(1)}s"); // Restore print statement

  // Send immediately for real-time updates
  _globalSendPort?.send({
    'type': 'progress',
    'step': step,
    'steps': steps,
    'time': time,
  });
}

class Img2ImgProcessor {
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
  final int clipSkip;
  final bool vaeTiling;
  final String? controlNetPath; // Add this
  final Uint8List? controlImageData; // Add this
  final int? controlImageWidth; // Add this
  final int? controlImageHeight;
  final double controlStrength;
  late Isolate _sdIsolate;
  late SendPort _sdSendPort;
  final Completer _uninitialized = Completer();
  final ReceivePort _receivePort = ReceivePort();
  final StreamController<ui.Image> _imageController =
      StreamController<ui.Image>.broadcast();
  final StreamController<List<String>> _logListController =
      StreamController<List<String>>.broadcast(); // Added for logs
  final _loadingController = StreamController<bool>.broadcast();

  Stream<bool> get loadingStream => _loadingController.stream;
  Stream<List<String>> get logListStream =>
      _logListController.stream; // Added getter for logs
  Stream<ui.Image> get imageStream => _imageController.stream;

  Img2ImgProcessor({
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
    this.clipSkip = 1,
    this.vaeTiling = false,
    this.controlNetPath, // Add this
    this.controlImageData, // Add this
    this.controlImageWidth, // Add this
    this.controlImageHeight, // Add this
    this.controlStrength = 0.9,
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
            'clipLPath': clipLPath,
            'clipGPath': clipGPath,
            't5xxlPath': t5xxlPath,
            'vaePath': vaePath,
            'embedDirPath': embedDirPath,
            'clipSkip': clipSkip,
            'vaeTiling': vaeTiling,
            'controlNetPath': controlNetPath,
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
          } else if (message['type'] == 'logs') {
            // Handle the collected logs
            _logListController.add(List<String>.from(message['logs']));
          } else if (message['type'] == 'error') {
            // Handle errors from isolate
            print(
                "Error from isolate (${message['errorType']}): ${message['message']}");
            // Propagate the specific error type and message to the UI via onLog
            if (onLog != null) {
              onLog!(LogMessage(
                  -1, // Indicate error level
                  "Error (${message['errorType']}): ${message['message']}"));
            }
            // The UI (_img2img_page) listens to onLog and calls _handleLoadingError
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
                  : "/".toNativeUtf8(); // Provide a valid default path
              final taesdPathUtf8 = (message['useTinyAutoencoder'] &&
                      message['taesdPath'] != null)
                  ? message['taesdPath'].toString().toNativeUtf8()
                  : emptyUtf8;
              final embedDirUtf8 = message['embedDirPath'] != null &&
                      message['embedDirPath'].toString().isNotEmpty
                  ? message['embedDirPath'].toString().toNativeUtf8()
                  : "".toNativeUtf8();
              final controlNetPathUtf8 = message['controlNetPath'] != null &&
                      message['controlNetPath'].toString().isNotEmpty
                  ? message['controlNetPath'].toString().toNativeUtf8()
                  : "".toNativeUtf8();

              ctx = FFIBindings.newSdCtx(
                modelPathUtf8,
                clipLPathUtf8,
                clipGPathUtf8,
                t5xxlPathUtf8,
                emptyUtf8, // diffusion_model_path, not used
                vaePathUtf8,
                taesdPathUtf8,
                controlNetPathUtf8,
                loraDirUtf8,
                embedDirUtf8,
                emptyUtf8, // stacked_id_embed_dir_c_str, not used
                message['useFlashAttention'],
                message['vaeTiling'],
                false, // free_params_immediately, not used
                FFIBindings.getCores() * 2,
                mapModelTypeToIndex(SDType.values[message['modelType']]),
                0, // rng_type, not used
                message['schedule'],
                false, // keep_clip_on_cpu, not used
                false, // keep_control_net_cpu, not used
                false, // keep_vae_on_cpu, not used
                message['clipSkip'],
                message[
                    'useFlashAttention'], // Added diffusion_flash_attn parameter
              );

              calloc.free(modelPathUtf8);
              calloc.free(loraDirUtf8);
              calloc.free(clipLPathUtf8);
              calloc.free(clipGPathUtf8);
              calloc.free(t5xxlPathUtf8);
              calloc.free(vaePathUtf8);
              calloc.free(embedDirUtf8);
              calloc.free(controlNetPathUtf8);
              if (message['useTinyAutoencoder'] &&
                  message['taesdPath'] != null &&
                  taesdPathUtf8 != emptyUtf8) {
                calloc.free(taesdPathUtf8);
              }
              calloc.free(emptyUtf8);

              if (ctx != null && ctx!.address != 0) {
                print("Model initialized successfully in isolate");
                mainSendPort.send({'type': 'modelLoaded'});
              } else {
                // Check if an error was already sent by the log callback
                print(
                    "Failed to initialize model in isolate (ctx is null or address is 0)");
                bool specificErrorSent = _collectedLogs.any((log) =>
                    log.contains("failed") || // Basic check
                    log.contains("error")); // Basic check

                if (!specificErrorSent) {
                  mainSendPort.send({
                    'type': 'error',
                    'errorType': 'modelError', // Generic model error
                    'message':
                        'Failed to initialize model context. Check model compatibility or file integrity.'
                  });
                }
              }
            } catch (e) {
              print("Error initializing model: $e");
              mainSendPort.send({'type': 'error', 'message': e.toString()});
            }
            break;

          case 'img2img':
            _collectedLogs.clear(); // Clear logs before starting generation
            if (ctx != null && ctx!.address != 0) {
              // Add null check for address too
              print(
                  "Starting img2img generation with context: ${ctx!.address}");
              Pointer<SDImage>? controlCondPtr = nullptr; // Declare before try
              Pointer<Uint8>? controlDataPtr = nullptr; // Also declare here

              try {
                final inputImageData = message['inputImageData'] as Uint8List;
                final inputWidth = message['inputWidth'] as int;
                final inputHeight = message['inputHeight'] as int;
                final channel = message['channel'] as int;
                final outputWidth = message['outputWidth'] as int;
                final outputHeight = message['outputHeight'] as int;

                final initImageDataPtr = malloc<Uint8>(inputImageData.length);
                final initImageDataList =
                    initImageDataPtr.asTypedList(inputImageData.length);
                initImageDataList.setAll(0, inputImageData);

                final initImage = malloc<SDImage>();
                initImage.ref
                  ..width = inputWidth
                  ..height = inputHeight
                  ..channel = channel
                  ..data = initImageDataPtr;

                final promptUtf8 = message['prompt'].toString().toNativeUtf8();
                final negPromptUtf8 =
                    message['negativePrompt'].toString().toNativeUtf8();
                final emptyUtf8 = "".toNativeUtf8();
                // Pointer<SDImage>? controlCondPtr; // Remove re-declaration
                if (message['controlImageData'] != null) {
                  final controlImageData =
                      message['controlImageData'] as Uint8List;
                  final controlWidth = message['controlImageWidth'] as int;
                  final controlHeight = message['controlImageHeight'] as int;
                  // final controlDataPtr = malloc<Uint8>(controlImageData.length); // Use outer declaration
                  controlDataPtr = malloc<Uint8>(controlImageData.length);
                  controlDataPtr!
                      .asTypedList(controlImageData.length) // Add null check
                      .setAll(0, controlImageData);

                  // In the _isolateEntryPoint function, modify the code for Canny processing:

                  // controlCondPtr = malloc<SDImage>(); // Use outer declaration
                  controlCondPtr = malloc<SDImage>();
                  controlCondPtr.ref
                    ..width = controlWidth
                    ..height = controlHeight
                    ..channel = 3
                    ..data = controlDataPtr!; // Add null check
                }

                Pointer<SDImage> maskImage;
                if (message['maskImageData'] != null) {
                  // User provided a mask image
                  final maskImageData = message['maskImageData'] as Uint8List;
                  final maskWidth = message['maskImageWidth'] as int;
                  final maskHeight = message['maskImageHeight'] as int;
                  final maskDataPtr = malloc<Uint8>(maskImageData.length);
                  maskDataPtr
                      .asTypedList(maskImageData.length)
                      .setAll(0, maskImageData);

                  maskImage = malloc<SDImage>();
                  maskImage.ref
                    ..width = maskWidth
                    ..height = maskHeight
                    ..channel = 1 // Mask should be grayscale - 1 channel
                    ..data = maskDataPtr;
                } else {
                  // Create default white mask (all 255 values)
                  final maskDataSize = inputWidth * inputHeight;
                  final maskDataPtr = malloc<Uint8>(maskDataSize);
                  final maskDataList = maskDataPtr.asTypedList(maskDataSize);
                  // Fill with 255 (white) values
                  for (int i = 0; i < maskDataSize; i++) {
                    maskDataList[i] = 255;
                  }

                  maskImage = malloc<SDImage>();
                  maskImage.ref
                    ..width = inputWidth
                    ..height = inputHeight
                    ..channel = 1 // Mask should be grayscale - 1 channel
                    ..data = maskDataPtr;
                }

                final result = FFIBindings.img2img(
                  ctx!,
                  initImage.ref,
                  maskImage.ref, // Added mask_image parameter
                  promptUtf8,
                  negPromptUtf8,
                  message['clipSkip'],
                  message['cfgScale'],
                  message['guidance'],
                  0.0, // Added eta parameter
                  outputWidth,
                  outputHeight,
                  message['sampleMethod'],
                  message['sampleSteps'],
                  message['strength'],
                  message['seed'],
                  message['batchCount'],
                  controlCondPtr ?? nullptr,
                  message['controlStrength'] ?? 0.0,
                  0.0, // style_strength
                  false, // normalize_input
                  emptyUtf8, // input_id_images_path
                  nullptr, // skip_layers
                  0, // skip_layers_count
                  0.0, // slg_scale
                  0.0, // skip_layer_start
                  0.0, // skip_layer_end
                );

                calloc.free(initImageDataPtr);
                malloc.free(initImage);
                malloc.free(maskImage.ref.data);
                malloc.free(maskImage);
                calloc.free(promptUtf8);
                calloc.free(negPromptUtf8);
                calloc.free(emptyUtf8);

                print("Generation result address: ${result.address}");

                if (result.address != 0) {
                  final image = result.cast<SDImage>().ref;
                  final bytes = image.data
                      .asTypedList(image.width * image.height * image.channel);
                  final rgbaBytes = Uint8List(image.width * image.height * 4);

                  for (var i = 0; i < image.width * image.height; i++) {
                    rgbaBytes[i * 4] = bytes[i * 3];
                    rgbaBytes[i * 4 + 1] = bytes[i * 3 + 1];
                    rgbaBytes[i * 4 + 2] = bytes[i * 3 + 2];
                    rgbaBytes[i * 4 + 3] = 255;
                  }

                  mainSendPort.send({
                    'type': 'image',
                    'bytes': rgbaBytes,
                    'width': image.width,
                    'height': image.height,
                  });

                  calloc.free(image.data);
                  calloc.free(result.cast<Void>());
                }
              } catch (e) {
                print(
                    "Error generating image in isolate: $e"); // More specific log
                mainSendPort.send({
                  'type': 'error',
                  'errorType': 'generationError', // Specific error type
                  'message': "Generation error: ${e.toString()}"
                });
                // Send logs even if there was an error during generation
                mainSendPort.send({
                  'type': 'logs',
                  'logs': _collectedLogs,
                });
              } finally {
                // Send collected logs regardless of success or failure
                mainSendPort.send({
                  'type': 'logs',
                  'logs': _collectedLogs,
                });

                // Free C memory for control cond if allocated
                if (controlCondPtr != null && controlCondPtr != nullptr) {
                  // Add null check
                  // Don't free controlCondPtr.ref.data here if it points to controlDataPtr
                  malloc.free(controlCondPtr); // Free the SDImage struct itself
                }
                if (controlDataPtr != null && controlDataPtr != nullptr) {
                  // Add null check
                  malloc.free(
                      controlDataPtr); // Free the image data buffer allocated for control image
                }

                // Free other native strings if they were allocated
                // Note: These were already freed earlier in the 'initialize' case,
                // but ensure correct freeing pattern if paths were passed differently
                // For img2img, promptUtf8, negPromptUtf8, etc. are freed inside the try block
              }
            } else {
              // This else corresponds to the 'if (ctx != null)' check
              print("Context is null in isolate, cannot generate image");
              mainSendPort.send({
                'type': 'error',
                'message': 'Model not initialized in isolate'
              });
            }
            break; // Break for the 'img2img' case

          case 'dispose':
            if (ctx != null && ctx!.address != 0) {
              FFIBindings.freeSdCtx(ctx!);
              ctx = null;
              print("Model context freed.");
            } else {
              print("No model context to free.");
            }
            mainSendPort.send({'type': 'disposed'});
            // Close the isolate's receive port to allow the isolate to terminate
            isolateReceivePort.close();
            break;
        }
      }
    });
  }

  Future<void> generateImg2Img({
    required Uint8List inputImageData,
    required int inputWidth,
    required int inputHeight,
    required int channel,
    required int outputWidth,
    required int outputHeight,
    required String prompt,
    String negativePrompt = "",
    int clipSkip = 1,
    double cfgScale = 7.0,
    double guidance = 1.0,
    double eta = 0.0, // Added eta parameter
    int sampleMethod = 0,
    int sampleSteps = 20,
    double strength = 0.5,
    int seed = 42,
    int batchCount = 1,
    Uint8List? controlImageData,
    int? controlImageWidth,
    int? controlImageHeight,
    double controlStrength = 0.9,
    Uint8List? maskImageData, // Add this
    int? maskWidth, // Add this
    int? maskHeight, // Add this
  }) async {
    _loadingController.add(true);
    try {
      await _uninitialized.future;
      await Future.delayed(const Duration(milliseconds: 500));
      _sdSendPort.send({
        'command': 'img2img',
        'inputImageData': inputImageData,
        'inputWidth': inputWidth,
        'inputHeight': inputHeight,
        'channel': channel,
        'outputWidth': outputWidth,
        'outputHeight': outputHeight,
        'prompt': prompt,
        'negativePrompt': negativePrompt,
        'clipSkip': clipSkip,
        'cfgScale': cfgScale,
        'guidance': guidance,
        'eta': eta, // Added eta parameter
        'sampleMethod': sampleMethod,
        'sampleSteps': sampleSteps,
        'strength': strength,
        'seed': seed,
        'batchCount': batchCount,
        'controlImageData': controlImageData,
        'controlImageWidth': controlImageWidth,
        'controlImageHeight': controlImageHeight,
        'controlStrength': controlStrength,
        if (maskImageData != null) 'maskImageData': maskImageData,
        if (maskWidth != null) 'maskImageWidth': maskWidth,
        if (maskHeight != null) 'maskImageHeight': maskHeight,
      });
    } finally {
      _loadingController.add(false);
    }
  }

  Future<String> saveGeneratedImage(ui.Image image, String prompt, int width,
      int height, SampleMethod sampleMethod) async {
    final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
    if (bytes == null) return 'Failed to encode image';

    final seedString = StableDiffusionService.lastUsedSeed != null
        ? '_seed${StableDiffusionService.lastUsedSeed}'
        : '';

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
    _logListController.close(); // Close the new controller
  }
}
