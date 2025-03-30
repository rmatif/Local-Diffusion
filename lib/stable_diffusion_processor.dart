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

// Global variables for the isolate
SendPort? _globalSendPort;
List<String> _collectedLogs = []; // Added to collect logs within the isolate

// FFI callback pointers
late final Pointer<NativeFunction<LogCallbackNative>> _logCallbackPtr;
late final Pointer<NativeFunction<ProgressCallbackNative>> _progressCallbackPtr;

// Static FFI log callback
void _staticLogCallback(int level, Pointer<Utf8> text, Pointer<Void> data) {
  final message = text.toDartString();

  // Also add to the collected logs list
  final logEntry = '[Log L$level] $message';
  _collectedLogs.add(logEntry);

  // Send immediately for real-time updates (optional, but kept for existing behavior)
  if (message.contains("generating image")) {
    final seedMatch = RegExp(r'seed (\d+)').firstMatch(message);
    if (seedMatch != null) {
      final extractedSeed = int.parse(seedMatch.group(1)!);
      _globalSendPort?.send({
        'type': 'log',
        'level': level,
        'message': message,
        'seed': extractedSeed // Keep sending seed if found
      });
      // Don't return here, let it fall through to send the basic log message too if needed
    }
  }

  _globalSendPort?.send({
    'type': 'log',
    'level': level,
    'message': message, // Send the original message for compatibility
  });
}

// Static FFI progress callback
void _staticProgressCallback(
    int step, int steps, double time, Pointer<Void> data) {
  final progressEntry =
      '[Progress] Step $step/$steps (${time.toStringAsFixed(1)}s)';
  _collectedLogs.add(progressEntry);

  // Send immediately for real-time updates
  _globalSendPort?.send({
    'type': 'progress',
    'step': step,
    'steps': steps,
    'time': time,
  });
}

// Main processor class
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
  final String? controlNetPath;
  final Uint8List? controlImageData;
  final int? controlImageWidth;
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
  Stream<ui.Image> get imageStream => _imageController.stream;
  Stream<List<String>> get logListStream =>
      _logListController.stream; // Added getter for logs

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
    this.controlNetPath,
    this.controlImageData,
    this.controlImageWidth,
    this.controlImageHeight,
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
            'controlNetPath': controlNetPath,
          });
        } else if (message is Map) {
          switch (message['type']) {
            case 'modelLoaded':
              _uninitialized.complete();
              if (onModelLoaded != null) onModelLoaded!();
              break;
            case 'log':
              if (message.containsKey('seed')) {
                StableDiffusionService.lastUsedSeed = message['seed'];
              }
              if (onLog != null) {
                onLog!(LogMessage(message['level'], message['message']));
              }
              break;
            case 'progress':
              if (onProgress != null) {
                onProgress!(ProgressUpdate(
                  message['step'],
                  message['steps'],
                  message['time'],
                ));
              }
              break;
            case 'image':
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
              break;
            case 'logs': // Handle the collected logs
              _logListController.add(List<String>.from(message['logs']));
              break;
            case 'error':
              print("Error from isolate: ${message['message']}");
              // Optionally propagate the error further
              break;
          }
        }
      });
    } finally {
      _loadingController.add(false);
    }
  }

  // Isolate entry point - runs in the separate isolate
  static void _isolateEntryPoint(Map<String, dynamic> args) {
    final SendPort mainSendPort = args['port'];
    _globalSendPort = mainSendPort; // Set the global send port for callbacks
    _logCallbackPtr =
        Pointer.fromFunction<LogCallbackNative>(_staticLogCallback);
    _progressCallbackPtr =
        Pointer.fromFunction<ProgressCallbackNative>(_staticProgressCallback);
    final ReceivePort isolateReceivePort = ReceivePort();
    mainSendPort.send(
        isolateReceivePort.sendPort); // Send the isolate's port back to main

    Pointer<Void>? ctx; // Stable Diffusion context pointer
    print("Isolate started");

    isolateReceivePort.listen((message) {
      print("Received message in isolate: ${message['command']}");
      if (message is Map) {
        switch (message['command']) {
          case 'initialize':
            print("Initializing SD model in isolate...");
            FFIBindings.setLogCallback(_logCallbackPtr, nullptr);
            FFIBindings.setProgressCallback(_progressCallbackPtr, nullptr);

            Pointer<Utf8>? modelPathUtf8;
            Pointer<Utf8>? clipLPathUtf8;
            Pointer<Utf8>? clipGPathUtf8;
            Pointer<Utf8>? t5xxlPathUtf8;
            Pointer<Utf8>? vaePathUtf8;
            Pointer<Utf8>? loraDirUtf8;
            Pointer<Utf8>? taesdPathUtf8;
            Pointer<Utf8>? stackedIdEmbedDirUtf8;
            Pointer<Utf8>? embedDirUtf8;
            Pointer<Utf8>? controlNetPathUtf8;
            Pointer<Utf8>? emptyUtf8;

            try {
              modelPathUtf8 = message['modelPath'].toString().toNativeUtf8();
              clipLPathUtf8 = message['clipLPath']?.toString().toNativeUtf8() ??
                  "".toNativeUtf8();
              clipGPathUtf8 = message['clipGPath']?.toString().toNativeUtf8() ??
                  "".toNativeUtf8();
              t5xxlPathUtf8 = message['t5xxlPath']?.toString().toNativeUtf8() ??
                  "".toNativeUtf8();
              vaePathUtf8 = message['vaePath']?.toString().toNativeUtf8() ??
                  "".toNativeUtf8();
              loraDirUtf8 = message['loraPath']?.toString().toNativeUtf8() ??
                  "/".toNativeUtf8(); // Default if null
              taesdPathUtf8 = (message['useTinyAutoencoder'] &&
                      message['taesdPath'] != null)
                  ? message['taesdPath'].toString().toNativeUtf8()
                  : "".toNativeUtf8();
              stackedIdEmbedDirUtf8 =
                  message['stackedIdEmbedDir']?.toString().toNativeUtf8() ??
                      "".toNativeUtf8();
              embedDirUtf8 =
                  message['embedDirPath']?.toString().toNativeUtf8() ??
                      "".toNativeUtf8();
              controlNetPathUtf8 =
                  message['controlNetPath']?.toString().toNativeUtf8() ??
                      "".toNativeUtf8();
              emptyUtf8 = "".toNativeUtf8(); // Reusable empty string

              ctx = FFIBindings.newSdCtx(
                modelPathUtf8,
                clipLPathUtf8,
                clipGPathUtf8,
                t5xxlPathUtf8,
                emptyUtf8, // id_embed_dir (assuming empty for now)
                vaePathUtf8,
                taesdPathUtf8,
                controlNetPathUtf8,
                loraDirUtf8,
                embedDirUtf8,
                stackedIdEmbedDirUtf8,
                message['useFlashAttention'],
                message['vaeTiling'],
                false, // free_params_immediately
                FFIBindings.getCores() * 2, // n_threads
                mapModelTypeToIndex(
                    SDType.values[message['modelType']]), // wtype
                0, // rng_type (STD_DEFAULT_RNG)
                message['schedule'], // schedule
                false, // keep_clip_on_cpu
                false, // keep_control_net_cpu
                false, // keep_vae_on_cpu
                message['clipSkip'],
                message['useFlashAttention'], // diffusion_flash_attn
              );

              if (ctx != null && ctx!.address != 0) {
                print("Model initialized successfully in isolate");
                mainSendPort.send({'type': 'modelLoaded'});
              } else {
                print("Failed to initialize model in isolate");
                mainSendPort.send({
                  'type': 'error',
                  'message': 'Failed to initialize model context'
                });
              }
            } catch (e) {
              print("Error initializing model in isolate: $e");
              mainSendPort.send({
                'type': 'error',
                'message': "Initialization error: ${e.toString()}"
              });
            } finally {
              // Free allocated memory
              if (modelPathUtf8 != null) calloc.free(modelPathUtf8);
              if (clipLPathUtf8 != null &&
                  clipLPathUtf8.address != emptyUtf8?.address)
                calloc.free(clipLPathUtf8);
              if (clipGPathUtf8 != null &&
                  clipGPathUtf8.address != emptyUtf8?.address)
                calloc.free(clipGPathUtf8);
              if (t5xxlPathUtf8 != null &&
                  t5xxlPathUtf8.address != emptyUtf8?.address)
                calloc.free(t5xxlPathUtf8);
              if (vaePathUtf8 != null &&
                  vaePathUtf8.address != emptyUtf8?.address)
                calloc.free(vaePathUtf8);
              if (loraDirUtf8 != null &&
                  loraDirUtf8.address != emptyUtf8?.address)
                calloc.free(loraDirUtf8); // Check against empty
              if (taesdPathUtf8 != null &&
                  taesdPathUtf8.address != emptyUtf8?.address)
                calloc.free(taesdPathUtf8);
              if (stackedIdEmbedDirUtf8 != null &&
                  stackedIdEmbedDirUtf8.address != emptyUtf8?.address)
                calloc.free(stackedIdEmbedDirUtf8);
              if (embedDirUtf8 != null &&
                  embedDirUtf8.address != emptyUtf8?.address)
                calloc.free(embedDirUtf8);
              if (controlNetPathUtf8 != null &&
                  controlNetPathUtf8.address != emptyUtf8?.address)
                calloc.free(controlNetPathUtf8);
              if (emptyUtf8 != null) calloc.free(emptyUtf8);
            }
            break;

          case 'generate':
            _collectedLogs.clear(); // Clear logs before starting generation
            if (ctx == null || ctx!.address == 0) {
              print("Context is null in isolate, cannot generate image");
              mainSendPort.send({
                'type': 'error',
                'message': 'Model not initialized in isolate'
              });
              break; // Exit case
            }

            print(
                "Starting image generation in isolate with context: ${ctx!.address}");
            Pointer<Utf8>? promptUtf8;
            Pointer<Utf8>? negPromptUtf8;
            Pointer<Utf8>? inputIdImagesPathUtf8;
            Pointer<SDImage>? controlCondPtr;
            Pointer<Uint8>? controlDataPtr;

            try {
              promptUtf8 = message['prompt'].toString().toNativeUtf8();
              negPromptUtf8 =
                  message['negativePrompt'].toString().toNativeUtf8();
              inputIdImagesPathUtf8 =
                  message['inputIdImagesPath']?.toString().toNativeUtf8() ??
                      "".toNativeUtf8();
              final styleStrength =
                  message['styleStrength'] ?? 1.0; // Default style strength

              // Prepare control image data if provided
              if (message['controlImageData'] != null) {
                final controlImageData =
                    message['controlImageData'] as Uint8List;
                final controlWidth = message['controlImageWidth'] as int;
                final controlHeight = message['controlImageHeight'] as int;

                controlDataPtr = malloc<Uint8>(controlImageData.length);
                controlDataPtr!
                    .asTypedList(controlImageData.length)
                    .setAll(0, controlImageData);

                controlCondPtr = malloc<SDImage>();
                controlCondPtr.ref
                  ..width = controlWidth
                  ..height = controlHeight
                  ..channel = 3 // Assuming RGB
                  ..data = controlDataPtr;
              }

              final result = FFIBindings.txt2img(
                ctx!,
                promptUtf8,
                negPromptUtf8,
                message['clipSkip'],
                message['cfgScale'],
                message['guidance'],
                0.0, // eta (new parameter, default 0.0)
                message['width'],
                message['height'],
                message['sampleMethod'],
                message['sampleSteps'],
                message['seed'],
                message['batchCount'],
                controlCondPtr ?? nullptr, // Pass control condition or null
                message['controlStrength'] ??
                    0.9, // Pass control strength or default
                styleStrength, // Pass style strength
                false, // normalize_input
                inputIdImagesPathUtf8, // input_id_images_path
                nullptr, // skip_layers
                0, // skip_layers_count
                0.0, // slg_scale
                0.0, // skip_layer_start
                0.0, // skip_layer_end
              );

              print("Generation result address in isolate: ${result.address}");

              if (result.address != 0) {
                final image = result.cast<SDImage>().ref;
                // Ensure channel is 3 (RGB) before processing
                if (image.channel == 3) {
                  final bytes = image.data
                      .asTypedList(message['width'] * message['height'] * 3);
                  final rgbaBytes =
                      Uint8List(message['width'] * message['height'] * 4);

                  for (var i = 0;
                      i < message['width'] * message['height'];
                      i++) {
                    rgbaBytes[i * 4] = bytes[i * 3]; // R
                    rgbaBytes[i * 4 + 1] = bytes[i * 3 + 1]; // G
                    rgbaBytes[i * 4 + 2] = bytes[i * 3 + 2]; // B
                    rgbaBytes[i * 4 + 3] = 255; // A
                  }

                  mainSendPort.send({
                    'type': 'image',
                    'bytes': rgbaBytes,
                    'width': message['width'],
                    'height': message['height'],
                  });
                } else {
                  print(
                      "Error: Generated image has unexpected channel count: ${image.channel}");
                  mainSendPort.send({
                    'type': 'error',
                    'message':
                        'Generated image has unexpected channel count: ${image.channel}'
                  });
                }
                // Free the native image data *after* processing
                calloc.free(image.data);
                calloc.free(
                    result.cast<Void>()); // Free the result pointer itself
              } else {
                print("Image generation failed (result address is 0)");
                mainSendPort.send({
                  'type': 'error',
                  'message': 'Image generation failed in isolate'
                });
              }

              // Send collected logs regardless of success or failure
              mainSendPort.send({
                'type': 'logs',
                'logs': _collectedLogs,
              });
            } catch (e) {
              print("Error generating image in isolate: $e");
              mainSendPort.send({
                'type': 'error',
                'message': "Generation error: ${e.toString()}"
              });
              // Send logs even if there was an error during generation
              mainSendPort.send({
                'type': 'logs',
                'logs': _collectedLogs,
              });
            } finally {
              // Free allocated memory for this generation
              if (promptUtf8 != null) calloc.free(promptUtf8);
              if (negPromptUtf8 != null) calloc.free(negPromptUtf8);
              if (inputIdImagesPathUtf8 != null &&
                  inputIdImagesPathUtf8.address != "".toNativeUtf8().address)
                calloc.free(inputIdImagesPathUtf8); // Check against empty
              if (controlCondPtr != null) {
                // Don't free controlCondPtr.ref.data here if it points to controlDataPtr
                // calloc.free(controlCondPtr.ref.data); // This would be double free if data is from malloc
                malloc.free(controlCondPtr); // Free the SDImage struct itself
              }
              if (controlDataPtr != null) {
                malloc.free(controlDataPtr); // Free the image data buffer
              }
            }
            break;

          case 'dispose':
            if (ctx != null && ctx?.address != 0) {
              FFIBindings.freeSdCtx(ctx!);
              ctx = null;
              print("Model context freed in isolate.");
            } else {
              print("No model context to free in isolate.");
            }
            mainSendPort.send({'type': 'disposed'});
            // Close the isolate's receive port to allow the isolate to terminate
            isolateReceivePort.close();
            break;
        }
      }
    });
  } // End of _isolateEntryPoint

  // --- Methods called from the main isolate ---

  Future<void> generateImage({
    required String prompt,
    String negativePrompt = "",
    int clipSkip = 1,
    double cfgScale = 7.0,
    double guidance = 1.0,
    int width = 512,
    int height = 512,
    int sampleMethod = 0, // Corresponds to SampleMethod enum index
    int sampleSteps = 20,
    int seed = 42,
    int batchCount = 1,
    String? inputIdImagesPath,
    double styleStrength = 1.0,
    Uint8List? controlImageData,
    int? controlImageWidth,
    int? controlImageHeight,
    double controlStrength = 0.9,
  }) async {
    _loadingController.add(true); // Indicate loading starts
    try {
      await _uninitialized.future; // Ensure isolate is initialized
      // Add a small delay if needed, though Completer should handle readiness
      // await Future.delayed(const Duration(milliseconds: 100));
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
        'inputIdImagesPath': inputIdImagesPath, // Pass null directly if null
        'styleStrength': styleStrength,
        'controlImageData': controlImageData,
        'controlImageWidth': controlImageWidth,
        'controlImageHeight': controlImageHeight,
        'controlStrength': controlStrength,
      });
    } catch (e) {
      print("Error sending generate command: $e");
      _loadingController.add(false); // Ensure loading stops on error
    } finally {
      // Loading state might be managed differently now, perhaps based on receiving image/logs/error
      // Consider removing this immediate false setting if loading should persist until completion/error
      // _loadingController.add(false);
    }
  }

  Future<String> saveGeneratedImage(ui.Image image, String prompt, int width,
      int height, SampleMethod sampleMethod) async {
    final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
    if (bytes == null) return 'Failed to encode image';

    print("Saving image with seed: ${StableDiffusionService.lastUsedSeed}");

    final seedString = StableDiffusionService.lastUsedSeed != null
        ? '_seed${StableDiffusionService.lastUsedSeed}'
        : '';

    final fileName =
        '${sanitizePrompt(prompt)}_${width}x${height}_${sampleMethod.displayName}${seedString}_${generateRandomSequence(5)}';

    try {
      await Gal.putImageBytes(bytes.buffer.asUint8List(), name: fileName);
      print('Image saved successfully as $fileName');
      return 'Image saved as $fileName';
    } catch (e) {
      print('Failed to save image: $e');
      return 'Failed to save image: $e';
    }
  }

  void dispose() {
    print("Disposing StableDiffusionProcessor...");
    _loadingController.close();
    // Check if _sdSendPort has been initialized before sending
    if (_uninitialized.isCompleted) {
      try {
        _sdSendPort.send({'command': 'dispose'});
      } catch (e) {
        print(
            "Error sending dispose command (isolate might already be gone): $e");
      }
    }

    // Give the isolate a moment to process the dispose command before killing
    Future.delayed(Duration(milliseconds: 200), () {
      print("Killing isolate...");
      _sdIsolate.kill(priority: Isolate.immediate);
    });

    _receivePort.close();
    _imageController.close();
    _logListController.close(); // Close the new controller
    print("StableDiffusionProcessor disposed.");
  }
}
