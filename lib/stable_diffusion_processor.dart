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
  final bool isDiffusionModelType; // Added flag for model type

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
    required this.isDiffusionModelType, // Added parameter
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
            'isDiffusionModelType': isDiffusionModelType, // Pass the flag
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
            case 'error': // Handle errors sent from the isolate
              print(
                  "Error from isolate (${message['errorType']}): ${message['message']}");
              // Propagate the specific error type and message to the main UI
              if (onLog != null) {
                // Use onLog or a dedicated onError callback if preferred
                onLog!(LogMessage(
                    -1, // Indicate error level
                    "Error (${message['errorType']}): ${message['message']}"));
              }
              // Send the error details to the main UI state handler
              _handleLoadingError(message['errorType'], message['message']);
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
            Pointer<Utf8>?
                diffusionModelPathUtf8; // Added for the diffusion model path
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
              emptyUtf8 = "".toNativeUtf8(); // Reusable empty string
              final bool isDiffusionModelType = message['isDiffusionModelType'];
              final String modelPathString = message['modelPath'].toString();

              // Assign path to the correct pointer based on the flag
              if (isDiffusionModelType) {
                modelPathUtf8 = emptyUtf8; // Pass empty for standard model path
                diffusionModelPathUtf8 = modelPathString.toNativeUtf8();
                print(
                    "Isolate: Using diffusion_model_path for $modelPathString");
              } else {
                modelPathUtf8 = modelPathString.toNativeUtf8();
                diffusionModelPathUtf8 =
                    emptyUtf8; // Pass empty for diffusion model path
                print("Isolate: Using model_path for $modelPathString");
              }

              // Prepare other paths (use emptyUtf8 if null/empty)
              clipLPathUtf8 =
                  message['clipLPath']?.toString().toNativeUtf8() ?? emptyUtf8;
              clipGPathUtf8 =
                  message['clipGPath']?.toString().toNativeUtf8() ?? emptyUtf8;
              t5xxlPathUtf8 =
                  message['t5xxlPath']?.toString().toNativeUtf8() ?? emptyUtf8;
              vaePathUtf8 =
                  message['vaePath']?.toString().toNativeUtf8() ?? emptyUtf8;
              loraDirUtf8 = message['loraPath']?.toString().toNativeUtf8() ??
                  "/".toNativeUtf8(); // Default if null
              taesdPathUtf8 = (message['useTinyAutoencoder'] &&
                      message['taesdPath'] != null)
                  ? message['taesdPath'].toString().toNativeUtf8()
                  : emptyUtf8;
              stackedIdEmbedDirUtf8 =
                  message['stackedIdEmbedDir']?.toString().toNativeUtf8() ??
                      emptyUtf8;
              embedDirUtf8 =
                  message['embedDirPath']?.toString().toNativeUtf8() ??
                      emptyUtf8;
              controlNetPathUtf8 =
                  message['controlNetPath']?.toString().toNativeUtf8() ??
                      emptyUtf8;

              ctx = FFIBindings.newSdCtx(
                modelPathUtf8, // First path arg
                clipLPathUtf8,
                clipGPathUtf8,
                t5xxlPathUtf8,
                diffusionModelPathUtf8, // <<< Pass the correct pointer here
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
                // Check if an error was already sent by the log callback
                // If not, send a generic model loading error.
                // This handles cases where newSdCtx returns null without a specific log.
                print(
                    "Failed to initialize model in isolate (ctx is null or address is 0)");
                // Avoid sending duplicate errors if one was already sent via log callback
                // A more robust way might involve tracking if an error was sent.
                // For simplicity, we'll rely on the log callback for specific errors.
                // If no specific error log was caught, send a generic one.
                // Check _collectedLogs for recent errors before sending generic one?
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
              print("Error initializing model in isolate: $e");
              mainSendPort.send({
                'type': 'error',
                'message': "Initialization error: ${e.toString()}"
              });
            } finally {
              // Free allocated memory
              // Only free if it's not pointing to the shared emptyUtf8
              if (modelPathUtf8 != null &&
                  modelPathUtf8.address != emptyUtf8?.address)
                calloc.free(modelPathUtf8);
              if (diffusionModelPathUtf8 != null &&
                  diffusionModelPathUtf8.address != emptyUtf8?.address)
                calloc.free(diffusionModelPathUtf8);
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
              // Special case for loraDirUtf8 which defaults to "/"
              if (loraDirUtf8 != null &&
                  loraDirUtf8.address != "/".toNativeUtf8().address)
                calloc.free(loraDirUtf8);
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
              // Free the reusable empty string pointer *once* at the end
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
            Pointer<Int32>? skipLayersPtr; // Pointer for skip_layers array
            int skipLayersCount = 0; // Count for skip_layers

            try {
              promptUtf8 = message['prompt'].toString().toNativeUtf8();
              negPromptUtf8 =
                  message['negativePrompt'].toString().toNativeUtf8();
              inputIdImagesPathUtf8 =
                  message['inputIdImagesPath']?.toString().toNativeUtf8() ??
                      "".toNativeUtf8();
              final styleStrength = message['styleStrength'] ?? 1.0;

              // --- Prepare skip_layers ---
              final String? skipLayersText = message['skipLayersText'];
              if (skipLayersText != null && skipLayersText.isNotEmpty) {
                try {
                  // Parse the string "[num1,num2,...]"
                  final numbersString =
                      skipLayersText.substring(1, skipLayersText.length - 1);
                  final layerIndices = numbersString
                      .split(',')
                      .map((s) => int.parse(s.trim()))
                      .toList();

                  if (layerIndices.isNotEmpty) {
                    skipLayersCount = layerIndices.length;
                    skipLayersPtr = malloc<Int32>(skipLayersCount);
                    for (int i = 0; i < skipLayersCount; i++) {
                      skipLayersPtr[i] = layerIndices[i];
                    }
                    print(
                        "Isolate: Parsed skip_layers: ${layerIndices.join(', ')} (Count: $skipLayersCount)");
                  } else {
                    skipLayersPtr =
                        nullptr; // Ensure it's null if parsing results in empty list
                  }
                } catch (e) {
                  print(
                      "Isolate: Error parsing skip_layers '$skipLayersText': $e");
                  skipLayersPtr = nullptr; // Set to null on error
                  skipLayersCount = 0;
                  // Optionally send an error back? For now, just log and proceed without skip_layers.
                }
              } else {
                skipLayersPtr =
                    nullptr; // Explicitly null if text is null or empty
              }
              // --- End Prepare skip_layers ---

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
                message['clipSkip'], // Already passed
                message['cfgScale'], // Already passed
                message['guidance'], // New
                message['eta'], // New
                message['width'], // Already passed
                message['height'], // Already passed
                message['sampleMethod'], // Already passed
                message['sampleSteps'], // Already passed
                message['seed'], // Already passed
                message['batchCount'], // Already passed
                controlCondPtr ?? nullptr, // Already passed
                message['controlStrength'], // Already passed
                styleStrength, // Already passed
                false, // normalize_input (Already passed)
                inputIdImagesPathUtf8, // Already passed
                skipLayersPtr ?? nullptr, // New - Pass pointer or null
                skipLayersCount, // New - Pass count
                message['slgScale'], // New
                message['skipLayerStart'], // New
                message['skipLayerEnd'], // New
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
                print(
                    "Image generation failed in isolate (result address is 0)");
                mainSendPort.send({
                  'type': 'error',
                  'errorType': 'generationError', // Specific error type
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
                'errorType': 'generationError', // Specific error type
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
                  inputIdImagesPathUtf8.address != "".toNativeUtf8().address) {
                calloc.free(inputIdImagesPathUtf8); // Check against empty
              }
              if (controlCondPtr != null) {
                // Don't free controlCondPtr.ref.data here if it points to controlDataPtr
                malloc.free(controlCondPtr); // Free the SDImage struct itself
              }
              if (controlDataPtr != null) {
                malloc.free(controlDataPtr); // Free the image data buffer
              }
              if (skipLayersPtr != null && skipLayersPtr != nullptr) {
                malloc.free(skipLayersPtr); // Free the skip_layers array
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
    int clipSkip = 0, // Default changed to 0 as per UI
    double cfgScale = 7.0,
    double guidance = 3.5, // Default changed as per UI
    double eta = 0.0, // New parameter with default
    int width = 512,
    int height = 512,
    int sampleMethod = 0,
    int sampleSteps = 20,
    int seed = -1, // Default changed as per UI
    int batchCount = 1,
    String? inputIdImagesPath,
    double styleStrength = 1.0,
    Uint8List? controlImageData,
    int? controlImageWidth,
    int? controlImageHeight,
    double controlStrength = 0.9,
    double slgScale = 0.0, // New parameter with default
    String? skipLayersText, // New parameter (nullable string)
    double skipLayerStart = 0.01, // New parameter with default
    double skipLayerEnd = 0.2, // New parameter with default
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
        'clipSkip': clipSkip, // Already passed
        'cfgScale': cfgScale, // Already passed
        'guidance': guidance, // New
        'eta': eta, // New
        'width': width, // Already passed
        'height': height, // Already passed
        'sampleMethod': sampleMethod, // Already passed
        'sampleSteps': sampleSteps, // Already passed
        'seed': seed, // Already passed
        'batchCount': batchCount, // Already passed
        'inputIdImagesPath': inputIdImagesPath, // Already passed
        'styleStrength': styleStrength, // Already passed
        'controlImageData': controlImageData, // Already passed
        'controlImageWidth': controlImageWidth, // Already passed
        'controlImageHeight': controlImageHeight, // Already passed
        'controlStrength': controlStrength, // Already passed
        'slgScale': slgScale, // New
        'skipLayersText': skipLayersText, // New (pass the string)
        'skipLayerStart': skipLayerStart, // New
        'skipLayerEnd': skipLayerEnd, // New
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

  // Helper in main class to pass error details to the UI state handler
  void _handleLoadingError(String errorType, String errorMessage) {
    // This method needs access to the main UI's state update mechanism.
    // Since this class doesn't directly hold the state, we'll rely on the
    // message listening mechanism in main.dart to update the state.
    // We just ensure the error message is sent correctly from the isolate.
    // The actual state update happens in main.dart's _receivePort.listen.
    print("Passing error to main thread: $errorType - $errorMessage");
  }
}
