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
String? _lastGenerationTime; // Added to store the last generation time
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

  // --- Extract generation time ---
  if (message.contains("img2img completed in")) {
    // Check for img2img
    final timeMatch =
        RegExp(r'img2img completed in ([\d.]+)s').firstMatch(message);
    if (timeMatch != null) {
      _lastGenerationTime = "${timeMatch.group(1)!}s";
      print(
          "Isolate (Img2Img): Extracted generation time: $_lastGenerationTime");
      // Don't send this specific log, just store the time.
      return; // Prevent sending this log message itself
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
  final bool isDiffusionModelType; // Added flag for model type
  late Isolate _sdIsolate;
  late SendPort _sdSendPort;
  final Completer _uninitialized = Completer();
  final ReceivePort _receivePort = ReceivePort();
  // final StreamController<ui.Image> _imageController = StreamController<ui.Image>.broadcast(); // Replaced
  final StreamController<Map<String, dynamic>> _generationResultController =
      StreamController<
          Map<String, dynamic>>.broadcast(); // New controller for image + time
  final StreamController<List<String>> _logListController =
      StreamController<List<String>>.broadcast(); // Added for logs
  final _loadingController = StreamController<bool>.broadcast();

  Stream<bool> get loadingStream => _loadingController.stream;
  Stream<List<String>> get logListStream =>
      _logListController.stream; // Added getter for logs
  // Stream<ui.Image> get imageStream => _imageController.stream; // Replaced
  Stream<Map<String, dynamic>> get generationResultStream =>
      _generationResultController.stream; // Getter for the new stream

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
            'isDiffusionModelType': isDiffusionModelType, // Pass the flag
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
            final generationTime = message['generationTime']; // Extract time
            // Add both image and time to the new stream
            _generationResultController.add({
              'image': image,
              'generationTime': generationTime,
            });
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

            Pointer<Utf8>? modelPathUtf8;
            Pointer<Utf8>? diffusionModelPathUtf8; // Added
            Pointer<Utf8>? clipLPathUtf8;
            Pointer<Utf8>? clipGPathUtf8;
            Pointer<Utf8>? t5xxlPathUtf8;
            Pointer<Utf8>? vaePathUtf8;
            Pointer<Utf8>? loraDirUtf8;
            Pointer<Utf8>? taesdPathUtf8;
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
                    "Isolate (Img2Img): Using diffusion_model_path for $modelPathString");
              } else {
                modelPathUtf8 = modelPathString.toNativeUtf8();
                diffusionModelPathUtf8 =
                    emptyUtf8; // Pass empty for diffusion model path
                print(
                    "Isolate (Img2Img): Using model_path for $modelPathString");
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
              embedDirUtf8 =
                  message['embedDirPath']?.toString().toNativeUtf8() ??
                      emptyUtf8;
              controlNetPathUtf8 =
                  message['controlNetPath']?.toString().toNativeUtf8() ??
                      emptyUtf8;

              ctx = FFIBindings.newSdCtx(
                modelPathUtf8, // 1
                clipLPathUtf8, // 2
                clipGPathUtf8, // 3
                t5xxlPathUtf8, // 4
                diffusionModelPathUtf8, // 5
                vaePathUtf8, // 6
                taesdPathUtf8, // 7
                controlNetPathUtf8, // 8
                loraDirUtf8, // 9
                embedDirUtf8, // 10
                emptyUtf8, // 11 stacked_id_embed_dir_c_str (Assuming empty is ok here)
                false, // 12 vae_decode_only (Assuming false)
                message['vaeTiling'], // 13
                false, // 14 free_params_immediately
                FFIBindings.getCores() * 2, // 15 n_threads
                mapModelTypeToIndex(
                    SDType.values[message['modelType']]), // 16 wtype
                0, // 17 rng_type (STD_DEFAULT_RNG)
                message['schedule'], // 18 schedule
                false, // 19 keep_clip_on_cpu
                false, // 20 keep_control_net_cpu
                false, // 21 keep_vae_on_cpu
                message['useFlashAttention'], // 22 diffusion_flash_attn
              );

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
              if (embedDirUtf8 != null &&
                  embedDirUtf8.address != emptyUtf8?.address)
                calloc.free(embedDirUtf8);
              if (controlNetPathUtf8 != null &&
                  controlNetPathUtf8.address != emptyUtf8?.address)
                calloc.free(controlNetPathUtf8);
              // Free the reusable empty string pointer *once* at the end
              if (emptyUtf8 != null) calloc.free(emptyUtf8);

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
            _lastGenerationTime = null; // Reset time before generation
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
                    ..channel = 3 // Assuming RGB for control image
                    ..data = controlDataPtr!; // Add null check
                }

                // --- Prepare skip_layers (Copied and modified from stable_diffusion_processor) ---
                Pointer<Int32>? skipLayersPtr; // Pointer for skip_layers array
                int skipLayersCount = 0; // Count for skip_layers
                final String? skipLayersText = message['skipLayersText'];
                if (skipLayersText != null && skipLayersText.isNotEmpty) {
                  // Existing logic to parse skipLayersText
                  try {
                    // Parse the string "[num1,num2,...]" or "num1,num2,..."
                    String numbersString = skipLayersText;
                    if (numbersString.startsWith('[') &&
                        numbersString.endsWith(']')) {
                      numbersString =
                          numbersString.substring(1, numbersString.length - 1);
                    }
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
                          "Isolate (Img2Img): Parsed skip_layers: ${layerIndices.join(', ')} (Count: $skipLayersCount)");
                    } else {
                      // Parsing resulted in empty list, use default
                      print(
                          "Isolate (Img2Img): Parsed skip_layers resulted in empty list, using default [7, 8, 9]");
                      skipLayersCount = 3;
                      skipLayersPtr = malloc<Int32>(skipLayersCount);
                      skipLayersPtr[0] = 7;
                      skipLayersPtr[1] = 8;
                      skipLayersPtr[2] = 9;
                    }
                  } catch (e) {
                    print(
                        "Isolate (Img2Img): Error parsing skip_layers '$skipLayersText': $e. Using default [7, 8, 9]");
                    // Error parsing, use default
                    skipLayersCount = 3;
                    skipLayersPtr = malloc<Int32>(skipLayersCount);
                    skipLayersPtr[0] = 7;
                    skipLayersPtr[1] = 8;
                    skipLayersPtr[2] = 9;
                  }
                } else {
                  // skipLayersText is null or empty, use default
                  print(
                      "Isolate (Img2Img): skipLayersText is null or empty, using default [7, 8, 9]");
                  skipLayersCount = 3;
                  skipLayersPtr = malloc<Int32>(skipLayersCount);
                  skipLayersPtr[0] = 7;
                  skipLayersPtr[1] = 8;
                  skipLayersPtr[2] = 9;
                }
                // --- End Prepare skip_layers ---

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
                  message['cfgScale'], // Already passed
                  message['guidance'], // New
                  message['eta'], // New
                  outputWidth, // Already passed
                  outputHeight, // Already passed
                  message['sampleMethod'], // Already passed
                  message['sampleSteps'], // Already passed
                  message['strength'], // Already passed
                  message['seed'], // Already passed
                  message['batchCount'], // Already passed
                  controlCondPtr ?? nullptr, // Already passed
                  message['controlStrength'] ?? 0.0, // Already passed
                  0.0, // style_strength (Not exposed in UI yet)
                  false, // normalize_input (Already passed)
                  emptyUtf8, // input_id_images_path (Not exposed in UI yet)
                  skipLayersPtr ?? nullptr, // New - Pass pointer or null
                  skipLayersCount, // New - Pass count
                  message['slgScale'], // New
                  message['skipLayerStart'], // New
                  message['skipLayerEnd'], // New
                );

                calloc.free(initImageDataPtr);
                malloc.free(initImage);
                malloc.free(maskImage.ref.data);
                malloc.free(maskImage);
                calloc.free(promptUtf8);
                calloc.free(negPromptUtf8);
                calloc.free(emptyUtf8);
                // Free skip_layers pointer if allocated
                if (skipLayersPtr != null && skipLayersPtr != nullptr) {
                  malloc.free(skipLayersPtr);
                }

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
                    'generationTime': _lastGenerationTime, // Include the time
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
    double guidance = 1.0, // Already passed
    double eta = 0.0, // Already passed
    int sampleMethod = 0, // Already passed
    int sampleSteps = 20, // Already passed
    double strength = 0.5, // Already passed
    int seed = 42, // Already passed
    int batchCount = 1, // Already passed
    Uint8List? controlImageData, // Already passed
    int? controlImageWidth, // Already passed
    int? controlImageHeight, // Already passed
    double controlStrength = 0.9, // Already passed
    Uint8List? maskImageData, // Already passed
    int? maskWidth, // Already passed
    int? maskHeight, // Already passed
    // New parameters from advanced sampling options
    double slgScale = 0.0,
    String? skipLayersText,
    double skipLayerStart = 0.01,
    double skipLayerEnd = 0.2,
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
        'guidance': guidance, // Already passed
        'eta': eta, // Already passed
        'sampleMethod': sampleMethod, // Already passed
        'sampleSteps': sampleSteps, // Already passed
        'strength': strength, // Already passed
        'seed': seed, // Already passed
        'batchCount': batchCount, // Already passed
        'controlImageData': controlImageData, // Already passed
        'controlImageWidth': controlImageWidth, // Already passed
        'controlImageHeight': controlImageHeight, // Already passed
        'controlStrength': controlStrength, // Already passed
        if (maskImageData != null)
          'maskImageData': maskImageData, // Already passed
        if (maskWidth != null) 'maskImageWidth': maskWidth, // Already passed
        if (maskHeight != null) 'maskImageHeight': maskHeight, // Already passed
        // Pass new parameters to isolate
        'slgScale': slgScale,
        'skipLayersText': skipLayersText,
        'skipLayerStart': skipLayerStart,
        'skipLayerEnd': skipLayerEnd,
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
    // _imageController.close(); // Replaced
    _generationResultController.close(); // Close the new result controller
    _logListController.close(); // Close the new controller
  }
}
