import 'dart:async';
import 'dart:isolate';
import 'dart:ui' as ui;
import 'dart:typed_data';
import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'package:image/image.dart' as img;
import 'ffi_bindings.dart';

class CannyParameters {
  final double highThreshold;
  final double lowThreshold;
  final double weak;
  final double strong;
  final bool inverse;

  CannyParameters({
    this.highThreshold = 100.0,
    this.lowThreshold = 50.0,
    this.weak = 1.0,
    this.strong = 255.0,
    this.inverse = false,
  });
}

class CannyProcessor {
  Isolate? _cannyIsolate;
  ReceivePort? _receivePort;
  SendPort? _sendPort;
  final _imageController = StreamController<ui.Image>.broadcast();
  final _loadingController = StreamController<bool>.broadcast();
  final _resultDataController = StreamController<Uint8List>.broadcast();
  Uint8List? _resultRgbBytes;
  int? _resultWidth;
  int? _resultHeight;
  final Completer<void> _initialized = Completer<void>();

  Stream<ui.Image> get imageStream => _imageController.stream;
  Stream<bool> get loadingStream => _loadingController.stream;
  Stream<Uint8List> get resultDataStream => _resultDataController.stream;

  Uint8List? get resultRgbBytes => _resultRgbBytes;
  int? get resultWidth => _resultWidth;
  int? get resultHeight => _resultHeight;

  Future<void> init() async {
    if (_initialized.isCompleted) return;

    _receivePort = ReceivePort();
    _cannyIsolate = await Isolate.spawn(
      _isolateEntryPoint,
      _receivePort!.sendPort,
    );

    // Set up a single listener for all messages
    _receivePort!.listen((message) {
      if (message is SendPort) {
        // First message is the SendPort
        _sendPort = message;
        _initialized.complete();
      } else if (message is Map) {
        if (message['type'] == 'cannyResult') {
          final bytes = message['bytes'] as Uint8List;
          final width = message['width'] as int;
          final height = message['height'] as int;

          // Store the processed image bytes for later use with ControlNet
          _resultRgbBytes = bytes;
          _resultWidth = width;
          _resultHeight = height;

          // Send to result data stream
          _resultDataController.add(bytes);

          // Create a UI image for display
          _createUIImage(bytes, width, height);

          // Notify that loading is complete
          _loadingController.add(false);
        } else if (message['type'] == 'error') {
          print("Canny error: ${message['message']}");
          _loadingController.add(false);
        }
      }
    });
  }

  Future<void> _createUIImage(Uint8List bytes, int width, int height) async {
    // Convert to RGBA for UI
    final rgbaBytes = Uint8List(width * height * 4);
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

    final image = await completer.future;
    _imageController.add(image);
  }

  Future<void> processImage(
    Uint8List imageData,
    int width,
    int height,
    CannyParameters params,
  ) async {
    if (!_initialized.isCompleted) {
      await init();
      await _initialized.future;
    }

    _loadingController.add(true);

    _sendPort!.send({
      'command': 'processCanny',
      'imageData': imageData,
      'width': width,
      'height': height,
      'highThreshold': params.highThreshold,
      'lowThreshold': params.lowThreshold,
      'weak': params.weak,
      'strong': params.strong,
      'inverse': params.inverse,
    });
  }

  static void _isolateEntryPoint(SendPort sendPort) {
    final receivePort = ReceivePort();
    sendPort.send(receivePort.sendPort);

    receivePort.listen((message) {
      if (message is Map && message['command'] == 'processCanny') {
        try {
          final imageData = message['imageData'] as Uint8List;
          final width = message['width'] as int;
          final height = message['height'] as int;
          final highThreshold = message['highThreshold'] as double;
          final lowThreshold = message['lowThreshold'] as double;
          final weak = message['weak'] as double;
          final strong = message['strong'] as double;
          final inverse = message['inverse'] as bool;

          // Allocate memory for the input image
          final inputPtr = malloc<Uint8>(imageData.length);
          inputPtr.asTypedList(imageData.length).setAll(0, imageData);

          // Call the Canny preprocessing function
          final resultPtr = FFIBindings.preprocessCanny(
            inputPtr,
            width,
            height,
            highThreshold,
            lowThreshold,
            weak,
            strong,
            inverse,
          );

          // Copy the result to a Dart list
          final resultBytes = resultPtr.asTypedList(width * height * 3);
          final dartResult = Uint8List(width * height * 3);
          dartResult.setAll(0, resultBytes);

          // Free the input memory
          malloc.free(inputPtr);

          // Send the result back to the main isolate
          sendPort.send({
            'type': 'cannyResult',
            'bytes': dartResult,
            'width': width,
            'height': height,
          });

          // Don't free resultPtr here, it's managed by the C++ code
        } catch (e) {
          sendPort.send({
            'type': 'error',
            'message': e.toString(),
          });
        }
      }
    });
  }

  void dispose() {
    _cannyIsolate?.kill();
    _receivePort?.close();
    _imageController.close();
    _loadingController.close();
    _resultDataController.close();
  }
}
