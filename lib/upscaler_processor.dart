import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'package:image/image.dart' as img;
import 'ffi_bindings.dart';
import 'sd_image.dart';

class UpscalerProcessor {
  final String modelPath;
  final int nThreads;
  final SDType wtype;
  late Isolate _isolate;
  late SendPort _sendPort;
  final ReceivePort _receivePort = ReceivePort();
  final StreamController<Uint8List> _imageController =
      StreamController<Uint8List>.broadcast();

  Stream<Uint8List> get imageStream => _imageController.stream;

  UpscalerProcessor({
    required this.modelPath,
    required this.nThreads,
    required this.wtype,
  }) {
    _initializeIsolate();
  }

  Future<void> _initializeIsolate() async {
    try {
      _isolate = await Isolate.spawn(
        _isolateEntryPoint,
        {
          'port': _receivePort.sendPort,
          'modelPath': modelPath,
          'nThreads': nThreads,
          'wtype': wtype.index,
        },
      );

      _receivePort.listen((message) {
        if (message is SendPort) {
          _sendPort = message;
        } else if (message is Map) {
          if (message['type'] == 'upscaled') {
            _imageController.add(message['data']);
          } else if (message['type'] == 'error') {
            _imageController.addError(Exception(message['message']));
          }
        }
      });
    } catch (e) {
      _imageController.addError(Exception('Failed to initialize upscaler: $e'));
    }
  }

  static void _isolateEntryPoint(Map<String, dynamic> args) {
    final SendPort mainSendPort = args['port'];
    final String modelPath = args['modelPath'];
    final int nThreads = args['nThreads'];
    final int wtype = args['wtype'];

    final ReceivePort isolateReceivePort = ReceivePort();
    mainSendPort.send(isolateReceivePort.sendPort);

    if (!File(modelPath).existsSync()) {
      mainSendPort.send(
          {'type': 'error', 'message': 'Model file not found: $modelPath'});
      return;
    }

    Pointer<Void> upscalerCtx = nullptr;

    try {
      final modelPathUtf8 = modelPath.toNativeUtf8();
      upscalerCtx = FFIBindings.newUpscalerCtx(modelPathUtf8, nThreads, wtype);
      malloc.free(modelPathUtf8);

      if (upscalerCtx.address == 0) {
        mainSendPort.send(
            {'type': 'error', 'message': 'Failed to load upscaler model'});
        return;
      }
    } catch (e) {
      mainSendPort.send({
        'type': 'error',
        'message': 'Error initializing upscaler context: $e'
      });
      return;
    }

    isolateReceivePort.listen((message) {
      if (message is Map) {
        switch (message['command']) {
          case 'upscale':
            try {
              final Uint8List inputData = message['data'];
              final int width = message['width'];
              final int height = message['height'];
              final int channel = message['channel'];
              final int upscaleFactor = message['upscaleFactor'];

              final inputDataPtr = malloc<Uint8>(inputData.length);
              final inputDataList = inputDataPtr.asTypedList(inputData.length);
              inputDataList.setAll(0, inputData);

              final inputImage = malloc<SDImage>();
              inputImage.ref
                ..width = width
                ..height = height
                ..channel = channel
                ..data = inputDataPtr;

              final upscaledImage = FFIBindings.upscale(
                  upscalerCtx, inputImage.ref, upscaleFactor);

              final outputWidth = upscaledImage.width;
              final outputHeight = upscaledImage.height;
              final outputChannel = upscaledImage.channel;
              final outputDataPtr = upscaledImage.data;
              final outputDataLength =
                  outputWidth * outputHeight * outputChannel;

              final outputBytes = Uint8List.fromList(
                  outputDataPtr.asTypedList(outputDataLength));

              final rgbaBuffer = Uint8List(outputWidth * outputHeight * 4);
              for (int i = 0; i < outputDataLength; i += 3) {
                final rgbaIndex = (i ~/ 3) * 4;
                rgbaBuffer[rgbaIndex] = outputBytes[i];
                rgbaBuffer[rgbaIndex + 1] = outputBytes[i + 1];
                rgbaBuffer[rgbaIndex + 2] = outputBytes[i + 2];
                rgbaBuffer[rgbaIndex + 3] = 255;
              }

              final image = img.Image.fromBytes(
                  width: outputWidth,
                  height: outputHeight,
                  bytes: rgbaBuffer.buffer,
                  numChannels: 4);
              final pngBytes = img.encodePng(image);

              mainSendPort.send({
                'type': 'upscaled',
                'data': pngBytes,
              });

              malloc.free(inputDataPtr);
              malloc.free(inputImage);
              malloc.free(outputDataPtr);
            } catch (e) {
              mainSendPort.send({
                'type': 'error',
                'message': 'Error during upscale operation: $e'
              });
            }
            break;

          case 'dispose':
            if (upscalerCtx.address != 0) {
              FFIBindings.freeUpscalerCtx(upscalerCtx);
            }
            isolateReceivePort.close();
            break;
        }
      }
    });
  }

  Future<void> upscaleImage({
    required Uint8List inputData,
    required int width,
    required int height,
    required int channel,
    required int upscaleFactor,
  }) async {
    _sendPort.send({
      'command': 'upscale',
      'data': inputData,
      'width': width,
      'height': height,
      'channel': channel,
      'upscaleFactor': upscaleFactor,
    });
  }

  void dispose() {
    _sendPort.send({'command': 'dispose'});
    _isolate.kill(priority: Isolate.immediate);
    _receivePort.close();
    _imageController.close();
  }
}
