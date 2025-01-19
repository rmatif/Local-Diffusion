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
  static String? taesdPath;
  static final _progressController =
      StreamController<ProgressUpdate>.broadcast();
  static final _logController = StreamController<LogMessage>.broadcast();
  static bool _useTinyAutoencoder = false;
  static int? lastUsedSeed;

  static Stream<ProgressUpdate> get progressStream =>
      _progressController.stream;
  static Stream<LogMessage> get logStream => _logController.stream;

  static bool setTinyAutoencoder(bool value) {
    if (value && taesdPath == null) {
      return false;
    }
    _useTinyAutoencoder = value;
    return true;
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
      return "TAESD model selected: ${taesdPath!.split('/').last}";
    } catch (e) {
      return "Error picking TAESD file: $e";
    }
  }

  static void dispose() {
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
