import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'dart:io' show Platform;
import 'sd_image.dart';

typedef LogCallbackNative = Void Function(
    Int32 level, Pointer<Utf8> text, Pointer<Void> data);
typedef LogCallback = void Function(
    int level, Pointer<Utf8> text, Pointer<Void> data);

typedef ProgressCallbackNative = Void Function(
    Int32 step, Int32 steps, Float time, Pointer<Void> data);
typedef ProgressCallback = void Function(
    int step, int steps, double time, Pointer<Void> data);

enum SDType {
  NONE, // No quantization
  SD_TYPE_Q8_0,
  SD_TYPE_Q8_1,
  SD_TYPE_Q8_K,
  SD_TYPE_Q6_K,
  SD_TYPE_Q5_0,
  SD_TYPE_Q5_1,
  SD_TYPE_Q5_K,
  SD_TYPE_Q4_0,
  SD_TYPE_Q4_1,
  SD_TYPE_Q4_K,
  SD_TYPE_Q3_K,
  SD_TYPE_Q2_K,
}

extension SDTypeExtension on SDType {
  String get displayName {
    switch (this) {
      case SDType.NONE:
        return 'None';
      case SDType.SD_TYPE_Q8_0:
        return 'Q8_0';
      case SDType.SD_TYPE_Q8_1:
        return 'Q8_1';
      case SDType.SD_TYPE_Q8_K:
        return 'Q8_K';
      case SDType.SD_TYPE_Q6_K:
        return 'Q6_K';
      case SDType.SD_TYPE_Q5_0:
        return 'Q5_0';
      case SDType.SD_TYPE_Q5_1:
        return 'Q5_1';
      case SDType.SD_TYPE_Q5_K:
        return 'Q5_K';
      case SDType.SD_TYPE_Q4_0:
        return 'Q4_0';
      case SDType.SD_TYPE_Q4_1:
        return 'Q4_1';
      case SDType.SD_TYPE_Q4_K:
        return 'Q4_K';
      case SDType.SD_TYPE_Q3_K:
        return 'Q3_K';
      case SDType.SD_TYPE_Q2_K:
        return 'Q2_K';
    }
  }
}

class FFIBindings {
  static final DynamicLibrary _lib = _loadLibrary();

  static DynamicLibrary _loadLibrary() {
    if (Platform.isAndroid) {
      return DynamicLibrary.open("libstable-diffusion.so");
    }
    return DynamicLibrary.process();
  }

  static final getCores = _lib.lookupFunction<Int32 Function(), int Function()>(
      'get_num_physical_cores',
      isLeaf: true);

  static final setLogCallback = _lib.lookupFunction<
      Void Function(Pointer<NativeFunction<LogCallbackNative>>, Pointer<Void>),
      void Function(Pointer<NativeFunction<LogCallbackNative>>,
          Pointer<Void>)>('sd_set_log_callback');

  static final setProgressCallback = _lib.lookupFunction<
      Void Function(
          Pointer<NativeFunction<ProgressCallbackNative>>, Pointer<Void>),
      void Function(Pointer<NativeFunction<ProgressCallbackNative>>,
          Pointer<Void>)>('sd_set_progress_callback');

  static final newSdCtx = _lib.lookupFunction<
      Pointer<Void> Function(
          Pointer<Utf8>,
          Pointer<Utf8>,
          Pointer<Utf8>,
          Pointer<Utf8>,
          Pointer<Utf8>,
          Pointer<Utf8>,
          Pointer<Utf8>,
          Pointer<Utf8>,
          Pointer<Utf8>,
          Pointer<Utf8>,
          Pointer<Utf8>,
          Bool,
          Bool,
          Bool,
          Int32,
          Int32,
          Int32,
          Int32,
          Bool,
          Bool,
          Bool),
      Pointer<Void> Function(
          Pointer<Utf8>,
          Pointer<Utf8>,
          Pointer<Utf8>,
          Pointer<Utf8>,
          Pointer<Utf8>,
          Pointer<Utf8>,
          Pointer<Utf8>,
          Pointer<Utf8>,
          Pointer<Utf8>,
          Pointer<Utf8>,
          Pointer<Utf8>,
          bool,
          bool,
          bool,
          int,
          int,
          int,
          int,
          bool,
          bool,
          bool)>('new_sd_ctx', isLeaf: true);

  static final freeSdCtx = _lib.lookupFunction<Void Function(Pointer<Void>),
      void Function(Pointer<Void>)>('free_sd_ctx', isLeaf: true);

  static final txt2img = _lib.lookupFunction<
      Pointer<SDImage> Function(
          Pointer<Void>,
          Pointer<Utf8>,
          Pointer<Utf8>,
          Int32,
          Float,
          Float,
          Int32,
          Int32,
          Int32,
          Int32,
          Int64,
          Int32,
          Pointer<Void>,
          Float,
          Float,
          Bool,
          Pointer<Utf8>),
      Pointer<SDImage> Function(
          Pointer<Void>,
          Pointer<Utf8>,
          Pointer<Utf8>,
          int,
          double,
          double,
          int,
          int,
          int,
          int,
          int,
          int,
          Pointer<Void>,
          double,
          double,
          bool,
          Pointer<Utf8>)>('txt2img', isLeaf: true);
}
