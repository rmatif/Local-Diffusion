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

enum SampleMethod {
  EULER_A,
  EULER,
  HEUN,
  DPM2,
  DPMPP2S_A,
  DPMPP2M,
  DPMPP2Mv2,
  IPNDM,
  IPNDM_V,
  LCM,
  DDIM_TRAILING, // New sampler
  TCD // New sampler
}

enum Schedule { DEFAULT, DISCRETE, KARRAS, EXPONENTIAL, AYS, GITS }

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

extension SampleMethodExtension on SampleMethod {
  String get displayName {
    return toString().split('.').last;
  }
}

extension ScheduleExtension on Schedule {
  String get displayName {
    return toString().split('.').last;
  }
}

class FFIBindings {
  static DynamicLibrary? _lib;
  static String _currentBackend = 'CPU'; // Default backend

  // Getter for the currently loaded backend
  static String getCurrentBackend() => _currentBackend;

  // Function to get the library name based on the backend
  static String _getLibraryName(String backend) {
    switch (backend) {
      case 'Vulkan':
        return "libstable-diffusion_vulkan.so";
      case 'OpenCL':
        return "libstable-diffusion_opencl.so";
      case 'CPU':
      default:
        return "libstable-diffusion.so";
    }
  }

  // Modified load library function
  static void _loadLibrary(String backend) {
    if (Platform.isAndroid) {
      final libraryName = _getLibraryName(backend);
      print("Attempting to load library: $libraryName");
      try {
        _lib = DynamicLibrary.open(libraryName);
        _currentBackend = backend;
        print("Successfully loaded: $libraryName");
      } catch (e) {
        print("Error loading library $libraryName: $e");
        // Fallback or handle error appropriately
        // Maybe try loading the default CPU library if the specific one fails?
        if (backend != 'CPU') {
          print("Falling back to CPU library.");
          _loadLibrary('CPU'); // Try loading default
        } else {
          // If even CPU fails, rethrow or handle as critical error
          rethrow;
        }
      }
    } else {
      // Handle other platforms if necessary, default to process()
      _lib = DynamicLibrary.process();
      _currentBackend = 'Process'; // Indicate process library used
      print(
          "Loaded library via DynamicLibrary.process() for non-Android platform.");
    }
  }

  // Public method to initialize or re-initialize bindings
  static void initializeBindings(String backend) {
    // Consider closing the old library if it exists and is supported
    // DynamicLibrary doesn't have a close method in dart:ffi standard
    _loadLibrary(backend);
    _lookupFunctions(); // Re-lookup functions with the new library
  }

  // REMOVED _initializeOnLoad - Initialization must be explicit via initializeBindings.

  // --- Function Lookups (late variables with specific types for reassignment) ---
  static late int Function() getCores;
  static late void Function(
      Pointer<NativeFunction<LogCallbackNative>>, Pointer<Void>) setLogCallback;
  static late void Function(
          Pointer<NativeFunction<ProgressCallbackNative>>, Pointer<Void>)
      setProgressCallback;
  static late Pointer<Void> Function(
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
      bool,
      bool) newSdCtx;
  static late void Function(Pointer<Void>) freeSdCtx;
  static late Pointer<Void> Function(Pointer<Utf8>, int, int) newUpscalerCtx;
  static late void Function(Pointer<Void>) freeUpscalerCtx;
  static late SDImage Function(Pointer<Void>, SDImage, int) upscale;
  static late Pointer<SDImage> Function(
      Pointer<Void>,
      Pointer<Utf8>,
      Pointer<Utf8>,
      int,
      double,
      double,
      double,
      int,
      int,
      int,
      int,
      int,
      int,
      Pointer<SDImage>,
      double,
      double,
      bool,
      Pointer<Utf8>,
      Pointer<Int32>,
      int,
      double,
      double,
      double) txt2img;
  static late Pointer<Uint8> Function(
          Pointer<Uint8>, int, int, double, double, double, double, bool)
      preprocessCanny;
  static late Pointer<SDImage> Function(
      Pointer<Void>,
      SDImage,
      SDImage,
      Pointer<Utf8>,
      Pointer<Utf8>,
      int,
      double,
      double,
      double,
      int,
      int,
      int,
      int,
      double,
      int,
      int,
      Pointer<SDImage>,
      double,
      double,
      bool,
      Pointer<Utf8>,
      Pointer<Int32>,
      int,
      double,
      double,
      double) img2img;

  // Method to perform all function lookups
  static void _lookupFunctions() {
    if (_lib == null) {
      throw StateError(
          "FFI library not loaded. Call initializeBindings first.");
    }

    getCores = _lib!.lookupFunction<Int32 Function(), int Function()>(
        'get_num_physical_cores',
        isLeaf: true);
    setLogCallback = _lib!.lookupFunction<
        Void Function(
            Pointer<NativeFunction<LogCallbackNative>>, Pointer<Void>),
        void Function(Pointer<NativeFunction<LogCallbackNative>>,
            Pointer<Void>)>('sd_set_log_callback');
    setProgressCallback = _lib!.lookupFunction<
        Void Function(
            Pointer<NativeFunction<ProgressCallbackNative>>, Pointer<Void>),
        void Function(Pointer<NativeFunction<ProgressCallbackNative>>,
            Pointer<Void>)>('sd_set_progress_callback');
    newSdCtx = _lib!.lookupFunction<
            Pointer<Void> Function(
                Pointer<Utf8>, // 1 model_path
                Pointer<Utf8>, // 2 clip_l_path
                Pointer<Utf8>, // 3 clip_g_path
                Pointer<Utf8>, // 4 t5xxl_path
                Pointer<Utf8>, // 5 diffusion_model_path
                Pointer<Utf8>, // 6 vae_path
                Pointer<Utf8>, // 7 taesd_path
                Pointer<Utf8>, // 8 control_net_path_c_str
                Pointer<Utf8>, // 9 lora_model_dir
                Pointer<Utf8>, // 10 embed_dir_c_str
                Pointer<Utf8>, // 11 stacked_id_embed_dir_c_str
                Bool, // 12 vae_decode_only
                Bool, // 13 vae_tiling
                Bool, // 14 free_params_immediately
                Int32, // 15 n_threads
                Int32, // 16 wtype (maps to sd_type_t enum)
                Int32, // 17 rng_type (maps to rng_type_t enum)
                Int32, // 18 s (maps to schedule_t enum)
                Bool, // 19 keep_clip_on_cpu
                Bool, // 20 keep_control_net_cpu
                Bool, // 21 keep_vae_on_cpu
                Bool), // 22 diffusion_flash_attn
            Pointer<Void> Function(
                Pointer<Utf8>, // 1
                Pointer<Utf8>, // 2
                Pointer<Utf8>, // 3
                Pointer<Utf8>, // 4
                Pointer<Utf8>, // 5
                Pointer<Utf8>, // 6
                Pointer<Utf8>, // 7
                Pointer<Utf8>, // 8
                Pointer<Utf8>, // 9
                Pointer<Utf8>, // 10
                Pointer<Utf8>, // 11
                bool, // 12
                bool, // 13
                bool, // 14
                int, // 15
                int, // 16
                int, // 17
                int, // 18
                bool, // 19
                bool, // 20
                bool, // 21
                bool)> // 22
        ('new_sd_ctx', isLeaf: false);

    freeSdCtx = _lib!.lookupFunction<Void Function(Pointer<Void>),
        void Function(Pointer<Void>)>('free_sd_ctx', isLeaf: false);
    newUpscalerCtx = _lib!.lookupFunction<
        Pointer<Void> Function(Pointer<Utf8>, Int32, Int32),
        Pointer<Void> Function(
            Pointer<Utf8>, int, int)>('new_upscaler_ctx', isLeaf: false);
    freeUpscalerCtx = _lib!.lookupFunction<Void Function(Pointer<Void>),
        void Function(Pointer<Void>)>('free_upscaler_ctx', isLeaf: false);
    upscale = _lib!.lookupFunction<
        SDImage Function(Pointer<Void>, SDImage, Uint32),
        SDImage Function(
            Pointer<Void>, SDImage, int)>('upscale', isLeaf: false);
    // Around line 135 in lib/ffi_bindings.dart (before txt2img binding)
    txt2img = _lib!.lookupFunction<
            Pointer<SDImage> Function(
                Pointer<Void>,
                Pointer<Utf8>,
                Pointer<Utf8>,
                Int32,
                Float,
                Float,
                Float, // eta (new parameter)
                Int32,
                Int32,
                Int32,
                Int32,
                Int64,
                Int32,
                Pointer<SDImage>,
                Float,
                Float,
                Bool,
                Pointer<Utf8>,
                Pointer<Int32>, // skip_layers (new)
                IntPtr, // skip_layers_count (new)
                Float, // slg_scale (new)
                Float, // skip_layer_start (new)
                Float), // skip_layer_end (new)
            Pointer<SDImage> Function(
                Pointer<Void>,
                Pointer<Utf8>,
                Pointer<Utf8>,
                int,
                double,
                double,
                double, // eta (new parameter)
                int,
                int,
                int,
                int,
                int,
                int,
                Pointer<SDImage>,
                double,
                double,
                bool,
                Pointer<Utf8>,
                Pointer<Int32>, // skip_layers (new)
                int, // skip_layers_count (new)
                double, // slg_scale (new)
                double, // skip_layer_start (new)
                double)> // skip_layer_end (new)
        ('txt2img', isLeaf: false);

// Add preprocess_canny binding around line 200 (after upscale binding)
    preprocessCanny = _lib!.lookupFunction<
        Pointer<Uint8> Function(
            Pointer<Uint8>, Int32, Int32, Float, Float, Float, Float, Bool),
        Pointer<Uint8> Function(Pointer<Uint8>, int, int, double, double,
            double, double, bool)>('preprocess_canny', isLeaf: false);
    img2img = _lib!.lookupFunction<
            Pointer<SDImage> Function(
                Pointer<Void>,
                SDImage,
                SDImage, // mask_image (new parameter)
                Pointer<Utf8>,
                Pointer<Utf8>,
                Int32,
                Float,
                Float,
                Float, // eta (new parameter)
                Int32,
                Int32,
                Int32,
                Int32,
                Float,
                Int64,
                Int32,
                Pointer<SDImage>,
                Float,
                Float,
                Bool,
                Pointer<Utf8>,
                Pointer<Int32>, // skip_layers (new)
                IntPtr, // skip_layers_count (new)
                Float, // slg_scale (new)
                Float, // skip_layer_start (new)
                Float), // skip_layer_end (new)
            Pointer<SDImage> Function(
                Pointer<Void>,
                SDImage,
                SDImage, // mask_image (new parameter)
                Pointer<Utf8>,
                Pointer<Utf8>,
                int,
                double,
                double,
                double, // eta (new parameter)
                int,
                int,
                int,
                int,
                double,
                int,
                int,
                Pointer<SDImage>,
                double,
                double,
                bool,
                Pointer<Utf8>,
                Pointer<Int32>, // skip_layers (new)
                int, // skip_layers_count (new)
                double, // slg_scale (new)
                double, // skip_layer_start (new)
                double)> // skip_layer_end (new)
        ('img2img', isLeaf: false);
  }
}

// Ensure bindings are initialized on load
// This needs to be called before any FFI function is accessed.
// A good place might be at the start of your main() function in main.dart
// or ensure FFIBindings._initializeOnLoad() is called implicitly.
// For simplicity, let's call it here, but be mindful of execution order.
// FFIBindings._initializeOnLoad(); // Removed direct call here, should be called explicitly from main.dart
