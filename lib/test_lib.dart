import 'dart:ffi';
import 'dart:io' show Platform;

class TestLib {
  static final DynamicLibrary _lib = Platform.isAndroid
      ? DynamicLibrary.open("libtest.so")
      : Platform.isWindows
          ? DynamicLibrary.open("test.dll")
          : DynamicLibrary.open("./libtest.so");

  static int getTestValue() {
    final getTest =
        _lib.lookupFunction<Int32 Function(), int Function()>('get_test_value');
    return getTest();
  }
}
