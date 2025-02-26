import 'dart:ffi';

base class SDImage extends Struct {
  @Uint32()
  external int width;

  @Uint32()
  external int height;

  @Uint32()
  external int channel;

  external Pointer<Uint8> data;

  external Pointer<Void> userdata;
}
