import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'dart:math' as math;

class ProcessedImageData {
  final Uint8List bytes;
  final int width;
  final int height;

  ProcessedImageData(
      {required this.bytes, required this.width, required this.height});
}

/// Resizes the input image bytes to the target dimensions.
/// Assumes input bytes are raw RGB.
ProcessedImageData resizeImage(Uint8List imageBytes, int currentWidth,
    int currentHeight, int targetWidth, int targetHeight) {
  // Construct Image from raw RGB bytes
  // Ensure the input byte length matches expected RGB size
  if (imageBytes.length != currentWidth * currentHeight * 3) {
    throw Exception(
        "Input byte length (${imageBytes.length}) does not match expected RGB size (${currentWidth * currentHeight * 3}) for resizing.");
  }
  img.Image originalImage = img.Image.fromBytes(
    width: currentWidth,
    height: currentHeight,
    bytes: imageBytes.buffer, // Use the buffer
    numChannels: 3, // Assuming 3 channels (RGB)
    order: img.ChannelOrder.rgb, // Specify the order
  );

  // The image is constructed as RGB, no further checks needed here.

  // Resize the image
  final img.Image resizedImage = img.copyResize(
    originalImage,
    width: targetWidth,
    height: targetHeight,
    interpolation:
        img.Interpolation.average, // Or linear, cubic for different quality
  );

  // Get raw RGB bytes directly
  final Uint8List finalBytes =
      resizedImage.getBytes(order: img.ChannelOrder.rgb);

  return ProcessedImageData(
    bytes: finalBytes,
    width: targetWidth,
    height: targetHeight,
  );
}

/// Crops the input image bytes to the target dimensions using a center-crop strategy.
/// Assumes input bytes are raw RGB.
ProcessedImageData cropImage(Uint8List imageBytes, int currentWidth,
    int currentHeight, int targetWidth, int targetHeight) {
  // Construct Image from raw RGB bytes
  // Ensure the input byte length matches expected RGB size
  if (imageBytes.length != currentWidth * currentHeight * 3) {
    throw Exception(
        "Input byte length (${imageBytes.length}) does not match expected RGB size (${currentWidth * currentHeight * 3}) for cropping.");
  }
  img.Image originalImage = img.Image.fromBytes(
    width: currentWidth,
    height: currentHeight,
    bytes: imageBytes.buffer, // Use the buffer
    numChannels: 3, // Assuming 3 channels (RGB)
    order: img.ChannelOrder.rgb, // Specify the order
  );

  // The image is constructed as RGB, no further checks needed here.

  int cropX = 0;
  int cropY = 0;
  int cropWidth = currentWidth;
  int cropHeight = currentHeight;

  double currentAspect = currentWidth / currentHeight;
  double targetAspect = targetWidth / targetHeight;

  if (currentAspect > targetAspect) {
    // Current image is wider than target aspect ratio, crop width
    cropWidth = (currentHeight * targetAspect).round();
    cropX = ((currentWidth - cropWidth) / 2).round();
  } else if (currentAspect < targetAspect) {
    // Current image is taller than target aspect ratio, crop height
    cropHeight = (currentWidth / targetAspect).round();
    cropY = ((currentHeight - cropHeight) / 2).round();
  }
  // else: aspect ratios match, crop the whole image (handled by copyCrop dimensions)

  // Crop the image
  final img.Image croppedImage = img.copyCrop(
    originalImage,
    x: cropX,
    y: cropY,
    width: cropWidth,
    height: cropHeight,
  );

  // Resize the (potentially already cropped) image to the exact target size
  // This handles cases where the aspect ratio crop didn't result in the *exact* target size
  // due to rounding, or if the original was smaller than the target.
  final img.Image finalImage = img.copyResize(
    croppedImage,
    width: targetWidth,
    height: targetHeight,
    interpolation:
        img.Interpolation.average, // Use average for smoother resize after crop
  );

  // Get raw RGB bytes directly
  final Uint8List finalBytes = finalImage.getBytes(order: img.ChannelOrder.rgb);

  return ProcessedImageData(
    bytes: finalBytes,
    width: targetWidth,
    height: targetHeight,
  );
}
