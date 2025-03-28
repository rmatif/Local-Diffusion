import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:shadcn_ui/shadcn_ui.dart';

// Helper function to find the largest multiple of 64 <= value
int largestMultipleOf64(int value) {
  if (value < 64) return 64; // Minimum size
  return (value ~/ 64) * 64;
}

class CroppedImageData {
  final Uint8List imageBytes;
  final Uint8List? maskBytes; // Cropped mask, if applicable
  final int width;
  final int height;

  CroppedImageData({
    required this.imageBytes,
    this.maskBytes,
    required this.width,
    required this.height,
  });
}

class ImageCropper extends StatefulWidget {
  final File imageFile;
  final Uint8List? initialMaskData; // Optional: Pass mask data to crop as well
  final int initialWidth;
  final int initialHeight;
  final Function(CroppedImageData) onCropComplete;

  const ImageCropper({
    super.key,
    required this.imageFile,
    this.initialMaskData,
    required this.initialWidth,
    required this.initialHeight,
    required this.onCropComplete,
  });

  @override
  State<ImageCropper> createState() => _ImageCropperState();
}

class _ImageCropperState extends State<ImageCropper> {
  late int _maxCropWidth;
  late int _maxCropHeight;
  late int _currentCropWidth;
  late int _currentCropHeight;

  // State for the cropping frame
  Rect _cropRect = Rect.zero; // Position relative to the displayed image widget
  Size _imageDisplaySize = Size.zero; // Size of the image widget as displayed
  final GlobalKey _imageKey = GlobalKey();

  // Offset for dragging the crop rectangle
  Offset _dragStartOffset = Offset.zero;
  Rect _dragStartCropRect = Rect.zero;

  @override
  void initState() {
    super.initState();
    _maxCropWidth = largestMultipleOf64(widget.initialWidth);
    _maxCropHeight = largestMultipleOf64(widget.initialHeight);

    // Initialize crop size, maintaining aspect ratio if possible, else default
    double initialAspectRatio = widget.initialWidth / widget.initialHeight;
    if (initialAspectRatio >= 1) {
      // Wider or square
      _currentCropWidth = math.min(512, _maxCropWidth);
      _currentCropHeight =
          largestMultipleOf64((_currentCropWidth / initialAspectRatio).round());
      _currentCropHeight = math.min(_currentCropHeight,
          _maxCropHeight); // Ensure height doesn't exceed max
      _currentCropWidth = largestMultipleOf64(
          (_currentCropHeight * initialAspectRatio)
              .round()); // Recalculate width based on clamped height
      _currentCropWidth = math.min(
          _currentCropWidth, _maxCropWidth); // Ensure width doesn't exceed max
    } else {
      // Taller
      _currentCropHeight = math.min(512, _maxCropHeight);
      _currentCropWidth = largestMultipleOf64(
          (_currentCropHeight * initialAspectRatio).round());
      _currentCropWidth = math.min(
          _currentCropWidth, _maxCropWidth); // Ensure width doesn't exceed max
      _currentCropHeight = largestMultipleOf64(
          (_currentCropWidth / initialAspectRatio)
              .round()); // Recalculate height based on clamped width
      _currentCropHeight = math.min(_currentCropHeight,
          _maxCropHeight); // Ensure height doesn't exceed max
    }

    // Ensure minimum size
    _currentCropWidth = math.max(64, _currentCropWidth);
    _currentCropHeight = math.max(64, _currentCropHeight);

    // Defer calculating initial cropRect until after the first frame
    WidgetsBinding.instance.addPostFrameCallback(_calculateInitialCropRect);
  }

  void _calculateInitialCropRect(_) {
    if (!mounted) return;
    final RenderBox? imageBox =
        _imageKey.currentContext?.findRenderObject() as RenderBox?;
    if (imageBox != null && imageBox.hasSize) {
      setState(() {
        _imageDisplaySize = imageBox.size;
        _updateCropRect(center: true); // Center the initial crop rect
      });
    } else {
      // Retry if the size wasn't available yet
      WidgetsBinding.instance.addPostFrameCallback(_calculateInitialCropRect);
    }
  }

  // Updates the _cropRect based on current dimensions and optionally centers it
  void _updateCropRect({bool center = false}) {
    if (_imageDisplaySize == Size.zero) return; // Not ready yet

    double displayAspect = _imageDisplaySize.width / _imageDisplaySize.height;
    double cropAspect = _currentCropWidth / _currentCropHeight;

    double rectWidth, rectHeight;

    // Calculate the dimensions of the crop rectangle within the image display bounds
    if (displayAspect > cropAspect) {
      // Image display is wider than crop aspect ratio
      rectHeight = _imageDisplaySize.height;
      rectWidth = rectHeight * cropAspect;
    } else {
      // Image display is taller or same aspect ratio
      rectWidth = _imageDisplaySize.width;
      rectHeight = rectWidth / cropAspect;
    }

    double left, top;

    if (center) {
      left = (_imageDisplaySize.width - rectWidth) / 2;
      top = (_imageDisplaySize.height - rectHeight) / 2;
    } else {
      // Keep the current center if possible, otherwise clamp to bounds
      double currentCenterX = _cropRect.left + _cropRect.width / 2;
      double currentCenterY = _cropRect.top + _cropRect.height / 2;

      left = currentCenterX - rectWidth / 2;
      top = currentCenterY - rectHeight / 2;
    }

    // Clamp the rectangle to the image bounds
    left = left.clamp(0.0, _imageDisplaySize.width - rectWidth);
    top = top.clamp(0.0, _imageDisplaySize.height - rectHeight);

    _cropRect = Rect.fromLTWH(left, top, rectWidth, rectHeight);
  }

  void _onPanStart(DragStartDetails details) {
    if (_cropRect.contains(details.localPosition)) {
      _dragStartOffset = details.localPosition;
      _dragStartCropRect = _cropRect;
    } else {
      _dragStartOffset = Offset.zero; // Indicate drag didn't start inside
    }
  }

  void _onPanUpdate(DragUpdateDetails details) {
    if (_dragStartOffset == Offset.zero) return; // Drag didn't start inside

    final Offset delta = details.localPosition - _dragStartOffset;
    double newLeft = _dragStartCropRect.left + delta.dx;
    double newTop = _dragStartCropRect.top + delta.dy;

    // Clamp movement within the image bounds
    newLeft = newLeft.clamp(0.0, _imageDisplaySize.width - _cropRect.width);
    newTop = newTop.clamp(0.0, _imageDisplaySize.height - _cropRect.height);

    setState(() {
      _cropRect =
          Rect.fromLTWH(newLeft, newTop, _cropRect.width, _cropRect.height);
    });
  }

  Future<void> _performCrop() async {
    // 1. Read original image bytes
    final Uint8List imageBytes = await widget.imageFile.readAsBytes();
    final img.Image? originalImage = img.decodeImage(imageBytes);

    img.Image? originalMask;
    if (widget.initialMaskData != null) {
      // Assuming mask data is grayscale (1 byte per pixel)
      // We need to convert it to an img.Image, potentially creating an RGBA image first
      // then converting to grayscale if the img library requires it.
      // For simplicity, let's assume the mask is loaded correctly into an img.Image format.
      // This part might need refinement based on how mask data is actually stored/handled.

      // Placeholder: Convert raw mask bytes to img.Image
      // This assumes mask data is raw grayscale bytes matching initialWidth * initialHeight
      if (widget.initialMaskData!.length ==
          widget.initialWidth * widget.initialHeight) {
        originalMask = img.Image.fromBytes(
          width: widget.initialWidth,
          height: widget.initialHeight,
          bytes: widget
              .initialMaskData!.buffer, // Need to ensure format compatibility
          format: img.Format.uint8, // Assuming grayscale
          numChannels: 1,
        );
      } else {
        print(
            "Warning: Mask data size does not match image dimensions. Skipping mask crop.");
      }
    }

    if (originalImage == null) {
      // Handle error: Could not decode image
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Error: Could not decode image.')),
      );
      return;
    }

    // 2. Calculate the source crop rectangle in original image coordinates
    // Need to map the _cropRect (relative to display size) back to original image pixels
    final double scaleX = originalImage.width / _imageDisplaySize.width;
    final double scaleY = originalImage.height / _imageDisplaySize.height;

    final int srcX = (_cropRect.left * scaleX).round();
    final int srcY = (_cropRect.top * scaleY).round();
    final int srcWidth = (_cropRect.width * scaleX).round();
    final int srcHeight = (_cropRect.height * scaleY).round();

    // 3. Crop the image using img.copyCrop
    // Note: img.copyCrop needs x, y, width, height
    final img.Image croppedImage = img.copyCrop(
      originalImage,
      x: srcX,
      y: srcY,
      width: srcWidth,
      height: srcHeight,
    );

    // 4. Resize the cropped image to the target _currentCropWidth, _currentCropHeight
    // Using NEAREST for potentially better pixel art/sharpness preservation,
    // or LINEAR for smoother results. Adjust as needed.
    final img.Image resizedImage = img.copyResize(
      croppedImage,
      width: _currentCropWidth,
      height: _currentCropHeight,
      interpolation: img.Interpolation.nearest,
    );

    // 5. Crop and resize the mask similarly if it exists
    img.Image? resizedMask;
    if (originalMask != null) {
      final img.Image croppedMask = img.copyCrop(
        originalMask,
        x: srcX,
        y: srcY,
        width: srcWidth,
        height: srcHeight,
      );
      resizedMask = img.copyResize(
        croppedMask,
        width: _currentCropWidth,
        height: _currentCropHeight,
        interpolation: img.Interpolation.nearest, // Use nearest for masks
      );
    }

    // 6. Encode the resized image (and mask) back to Uint8List (e.g., PNG)
    // Using PNG as it's lossless and common.
    final Uint8List finalImageBytes =
        Uint8List.fromList(img.encodePng(resizedImage));
    Uint8List? finalMaskBytes;
    if (resizedMask != null) {
      // Encode mask - assuming we want raw grayscale bytes again
      // The img.encodePng might save as RGBA, need to extract grayscale if needed by backend
      // For now, let's encode as PNG and see. If backend needs raw bytes, adjust here.

      // Option 1: Encode as PNG (might be RGBA)
      // finalMaskBytes = Uint8List.fromList(img.encodePng(resizedMask));

      // Option 2: Get raw grayscale bytes (if resizedMask is grayscale)
      if (resizedMask.numChannels == 1 &&
          resizedMask.format == img.Format.uint8) {
        finalMaskBytes = resizedMask.getBytes(
            order: img.ChannelOrder
                .red); // For grayscale, red channel holds the value
      } else {
        // If mask isn't grayscale, convert it first or handle error
        print(
            "Warning: Resized mask is not in expected grayscale format. Encoding as PNG.");
        finalMaskBytes =
            Uint8List.fromList(img.encodePng(resizedMask)); // Fallback to PNG
      }
    }

    // 7. Call the callback
    widget.onCropComplete(CroppedImageData(
      imageBytes: finalImageBytes,
      maskBytes: finalMaskBytes,
      width: _currentCropWidth,
      height: _currentCropHeight,
    ));
  }

  @override
  Widget build(BuildContext context) {
    final theme = ShadTheme.of(context);
    // Calculate aspect ratio for the SizedBox containing the image
    double imageAspectRatio = widget.initialWidth / widget.initialHeight;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Crop &amp; Resize Image'),
        actions: [
          ShadButton.ghost(
            onPressed: _performCrop,
            child: const Text('Apply'),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // --- Image Display Area ---
            Expanded(
              child: Center(
                // Center the image container
                child: AspectRatio(
                  aspectRatio: imageAspectRatio,
                  child: Container(
                    color: Colors.grey[300], // Background for image area
                    child: LayoutBuilder(
                        // Use LayoutBuilder to get actual display size
                        builder: (context, constraints) {
                      // Update display size if it changed (e.g., on rotation or resize)
                      // Check added to prevent setState during build errors
                      WidgetsBinding.instance.addPostFrameCallback((_) {
                        if (mounted &&
                            _imageDisplaySize != constraints.biggest) {
                          setState(() {
                            _imageDisplaySize = constraints.biggest;
                            _updateCropRect(); // Recalculate rect based on new size
                          });
                        }
                      });

                      return GestureDetector(
                        onPanStart: _onPanStart,
                        onPanUpdate: _onPanUpdate,
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            // Image
                            Image.file(
                              widget.imageFile,
                              key: _imageKey,
                              fit: BoxFit
                                  .contain, // Ensure whole image is visible
                            ),
                            // Cropping Frame Overlay
                            if (_cropRect != Rect.zero)
                              Positioned.fromRect(
                                rect: _cropRect,
                                child: Container(
                                  decoration: BoxDecoration(
                                    border: Border.all(
                                      color: Colors.white.withOpacity(0.8),
                                      width: 2,
                                    ),
                                  ),
                                  // Optional: Add handles or visual cues
                                ),
                              ),
                            // Optional: Dimming outside the crop area
                            if (_cropRect != Rect.zero)
                              ClipPath(
                                clipper:
                                    InvertedRectClipper(clipRect: _cropRect),
                                child: Container(
                                  color: Colors.black.withOpacity(0.5),
                                ),
                              ),
                          ],
                        ),
                      );
                    }),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 20),

            // --- Sliders ---
            Text('Width: $_currentCropWidth px'),
            ShadSlider(
              min: 64,
              max: _maxCropWidth.toDouble(),
              divisions: (_maxCropWidth - 64) ~/ 64,
              initialValue: _currentCropWidth.toDouble(),
              onChanged: (value) {
                int newWidth = largestMultipleOf64(value.round());
                if (newWidth != _currentCropWidth) {
                  setState(() {
                    _currentCropWidth = newWidth;
                    // Maintain aspect ratio
                    double aspect = _currentCropWidth / _currentCropHeight;
                    _currentCropHeight = largestMultipleOf64(
                        (_currentCropWidth / aspect).round());
                    _currentCropHeight = _currentCropHeight.clamp(
                        64, _maxCropHeight); // Clamp height
                    // Recalculate width based on clamped height to be precise
                    _currentCropWidth = largestMultipleOf64(
                        (_currentCropHeight * aspect).round());
                    _currentCropWidth = _currentCropWidth.clamp(
                        64, _maxCropWidth); // Clamp width again

                    _updateCropRect();
                  });
                }
              },
            ),
            const SizedBox(height: 10),
            Text('Height: $_currentCropHeight px'),
            ShadSlider(
              min: 64,
              max: _maxCropHeight.toDouble(),
              divisions: (_maxCropHeight - 64) ~/ 64,
              initialValue: _currentCropHeight.toDouble(),
              onChanged: (value) {
                int newHeight = largestMultipleOf64(value.round());
                if (newHeight != _currentCropHeight) {
                  setState(() {
                    _currentCropHeight = newHeight;
                    // Maintain aspect ratio
                    double aspect = _currentCropWidth / _currentCropHeight;
                    _currentCropWidth = largestMultipleOf64(
                        (_currentCropHeight * aspect).round());
                    _currentCropWidth = _currentCropWidth.clamp(
                        64, _maxCropWidth); // Clamp width
                    // Recalculate height based on clamped width
                    _currentCropHeight = largestMultipleOf64(
                        (_currentCropWidth / aspect).round());
                    _currentCropHeight = _currentCropHeight.clamp(
                        64, _maxCropHeight); // Clamp height again

                    _updateCropRect();
                  });
                }
              },
            ),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }
}

// Custom Clipper for dimming effect outside the crop rectangle
class InvertedRectClipper extends CustomClipper<Path> {
  final Rect clipRect;

  InvertedRectClipper({required this.clipRect});

  @override
  Path getClip(Size size) {
    // Path for the outer rectangle (full size)
    Path outerPath = Path()
      ..addRect(Rect.fromLTWH(0, 0, size.width, size.height));
    // Path for the inner rectangle (the crop area)
    Path innerPath = Path()..addRect(clipRect);

    // Subtract the inner path from the outer path
    return Path.combine(PathOperation.difference, outerPath, innerPath);
  }

  @override
  bool shouldReclip(CustomClipper<Path> oldClipper) =>
      true; // Reclip whenever rect changes
}
