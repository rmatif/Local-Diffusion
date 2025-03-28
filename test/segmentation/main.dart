import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:file_picker/file_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart'; // Added
import 'package:share_plus/share_plus.dart'; // Added

// Define a type for the segmentation result
typedef SegmentationResult = ({Uint8List blendedImage, Uint8List labelImage});

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Interactive Segmentation',
      theme: ThemeData(
        primarySwatch: Colors.teal,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: SegmentationScreen(),
    );
  }
}

class SegmentationScreen extends StatefulWidget {
  @override
  _SegmentationScreenState createState() => _SegmentationScreenState();
}

// --- COMPLETE Updated _SegmentationScreenState Class ---

class _SegmentationScreenState extends State<SegmentationScreen> {
  // --- State Variables ---
  File? imageFile;
  String? modelPath;
  Uint8List?
      segmentedBlendedImage; // Stores the visual blended output from processOutput
  Uint8List?
      segmentationLabelImage; // Stores the raw label map from processOutput
  Uint8List?
      exportedMaskImageBytes; // ADDED: Stores the B/W mask returned from Interactive view
  int originalImageWidth = 0;
  int originalImageHeight = 0;
  bool isProcessing = false;

  // --- Image and Model Picking ---

  Future<void> pickImage() async {
    final picker = ImagePicker();
    // Limit image size for stability if needed, though not strictly necessary here
    // final pickedFile = await picker.pickImage(source: ImageSource.gallery, imageQuality: 85, maxWidth: 1024);
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        imageFile = File(pickedFile.path);
        // Reset all results when a new image is picked
        segmentedBlendedImage = null;
        segmentationLabelImage = null;
        exportedMaskImageBytes = null;
      });
    }
  }

  Future<void> pickModel() async {
    setState(() {
      // Optionally indicate picking state in UI if needed
    });
    try {
      // Use FileType.any for broader compatibility and manual check
      final result = await FilePicker.platform.pickFiles(
        type: FileType.any,
      );

      if (result != null && result.files.single.path != null) {
        final path = result.files.single.path!;
        final fileName = result.files.single.name;

        // Manual extension check
        if (path.toLowerCase().endsWith('.tflite')) {
          setState(() {
            modelPath = path;
            // Reset results related to previous model/inference
            segmentedBlendedImage = null;
            segmentationLabelImage = null;
            exportedMaskImageBytes = null;
          });
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
                content: Text('Model selected: $fileName'),
                duration: Duration(seconds: 2)),
          );
        } else {
          // Invalid file type selected
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                  'Invalid file type: Please select a .tflite model file.'),
              backgroundColor: Colors.red,
            ),
          );
          setState(() {
            modelPath = null; // Clear invalid selection
            exportedMaskImageBytes = null; // Also clear mask if model changes
          });
        }
      } else {
        // User canceled the picker
        print('Model picking cancelled.');
      }
    } catch (e) {
      print('Error picking model: $e');
      String errorMessage = 'Error selecting model file.';
      if (e is PlatformException) {
        // Provide specific platform error if available
        errorMessage = 'Error selecting model file:\n${e.message ?? e.code}';
      } else {
        errorMessage = 'Error selecting model file:\n${e.toString()}';
      }
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(errorMessage), backgroundColor: Colors.red),
      );
      setState(() {
        modelPath = null; // Clear model path on error
        exportedMaskImageBytes = null; // Clear mask on error
      });
    }
  }

  // --- Inference Logic ---

  Future<void> runInference() async {
    // Input validation
    if (imageFile == null) {
      ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Please select an image first.')));
      return;
    }
    if (modelPath == null) {
      ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Please select a .tflite model first.')));
      return;
    }

    // Update UI to show processing state
    setState(() {
      isProcessing = true;
      // Clear previous results before starting new inference
      segmentedBlendedImage = null;
      segmentationLabelImage = null;
      exportedMaskImageBytes = null;
    });

    try {
      // --- Image Loading and Preprocessing ---
      final imageBytes = await imageFile!.readAsBytes();
      // Use image package to decode (ensure robust decoding)
      final originalImage = img.decodeImage(imageBytes);
      if (originalImage == null) {
        throw Exception(
            "Could not decode image file. Ensure it's a supported format (JPG, PNG, etc).");
      }
      originalImageWidth = originalImage.width;
      originalImageHeight = originalImage.height;

      // Resize with padding (assuming model expects 640x640)
      final (paddedImage, paddingInfo) = resizeWithPadding(originalImage, 640);
      final input =
          preprocessImage(paddedImage); // Normalize to Float32List [0,1]

      // --- TFLite Interpreter Setup ---
      final options = InterpreterOptions()
        ..useNnApiForAndroid =
            false; // Disable NNAPI if causing issues, enable for potential speedup
      final interpreter =
          await Interpreter.fromFile(File(modelPath!), options: options);

      // --- Get Input/Output Details (Crucial for Debugging) ---
      final inputShape = interpreter.getInputTensor(0).shape;
      print('--> Actual Input Shape: $inputShape');
      final outputTensors = interpreter.getOutputTensors();
      print('--> Number of Output Tensors: ${outputTensors.length}');
      if (outputTensors.length < 2) {
        interpreter.close(); // Close interpreter on error
        throw Exception(
            "Model requires at least 2 output tensors for segmentation, but found ${outputTensors.length}.");
      }
      final outputShape0 = outputTensors[0].shape; // Boxes/Scores/Coeffs
      final outputShape1 = outputTensors[1].shape; // Proto Masks
      print(
          '--> Actual Output Tensor #0 Shape: $outputShape0, Type: ${outputTensors[0].type}');
      print(
          '--> Actual Output Tensor #1 Shape: $outputShape1, Type: ${outputTensors[1].type}');

      // Validate input tensor size against preprocessed data
      int expectedInputElements = inputShape.fold(1, (a, b) => a * b);
      if (input.length != expectedInputElements) {
        interpreter.close();
        throw Exception(
            "Input size mismatch: Preprocessed data has ${input.length} elements, but model expects $expectedInputElements (Shape: $inputShape).");
      }

      // --- Prepare Output Buffers ---
      final outputBuffers = <int, ByteBuffer>{};
      int output0Size = outputShape0.fold<int>(1, (acc, dim) => acc * dim);
      int output1Size = outputShape1.fold<int>(1, (acc, dim) => acc * dim);

      // Assuming Float32 output based on typical models, verify with Type printout if needed
      outputBuffers[0] = Float32List(output0Size).buffer;
      outputBuffers[1] = Float32List(output1Size).buffer;

      // --- Run Inference ---
      print("Running inference...");
      interpreter.runForMultipleInputs([input.buffer], outputBuffers);
      print("Inference complete.");

      // --- Process Outputs ---
      final outputs = <int, Float32List>{};
      outputBuffers.forEach((key, value) {
        outputs[key] = Float32List.view(value);
      });

      if (outputs[0] == null || outputs[1] == null) {
        interpreter.close();
        throw Exception(
            'Expected outputs not found in buffer map after inference.');
      }

      // Call the processing function (make sure it uses correct indexing for YOUR model)
      print("Processing model outputs...");
      final SegmentationResult result = await processOutput(
        outputs[0]!,
        outputs[1]!,
        outputShape0, // Pass shapes for validation/use within processOutput
        outputShape1,
        paddingInfo,
        originalImageWidth,
        originalImageHeight,
        originalImage, // Pass the original img.Image for blending
      );
      print("Output processing complete.");

      // --- Update State with Results ---
      setState(() {
        segmentedBlendedImage = result.blendedImage;
        segmentationLabelImage = result.labelImage;
        isProcessing = false; // Mark processing as finished
      });

      // --- Clean up ---
      interpreter.close();
    } catch (e, stackTrace) {
      // Catch errors and stack trace
      print('--- ERROR DURING INFERENCE OR PROCESSING ---');
      print('Error: $e');
      print('Stack Trace:\n$stackTrace');
      print('--- END ERROR ---');
      String errorMessage = 'Error during processing: ${e.toString()}';
      // Add specific error messages if helpful
      if (e is FileSystemException) {
        errorMessage = "Error accessing file (image or model): ${e.message}";
      } else if (e is ArgumentError) {
        errorMessage =
            'Error: Invalid argument during processing. Check model output interpretation logic.';
      } else if (e.toString().contains('Interpreter has already been closed')) {
        errorMessage = 'Error: Model interpreter was closed prematurely.';
      }
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
            content: Text(errorMessage),
            duration: Duration(seconds: 5),
            backgroundColor: Colors.red),
      );
      setState(() =>
          isProcessing = false); // Ensure processing state is reset on error
      // interpreter.close(); // Ensure interpreter is closed if error happened after loading
    }
  }

  // --- Image Preprocessing Helpers ---
  (img.Image, List<int>) resizeWithPadding(img.Image image, int targetSize) {
    int h = image.height;
    int w = image.width;
    // Calculate scale to fit whilst maintaining aspect ratio
    double scale = min(targetSize / h, targetSize / w);
    int newH = (h * scale).round();
    int newW = (w * scale).round();
    // Resize using linear interpolation for better quality than default nearest
    img.Image resizedImage = img.copyResize(image,
        width: newW, height: newH, interpolation: img.Interpolation.linear);
    // Create background image
    img.Image paddedImage =
        img.Image(width: targetSize, height: targetSize); // Defaults to black
    // Calculate padding offsets
    int padH = (targetSize - newH) ~/ 2;
    int padW = (targetSize - newW) ~/ 2;
    // Composite resized image onto the black background
    img.compositeImage(paddedImage, resizedImage, dstX: padW, dstY: padH);
    // Return padded image and padding info [top, left, new_height, new_width]
    return (paddedImage, [padH, padW, newH, newW]);
  }

  // Normalizes image pixels to Float32List [0, 1]
  Float32List preprocessImage(img.Image image) {
    int height = image.height;
    int width = image.width;
    // Assuming input format [1, H, W, 3]
    Float32List input = Float32List(1 * height * width * 3);
    int index = 0;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final pixel = image.getPixel(x, y);
        // Use normalized getters and ensure cast to double
        input[index++] = pixel.rNormalized.toDouble();
        input[index++] = pixel.gNormalized.toDouble();
        input[index++] = pixel.bNormalized.toDouble();
      }
    }
    return input;
  }

  // --- Output Processing Function ---
  // IMPORTANT: This function MUST correctly interpret your specific model's output shapes and structure.
  // The version below assumes Output0=[1, 37, 8400] (Box+Score+Coeffs) and Output1=[1, 160, 160, 32] (Protos HWC)
  Future<SegmentationResult> processOutput(
    Float32List output0,
    Float32List output1,
    List<int> outputShape0,
    List<int> outputShape1,
    List<int> paddingInfo,
    int originalWidth,
    int originalHeight,
    img.Image originalImage,
  ) async {
    // --- Verify Sizes --- (Good practice)
    int expectedSize0 = outputShape0.fold(1, (a, b) => a * b);
    int expectedSize1 = outputShape1.fold(1, (a, b) => a * b);
    if (output0.length != expectedSize0)
      throw ArgumentError('Output 0 size mismatch');
    if (output1.length != expectedSize1)
      throw ArgumentError('Output 1 size mismatch');

    // --- Constants based on YOUR model's output shapes ---
    final int numBoxes = outputShape0[2]; // 8400 from [1, 37, 8400]
    final int boxDim = outputShape0[1]; // 37 from [1, 37, 8400]
    // Assuming structure: 4 box coords, 1 conf score, 32 coeffs = 37
    final int scoreIndex = 4;
    final int coeffStartIndex = 5;
    final int numCoeffs = boxDim - coeffStartIndex; // Should be 32

    final int protoMaskHeight = outputShape1[1]; // 160 from [1, 160, 160, 32]
    final int protoMaskWidth = outputShape1[2]; // 160 from [1, 160, 160, 32]
    final int protoMaskChannels = outputShape1[3]; // 32 from [1, 160, 160, 32]
    if (numCoeffs != protoMaskChannels)
      print(
          "Warning: Coeff count ($numCoeffs) doesn't match proto channels ($protoMaskChannels).");

    // --- Step 1: Process Detections (Boxes, Scores, Coeffs) ---
    List<List<double>> boxesProcessed = [];
    List<double> scores = [];
    double confidenceThreshold = 0.3; // Adjust confidence threshold as needed

    for (int i = 0; i < numBoxes; i++) {
      // Iterate through boxes
      List<double> currentBoxData =
          List.filled(boxDim, 0.0); // Pre-allocate for efficiency
      double score = 0;
      // Access data assuming [1, BoxDim, NumBoxes] format -> idx = dim * NumBoxes + box_idx
      for (int j = 0; j < boxDim; j++) {
        // Iterate through dimensions (coords, score, coeffs)
        double val = output0[j * numBoxes + i];
        currentBoxData[j] = val;
        if (j == scoreIndex) score = val;
      }
      // Keep boxes exceeding confidence threshold
      if (score > confidenceThreshold) {
        boxesProcessed.add(currentBoxData);
        scores.add(score);
        // Note: We don't store original indices here, rely on filtered lists directly
      }
    }
    print(
        "Boxes after confidence threshold ($confidenceThreshold): ${boxesProcessed.length}");

    // (Optional but Recommended) Add Non-Maximum Suppression (NMS) here
    // If NMS is added, it would further filter `boxesProcessed` and `scores`

    // Select top N detections based on score *after* thresholding
    const int N = 10; // Limit number of masks for performance/clarity
    List<int> sortedIndices = List.generate(scores.length, (i) => i)
      ..sort(
          (a, b) => scores[b].compareTo(scores[a])); // Sort descending by score
    List<int> topNIndices = sortedIndices.take(N).toList();

    // Get the boxes and coefficients for the top N detections
    List<List<double>> topBoxes =
        topNIndices.map((i) => boxesProcessed[i]).toList();
    List<List<double>> topCoeffs = topBoxes
        .map((box) => box.sublist(coeffStartIndex, coeffStartIndex + numCoeffs))
        .toList();

    print("Number of top masks selected (Top ${N}): ${topBoxes.length}");
    if (topBoxes.isEmpty) {
      // Handle case where no boxes pass threshold
      print("No masks found after filtering.");
      Uint8List emptyLabels = Uint8List(originalHeight * originalWidth);
      img.Image emptyBlended = img.copyResize(originalImage,
          width: originalWidth, height: originalHeight);
      return (
        blendedImage: Uint8List.fromList(img.encodePng(emptyBlended)),
        labelImage: emptyLabels
      );
    }
    print("Coefficients shape: ${topCoeffs.length}x${topCoeffs[0].length}");

    // --- Step 2: Process Prototype Masks ---
    int padH = paddingInfo[0],
        padW = paddingInfo[1],
        newH = paddingInfo[2],
        newW = paddingInfo[3];
    // Calculate scale factor between input image (640) and proto mask (160)
    const int maskScale = 4; // = 640 / 160. Verify if your model differs.
    // Calculate crop region on the 160x160 proto masks based on padding
    int cropHStart = max(0, padH ~/ maskScale);
    int cropHEnd = min(protoMaskHeight, (padH + newH) ~/ maskScale);
    int cropWStart = max(0, padW ~/ maskScale);
    int cropWEnd = min(protoMaskWidth, (padW + newW) ~/ maskScale);
    int cropMaskHeight = cropHEnd - cropHStart;
    int cropMaskWidth = cropWEnd - cropWStart;

    if (cropMaskHeight <= 0 || cropMaskWidth <= 0) {
      // Check for invalid crop dimensions
      print(
          "Warning: Invalid calculated crop dimensions ($cropMaskWidth x $cropMaskHeight). Check padding or mask scale.");
      // Return empty results if crop is invalid
      Uint8List emptyLabels = Uint8List(originalHeight * originalWidth);
      img.Image emptyBlended = img.copyResize(originalImage,
          width: originalWidth, height: originalHeight);
      return (
        blendedImage: Uint8List.fromList(img.encodePng(emptyBlended)),
        labelImage: emptyLabels
      );
    }

    // Extract, Crop, and Resize each of the 32 prototype masks
    List<Float32List> finalProtoMasks = [];
    for (int i = 0; i < protoMaskChannels; i++) {
      // Loop through channels (0 to 31)
      Float32List croppedMask = Float32List(cropMaskHeight * cropMaskWidth);
      int cropIdx = 0;
      // Extract data for channel `i` from the cropped region (y, x)
      // using the correct indexing for [1, H, W, C] format
      for (int y = cropHStart; y < cropHEnd; y++) {
        for (int x = cropWStart; x < cropWEnd; x++) {
          // Index = y_coord * Width * Channels + x_coord * Channels + channel_index
          int sourceIndex = y * protoMaskWidth * protoMaskChannels +
              x * protoMaskChannels +
              i;
          // Bounds check for safety
          if (sourceIndex >= 0 && sourceIndex < output1.length) {
            croppedMask[cropIdx++] = output1[sourceIndex];
          } else {
            // Should not happen if shapes/constants are correct, but good to handle
            print(
                'Warning: Index out of bounds $sourceIndex accessing prototype masks (size ${output1.length}). H=$protoMaskHeight, W=$protoMaskWidth, C=$protoMaskChannels, y=$y, x=$x, i=$i');
            if (cropIdx < croppedMask.length)
              croppedMask[cropIdx++] = 0.0; // Fill with 0 if out of bounds
          }
        }
      }
      // Resize the cropped mask (now size cropW x cropH) up to the original image size
      Float32List resizedMask = bilinearResize(croppedMask, cropMaskWidth,
          cropMaskHeight, originalWidth, originalHeight);
      finalProtoMasks.add(resizedMask);
    }
    print(
        "Processed ${finalProtoMasks.length} final prototype masks (resized).");

    // --- Step 3: Generate Instance Masks ---
    // Multiply coefficients with resized prototype masks and apply sigmoid
    List<Float32List> instanceMasks = [];
    int numPixels = originalHeight * originalWidth;
    for (int i = 0; i < topCoeffs.length; i++) {
      // For each of the top N instances
      Float32List instanceMask =
          Float32List(numPixels); // Mask for this instance
      for (int p = 0; p < numPixels; p++) {
        // For each pixel in the output mask
        double sum = 0.0;
        // Dot product: coeff[j] * protoMask[j][pixel p]
        for (int j = 0; j < numCoeffs; j++) {
          // Ensure indices are valid
          if (j < topCoeffs[i].length && j < finalProtoMasks.length) {
            sum += topCoeffs[i][j] * finalProtoMasks[j][p];
          }
        }
        // Apply sigmoid activation to get pixel probability [0, 1]
        instanceMask[p] = 1.0 / (1.0 + exp(-sum));
      }
      instanceMasks.add(instanceMask);
    }
    print("Generated ${instanceMasks.length} instance masks (after sigmoid).");

    // --- Step 4: Create Label Image ---
    // Assign a unique label (1, 2, ...) to pixels based on thresholded instance masks
    double maskThreshold =
        0.5; // Threshold for converting probability to binary mask
    Uint8List labelImage =
        Uint8List(numPixels); // Initialize with 0 (background)

    // Iterate through instance masks (which are sorted by confidence implicitly)
    for (int i = 0; i < instanceMasks.length; i++) {
      int maskLabel = i + 1; // Assign labels 1, 2, ..., N
      int pixelCount = 0;
      for (int p = 0; p < numPixels; p++) {
        // If pixel probability > threshold AND pixel is currently background (0)
        if (instanceMasks[i][p] > maskThreshold && labelImage[p] == 0) {
          labelImage[p] = maskLabel; // Assign the label for this instance
          pixelCount++;
        }
      }
      print(
          "Mask ${i + 1} (Label ${maskLabel}) has $pixelCount pixels assigned (threshold $maskThreshold).");
    }

    // --- Step 5: Generate Colormap for Blending ---
    List<List<int>> colormap = [
      [0, 0, 0]
    ]; // Background is black
    Random colorRandom = Random(42); // Use a seed for deterministic colors
    int numInstancesFound = topBoxes.length;
    for (int i = 0; i < numInstancesFound; i++) {
      // Generate somewhat distinct random colors (avoiding very dark ones)
      colormap.add([
        colorRandom.nextInt(156) + 100, // R (100-255)
        colorRandom.nextInt(156) + 100, // G (100-255)
        colorRandom.nextInt(156) + 100, // B (100-255)
      ]);
    }

    // --- Step 6: Create Blended Visualization Image ---
    // Overlay colored masks onto the original image
    img.Image blendedImage = img.Image(
        width: originalWidth,
        height: originalHeight,
        numChannels: originalImage.numChannels, // Match original (RGB or RGBA)
        format: originalImage.format // Preserve format if possible
        );
    double alpha = 0.4; // Transparency of the overlay
    bool imageHasAlpha =
        originalImage.numChannels == 4; // Check if original has alpha

    for (int y = 0; y < originalHeight; y++) {
      for (int x = 0; x < originalWidth; x++) {
        int index = y * originalWidth + x;
        int label =
            labelImage[index]; // Get the assigned label (0 for background)
        final origPixel =
            originalImage.getPixel(x, y); // Get original pixel data

        if (label == 0) {
          // If background, just copy the original pixel
          blendedImage.setPixel(x, y, origPixel);
        } else {
          // If part of a mask, get the color and blend it
          // Use default white if label somehow exceeds colormap size (shouldn't happen)
          List<int> color =
              (label < colormap.length) ? colormap[label] : [255, 255, 255];
          // Blend: (original * (1-alpha)) + (mask_color * alpha)
          int r = (origPixel.r * (1 - alpha) + color[0] * alpha)
              .round()
              .clamp(0, 255);
          int g = (origPixel.g * (1 - alpha) + color[1] * alpha)
              .round()
              .clamp(0, 255);
          int b = (origPixel.b * (1 - alpha) + color[2] * alpha)
              .round()
              .clamp(0, 255);
          // Preserve original alpha if present, otherwise assume opaque
          int a = imageHasAlpha ? origPixel.a.round().clamp(0, 255) : 255;

          // Set the blended pixel, using RGBA if alpha channel exists
          if (imageHasAlpha) {
            blendedImage.setPixelRgba(x, y, r, g, b, a);
          } else {
            blendedImage.setPixelRgb(x, y, r, g, b);
          }
        }
      }
    }

    // --- Step 7: Return Results ---
    // Encode the blended image to PNG bytes and return it along with the raw label map
    return (
      blendedImage: Uint8List.fromList(img.encodePng(blendedImage)),
      labelImage: labelImage // This is the map of labels (0, 1, 2...)
    );
  }

  // --- Bilinear Resize Helper --- (Essential for resizing proto masks)
  Float32List bilinearResize(Float32List input, int inputWidth, int inputHeight,
      int outputWidth, int outputHeight) {
    Float32List output = Float32List(outputWidth * outputHeight);
    // Handle edge case of zero dimensions
    if (inputWidth <= 0 ||
        inputHeight <= 0 ||
        outputWidth <= 0 ||
        outputHeight <= 0) return output;

    double scaleX = inputWidth / outputWidth;
    double scaleY = inputHeight / outputHeight;

    for (int y = 0; y < outputHeight; y++) {
      // Loop through output pixels
      for (int x = 0; x < outputWidth; x++) {
        // Calculate corresponding source coordinates (center sampling)
        double srcX = (x + 0.5) * scaleX - 0.5;
        double srcY = (y + 0.5) * scaleY - 0.5;

        // Get integer coordinates of surrounding source pixels
        int x0 = srcX.floor();
        int y0 = srcY.floor();
        // Clamp coordinates to be within source bounds
        x0 = max(0, x0);
        y0 = max(0, y0);
        int x1 = min(x0 + 1, inputWidth - 1);
        int y1 = min(y0 + 1, inputHeight - 1);

        // Calculate interpolation weights
        double wx = srcX - x0; // Weight for x1
        double wy = srcY - y0; // Weight for y1

        // Get source pixel indices (bounds checked)
        int idx00 = y0 * inputWidth + x0;
        int idx10 = y1 * inputWidth + x0;
        int idx01 = y0 * inputWidth + x1;
        int idx11 = y1 * inputWidth + x1;

        // Get surrounding pixel values (handle potential index issues, though clamping helps)
        double p00 = (idx00 >= 0 && idx00 < input.length) ? input[idx00] : 0.0;
        double p10 = (idx10 >= 0 && idx10 < input.length) ? input[idx10] : 0.0;
        double p01 = (idx01 >= 0 && idx01 < input.length) ? input[idx01] : 0.0;
        double p11 = (idx11 >= 0 && idx11 < input.length) ? input[idx11] : 0.0;

        // Perform bilinear interpolation
        double valTop = p00 * (1.0 - wx) + p01 * wx; // Interpolate top row
        double valBottom =
            p10 * (1.0 - wx) + p11 * wx; // Interpolate bottom row
        double value =
            valTop * (1.0 - wy) + valBottom * wy; // Interpolate vertically

        // Assign interpolated value to output pixel
        output[y * outputWidth + x] = value;
      }
    }
    return output;
  }

  // --- HSV to RGB Helper --- (Used for colormap generation if needed)
  List<int> hsvToRgb(int h, int s, int v) {
    // Simplified implementation (can use Flutter's HSVColor.fromAHSV(1.0, h.toDouble(), s / 255.0, v / 255.0).toColor() if preferred)
    if (s == 0) return [v, v, v];
    double hh = h / 60.0;
    int i = hh.floor();
    double ff = hh - i;
    double p = v * (1.0 - s / 255.0);
    double q = v * (1.0 - (s / 255.0) * ff);
    double t = v * (1.0 - (s / 255.0) * (1.0 - ff));
    int r, g, b;
    switch (i % 6) {
      // Use modulo 6 for safety
      case 0:
        r = v;
        g = t.round();
        b = p.round();
        break;
      case 1:
        r = q.round();
        g = v;
        b = p.round();
        break;
      case 2:
        r = p.round();
        g = v;
        b = t.round();
        break;
      case 3:
        r = p.round();
        g = q.round();
        b = v;
        break;
      case 4:
        r = t.round();
        g = p.round();
        b = v;
        break;
      default:
        /* case 5 */ r = v;
        g = p.round();
        b = q.round();
        break;
    }
    return [r.clamp(0, 255), g.clamp(0, 255), b.clamp(0, 255)];
  }

  // --- Navigation to Interactive View (Awaits Result) ---
  Future<void> _navigateToInteractiveView() async {
    // Ensure we have the necessary data before navigating
    if (imageFile == null) {
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('Original image is missing.')));
      return;
    }
    if (segmentationLabelImage == null) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('Run inference first to get segmentation results.')));
      return;
    }
    if (originalImageWidth <= 0 || originalImageHeight <= 0) {
      ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Invalid image dimensions after inference.')));
      return;
    }

    // Read original image bytes to pass to the interactive view
    Uint8List originalImageBytes;
    try {
      originalImageBytes = await imageFile!.readAsBytes();
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error reading original image file: $e')));
      return;
    }

    // Navigate and wait for a Uint8List? (the mask bytes or null) to be popped
    final result = await Navigator.push<Uint8List?>(
      context,
      MaterialPageRoute(
        builder: (context) => InteractiveSegmentationView(
          // Assumes this Widget exists
          originalImageBytes: originalImageBytes,
          labelImageBytes: segmentationLabelImage!,
          imageWidth: originalImageWidth,
          imageHeight: originalImageHeight,
        ),
      ),
    );

    // Handle the returned result
    if (result != null) {
      // If mask bytes were returned, update the state to display it
      setState(() {
        exportedMaskImageBytes = result;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
            content: Text('Mask generated successfully!'),
            duration: Duration(seconds: 1)),
      );
    } else {
      // If null was returned (e.g., user pressed back button), do nothing or log it
      print("Navigation returned without exporting mask.");
    }
  }

  // --- Build Method ---
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Image Segmentation')),
      body: Center(
        // Center the content vertically
        child: SingleChildScrollView(
          // Allow scrolling if content overflows
          padding: const EdgeInsets.all(16.0), // Add padding around content
          child: Column(
            mainAxisAlignment:
                MainAxisAlignment.center, // Center column content
            crossAxisAlignment:
                CrossAxisAlignment.center, // Center items horizontally
            children: [
              // --- Action Buttons ---
              Wrap(
                // Use Wrap for button responsiveness
                spacing: 10.0, // Horizontal gap between buttons
                runSpacing: 10.0, // Vertical gap if buttons wrap
                alignment: WrapAlignment.center, // Center buttons if they wrap
                children: [
                  ElevatedButton.icon(
                      icon: Icon(Icons.image),
                      onPressed: pickImage,
                      label: Text('Pick Image')),
                  ElevatedButton.icon(
                      icon: Icon(
                          Icons.model_training), // Icon for model selection
                      onPressed: pickModel,
                      label: Text('Pick Model (.tflite)')),
                  ElevatedButton.icon(
                    icon: Icon(Icons.play_arrow),
                    // Disable button while processing
                    onPressed: isProcessing ? null : runInference,
                    label: Text('Run Inference'),
                    // Style the run button for emphasis
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors
                          .green, // Or Theme.of(context).colorScheme.primary
                      foregroundColor: Colors
                          .white, // Or Theme.of(context).colorScheme.onPrimary
                      disabledBackgroundColor:
                          Colors.grey, // Indicate disabled state
                    ),
                  ),
                ],
              ),
              SizedBox(height: 20), // Spacing below buttons

              // --- Loading Indicator ---
              if (isProcessing)
                Padding(
                  padding: const EdgeInsets.symmetric(
                      vertical: 30.0), // Add vertical padding
                  child: Column(
                    children: [
                      CircularProgressIndicator(),
                      SizedBox(height: 15), // Space between spinner and text
                      Text('Processing, please wait...',
                          style: TextStyle(fontSize: 16)),
                    ],
                  ),
                ),

              // --- Results Display Area ---
              // Only show this section if not processing AND there's something to show
              if (!isProcessing &&
                  (imageFile != null ||
                      segmentedBlendedImage != null ||
                      exportedMaskImageBytes != null))
                Padding(
                  padding: const EdgeInsets.only(top: 20.0),
                  child: Wrap(
                    // Use Wrap for results layout too
                    spacing: 15.0,
                    runSpacing: 15.0,
                    alignment: WrapAlignment.center,
                    children: [
                      // Display Original Image (if available)
                      if (imageFile != null)
                        _buildImageColumn(
                            "Original",
                            Image.file(imageFile!,
                                fit: BoxFit.contain) // Use fit: contain
                            ),

                      // Display Segmentation Result (if available and tappable)
                      if (segmentedBlendedImage != null)
                        GestureDetector(
                          // Make tappable
                          onTap: _navigateToInteractiveView, // Navigate on tap
                          child: _buildImageColumn(
                            "Segmentation Result\n(Tap to interact)", // Multi-line title
                            Stack(
                              // Overlay touch icon
                              alignment: Alignment.center,
                              children: [
                                Image.memory(segmentedBlendedImage!,
                                    fit: BoxFit.contain),
                                // Add a visual hint to tap
                                Container(
                                    padding: EdgeInsets.all(8),
                                    decoration: BoxDecoration(
                                      color: Colors.black45,
                                      shape: BoxShape.circle,
                                    ),
                                    child: Icon(Icons.touch_app,
                                        color: Colors.white70, size: 35)),
                              ],
                            ),
                          ),
                        ),

                      // Display Exported Black & White Mask (if available)
                      if (exportedMaskImageBytes != null)
                        _buildImageColumn(
                            "Exported Mask (B/W)",
                            Image.memory(exportedMaskImageBytes!,
                                fit: BoxFit.contain)),
                    ],
                  ),
                )
              // Placeholder text if not processing and nothing to show yet
              else if (!isProcessing)
                Padding(
                  padding: const EdgeInsets.only(top: 50.0),
                  child: Text(
                      "1. Pick an Image\n2. Pick a .tflite Model\n3. Run Inference",
                      textAlign: TextAlign.center,
                      style: TextStyle(fontSize: 16, color: Colors.grey[600])),
                ),
            ],
          ),
        ),
      ),
    );
  }

  // --- Helper Widget for Displaying Images with Titles ---
  Widget _buildImageColumn(String title, Widget imageWidget) {
    return Column(
      mainAxisSize: MainAxisSize.min, // Use minimum space
      children: [
        Padding(
          padding: const EdgeInsets.only(bottom: 8.0), // Space below title
          child: Text(title,
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
              textAlign: TextAlign.center),
        ),
        ConstrainedBox(
          // Define max size for previews to maintain layout consistency
          constraints: BoxConstraints(
              maxHeight: 220, // Slightly larger max height
              // Adjust max width relative to screen, prevents stretching too wide
              maxWidth: max(150, MediaQuery.of(context).size.width * 0.4)),
          child: Card(
            // Add a slight card elevation/border
            elevation: 2.0,
            clipBehavior: Clip.antiAlias, // Clip image to card bounds
            child: imageWidget,
          ),
        ),
      ],
    );
  }
} // End _SegmentationScreenState Class

// ----------------------------------------------------
// NEW Interactive Segmentation View Screen
// ----------------------------------------------------
// ----------------------------------------------------
// Interactive Segmentation View Screen - MODIFIED
// ----------------------------------------------------
class InteractiveSegmentationView extends StatefulWidget {
  //final Uint8List blendedImageBytes; // REMOVED
  final Uint8List originalImageBytes; // ADDED
  final Uint8List labelImageBytes;
  final int imageWidth;
  final int imageHeight;

  const InteractiveSegmentationView({
    Key? key,
    required this.originalImageBytes, // MODIFIED
    required this.labelImageBytes,
    required this.imageWidth,
    required this.imageHeight,
  }) : super(key: key);

  @override
  _InteractiveSegmentationViewState createState() =>
      _InteractiveSegmentationViewState();
}

class _InteractiveSegmentationViewState
    extends State<InteractiveSegmentationView> {
  Set<int> selectedLabels = {}; // Store IDs of selected segments
  GlobalKey imageKey = GlobalKey(); // To get image render size/position
  TransformationController transformationController =
      TransformationController();

  // --- Tap Handling Logic ---
  void _handleTap(TapUpDetails details) {
    final RenderBox? imageBox =
        imageKey.currentContext?.findRenderObject() as RenderBox?;
    if (imageBox == null || !imageBox.hasSize) return;
    final localPosition = imageBox.globalToLocal(details.globalPosition);
    final imageWidgetSize = imageBox.size;
    final double scaleX = widget.imageWidth / imageWidgetSize.width;
    final double scaleY = widget.imageHeight / imageWidgetSize.height;
    final double scale = max(scaleX, scaleY);
    final double displayedImageWidth = widget.imageWidth / scale;
    final double displayedImageHeight = widget.imageHeight / scale;
    final double offsetX = (imageWidgetSize.width - displayedImageWidth) / 2.0;
    final double offsetY =
        (imageWidgetSize.height - displayedImageHeight) / 2.0;
    final double imageX = (localPosition.dx - offsetX) * scale;
    final double imageY = (localPosition.dy - offsetY) * scale;
    final int pixelX = imageX.clamp(0, widget.imageWidth - 1).floor();
    final int pixelY = imageY.clamp(0, widget.imageHeight - 1).floor();
    final int index = pixelY * widget.imageWidth + pixelX;
    if (index < 0 || index >= widget.labelImageBytes.length) {
      print("Error: Calculated index out of bounds.");
      return;
    }
    final int tappedLabel = widget.labelImageBytes[index];

    print(
        "Tapped Local: ${localPosition.dx.toStringAsFixed(1)}, ${localPosition.dy.toStringAsFixed(1)} -> "
        "Image Pixel: $pixelX, $pixelY -> Label: $tappedLabel");

    if (tappedLabel != 0) {
      setState(() {
        if (selectedLabels.contains(tappedLabel)) {
          selectedLabels.remove(tappedLabel);
          print("Deselected Label: $tappedLabel");
        } else {
          selectedLabels.add(tappedLabel);
          print("Selected Label: $tappedLabel");
        }
      });
    }
  }

  // --- MODIFIED Export Logic ---
  // Returns the mask bytes via Navigator.pop
  // In class _InteractiveSegmentationViewState

  // --- MODIFIED Export Logic ---
  // Returns the mask bytes via Navigator.pop
  // In class _InteractiveSegmentationViewState

  // --- MODIFIED Export Logic ---
  // Temporarily saves the file AND pops
  // In class _InteractiveSegmentationViewState
// Modify _exportMask again

  void _exportMask() async {
    // Add async
    print(
        "[Export Mask] Starting export function. Current selectedLabels: $selectedLabels");
    if (selectedLabels.isEmpty) {
      /* ... */ return;
    }

    // 1. Create RGB mask image
    final maskImage = img.Image(
        width: widget.imageWidth,
        height: widget.imageHeight,
        numChannels: 3 // *** CHANGE: Create 3 channels (RGB) ***
        );
    // Fill with black (0, 0, 0)
    img.fill(maskImage,
        color: img.ColorRgb8(0, 0, 0)); // *** CHANGE: Use ColorRgb8 ***
    print(
        "[Export Mask] Created ${widget.imageWidth}x${widget.imageHeight} RGB mask image, filled black.");

    // 2. Iterate and set pixels
    int whitePixelCount = 0;
    for (int y = 0; y < widget.imageHeight; y++) {
      for (int x = 0; x < widget.imageWidth; x++) {
        final int index = y * widget.imageWidth + x;
        if (index < widget.labelImageBytes.length) {
          final int label = widget.labelImageBytes[index];
          if (label != 0 && selectedLabels.contains(label)) {
            // *** CHANGE: Set pixel to white RGB ***
            maskImage.setPixelRgb(x, y, 255, 255, 255);
            whitePixelCount++;
          }
        }
      }
    }
    print(
        "[Export Mask] Pixel iteration complete. Total pixels set to white: $whitePixelCount");
    if (whitePixelCount == 0 && selectedLabels.isNotEmpty) {
      /* ... Warning ... */
    }

    // 3. Encode as PNG bytes (now encoding an RGB image)
    print("[Export Mask] Encoding the RGB mask image to PNG...");
    final Uint8List pngBytes = Uint8List.fromList(img.encodePng(maskImage));
    print(
        "[Export Mask] PNG Encoding complete. Byte length: ${pngBytes.length}");

    // (Optional: Keep the temporary save from Step 1 here to verify the RGB PNG)
    try {
      final tempDir = await getTemporaryDirectory();
      final filePath = '${tempDir.path}/_debug_mask_export_rgb.png';
      final file = File(filePath);
      await file.writeAsBytes(pngBytes);
      print('[Export Mask] DEBUG: RGB Mask temporary saved to: $filePath');
    } catch (e) {
      print('[Export Mask] DEBUG: Error saving temporary RGB mask: $e');
    }

    // 4. Pop screen and return mask bytes
    Navigator.pop(context, pngBytes);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // --- MODIFIED AppBar ---
      appBar: AppBar(
        title: Text('Select Regions'),
        actions: [
          IconButton(
            icon: Icon(Icons.check_circle_outline), // Changed Icon
            tooltip: 'Confirm Selection & Export Mask', // Updated tooltip
            onPressed: selectedLabels.isEmpty
                ? null
                : _exportMask, // Calls modified function
          ),
        ],
      ),
      body: GestureDetector(
        onTapUp: _handleTap,
        child: Container(
          color: Colors.black,
          child: InteractiveViewer(
            transformationController: transformationController,
            minScale: 0.5,
            maxScale: 4.0,
            child: Stack(
              alignment: Alignment.center,
              children: [
                // Base Image (Original)
                Image.memory(
                  widget.originalImageBytes,
                  key: imageKey,
                  fit: BoxFit.contain,
                  gaplessPlayback: true,
                ),
                // Selection Overlay Painter
                Positioned.fill(
                  child: CustomPaint(
                    painter: SelectionPainter(
                      // Assumes SelectionPainter class exists
                      labelImage: widget.labelImageBytes,
                      selectedLabels: selectedLabels,
                      imageWidth: widget.imageWidth,
                      imageHeight: widget.imageHeight,
                    ),
                    child: Container(),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
      bottomNavigationBar: selectedLabels.isNotEmpty
          ? Container(
              padding: EdgeInsets.all(8),
              color: Colors.black87,
              child: Text(
                "Selected Labels: ${selectedLabels.join(', ')}",
                style: TextStyle(color: Colors.white),
                textAlign: TextAlign.center,
              ),
            )
          : null,
    );
  }
}

// --- SelectionPainter class - REMAINS THE SAME ---
// (It correctly overlays based on labelImage, doesn't need blended image)
class SelectionPainter extends CustomPainter {
  final Uint8List labelImage;
  final Set<int> selectedLabels;
  final int imageWidth;
  final int imageHeight;

  SelectionPainter({
    required this.labelImage,
    required this.selectedLabels,
    required this.imageWidth,
    required this.imageHeight,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (selectedLabels.isEmpty) return;

    final double scaleX = imageWidth / size.width;
    final double scaleY = imageHeight / size.height;
    final double scale = max(scaleX, scaleY);
    final double displayedImageWidth = imageWidth / scale;
    final double displayedImageHeight = imageHeight / scale;
    final double offsetX = (size.width - displayedImageWidth) / 2.0;
    final double offsetY = (size.height - displayedImageHeight) / 2.0;

    final paint = Paint()
      ..color = Colors.yellow.withOpacity(0.5) // Increased opacity slightly
      ..style = PaintingStyle.fill;

    // Pixel-by-pixel drawing (can be slow, consider optimizations for production)
    for (int y = 0; y < imageHeight; y++) {
      for (int x = 0; x < imageWidth; x++) {
        final int index = y * imageWidth + x;
        if (index < labelImage.length) {
          final int label = labelImage[index];
          if (selectedLabels.contains(label)) {
            final double canvasX = (x / scale) + offsetX;
            final double canvasY = (y / scale) + offsetY;
            // Draw slightly larger rectangles to fill gaps at high zoom
            final double pixelWidth =
                (1.0 / scale) + 0.5; // Heuristic adjustment
            final double pixelHeight =
                (1.0 / scale) + 0.5; // Heuristic adjustment
            canvas.drawRect(
              Rect.fromLTWH(canvasX, canvasY, pixelWidth, pixelHeight),
              paint,
            );
          }
        }
      }
    }
  }

  @override
  bool shouldRepaint(covariant SelectionPainter oldDelegate) {
    return oldDelegate.selectedLabels != selectedLabels;
  }
}
