import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:io';
import 'package:flutter/material.dart';

class Stroke {
  final List<Offset> points;
  final double brushSize;

  Stroke(this.points, this.brushSize);
}

class MaskPainter extends CustomPainter {
  final List<Stroke> strokes;
  final double opacity;

  MaskPainter(this.strokes, {this.opacity = 1.0});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Color.fromARGB((255 * opacity).toInt(), 255, 255, 255)
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    for (final stroke in strokes) {
      paint.strokeWidth = stroke.brushSize;

      if (stroke.points.isEmpty) continue;

      final path = Path();
      path.moveTo(stroke.points.first.dx, stroke.points.first.dy);

      for (int i = 1; i < stroke.points.length; i++) {
        path.lineTo(stroke.points[i].dx, stroke.points[i].dy);
      }

      canvas.drawPath(path, paint);
    }
  }

  @override
  bool shouldRepaint(MaskPainter oldDelegate) => true;
}

class MaskEditor extends StatefulWidget {
  final File imageFile;
  final int width;
  final int height;

  const MaskEditor({
    required this.imageFile,
    required this.width,
    required this.height,
    super.key,
  });

  @override
  State<MaskEditor> createState() => _MaskEditorState();
}

class _MaskEditorState extends State<MaskEditor> {
  ui.Image? image;
  final List<Stroke> strokes = [];
  Stroke? currentStroke;
  double brushSize = 10.0;

  @override
  void initState() {
    super.initState();
    _loadImage();
  }

  Future<void> _loadImage() async {
    final bytes = await widget.imageFile.readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    setState(() {
      image = frame.image;
    });
  }

  Future<Uint8List> _generateMaskData() async {
    // Create a single-channel mask with all zeros (black)
    final maskData = Uint8List(widget.width * widget.height);

    // Create an in-memory canvas to draw the mask
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder,
        Rect.fromLTWH(0, 0, widget.width.toDouble(), widget.height.toDouble()));

    // Fill with black background
    canvas.drawRect(
      Rect.fromLTWH(0, 0, widget.width.toDouble(), widget.height.toDouble()),
      Paint()..color = Colors.black,
    );

    // Draw the strokes in white
    final paint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    for (final stroke in strokes) {
      if (stroke.points.isEmpty) continue;

      paint.strokeWidth = stroke.brushSize;

      final path = Path();
      path.moveTo(stroke.points.first.dx, stroke.points.first.dy);

      for (int i = 1; i < stroke.points.length; i++) {
        path.lineTo(stroke.points[i].dx, stroke.points[i].dy);
      }

      canvas.drawPath(path, paint);
    }

    // Convert the drawing to an image
    final picture = recorder.endRecording();
    final maskImage = await picture.toImage(widget.width, widget.height);
    final byteData =
        await maskImage.toByteData(format: ui.ImageByteFormat.rawRgba);

    if (byteData == null) {
      throw Exception('Failed to generate mask data');
    }

    // Convert RGBA data to single-channel binary mask (only 0 or 255)
    final bytes = byteData.buffer.asUint8List();
    for (int i = 0; i < maskData.length; i++) {
      // Use red channel for threshold (white = 255, black = 0)
      maskData[i] = bytes[i * 4] > 128 ? 255 : 0;
    }

    return maskData;
  }

  @override
  Widget build(BuildContext context) {
    if (image == null) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Create Mask'),
        actions: [
          IconButton(
            icon: const Icon(Icons.undo),
            onPressed: strokes.isNotEmpty
                ? () {
                    setState(() {
                      strokes.removeLast();
                    });
                  }
                : null,
            tooltip: 'Undo Last Stroke',
          ),
          IconButton(
            icon: const Icon(Icons.clear_all),
            onPressed: strokes.isNotEmpty
                ? () {
                    setState(() {
                      strokes.clear();
                    });
                  }
                : null,
            tooltip: 'Clear All',
          ),
          IconButton(
            icon: const Icon(Icons.save),
            onPressed: () async {
              final maskData = await _generateMaskData();
              if (mounted) {
                Navigator.pop(context, maskData);
              }
            },
            tooltip: 'Save Mask',
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: Center(
              child: FittedBox(
                child: SizedBox(
                  width: widget.width.toDouble(),
                  height: widget.height.toDouble(),
                  child: Stack(
                    children: [
                      Image.file(
                        widget.imageFile,
                        width: widget.width.toDouble(),
                        height: widget.height.toDouble(),
                        fit: BoxFit.fill,
                      ),
                      // This is key: using exact dimensions of the original image
                      SizedBox(
                        width: widget.width.toDouble(),
                        height: widget.height.toDouble(),
                        child: ClipRect(
                          child: Stack(
                            children: [
                              CustomPaint(
                                painter: MaskPainter(strokes, opacity: 0.7),
                                size: Size(widget.width.toDouble(),
                                    widget.height.toDouble()),
                              ),
                              GestureDetector(
                                onPanStart: (details) {
                                  RenderBox box =
                                      context.findRenderObject() as RenderBox;
                                  Offset localPosition =
                                      box.globalToLocal(details.globalPosition);

                                  // Adjust for FittedBox scaling
                                  RenderBox sizedBox =
                                      context.findRenderObject() as RenderBox;
                                  Rect sizedBoxRect =
                                      Offset.zero & sizedBox.size;
                                  Rect imageRect = Rect.fromLTWH(
                                    (sizedBoxRect.width - widget.width) / 2,
                                    (sizedBoxRect.height - widget.height) / 2,
                                    widget.width.toDouble(),
                                    widget.height.toDouble(),
                                  );

                                  // Check if within the image bounds
                                  if (!imageRect.contains(localPosition)) {
                                    return;
                                  }

                                  // Normalize coordinates to the original image dimensions
                                  double x =
                                      (localPosition.dx - imageRect.left);
                                  double y = (localPosition.dy - imageRect.top);

                                  setState(() {
                                    currentStroke =
                                        Stroke([Offset(x, y)], brushSize);
                                    strokes.add(currentStroke!);
                                  });
                                },
                                onPanUpdate: (details) {
                                  if (currentStroke == null) return;

                                  RenderBox box =
                                      context.findRenderObject() as RenderBox;
                                  Offset localPosition =
                                      box.globalToLocal(details.globalPosition);

                                  // Adjust for FittedBox scaling
                                  RenderBox sizedBox =
                                      context.findRenderObject() as RenderBox;
                                  Rect sizedBoxRect =
                                      Offset.zero & sizedBox.size;
                                  Rect imageRect = Rect.fromLTWH(
                                    (sizedBoxRect.width - widget.width) / 2,
                                    (sizedBoxRect.height - widget.height) / 2,
                                    widget.width.toDouble(),
                                    widget.height.toDouble(),
                                  );

                                  // Normalize coordinates to the original image dimensions
                                  double x =
                                      (localPosition.dx - imageRect.left);
                                  double y = (localPosition.dy - imageRect.top);

                                  // Clamp to image boundaries
                                  x = x.clamp(0, widget.width.toDouble());
                                  y = y.clamp(0, widget.height.toDouble());

                                  setState(() {
                                    currentStroke!.points.add(Offset(x, y));
                                  });
                                },
                                onPanEnd: (_) {
                                  setState(() {
                                    currentStroke = null;
                                  });
                                },
                              ),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Row(
              children: [
                const Text('Brush Size:'),
                Expanded(
                  child: Slider(
                    value: brushSize,
                    min: 1.0,
                    max: 50.0,
                    onChanged: (value) {
                      setState(() {
                        brushSize = value;
                      });
                    },
                  ),
                ),
                Text('${brushSize.toInt()}px'),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
