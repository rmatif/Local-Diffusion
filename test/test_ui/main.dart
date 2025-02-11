import 'package:flutter/material.dart';
import 'package:shadcn_ui/shadcn_ui.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return ShadApp(
      darkTheme: ShadThemeData(
        brightness: Brightness.dark,
        colorScheme: const ShadSlateColorScheme.dark(),
      ),
      home: const StableDiffusionApp(),
    );
  }
}

class StableDiffusionApp extends StatefulWidget {
  const StableDiffusionApp({super.key});
  @override
  State<StableDiffusionApp> createState() => _StableDiffusionAppState();
}

class _StableDiffusionAppState extends State<StableDiffusionApp> {
  bool useTAESD = false;
  bool useVAETiling = false;
  double clipSkip = 0;
  bool useVAE = false;
  String samplingMethod = 'euler_a';
  double cfg = 7;
  int steps = 25;
  int width = 512;
  int height = 512;
  String seed = "-1";
  String prompt = '';
  String negativePrompt = '';
  double progress = 0;
  String status = '';
  Map<String, bool> loadedComponents = {};
  String loadingText = '';

  final List<String> samplingMethods = const [
    'euler_a',
    'euler',
    'heun',
    'dpm2',
    'dmp ++2s_a',
    'dmp++2m',
    'dpm++2mv2',
    'ipndm',
    'ipndm_v',
    'lcm'
  ];

  List<int> getWidthOptions() {
    List<int> opts = [];
    for (int i = 128; i <= 512; i += 64) {
      opts.add(i);
    }
    return opts;
  }

  List<int> getHeightOptions() {
    return getWidthOptions();
  }

  void simulateLoading(String component) {
    setState(() {
      loadingText = 'Loading $component...';
    });

    // Simulate loading completion after 2 seconds
    Future.delayed(const Duration(seconds: 2), () {
      setState(() {
        loadedComponents[component] = true;
        loadingText = '';
      });
    });
  }

  void showModelLoadDialog() {
    String selectedQuantization = 'NONE';
    String selectedSchedule = 'DEFAULT';
    bool useFlashAttention = false;

    final List<String> quantizationOptions = [
      'NONE',
      'Q8_0',
      'Q8_1',
      'Q8_K',
      'Q6_K',
      'Q5_0',
      'Q5_1',
      'Q5_K',
      'Q4_0',
      'Q4_1',
      'Q4_K',
      'Q3_K',
      'Q2_K'
    ];

    final List<String> scheduleOptions = [
      'DEFAULT',
      'DISCRETE',
      'KARRAS',
      'EXPONENTIAL',
      'AYS'
    ];

    showShadDialog(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setState) => ShadDialog.alert(
          constraints: const BoxConstraints(maxWidth: 300),
          title: const Text('Load Model Settings'),
          description: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Text('Quantization Type:'),
                  const SizedBox(width: 8),
                  Expanded(
                    child: ShadSelect<String>(
                      placeholder: Text(selectedQuantization),
                      onChanged: (value) => setState(
                          () => selectedQuantization = value ?? 'NONE'),
                      options: quantizationOptions
                          .map((type) => ShadOption(
                                value: type,
                                child: Text(type),
                              ))
                          .toList(),
                      selectedOptionBuilder: (context, value) => Text(value),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              Row(
                children: [
                  const Text('Schedule:'),
                  const SizedBox(width: 8),
                  Expanded(
                    child: ShadSelect<String>(
                      placeholder: Text(selectedSchedule),
                      onChanged: (value) =>
                          setState(() => selectedSchedule = value ?? 'DEFAULT'),
                      options: scheduleOptions
                          .map((schedule) => ShadOption(
                                value: schedule,
                                child: Text(schedule),
                              ))
                          .toList(),
                      selectedOptionBuilder: (context, value) => Text(value),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              ShadSwitch(
                value: useFlashAttention,
                onChanged: (v) => setState(() => useFlashAttention = v),
                label: const Text('Use Flash Attention'),
              ),
            ],
          ),
          actions: [
            ShadButton.outline(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Cancel'),
            ),
            ShadButton(
              onPressed: () {
                simulateLoading('Model');
                Navigator.of(context).pop();
              },
              child: const Text('Load Model'),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = ShadTheme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Local Diffusion',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        backgroundColor: theme.colorScheme.background,
        elevation: 0,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Status Section at the top
            if (loadingText.isNotEmpty || loadedComponents.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(bottom: 16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: loadedComponents.entries
                          .map((entry) => Text(
                                '${entry.key} loaded ✓',
                                style: theme.textTheme.p.copyWith(
                                  color: Colors.green,
                                  fontWeight: FontWeight.bold,
                                ),
                              )
                                  .animate()
                                  .fadeIn(
                                      duration:
                                          const Duration(milliseconds: 500))
                                  .slideY(begin: -0.2, end: 0))
                          .toList(),
                    ),
                    if (loadingText.isNotEmpty) const SizedBox(height: 8),
                    if (loadingText.isNotEmpty)
                      TweenAnimationBuilder(
                        duration: const Duration(milliseconds: 800),
                        tween: Tween(begin: 0.0, end: 1.0),
                        builder: (context, value, child) {
                          return Text(
                            '$loadingText${'..' * ((value * 3).floor())}',
                            style: theme.textTheme.p.copyWith(
                              color: Colors.orange,
                              fontWeight: FontWeight.bold,
                            ),
                          );
                        },
                      ).animate().fadeIn(),
                  ],
                ),
              ),

            Row(
              children: [
                ShadButton(
                  child: const Text('Load Model'),
                  onPressed: showModelLoadDialog,
                ),
                const SizedBox(width: 8),
                IconButton(
                  icon: const Icon(LucideIcons.circleHelp,
                      size: 24, color: Colors.white),
                  onPressed: () {
                    showShadDialog(
                      context: context,
                      builder: (context) => ShadDialog.alert(
                        title: const Text('Model Information'),
                        constraints: const BoxConstraints(maxWidth: 400),
                        description: const Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('Load an SD or Flux model'),
                            SizedBox(height: 8),
                            Text('Supported models:',
                                style: TextStyle(fontWeight: FontWeight.bold)),
                            Text('\nSD 1.x, SD 2.x, SDXL, SDXL Turbo'),
                            Text('SD 3 Medium/Large, SD 3.5 Medium/Large'),
                            Text('Flux 1 Dev, Flux 1 Schnell, Flux Lite'),
                            SizedBox(height: 16),
                            Text('Supported formats:',
                                style: TextStyle(fontWeight: FontWeight.bold)),
                            Text('\nSafeTensors, CKPT, GGUF'),
                            Text('FP32/FP16 and quantized GGUF formats'),
                            Text(
                                'Distilled formats: Turbo, LCM, Lightning, Hyper'),
                            SizedBox(height: 16),
                            Text('Where to download models?',
                                style: TextStyle(fontWeight: FontWeight.bold)),
                            Text('\nRecommended websites:'),
                            Text('• civitai.com'),
                            Text('• huggingface.co'),
                          ],
                        ),
                        actions: [
                          ShadButton(
                            child: const Text('Close'),
                            onPressed: () => Navigator.of(context).pop(),
                          ),
                        ],
                      ),
                    );
                  },
                ),
              ],
            ),

            const SizedBox(height: 8),
            Row(
              children: [
                ShadButton(
                  child: const Text('Load TAESD'),
                  onPressed: () => simulateLoading('TAESD'),
                ),
                const SizedBox(width: 8),
                ShadCheckbox(
                  value: useTAESD,
                  onChanged: (bool v) => setState(() => useTAESD = v),
                  label: const Text('Use TAESD'),
                ),
              ],
            ),
            const SizedBox(height: 16),
            ShadAccordion<Map<String, dynamic>>(
              children: [
                ShadAccordionItem<Map<String, dynamic>>(
                  value: const {},
                  title: const Text('Advanced Options'),
                  child: Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Column(
                      children: [
                        Row(
                          children: [
                            Expanded(
                              child: ShadButton(
                                child: const Text('Load Lora'),
                                onPressed: () => simulateLoading('Lora'),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        Row(
                          children: [
                            const Text('Clip Skip'),
                            const SizedBox(width: 8),
                            Expanded(
                              child: ShadSlider(
                                initialValue: clipSkip,
                                min: 0,
                                max: 2,
                                divisions: 2,
                                onChanged: (v) => setState(() => clipSkip = v),
                              ),
                            ),
                            Text(clipSkip.toInt().toString()),
                          ],
                        ),
                        const SizedBox(height: 16),
                        Row(
                          children: [
                            Expanded(
                              child: ShadButton(
                                child: const Text('Load Clip_L'),
                                onPressed: () => simulateLoading('Clip_L'),
                              ),
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: ShadButton(
                                child: const Text('Load Clip_G'),
                                onPressed: () => simulateLoading('Clip_G'),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        Row(
                          children: [
                            Expanded(
                              child: ShadButton(
                                child: const Text('Load T5XXL'),
                                onPressed: () => simulateLoading('T5XXL'),
                              ),
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: ShadButton(
                                child: const Text('Load Embed'),
                                onPressed: () => simulateLoading('Embeddings'),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        Row(
                          children: [
                            Expanded(
                              child: ShadButton(
                                child: const Text('Load VAE'),
                                onPressed: () => simulateLoading('VAE'),
                              ),
                            ),
                            const SizedBox(width: 8),
                            ShadCheckbox(
                              value: useVAE,
                              onChanged: (bool v) => setState(() => useVAE = v),
                              label: const Text('Use VAE'),
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        ShadCheckbox(
                          value: useVAETiling,
                          onChanged: (bool v) =>
                              setState(() => useVAETiling = v),
                          label: const Text('VAE Tiling'),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            ShadInput(
              placeholder: const Text('Prompt'),
              onChanged: (String? v) => setState(() {
                prompt = v ?? '';
              }),
            ),
            const SizedBox(height: 16),
            ShadInput(
              placeholder: const Text('Negative Prompt'),
              onChanged: (String? v) => setState(() {
                negativePrompt = v ?? '';
              }),
            ),
            Row(
              children: [
                const Text('Sampling Method'),
                const SizedBox(width: 8),
                Expanded(
                  child: ShadSelect<String>(
                    placeholder: const Text('euler_a'),
                    options: samplingMethods
                        .map((method) => ShadOption(
                              value: method,
                              child: Text(method),
                            ))
                        .toList(),
                    selectedOptionBuilder: (context, value) => Text(value),
                    onChanged: (String? value) =>
                        setState(() => samplingMethod = value ?? 'euler_a'),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 16),
            Row(
              children: [
                const Text('CFG'),
                const SizedBox(width: 8),
                Expanded(
                  child: ShadSlider(
                    initialValue: cfg,
                    min: 1,
                    max: 20,
                    divisions: 38,
                    onChanged: (v) => setState(() => cfg = v),
                  ),
                ),
                Text(cfg.toStringAsFixed(1)),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                const Text('Steps'),
                const SizedBox(width: 8),
                Expanded(
                  child: ShadSlider(
                    initialValue: steps.toDouble(),
                    min: 1,
                    max: 50,
                    divisions: 49,
                    onChanged: (v) => setState(() => steps = v.toInt()),
                  ),
                ),
                Text(steps.toString()),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                const Text('Width'),
                const SizedBox(width: 8),
                Expanded(
                  child: ShadSelect<int>(
                    placeholder: const Text('512'),
                    options: getWidthOptions()
                        .map((w) => ShadOption(
                              value: w,
                              child: Text(w.toString()),
                            ))
                        .toList(),
                    selectedOptionBuilder: (context, value) =>
                        Text(value.toString()),
                    onChanged: (int? value) {
                      if (value != null) {
                        setState(() => width = value);
                      }
                    },
                  ),
                ),
                const SizedBox(width: 16),
                const Text('Height'),
                const SizedBox(width: 8),
                Expanded(
                  child: ShadSelect<int>(
                    placeholder: const Text('512'),
                    options: getHeightOptions()
                        .map((h) => ShadOption(
                              value: h,
                              child: Text(h.toString()),
                            ))
                        .toList(),
                    selectedOptionBuilder: (context, value) =>
                        Text(value.toString()),
                    onChanged: (int? value) {
                      if (value != null) {
                        setState(() => height = value);
                      }
                    },
                  ),
                ),
              ],
            ),

            const SizedBox(height: 16),
            const Text('Seed (-1 for random)'),
            const SizedBox(height: 8),
            ShadInput(
              placeholder: const Text('Seed'),
              keyboardType: TextInputType.number,
              onChanged: (String? v) => setState(() {
                seed = v ?? "-1";
              }),
              initialValue: seed,
            ),
            const SizedBox(height: 16),
            ShadButton(
              child: const Text('Generate'),
              onPressed: () {
                setState(() {
                  status = 'Generating image...';
                  progress = 0;
                });
                Future.delayed(
                    const Duration(milliseconds: 100), updateProgress);
              },
            ),
            const SizedBox(height: 16),
            LinearProgressIndicator(
              value: progress,
              backgroundColor: theme.colorScheme.background,
              color: theme.colorScheme.primary,
            ),
            const SizedBox(height: 8),
            Text(status, style: theme.textTheme.p),
          ],
        ),
      ),
    );
  }

  void updateProgress() {
    if (progress < 1) {
      setState(() {
        progress += 0.1;
        status = 'Generating image... ${(progress * 100).toInt()}%';
      });
      Future.delayed(const Duration(milliseconds: 300), updateProgress);
    } else {
      setState(() {
        status = 'Image generated';
      });
    }
  }
}
