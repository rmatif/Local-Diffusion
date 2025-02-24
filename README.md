# Local Diffusion

Flutter GUI wrapper for stable-diffusion.cpp - Run Stable Diffusion locally on Android

## Features

- Pure Flutter implementation using FFI bindings to stable-diffusion.cpp
- Android support (more platforms coming soon)
- Super lightweight with minimal dependencies
- Supports multiple model architectures:
    - SD1.x
    - SD2.x
    - SDXL
    - SD3/SD3.5
    - Flux/Flux-schnell
    - SD-Turbo and SDXL-Turbo

- Performance optimizations:
    - OpenBLAS acceleration
    - TAESD for faster decoding (reduces latent decode time from ~48s to ~2s)

- Advanced features:
    - LoRA support
    - On-the-fly model quantization (q8_0, q6_k, q5_0, q5_1, q5_1k ,q4_0, q4_1, q4_k, q3_k, q2_k)
    - Negative prompts
    - Token weighting

- Sampling methods:
    - Euler A
    - Euler
    - Heun
    - DPM2
    - DPM++ 2M
    - DPM++ 2M v2
    - DPM++ 2S a
    - LCM

## Flash Attention Support

When initializing a model, you can choose to load it with or without Flash Attention:

# RAM Usage Benchmarking for SD 1.5

| Precision Level | Without Flash Attention (MB) | With Flash Attention (MB) |
|------------------|-----------------------------|----------------------------|
| FP16            | 2035                        | 1969                       |
| Q8_0            | 1683                        | 1618                       |
| Q6_K            | 1637                        | 1572                       |
| Q5_0            | 1543                        | 1477                       |
| Q4_0            | 1496                        | 1431                       |
| Q3_K            | 1498                        | 1433                       |
| Q2_K            | 1462                        | 1397                       |


These measurements were taken using SD 1.5 model, benchmark on larger models is coming soon.
Other Quantization method can be implemented if really needed.
## Roadmap

- [ ] Vulkan backend for GPU acceleration
- [ ] ControlNet support
- [✓] img2img generation
- [ ] inpainting/outpainting
- [x] RealESRGAN upscaling
- [ ] iOS support

## Building

### Run in release mode
`flutter run --release`

### Release APK

`flutter build apk --release`

The APK will be available at `build/app/outputs/flutter-apk/app-release.apk`

## Credits

Based on [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
