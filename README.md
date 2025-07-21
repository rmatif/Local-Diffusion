
# Local Diffusion
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Flutter Badge](https://img.shields.io/badge/Flutter-%2302569B.svg?style=flat&logo=Flutter&logoColor=white)](https://flutter.dev) [![Android Badge](https://img.shields.io/badge/Android-3DDC84?style=flat&logo=android&logoColor=white)](https://developer.android.com) [![Ko-fi Badge](https://img.shields.io/badge/Ko--fi-F16061?style=flat&logo=ko-fi&logoColor=white)](https://ko-fi.com/rmatif)<br>
[![Latest Release](https://img.shields.io/github/v/release/rmatif/Local-Diffusion?label=Latest%20Release&color=brightgreen)](https://github.com/rmatif/Local-Diffusion/releases/latest)

<br>

**Run the latest diffusion models directly on your mobile device.**

<br> 
<p align="center"> 
  <img src="assets/icon/icon.png" alt="App Icon" width="256" style="border-radius: 35%;"> 
</p>
<br>

Local Diffusion is a Flutter application that brings Diffusion models to Android, powered by the amazing stable-diffusion.cpp

## Download

Download and install the latest `.apk` from the GitHub Releases page:

[![Latest Release](https://img.shields.io/github/v/release/rmatif/Local-Diffusion?label=Download%20Latest%20APK&color=brightgreen)](https://github.com/rmatif/Local-Diffusion/releases/latest)

## ✨ Key Features

-   **📱 Truly Local Inference:** Generate images entirely on your Android device.
    
-   **🚀 Broad Model Compatibility:**
    
    -   Supports a wide range of architectures: SD1.x, SD2.x, SDXL, SD3/SD3.5, Flux/Flux-schnell, SD-Turbo, SDXL-Turbo, and more.
        
    -   Load models directly from popular sources like **Hugging Face** and **Civitai**.
        
    -   Works with common model formats: .safetensors and .ckpt.
        
-   **⚡ On-the-Fly Quantization:** Automatically quantize full-precision models during loading to save memory and potentially increase speed. Supported formats: q8_0, q6_k, q5_0, q5_1, q5_1k, q4_0, q4_1, q4_k, q3_k, q2_k.
    
-   **🏎️ Performance Optimizations:**
    
    -   **Flash Attention:** Reduces memory usage during inference 
        
    -   **TAESD:** Significantly speeds up the image decoding process.
        
    -   **VAE Tiling:** Reduces memory consumption during the VAE decoding stage, crucial for larger images on memory-constrained devices.
  
        
-   **🎨 Advanced Generation Capabilities:**
    
    -   **ControlNet:** Guide image generation with precise control (includes Scribble-to-Image).
        
    -   **PhotoMaker:** Generate custom portraits from reference images.
        
    -   **Img2Img:** Generate images based on an initial input image.
        
    -   **Inpainting/Outpainting:** Modify specific parts of an image or expand its canvas.
        
    -   **LoRA Support:** Apply Low-Rank Adaptations to customize model outputs.
        
    -   **Negative Prompts:** Specify what you don't want to see in the image.
        
    -   **Token Weighting:** Emphasize or de-emphasize specific parts of your prompt.
        
    -   **Multiple Sampling Methods:** Euler A, Euler, Heun, DPM2, DPM++ 2M, DPM++ 2M v2, DPM++ 2S a, LCM, TCD, DDIM
        
-   **💻 GPU Acceleration (Experimental):**
    
    -   **Vulkan:** Currently performs ~2x slower than CPU. Performance improvements are planned.
        
    -   **OpenCL:** Only supports Adreno 7xx GPUs for now. Optimized for Q4_0 quantization; also supports Q8_0 and FP16. Operations or devices outside these specifics may fall back to CPU execution.
        
    

## 📊 Memory Usage


### Peak Memory Usage in MB
 Benchmark settings: TAE + Flash Attention + VAE Tiling + only Clip_L Q8_0 for DiT/MMDiT architecture (Clip_G and T5_XXL were skipped)
| Model          | Resolution | Q2_K (MB) | Q4_0 (MB) | Q8_0 (MB) | fp16 (MB) |
|----------------|------------|-----------|-----------|-----------|-----------|
| **SD 1.5**     | 256x256    | 1431      | 1461      | 1648      | 1997      |
|                | 512x512    | 1870      | 1900      | 2087      | 2436      |
|                | 768x768    | 4001      | 4031      | 4218      | 4567      |
| **SDXL**       | 256x256    | 1868      | 2450      | 3889      | 6586      |
|                | 512x512    | 2228      | 2810      | 4249      | 6946      |
|                | 768x768    | 2228      | 2810      | 4249      | 6946      |
|                | 1024x1024  | 2228      | 2810      | 4249      | 6946      |
| **SD 2.1**     | 256x256    | 1510      | 1593      | 1892      | 2452      |
|                | 512x512    | 1870      | 1953      | 2252      | 2812      |
|                | 768x768    | 1870      | 1953      | 2252      | 2812      |
| **SD3.5 Medium**| 256x256    | 1790      | 2315      | 3433      | 5420      |
|                | 512x512    | 2150      | 2675      | 3793      | 5780      |
|                | 768x768    | 2289      | 2814      | 3932      | 5919      |
|                | 1024x1024  | 3437     | 3962      | 5080      | 7067      |
| **SD3.5 Large** | 256x256    | 4630      | 5027      | 8848      | -         |
|                | 512x512    | 4990      | 5387      | 9208      | -         |
|                | 768x768    | 4990      | 5387      | 9208      | -         |
|                | 1024x1024  | 7271      | 7668      | 11489     | -         |
| **FLUX.1**     | 256x256    | 4168      | 6813      | 12456     | -         |
|                | 512x512    | 4528      | 7173      | 12816     | -         |
|                | 768x768    | 4551      | 7196      | 12839     | -         |
|                | 1024x1024  | 4889      | 7534      | 13177     | -         |

## 🗺️ Roadmap

-   iOS Support
    
-   Performance Improvements
    

## 🛠️ Building

Ensure you have the Flutter SDK and Android development environment set up.

### Run in Debug/Release Mode

`flutter run`

# Release
`flutter run --release`
    

### Build Release APK

      `flutter build apk --release`
    
The output APK will be located at build/app/outputs/flutter-apk/app-release.apk.

## Credits

-   **Core Engine:**  [stable-diffusion.cpp by leejet](https://www.github.com/leejet/stable-diffusion.cpp)
    
-   **UI Components:**  [shadcn/ui & flutter-shadcn-ui by nank1ro](https://www.github.com/nank1ro/flutter-shadcn-ui)
