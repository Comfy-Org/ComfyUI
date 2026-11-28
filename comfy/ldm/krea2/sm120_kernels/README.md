# Krea 2 Turbo SM120 profile

This optional profile targets Windows x64, GeForce RTX 5090, CUDA 13 PyTorch, and Krea 2 Turbo FP8 scaled or INT8 ConvRot. It supports single-image sampling at variable resolutions and prompt lengths with FP16 compute and accumulation. Use Euler, Heun, or DPM++ 2M with a full `simple` schedule, CFG 1, and denoise 1. Unsupported workflows use stock sampling.

## Install and enable

Run these commands from the ComfyUI directory with the Python executable used to launch ComfyUI.

```powershell
python -m pip install --only-binary=:all: -r requirements-krea2-sm120.txt
python main.py --fp16-unet --fast fp16_accumulation --krea2-sm120
```

The optional requirements install a prebuilt adapter wheel and NVIDIA's cuBLAS package from PyPI. Allow approximately 475 MB of downloads for a first install. The profile accepts automatically selected PyTorch attention as well as either explicit `--use-pytorch-cross-attention` or `--use-sage-attention`. Other attention backends use stock sampling.

After warming the model, an eligible render logs its MXFP8, FP16, normalization, modulation, and text-preparation counts. You can add `--fast-unfiltered-rgb-png` to write lossless uncompressed RGB `SaveImage` output with text metadata. These files are larger than compressed PNGs.

## Build from source

To rebuild the adapters, use CUDA Toolkit 13.0 and the MSVC 2022 C++ x64 tools. Set `CUDA_PATH` to the toolkit or supply `-CudaRoot`. The build verifies the SHA-256 of NVIDIA's cuBLASLt 13.8.0.4 DLL. [CUDA redistributables manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.4.3.json)

```powershell
& .\comfy\ldm\krea2\sm120_kernels\build.ps1 -CublasLt C:\cuda-sdk\cublasLt64_13.dll -OutputDirectory C:\krea2-sm120
python -m pip install wheel
python comfy/ldm/krea2/sm120_kernels/pack_wheel.py C:\krea2-sm120 dist
```

For a local build, pass the output directory directly as `--krea2-sm120 C:\krea2-sm120`. The wheel contains the four compiled adapters and their source.

Projection shapes with 16,862 rows can use tuned cuBLASLt algorithms. Other shapes can use the generic MXFP8 path linked against the PyTorch cuBLASLt DLL. The FP8 projections retain the original E4M3 codes and FP32 scales. INT8 ConvRot keeps the existing Comfy Kitchen INT8 projections and can reuse the accelerated normalization, modulation, and text preparation. The CUDA modulation kernel rounds the multiply to FP16 before addition.
