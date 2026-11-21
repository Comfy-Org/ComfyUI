# LTX on Intel Iris Xe (Linux)

This opt-in configuration targets the Iris Xe integrated GPU in the Core i5-1340P. PyTorch 2.15.0+xpu can enumerate this device but reports it as unsupported. The LTX and GGUF workarounds run selected operations on the CPU when handling XPU tensors. The diffusion model still runs on the GPU.

## Install

If importing the four exported ComfyUI commits into another checkout, their base is `995456d0188ffb51032c28f41527cfa82d87d8a9`. On a branch at that revision, import the numbered mail patches with the following command. The embedded GGUF diffs contain CRLF context matching the pinned custom node. Preserving those bytes lets the subsequent GGUF installation apply cleanly.

```bash
git am --keep-cr --3way /path/to/outpatches/000[1-4]-*.patch
```

From the ComfyUI repository root, use a Python 3.12 environment with an Intel compute driver and access to the GPU render device:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r extra/iris-xe/requirements-xpu.txt
git clone https://github.com/city96/ComfyUI-GGUF.git custom_nodes/ComfyUI-GGUF
git -C custom_nodes/ComfyUI-GGUF switch --detach 6ea2651e7df66d7585f6ffee804b20e92fb38b8a
git -C custom_nodes/ComfyUI-GGUF apply --check "$PWD/extra/iris-xe/gguf-xpu.patch"
git -C custom_nodes/ComfyUI-GGUF apply "$PWD/extra/iris-xe/gguf-xpu.patch"
python -m pip install -r requirements.txt -r custom_nodes/ComfyUI-GGUF/requirements.txt
python -m pip check
python -c 'import torch; print(torch.__version__, torch.xpu.get_device_name(0))'
```

The separate `gguf-gemma4.patch` adds Gemma 4 to the custom node's GGUF text loader. Apply it in the same checkout for Gemma 4 GGUF text encoders that embed the full Hugging Face tokenizer in the `tokenizer.huggingface.json` metadata field. The LTX workflow using a safetensors text encoder uses the built-in CLIPLoader.

## Run

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
python -u main.py --intel-iris-xe --cpu-vae --fp32-unet --fp32-vae \
    --lowvram --disable-dynamic-vram
```

The switch enables the same workarounds as `LTX25_IRIS_CPU_UNPACK=1`. They cover the sampler's empty-latent check, LTX positional tables and keyframe masks, GGUF Q4_K/Q6_K unpacking and GGUF FP16-to-FP32 weight casts. For the 9-frame LTX-2.5 workflow use the CPU device in CLIPLoader. With these flags VideoVAE decodes on the CPU and the latent upscaler runs on XPU in FP32.

For the 9-frame 640×384 text-to-video benchmark on the i5-1340P, omit `--cpu-vae` to run the FP32 VideoVAE on XPU as well. Use the same VAE device for both sides of a sampler comparison.

For longer videos, add `--use-quad-cross-attention` to the Iris Xe launch command. On hp8, the default PyTorch SDPA path raised an unsupported free-memory query during the first sampling step at 960×544 and 81 frames. Sub-quadratic attention sizes its blocks using both the XPU allocator estimate and the available system RAM (including the cgroup limit). This accounts for memory shared with CPU models. `--cache-lru 5` controls reuse of node results between prompts in the same process.

## Mixed-quantization LTX GGUF files

For `ltx-2.5-22b-distilled-UD-Q5_K_XL.gguf`, apply the following additional diff to the custom node after the Iris Xe and native unpack patches. The checkpoint also uses BF16, Q2_K, Q3_K and Q8_0. On the tested Iris Xe, GPU unpacking of these four formats and Q5_K fails with Level Zero error 45. This fallback uses the plugin's existing CPU decoder for those formats and transfers the result back to the XPU. Q4_K and Q6_K retain the previously selected path.

```bash
node=custom_nodes/ComfyUI-GGUF
git -C "$node" apply --check "$PWD/extra/iris-xe/gguf-mixed-qtypes.patch"
git -C "$node" apply "$PWD/extra/iris-xe/gguf-mixed-qtypes.patch"
```

Restart the server to load the plugin edit. Keep the original CFG++ workflow to use both positive and negative conditioning at CFG 1.

## Optional native GGUF weight unpacking

On Linux x86-64 with AVX2 and F16C, an optional CPU decoder unpacks Q4_K and Q6_K weights directly into the FP32 buffer consumed by the XPU denoiser. It preserves the default GGUF decoder's intermediate FP16 rounding. Install it after applying `gguf-xpu.patch`, from the ComfyUI repository root, using GCC with OpenMP support.

```bash
node=custom_nodes/ComfyUI-GGUF
git -C "$node" apply --check "$PWD/extra/iris-xe/gguf-native-unpack.patch"
g++ -O3 -fPIC -shared -fopenmp -ffp-contract=off extra/iris-xe/unpack.cpp -o "$node/libiris_unpack.so"
git -C "$node" apply "$PWD/extra/iris-xe/gguf-native-unpack.patch"
```

Set `LTX25_IRIS_NATIVE_UNPACK=1` in the server environment and launch with the same Iris Xe settings as above. Set it to `0` for the standard CPU unpacker. Restart the server between modes. The first eligible tensor logs `Iris Xe: native GGUF CPU unpack active`. This path applies to FP32 output with default or FP16 intermediate precision.

For an end-to-end comparison, render the same fixed-seed 9-frame, 640×384 graph with eight sampling steps in each fresh process. Record both the eight-step sampler time and the complete prompt time. HD refinement can be measured separately by running the matching HD graph immediately after the preview in each process. Verify that the HD log shows just the three refinement steps.
