# TensorFold MOSS Image

`moss_image.workflow.json` uses ComfyUI's text encoder, VAE and image saving nodes around one TensorFold CUDA denoiser node. `moss_image.api.json` is the matching API prompt. Preview as Text displays the runtime receipt.

## Setup

The first MOSS-only PR covers experimental BF16 generation on Linux aarch64 with NVIDIA GB10, SM121. Install the separate TensorFold MOSS runtime before opening the workflow. It needs Git, Zig 0.17.0, the CUDA 13 toolkit and cuBLASLt 13. The companion must be available before this PR is submitted upstream.

SM120 is unvalidated and outside this first PR's generation support scope. The worker still compiles and admits SM120/SM121. Linux x86_64 has software, native-build and install checks only; no SM120 image workflow, numerical comparison or live kernel-contract check has run.

```sh
python /path/to/TensorFold/tools/comfyui_runtime/tensorfold_runtime.py install \
  --comfy-root /path/to/ComfyUI --backend cuda \
  --zig /path/to/zig --nvcc /usr/local/cuda/bin/nvcc
```

The node never downloads files or checks for updates. It verifies `models/tensorfold/runtime.json`, the pinned media revision, the numerical compatibility patch identity and the installed executable's checksum. Restart ComfyUI after runtime updates.

## Models and generation

Use the official single-file BF16 MOSS-Image-Base transformer, its compatible Qwen3.5 text encoder and Flux2-format VAE in ComfyUI's usual model directories. Existing configured model paths work. The installer does not fetch weights.

The official transformer has fused QKV weights. Prepare a separate file once with the offline companion tool, which preserves the tensor payload and requires about 10 GB of extra disk space:

```sh
python /path/to/TensorFold/tools/comfyui_runtime/prepare_moss.py \
  /path/to/ComfyUI/models/diffusion_models/moss_image_base_bf16.safetensors \
  /path/to/ComfyUI/models/diffusion_models/moss_image_base_tensorfold_bf16.safetensors
```

Open the example and replace the three `CHOOSE_...` model selections. Select the prepared transformer in TensorFold MOSS Image, keep CLIPLoader's type at `moss_image`, enter both prompts and queue generation. The LATENT connects to VAE Decode and Save Image. The receipt reports the runtime and patch identities, timings, requested/effective steps, seed and noise/sigma hashes.

Sampling uses the MOSS V2 shift-5 schedule snapped to its 1000-point training grid, guidance 3 and negative skip layer 22. Some requested step counts collapse after snapping. The output is `[1, 128, height/16, width/16]`. ComfyUI unloads its models before the native worker starts. Cancellation stops the worker process group and removes temporary files. The worker sees the device selected by ComfyUI.

The node accepts one unscheduled conditioning entry per branch and binary attention masks. Quantized/sharded checkpoints, LoRA, edit/reference latents, controls, regional/scheduled conditioning and batches larger than one are unsupported. Only the transformer is replaced; the standard encoder and VAE remain in ComfyUI.

## Validation

```sh
python -m pytest -q tests-unit/tensorfold
python -m ruff check comfy/tensorfold comfy_extras/nodes_tensorfold.py tests-unit/tensorfold
```

Compare complete native/reference latents with identical weights, conditioning, noise, sigmas and settings. Decode both with the same VAE. Record cold load, denoising and total wall time separately before making performance claims. The local qualification tool and receipts are supplied with the review package.
