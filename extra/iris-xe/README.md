# LTX on Intel Iris Xe (Linux)

This opt-in configuration targets the Iris Xe integrated GPU in the Core i5-1340P. PyTorch 2.15.0+xpu can enumerate this device but reports it as unsupported. The LTX and GGUF workarounds run selected operations on the CPU when handling XPU tensors. The diffusion model still runs on the GPU.

## Install

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
