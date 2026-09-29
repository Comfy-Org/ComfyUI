<div align="center">

# ComfyUI

**The most powerful and modular AI engine for content creation.**

[![Website](https://img.shields.io/badge/ComfyOrg-4285F4?style=flat)](https://www.comfy.org/)
[![Discord](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fcomfyorg%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&logo=discord&logoColor=white&label=Discord&color=green&suffix=%20total)](https://discord.com/invite/comfyorg)
[![Twitter](https://img.shields.io/twitter/follow/ComfyUI)](https://x.com/ComfyUI)
[![Matrix](https://img.shields.io/badge/Matrix-000000?style=flat&logo=matrix&logoColor=white)](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)
[![Release](https://img.shields.io/github/v/release/Comfy-Org/ComfyUI?style=flat&sort=semver)](https://github.com/Comfy-Org/ComfyUI/releases)
[![Downloads](https://img.shields.io/github/downloads/Comfy-Org/ComfyUI/total?style=flat)](https://github.com/Comfy-Org/ComfyUI/releases)

<img width="1590" height="795" alt="ComfyUI Screenshot" src="https://github.com/user-attachments/assets/36e065e0-bfae-4456-8c7f-8369d5ea48a2" />

</div>

ComfyUI is a node graph engine for generative AI. You wire up the models, samplers and
conditioning yourself, so every parameter stays under your control. It generates images,
video, audio, 3D and text on Windows, Linux and macOS.

- Native support for current open source models, with [ready-to-run templates](https://comfy.org/workflows) for each. The template library is the maintained list; the engine gains new architectures most weeks.
- Asynchronous weight streaming runs large models on as little as 4GB VRAM plus 8GB RAM while keeping the GPU saturated, alongside model offloading, quantized weights, async queueing, and re-execution of only the parts of a graph that changed.
- Reusable subgraphs, [App Mode](https://docs.comfy.org/interface/app-mode) to expose a workflow as a simple UI, and a [local API](https://docs.comfy.org/development/comfyui-server/api-examples) for production pipelines.
- Runs fully offline. The core downloads nothing unless you ask it to. [Partner nodes](https://docs.comfy.org/tutorials/partner-nodes/overview#partner-nodes) and [API nodes](https://docs.comfy.org/tutorials/api-nodes/overview) reach closed source models such as Nano Banana and Seedance; `--disable-api-nodes` turns them off.
- Extend it with [custom nodes](https://docs.comfy.org/custom-nodes/overview), managed by [ComfyUI-Manager](https://docs.comfy.org/manager/overview) (`pip install -r manager_requirements.txt`, then run with `--enable-manager`).

## Install

The [desktop app](https://www.comfy.org/download) is the easiest way in, and the right
answer for most people on Windows and macOS.

| Path | Use it when |
| --- | --- |
| [Desktop app](https://docs.comfy.org/installation/desktop/overview) | You want a normal installer. Windows and macOS. |
| [Windows portable](https://docs.comfy.org/installation/comfyui_portable_windows) | You want a self-contained build. NVIDIA, AMD, Intel or CPU only. |
| [Manual install](https://docs.comfy.org/installation/manual_install) | Linux, or you need control over Python and PyTorch. |
| [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started) | You prefer a CLI: `pip install comfy-cli && comfy install`. |
| [Comfy Cloud](https://www.comfy.org/cloud) | You have no local GPU. |

To run from this repo, first install PyTorch for your GPU using the
[per-vendor commands](https://docs.comfy.org/installation/manual_install) (NVIDIA CUDA,
AMD ROCm on Linux and Windows, Apple Silicon; Intel Arc is covered under
[Intel GPU issues](https://docs.comfy.org/troubleshooting/overview#intel-gpu-issues)),
then:

```bash
git clone https://github.com/Comfy-Org/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
python main.py
```

Python 3.13 is best supported and 3.14 mostly works. PyTorch 2.7 is the floor, cu130 or
newer is required on NVIDIA 20 series and above, and anything older than six months
should be updated. Put checkpoints in `models/checkpoints` and VAEs in `models/vae`, or
point ComfyUI at folders you already have via [`extra_model_paths.yaml`](extra_model_paths.yaml.example).

Ascend NPU, Cambricon MLU and Iluvatar Corex work through their vendor PyTorch builds:
install [torch_npu](https://ascend.github.io/docs/sources/pytorch/install.html#pytorch),
[torch_mlu](https://www.cambricon.com/docs/sdk_1.15.0/cambricon_pytorch_1.17.0/user_guide_1.9/index.html)
or [Iluvatar](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536),
then follow the steps above.

## Documentation

Full documentation lives at **[docs.comfy.org](https://docs.comfy.org)**, in English,
Chinese, Japanese and Korean.

| | |
| --- | --- |
| [Your first generation](https://docs.comfy.org/get_started/first_generation) | [Keyboard shortcuts](https://docs.comfy.org/interface/shortcuts) |
| [Workflow templates](https://comfy.org/workflows) | [Startup flags](https://docs.comfy.org/development/comfyui-server/startup-flags) |
| [Built-in node reference](https://docs.comfy.org/built-in-nodes/overview) | [Troubleshooting](https://docs.comfy.org/troubleshooting/overview) |
| [Writing custom nodes](https://docs.comfy.org/custom-nodes/overview) | [Which GPU should I buy?](https://github.com/Comfy-Org/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI) |

A few things that are easy to miss:

- Drag a generated PNG onto the canvas to [recover the whole workflow](https://docs.comfy.org/development/api-development/workflow-metadata), seeds included.
- [Prompt syntax](https://docs.comfy.org/built-in-nodes/ClipTextEncode) covers weighting with `(good code:1.2)`, wildcards with `{day|night}` and `embedding:name` for textual inversions. Dynamic prompts also accept `// C-style comments`.
- `--preview-method auto` enables live sampler previews. For higher quality, drop the [TAESD decoders](https://github.com/madebyollin/taesd/) into `models/vae_approx` and use `--preview-method taesd`.
- For TLS, generate a cert with `openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/CN=localhost"`, then pass `--tls-keyfile key.pem --tls-certfile cert.pem`.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) for how to report issues, open pull requests,
and how releases are cut.

The UI lives in [ComfyUI_frontend](https://github.com/Comfy-Org/ComfyUI_frontend) and
ships here as a [pip package](https://pypi.org/project/comfyui-frontend-package). File
UI bugs there. To run a newer frontend than the one pinned in this repo, launch with
`--front-end-version Comfy-Org/ComfyUI_frontend@latest`.

## Community

Ask questions in [#help or #feedback on Discord](https://comfy.org/discord), or on
[Matrix](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org).

_psst, we're hiring:_ [comfy.org/careers](https://www.comfy.org/careers)
