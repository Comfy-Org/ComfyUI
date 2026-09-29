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

ComfyUI is the AI creation engine for visual professionals who demand control over every model, every parameter, and every output. Its powerful and modular node graph interface empowers creatives to generate images, videos, 3D models, audio, and more...

- ComfyUI natively supports the latest open-source state of the art models.
- It is by far the most optimized inference engine for diffusion models in the world with a focus on local consumer hardware.
- Can run even the biggest open source models on as low as 4GB vram + 8GB ram relatively quickly (saturating your GPU compute) using our state of the art asynchronous weight streaming technology.
- [Partner nodes](https://docs.comfy.org/tutorials/partner-nodes/overview#partner-nodes) provide access to the best closed source models such as Nano Banana, Seedance, Hunyuan3D, etc.
- It is available on Windows, Linux, and macOS, locally with our [desktop application](https://www.comfy.org/download), our [portable install](docs/installation.md#windows-portable) or on our [cloud](https://www.comfy.org/cloud).
- The most sophisticated workflows can be exposed through a simple UI thanks to App Mode.
- It integrates seamlessly into production pipelines with our API endpoints.

## Features

- A visual node graph for building and reusing image, video, audio, 3D, and text workflows without code.
- Broad native model support. Browse the [workflow library](https://comfy.org/workflows/) for maintained, ready-to-run templates covering image generation and editing, video, audio, 3D, vision, and text.
- Efficient local execution with asynchronous queueing, partial graph re-execution, smart VRAM and RAM management, model offloading, and support for quantized models.
- Reusable subgraphs, workflow templates, [App Mode](https://docs.comfy.org/interface/app-mode), and a [local API](https://docs.comfy.org/development/comfyui-server/api-examples) for integrating workflows into applications.
- Load complete checkpoints or separate diffusion models, VAEs, text encoders, LoRAs, ControlNets, adapters, and upscalers from supported model formats.
- Built-in tools for inpainting, outpainting, reference conditioning, masks and compositing, model merging, upscaling, frame interpolation, segmentation, depth estimation, and media processing.
- Save and load workflows as JSON, or recover complete workflows and seeds from supported generated media.
- Runs fully offline: core does not download anything unless you request it. Use `--disable-api-nodes` to disable the optional paid [Comfy API nodes](https://docs.comfy.org/tutorials/api-nodes/overview) and force all built-in functionality to stay offline.
- Extend ComfyUI with custom nodes, installed and updated through [ComfyUI-Manager](docs/installation.md#comfyui-manager).
- Configure additional model locations with [`extra_model_paths.yaml`](extra_model_paths.yaml.example).
- Support for saving and loading high bit depth images and videos: 16 bit PNG images, 32 bit EXR, 10 bit AVIF are supported and more, including HDR.

## Products

| Product | What it is |
| --- | --- |
| [Comfy Desktop](https://www.comfy.org/download) | The desktop application. The easiest way to run ComfyUI locally. |
| [Comfy Cloud](https://www.comfy.org/cloud) | Run ComfyUI without local hardware. |
| [Developer Platform](https://www.comfy.org/platform) | Run workflows and models from your own applications. |
| [Comfy Enterprise](https://www.comfy.org/enterprise) | Deployment, support, and scale for teams. |
| [Comfy MCP](https://www.comfy.org/mcp) | Drive ComfyUI from MCP-compatible clients. |
| [Comfy Agent](https://www.comfy.org/agent) | Agentic workflow generation. |
| [Comfy CLI](https://www.comfy.org/cli) | Install, launch, and manage ComfyUI from the terminal. |
| [Pricing](https://www.comfy.org/pricing) | Plans across Cloud and the Developer Platform. |

## Install

**[Comfy Desktop](https://www.comfy.org/download) is the recommended way to install
ComfyUI.** It is the easiest and best way to get started, on Windows and macOS.

Other paths:

- **[Comfy Cloud](https://www.comfy.org/cloud)** if you would rather not run it locally at all.
- **[comfy-cli](https://www.comfy.org/cli)** if you prefer a CLI: `pip install comfy-cli && comfy install`.
- **Manual install**, below, when you need control over your Python and Torch versions.
- **[Windows Portable](docs/installation.md#windows-portable)** for a self-contained build you can unzip and run.

### Manual install

First install PyTorch for your GPU, then:

```bash
git clone https://github.com/Comfy-Org/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
python main.py
```

Python 3.13 is very well supported and 3.14 works, though some custom nodes may have
issues. torch 2.7 is minimally supported and cu130 or above is required on Nvidia 20
series and above; if your pytorch is more than 6 months old, please update it.

Put your checkpoints in `models/checkpoints` and your VAE in `models/vae`.

**[docs/installation.md](docs/installation.md) has the full detail:** PyTorch commands
for NVIDIA, AMD ROCm on Linux and Windows, Intel Arc, Apple Silicon, Ascend NPU,
Cambricon MLU and Iluvatar Corex, plus ComfyUI-Manager setup, portable builds, previews
and TLS.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to ask questions, report issues, open pull
requests, and how releases are cut across core, Desktop and the frontend.

## Community

Ask questions in [#help or #feedback on Discord](https://comfy.org/discord), or on
[Matrix](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org), which is like
Discord but open source.

_psst, we're hiring:_ [comfy.org/careers](https://www.comfy.org/careers)

## References

| Resource | Link |
| --- | --- |
| Documentation | https://docs.comfy.org |
| Installing (this repo) | [docs/installation.md](docs/installation.md) |
| Keyboard shortcuts | https://docs.comfy.org/interface/shortcuts |
| Launch / startup flags | https://docs.comfy.org/development/comfyui-server/startup-flags |
| Changelog | https://docs.comfy.org/changelog |
| Workflow templates | https://comfy.org/workflows |
| Built-in node reference | https://docs.comfy.org/built-in-nodes/overview |
| Writing custom nodes | https://docs.comfy.org/custom-nodes/overview |
| Prompt syntax: weighting, `{wildcards}`, `embedding:`, `//` comments | https://docs.comfy.org/built-in-nodes/ClipTextEncode |
| Recovering a workflow from generated media | https://docs.comfy.org/development/api-development/workflow-metadata |
| Troubleshooting | https://docs.comfy.org/troubleshooting/overview |
| Which GPU should I buy? | https://github.com/Comfy-Org/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI |
| Blog | https://blog.comfy.org |
| Discord | https://comfy.org/discord |
| Frontend repository | https://github.com/Comfy-Org/ComfyUI_frontend |
