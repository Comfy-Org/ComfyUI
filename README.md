# HLW Internal FOR R&D Comfy Hub


Custom ComfyUI fork with API node integration and environment variable support.

## Features

### COMFY_API_KEY Environment Variable Integration

All API nodes in the `comfy_api_nodes` directory now automatically use the `COMFY_API_KEY` environment variable for authentication. This eliminates the need to configure API keys individually for each node.
=======
# ComfyUI
**The most powerful and modular AI engine for content creation.**


[![Website][website-shield]][website-url]
[![Dynamic JSON Badge][discord-shield]][discord-url]
[![Twitter][twitter-shield]][twitter-url]
[![Matrix][matrix-shield]][matrix-url]
<br>
[![][github-release-shield]][github-release-link]
[![][github-release-date-shield]][github-release-link]
[![][github-downloads-shield]][github-downloads-link]
[![][github-downloads-latest-shield]][github-downloads-link]

[matrix-shield]: https://img.shields.io/badge/Matrix-000000?style=flat&logo=matrix&logoColor=white
[matrix-url]: https://app.element.io/#/room/%23comfyui_space%3Amatrix.org
[website-shield]: https://img.shields.io/badge/ComfyOrg-4285F4?style=flat
[website-url]: https://www.comfy.org/
<!-- Workaround to display total user from https://github.com/badges/shields/issues/4500#issuecomment-2060079995 -->
[discord-shield]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fcomfyorg%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&logo=discord&logoColor=white&label=Discord&color=green&suffix=%20total
[discord-url]: https://discord.com/invite/comfyorg
[twitter-shield]: https://img.shields.io/twitter/follow/ComfyUI
[twitter-url]: https://x.com/ComfyUI

[github-release-shield]: https://img.shields.io/github/v/release/comfyanonymous/ComfyUI?style=flat&sort=semver
[github-release-link]: https://github.com/comfyanonymous/ComfyUI/releases
[github-release-date-shield]: https://img.shields.io/github/release-date/comfyanonymous/ComfyUI?style=flat
[github-downloads-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/total?style=flat
[github-downloads-latest-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest
[github-downloads-link]: https://github.com/comfyanonymous/ComfyUI/releases

<img width="1590" height="795" alt="ComfyUI Screenshot" src="https://github.com/user-attachments/assets/36e065e0-bfae-4456-8c7f-8369d5ea48a2" />
<br>
</div>

ComfyUI is the AI creation engine for visual professionals who demand control over every model, every parameter, and every output. Its powerful and modular node graph interface empowers creatives to generate images, videos, 3D models, audio, and more...
- ComfyUI natively supports the latest open-source state of the art models.
- [Partner nodes](https://docs.comfy.org/tutorials/partner-nodes/overview#partner-nodes) provide access to the best closed source models such as Nano Banana, Seedance, Hunyuan3D, etc.
- It is available on Windows, Linux, and macOS, locally with our [desktop application](https://www.comfy.org/download), our [portable install](#installing) or on our [cloud](https://www.comfy.org/cloud).
- The most sophisticated workflows can be exposed through a simple UI thanks to App Mode.
- It integrates seamlessly into production pipelines with our API endpoints.

## Get Started

### Local

#### [Desktop Application](https://www.comfy.org/download)
- The easiest way to get started.
- Available on Windows & macOS.

#### [Manual Install](#manual-install-windows-linux)
Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).

### Cloud

#### [Comfy Cloud](https://www.comfy.org/cloud)
- Our official paid cloud version for those who can't afford local hardware.

## Examples
See what ComfyUI can do with the [newer template workflows](https://comfy.org/workflows) or old [example workflows](https://comfyanonymous.github.io/ComfyUI_examples/).

## Features
- A visual node graph for building and reusing image, video, audio, 3D, and text workflows without code.
- Reusable subgraphs, workflow templates, App Mode, and a local API for integrating workflows into applications.
- Efficient local execution with asynchronous queueing, partial graph re-execution, smart VRAM and RAM management, model offloading, and support for quantized models.
- Broad native model support. This is a representative list; browse the [workflow library](https://comfy.org/workflows/) for maintained, ready-to-run templates.
  - [Image generation](https://comfy.org/workflows/tag/text-to-image/): Stable Diffusion 1.5, SDXL, SD3.5, Flux.1, Flux.2, Qwen Image, Z-Image, Hunyuan Image 2.1, HiDream, Lumina Image 2.0, Chroma, Anima, LongCat Image, Ideogram 4, Krea 2, MageFlow, Microsoft Lens, PixelDiT, Kandinsky 5, and Ernie Image.
  - [Image editing](https://comfy.org/workflows/tag/image-edit/): Flux Kontext, Flux.2 Klein, Qwen Image Edit, HiDream E1.1 and O1, OmniGen2, Boogu, JoyImage Edit, MageFlow Edit, and LongCat Image Edit.
  - [Video generation](https://comfy.org/workflows/tag/video-generation/): Wan 2.1 and 2.2, LTX-Video 2 and 2.3, HunyuanVideo 1.5, Kandinsky 5 Video, CogVideoX, Cosmos Predict2, Bernini-R, SCAIL 2, and Mochi.
  - [Audio and video generation](https://comfy.org/workflows/): MiniMax H3 and LTX-AV.
  - [Audio generation](https://comfy.org/workflows/tag/text-to-audio/): ACE-Step 1.5, Stable Audio 3, MiniMax Music 3 and Yue 2.
  - [3D and vision](https://comfy.org/workflows/): Hunyuan3D 2.1, TripoSplat, SeedVR2, SUPIR, Depth Anything 3, MoGe, SAM 3 and 3.1, RT-DETRv4, and BiRefNet.
  - [Text generation](https://comfy.org/workflows/tag/text-generation/): Gemma 3 and 4, Qwen3, Qwen3.5, and Qwen3-VL, including multimodal inputs.
- Load complete checkpoints or separate diffusion models, VAEs, text encoders, LoRAs, ControlNets, adapters, and upscalers from supported model formats.
- Built-in tools for inpainting, outpainting, reference conditioning, masks and compositing, model merging, upscaling, frame interpolation, segmentation, depth estimation, and media processing.
- Save and load workflows as JSON, or recover complete workflows and seeds from supported generated media.
- Runs fully offline: core does not download anything unless you request it. Use `--disable-api-nodes` to disable the optional paid [Comfy API nodes](https://docs.comfy.org/tutorials/api-nodes/overview) and force all built-in functionality to stay offline.
- Extend ComfyUI with custom nodes
- Configure additional model locations with [`extra_model_paths.yaml`](extra_model_paths.yaml.example).
- Support for saving and loading high bit depth images and videos: 16 bit PNG images, 32 bit EXR, 10 bit AVIF are supported and more.
- Support for saving and loading HDR videos and images in various formats.


## Release Process

ComfyUI follows a weekly release cycle targeting Monday but this regularly changes because of model releases or large changes to the codebase. There are three interconnected repositories:

1. **[ComfyUI Core](https://github.com/comfyanonymous/ComfyUI)**
   - Releases a new major stable version (e.g., v0.7.0) roughly every 2 weeks.
   - Starting from v0.4.0 patch versions will be used for fixes backported onto the current stable release.
   - Minor versions will be used for releases off the master branch.
   - Patch versions may still be used for releases on the master branch in cases where a backport would not make sense.
   - Commits outside of the stable release tags may be very unstable and break many custom nodes.
   - Serves as the foundation for the desktop release

2. **[Comfy Desktop](https://github.com/Comfy-Org/Comfy-Desktop)**
   - Builds a new release using the latest stable core version

3. **[ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend)**
   - Every 2+ weeks frontend updates are merged into the core repository
   - Features are frozen for the upcoming core release
   - Development continues for the next release cycle

## Shortcuts

| Keybind                            | Explanation                                                                                                        |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `Ctrl` + `Enter`                      | Queue up current graph for generation                                                                              |
| `Ctrl` + `Shift` + `Enter`              | Queue up current graph as first for generation                                                                     |
| `Ctrl` + `Alt` + `Enter`                | Cancel current generation                                                                                          |
| `Ctrl` + `Z`/`Ctrl` + `Y`                 | Undo/Redo                                                                                                          |
| `Ctrl` + `S`                          | Save workflow                                                                                                      |
| `Ctrl` + `O`                          | Load workflow                                                                                                      |
| `Ctrl` + `A`                          | Select all nodes                                                                                                   |
| `Alt `+ `C`                           | Collapse/uncollapse selected nodes                                                                                 |
| `Ctrl` + `M`                          | Mute/unmute selected nodes                                                                                         |
| `Ctrl` + `B`                           | Bypass selected nodes (acts like the node was removed from the graph and the wires reconnected through)            |
| `Delete`/`Backspace`                   | Delete selected nodes                                                                                              |
| `Ctrl` + `Backspace`                   | Delete the current graph                                                                                           |
| `Space`                              | Move the canvas around when held and moving the cursor                                                             |
| `Ctrl`/`Shift` + `Click`                 | Add clicked node to selection                                                                                      |
| `Ctrl` + `C`/`Ctrl` + `V`                  | Copy and paste selected nodes (without maintaining connections to outputs of unselected nodes)                     |
| `Ctrl` + `C`/`Ctrl` + `Shift` + `V`          | Copy and paste selected nodes (maintaining connections from outputs of unselected nodes to inputs of pasted nodes) |
| `Shift` + `Drag`                       | Move multiple selected nodes at the same time                                                                      |
| `Ctrl` + `D`                           | Load default graph                                                                                                 |
| `Alt` + `+`                          | Canvas Zoom in                                                                                                     |
| `Alt` + `-`                          | Canvas Zoom out                                                                                                    |
| `Ctrl` + `Shift` + LMB + Vertical drag | Canvas Zoom in/out                                                                                                 |
| `P`                                  | Pin/Unpin selected nodes                                                                                           |
| `Ctrl` + `G`                           | Group selected nodes                                                                                               |
| `Q`                                 | Toggle visibility of the queue                                                                                     |
| `H`                                  | Toggle visibility of history                                                                                       |
| `R`                                  | Refresh graph                                                                                                      |
| `F`                                  | Show/Hide menu                                                                                                      |
| `.`                                  | Fit view to selection (Whole graph when nothing is selected)                                                        |
| Double-Click LMB                   | Open node quick search palette                                                                                     |
| `Shift` + Drag                       | Move multiple wires at once                                                                                        |
| `Ctrl` + `Alt` + LMB                   | Disconnect all wires from clicked slot                                                                             |

`Ctrl` can also be replaced with `Cmd` instead for macOS users

# Installing

## Windows and Mac

We highly recommend using the [desktop app](https://comfy.org/download):

### [Link to Download](https://comfy.org/download)

The desktop app is the easiest and best way to use ComfyUI for new users.

## Windows Portable

There is a portable standalone build for Windows that should work for running on Nvidia GPUs or for running on your CPU only. It is not recommended for regular users. Regular users should use the desktop app above.

[Direct link to download (nvidia)](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)

Simply download, extract with [7-Zip](https://7-zip.org) or with the windows explorer on recent windows versions and run. For smaller models you normally only need to put the checkpoints (the huge ckpt/safetensors files) in: ComfyUI\models\checkpoints but many of the larger models have multiple files. Make sure to follow the instructions to know which subfolder to put them in ComfyUI\models\

If you have trouble extracting it, right click the file -> properties -> unblock

The portable above currently comes with python 3.13 and pytorch cuda 13.0. Update your Nvidia drivers if it doesn't start.

#### All Official Portable Downloads:

[Portable for AMD GPUs](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_amd.7z)

[Portable for Intel GPUs](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_intel.7z)

[Portable for Nvidia GPUs](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z) (supports 20 series and above).

[Portable for Nvidia GPUs with pytorch cuda 12.6 and python 3.12](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia_cu126.7z) (Supports Nvidia 10 series and older GPUs, DO NOT USE THIS ON NEWER 20 SERIES AND ABOVE GPUS).

#### How do I share models between another UI and ComfyUI?

See the [Config file](extra_model_paths.yaml.example) to set the search paths for models. In the standalone windows build you can find this file in the ComfyUI directory. Rename this file to extra_model_paths.yaml and edit it with your favorite text editor.


## [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

You can install and start ComfyUI using comfy-cli:
```bash
pip install comfy-cli
comfy install
```

## Manual Install (Windows, Linux)

Python 3.14 works but some custom nodes may have issues. The free threaded variant works but some dependencies will enable the GIL so it's not fully supported.

Python 3.13 is very well supported. If you have trouble with some custom node dependencies on 3.13 you can try 3.12

torch 2.7 is minimally supported but using a newer version is extremely recommended. Using a cu130 or above version of pytorch is required on Nvidia 20 series and above. Some features and optimizations might only work on newer versions. We generally recommend using the latest major version of pytorch with the latest cuda version unless it is less than 2 weeks old. If your pytorch is more than 6 months old, please update it.

### Instructions:

Git clone this repo.

Put your SD checkpoints (the huge ckpt/safetensors files) in: models/checkpoints

Put your VAE in: models/vae


### AMD GPUs (Linux)

AMD users can install rocm and pytorch with pip if you don't have it already installed, this is the command to install the stable version:

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2```

This is the command to install the nightly with ROCm 7.2 which might have some performance improvements:

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.2```


### AMD GPUs (Windows, ROCm 10.0)

Use AMD's [multi-architecture PyTorch packages](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html). The `device-*` extras install your GPU's kernels and the matching ROCm runtime automatically; a separate HIP SDK installation is not needed.

Use Windows 11, a current [AMD graphics driver](https://www.amd.com/en/support/download/drivers.html), and 64-bit Python 3.13.

The install command below uses `device-all` to install kernels for all supported GPUs. To reduce download size and disk usage, optionally replace **both** occurrences of `device-all` with the target for your GPU:

| GPU | Device extra |
| --- | --- |
| RX 9070 / XT, Radeon AI PRO R9700 | `device-gfx1201` |
| RX 9060 / XT | `device-gfx1200` |
| RX 7900 XT / XTX | `device-gfx1100` |
| RX 7700 XT / 7800 XT | `device-gfx1101` |
| RX 7600 / XT | `device-gfx1102` |
| Ryzen AI Max / Max+ (Strix Halo) | `device-gfx1151` |

**Note:** This table only lists examples. A GPU missing from it may still be supported: supported architectures include RDNA 2, RDNA 3, RDNA 3.5, and RDNA 4. Keep `device-all` to install kernels for all supported targets. For other models, see AMD's [GPU target table](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#gfx-target-lookup-table) and [ROCm compatibility matrix](https://rocm.docs.amd.com/en/docs-10.0.0/compatibility/compatibility-matrix.html).

**ROCm 10.0.0 with PyTorch 2.13:**

```bat
pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-all]==2.13.0+rocm10.0.0" "torchvision[device-all]==0.28.0+rocm10.0.0" "torchaudio==2.11.0.2+rocm10.0.0"
```


#### Setup

The `COMFY_API_KEY` is configured in `user/HLWUser.bat`:

```batch
set COMFY_API_KEY=comfyui-afb307d0765bbae8c3f6e28cff7e1ce5fdfff1c9a67a0f09c05d409c09fab7c3
```

#### How It Works

**Authentication Priority Order:**
1. Bearer token from `IO.Hidden.auth_token_comfy_org` (highest priority)
2. **`COMFY_API_KEY` environment variable** (automatic)
3. API key from `IO.Hidden.api_key_comfy_org` (fallback)

**Implementation:**
- Modified `comfy_api_nodes/util/_helpers.py` → `get_auth_header()` function
- Automatically applies to all 26 API nodes without individual modifications
- Fully backward compatible with existing workflows

**Supported API Nodes:**
- Gemini (nodes_gemini.py)
- OpenAI (nodes_openai.py)
- Stability AI (nodes_stability.py)
- And 23 other API providers

#### Usage

1. Start ComfyUI using `user/HLWUser.bat` (sets the environment variable)
2. Load any workflow with API nodes
3. API authentication happens automatically - no manual configuration needed

### Quantized Model Support

This installation includes **`comfy_kitchen`** (v0.1.0) for loading quantized models like Flux and Qwen.

**Supported Quantization Formats:**
- **FP8** (float8_e4m3fn, float8_e5m2) - 8-bit floating point
- **NVFP4** - 4-bit quantization with block scales

**Benefits:**
- 2-4x reduction in VRAM usage
- Faster inference with tensor core acceleration
- Automatic dequantization during forward pass

**Requirements:**
- CUDA 13.0+ recommended for optimized operations
- Falls back to CPU implementations on older CUDA versions

## Post-Update Import Fix Procedure

After pulling upstream ComfyUI updates, some `comfy_api_nodes` files may fail to import if modules have been renamed or removed. The startup log will show `IMPORT FAILED` and `ModuleNotFoundError` entries. Follow this procedure to resolve them.

### Known Module Relocations

| Old import path | New import path | Affected files |
|---|---|---|
| `comfy_api_nodes.apinode_utils` | `comfy_api_nodes.util` | `nodes_gemini.py`, `nodes_kling.py` |

### How to Fix a `ModuleNotFoundError` After an Update

1. **Identify the failing file** from the startup log (`IMPORT FAILED: nodes_xyz.py`).
2. **Find the bad import** — look for `from comfy_api_nodes.<old_module> import ...`.
3. **Locate where the symbol moved** — check `comfy_api_nodes/util/__init__.py` and its sub-modules (`conversions.py`, `validation_utils.py`, `upload_helpers.py`, `download_helpers.py`).
4. **Update the import** to point at the new location (usually `from comfy_api_nodes.util import ...`).

### Missing Utility Modules (`mapper_utils`, etc.)

If a utility module referenced in a node file no longer exists in the upstream source, create it locally inside `comfy_api_nodes/`. Document it here so it survives future merges.

| File | Purpose | Status |
|---|---|---|
| `comfy_api_nodes/mapper_utils.py` | `model_field_to_node_input()` — maps Pydantic model fields to ComfyUI INPUT_TYPES entries | Created locally; not in upstream |

**`model_field_to_node_input` signature:**
```python
model_field_to_node_input(io_type, model_cls, field_name, enum_type=None, **kwargs)
# Returns (io_type, {tooltip, **kwargs}) or ([enum_values], {tooltip, **kwargs}) for IO.COMBO
```

## Recent Changes

### 2026-03-11: Upgrade to ComfyUI v0.16.4

**Updated:**
- Merged upstream ComfyUI v0.16.4 (branch renamed from `update/v0.16.2` → `update/v0.16.4`, then merged to `master`)
- New Gemini image nodes: **Nano Banana** (`GeminiImage`), **Nano Banana Pro** (`GeminiImage2`), **Nano Banana 2** (`GeminiNanoBanana2`) backed by Gemini 3.x image models
- New **Math Expression** node with simpleeval evaluation
- New **TencentSmartTopology** node
- Gemini LLM model list expanded with Gemini 3.x models (gemini-3-pro-preview, gemini-3-1-pro, gemini-3-1-flash-lite)

**Fixed:**
- `execution.py`: Guard `COMFY_API_KEY` injection to **v1 nodes only** — v3-style `IO.ComfyNode` nodes (`GeminiImage`, etc.) crashed with `unexpected keyword argument 'comfy_api_key'` because they handle auth internally via `sync_op()`. Fix: check `v3_data is None` before injecting. See [Post-Update section](#v3-node-comfy_api_key-injection-guard) below.

### 2026-01-28: COMFY_API_KEY Integration & Bug Fixes

**Added:**
- Environment variable support for API authentication in `comfy_api_nodes/util/_helpers.py`
- Automatic API key injection for all 26 API node providers
- Comprehensive documentation in walkthrough.md
- **`comfy_kitchen` dependency** (v0.1.0) for quantized model support

**Fixed:**
- Removed obsolete API key injection code from `execution.py` (lines 603-632)
- Resolved `NameError: name 'hidden_inputs' is not defined` error
- Cleaned up redundant authentication logic
- **Model loading error** for Flux and Qwen quantized models (`AttributeError: 'NoneType' object has no attribute 'Params'`)

**Technical Details:**
- Single function modification (`get_auth_header()`) applies globally
- No async/sync compatibility issues
- Maintains full backward compatibility
- Installed `comfy_kitchen` enables FP8/NVFP4 quantization support (2-4x memory reduction)


## Project Structure

```
HLWComfy/
├── comfy_api_nodes/          # API node implementations
│   ├── util/
│   │   └── _helpers.py       # Authentication logic (modified)
│   ├── nodes_gemini.py       # Google Gemini API
│   ├── nodes_openai.py       # OpenAI API
│   ├── nodes_stability.py    # Stability AI API
│   └── [23 other API nodes]
├── execution.py              # Execution engine (cleaned up)
├── user/
│   └── HLWUser.bat          # Environment setup script
└── README.md                # This file
```

## Development Notes

### V3 Node `COMFY_API_KEY` Injection Guard

ComfyUI v0.16+ introduces **v3-style nodes** that extend `IO.ComfyNode` (e.g., `GeminiImage`, `GeminiNode`, `GeminiNanoBanana2`). These nodes use the new `define_schema()` / `execute()` pattern and handle API auth **internally** via `sync_op()` — the framework reads their `IO.Hidden.api_key_comfy_org` declaration automatically.

#### The Problem

The local `execution.py` injection block (Krita API integration) injects `comfy_api_key` into **all** nodes marked `API_NODE=True`. When this runs for a v3 node, the kwarg is passed into the v3 dispatch chain (`EXECUTE_NORMALIZED_ASYNC`) which calls `execute()` — which does **not** accept `comfy_api_key` as a parameter:

```
TypeError: GeminiImage.execute() got an unexpected keyword argument 'comfy_api_key'
```

#### The Fix — `execution.py`

Guard the injection with `v3_data is None`. `v3_data` is only set for v3 nodes, so this precisely targets v1 nodes:

```python
# execution.py  — inside _async_map_node_over_list, before the coroutine dispatch

# Only inject for v1 API nodes — v3 nodes (IO.ComfyNode) handle auth internally
if v3_data is None and hasattr(obj, 'API_NODE') and getattr(obj, 'API_NODE', False):
    comfy_api_key = os.getenv('COMFY_API_KEY')
    if comfy_api_key:
        inputs = dict(inputs)
        if inputs.get('comfy_api_key') is None:
            inputs['comfy_api_key'] = comfy_api_key
```

**Apply this fix every time upstream updates introduce new v3 API nodes** and the injection block causes `unexpected keyword argument` errors.

#### How to detect v3 vs v1 nodes

| Trait | v1 node | v3 node |
|---|---|---|
| Base class | `ComfyNodeABC` | `IO.ComfyNode` |
| Schema | `INPUT_TYPES()` classmethod | `define_schema()` classmethod |
| Execution | `FUNCTION = "api_call"`, `api_call(self, ...)` | `async def execute(cls, ...)` |
| Auth wiring | `"hidden": {"comfy_api_key": "API_KEY_COMFY_ORG"}` | `IO.Hidden.api_key_comfy_org` in `define_schema` |

### API Authentication Architecture

The authentication system uses a centralized approach:

1. **Request Layer**: API nodes call `sync_op()` or `poll_op()` from `util/client.py`
2. **Authentication Layer**: These functions call `get_auth_header()` from `util/_helpers.py`
3. **Environment Layer**: `get_auth_header()` reads `COMFY_API_KEY` from environment
4. **Header Injection**: API key is automatically added to HTTP request headers

This design ensures:
- Single source of truth for API authentication
- No need to modify individual node files
- Easy to maintain and update
- Proper separation of concerns

### Async/Sync Handling

Both `sync_op()` and `poll_op()` are async functions. The `os.environ.get()` call in `get_auth_header()` is synchronous but safe to use within async contexts, following Python async best practices.

## License

Internal R&D project for HLW.

