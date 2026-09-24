<div align="center">

# ComfyUI
**The most powerful and modular AI engine for content creation.**

## RUN

```
export HF_TOKEN="your_hf_token"
python3 main.py --preview-method auto --auto-download-models
```

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
- API nodes provide access to the best closed source models such as Nano Banana, Seedance, Hunyuan3D, etc.
- It is available on Windows, Linux, and macOS, locally with our [desktop application](https://www.comfy.org/download), our [portable install](#installing) or on our [cloud](https://www.comfy.org/cloud).
- The most sophisticated workflows can be exposed through a simple UI thanks to App Mode.
- It integrates seamlessly into production pipelines with our API endpoints.

## Get Started

### Quickstart (uv)

The fastest local setup is the included `start_uv.sh` launcher, which uses [uv](https://docs.astral.sh/uv/) to create the virtualenv and install all dependencies automatically on first run:

```bash
./start_uv.sh                  # first run: creates .venv, installs requirements.txt, starts on :8188
./start_uv.sh --port 8588 --disable-auto-launch   # extra args are passed to main.py
```

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) (`brew install uv` on macOS). Later runs skip installation and start the server immediately.

### Local

#### [Desktop Application](https://www.comfy.org/download)
- The easiest way to get started.
- Available on Windows & macOS.

#### [Windows Portable Package](#installing)
- Get the latest commits and completely portable.
- Available on Windows.

#### [Manual Install](#manual-install-windows-linux)
Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).

### Cloud

#### [Comfy Cloud](https://www.comfy.org/cloud)
- Our official paid cloud version for those who can't afford local hardware.

## Examples
See what ComfyUI can do with the [newer template workflows](https://comfy.org/workflows) or old [example workflows](https://comfyanonymous.github.io/ComfyUI_examples/).

## Wrapper API

A simplified REST API on top of the regular ComfyUI API (`api_wrapper/`). It hides the node graph and model setup: each workflow has a dedicated synchronous endpoint that returns the final artifact (image, video, audio, 3D asset, ...) directly as a downloadable file. Missing model files for the requested workflow are downloaded automatically on first use (requires starting ComfyUI with `--auto-download-models`; gated models need `HF_TOKEN`).

- `POST /api/wrapper/{workflow}/generate` — multipart form with `prompt` and `image` (optional `negative_prompt`, `seed`, `steps`, `cfg`, `megapixels`, `timeout`, `free_vram`, `quantization`). Runs the workflow to completion and returns the final file as a download (`Content-Disposition: attachment`). By default the request waits as long as the job takes; set `timeout` (5–86400 seconds) if you prefer a cap — the timeout response includes the `job_id` to pick the result up via `/jobs/{job_id}`. `GET /api/wrapper/workflows` lists the available `{workflow}` names (currently `flux2klein9b`).
- `GET /api/wrapper/jobs/{job_id}` — job status (`pending` / `in_progress` / `completed` / `failed` / `cancelled`) with output image URLs; also useful to pick up a generate call that timed out (the timeout response includes the `job_id`).
- `GET /api/wrapper/jobs/{job_id}/image` — redirects to the generated image.
- `DELETE /api/wrapper/jobs/{job_id}` — worker cleanup once you have stored the result (or after a failure/cancel): deletes the job's output files, the uploads saved for it and its history entry, and answers `{"deleted": {"outputs": n, "inputs": m, "history": true|false}, "refused": 0}`. Idempotent (unknown job → zeros), `409` while the job is pending/running, and it never deletes anything outside ComfyUI's output/input/temp directories (real paths are checked; symlinks pointing out are refused). Each job's uploads and outputs live in their own `wrapper/<job_id>/` folder under `input/` and `output/`.
- `POST /api/wrapper/free` — releases the GPU VRAM and RAM used by model execution (unloads all models, empties torch caches). When a job is running/queued the release happens automatically right after that job finishes.
- `GET /api/wrapper/docs` — Swagger UI for the wrapper API (spec at `/api/wrapper/openapi.json`).

Example:

```bash
curl -o result.png -X POST http://127.0.0.1:8188/api/wrapper/flux2klein9b/generate \
  -F "prompt=make it snow" -F "image=@input.png" -F "steps=20"
```

**Memory:** by default the wrapper releases VRAM and RAM as soon as a job finishes — models are unloaded and caches emptied automatically, so the server stays resource-efficient between requests. Send `free_vram=false` on a generate request to keep models loaded across consecutive jobs (e.g. batch runs).

**Precision:** `quantization=fp8` (default) uses the shipped fp8 checkpoint — on Apple Silicon it loads as bf16 automatically. `quantization=fp4` converts the checkpoint to NVFP4 on first use (~half the file size, best on CUDA: native on Blackwell, emulated on older GPUs and CPU); on MPS the wrapper automatically falls back to fp8 since NVFP4 needs fp8 block scales that MPS cannot handle. MiniMax H3 has its own set (`fp8` / `int8` / `bf16` / `nvfp4`, default `int8`) — see below. Every generate response reports the checkpoints that actually ran in the `X-Wrapper-Models` header and the resolved parameters in `X-Wrapper-Settings`, so you can confirm what produced a file without reading the server log.

**MiniMax H3 prompt rewriting:** H3 does not want a one-line description — it wants an alignment instruction, `integrated_multimodal_description` split into timed shots, `overall_soundscape` and `non_diegetic_music` (or six labelled sections in reference mode). So the H3 endpoints take `prompt` as a plain description in your own words ("two wrestlers in a gym, one chokeslams the other onto a crash mat, heavy metal soundtrack") and have a Xiaomi MiMo model rewrite it into the real thing, following [MiniMax's own H3 prompt-writing guide](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing) (vendored in `api_wrapper/h3_prompt_skill/`) and this request's actual mode, duration, canvas and reference labels. Set `MIMO_API_KEY` to enable it.

- `raw_prompt` — a ready-made H3 prompt, used verbatim. Skips the rewrite entirely and needs no API key, so existing callers keep working by renaming `prompt` to `raw_prompt`.
- `llm_image` — an image given to the rewriter as visual context, so it describes what is actually in your footage. It never reaches H3 itself. When omitted, the task's own first image (the `image` first frame, or `ref_images` #1) is used, so image- and reference-driven jobs get visual grounding for free.
- `llm_model` — `auto` (default) routes image-bearing rewrites to `mimo-v2.5`, the omnimodal build, and text-only ones to `mimo-v2.5-pro` for its stronger reasoning. **`mimo-v2.5-pro` cannot read images** — the API rejects any request carrying one (`HTTP 404: No endpoints found that support image input`), so pinning it alongside an uploaded `llm_image` returns a 400, while the context image the wrapper adds on its own is simply skipped and reported. Leave it on `auto` unless you have a reason not to.
- `POST /api/wrapper/minimaxh3/prompt` turns a free-form prompt into an H3 prompt and returns it as JSON — no GPU work, no render. The task is inferred from what you attach (reference assets → ref2va, a keyframe → image-to-video, nothing → text-to-video), so a prompt on its own is a complete request; `POST /api/wrapper/minimaxh3/{task}/prompt` pins one. Iterate there, then send the result back to `/generate` as `raw_prompt`. Accepts multipart, form-urlencoded or JSON.

```bash
curl -s --max-time 60 -X POST http://127.0.0.1:8188/api/wrapper/minimaxh3/prompt \
  -d 'prompt=two wrestlers in a gym, one chokeslams the other onto a crash mat, heavy metal soundtrack'
```

A rewrite takes 10–20 s (it is one call to a large model), so allow at least 60 s — a short client timeout aborts mid-flight and looks like a missing endpoint. `/generate` is synchronous and streams the MP4 back in the response body, so it needs `-o out.mp4` and a timeout in minutes, not seconds.

**Generated prompts are recorded.** Because the H3 prompt is written by a model rather than supplied, every generation logs the full prompt it ran with to the server log (job id, mode, model, and the free-form input it came from), and saves a `<video>.prompt.json` sidecar next to the output — input prompt, generated prompt, model, mode, settings and checkpoints. The `X-Wrapper-Prompt-File` response header gives its path. `raw_prompt` runs are recorded the same way, marked `"source": "raw_prompt"`, so every video in the output directory can be traced back to the exact prompt that made it.

**MiniMax H3 memory limits:** H3's four checkpoints total ~43 GB (21 GB UNET + 16 GB text encoder + 5.8 GB VAEs), so on a 32 GB RAM box every job runs within ~2 GB of the ceiling and an over-large request does not fail gracefully — it OOM-kills the server and takes the queue with it. The wrapper therefore checks clip length and canvas×frames *before* queueing and returns a 400 for anything past the host's envelope. Frame count is the limit that bites: the video VAE tiles spatially but not temporally, so a longer clip costs far more memory than a wider one and no resolution is small enough to compensate. Defaults (`864x480`, 5.17 s, 20 steps, `int8`) are the largest values measured to complete with memory to spare on a 32 GB RTX 5090; raise `COMFY_MINIMAX_H3_MAX_FRAMES` / `COMFY_MINIMAX_H3_MAX_PIXEL_FRAMES` on a host with more RAM, swap, or running with `--fast-disk`. See `.env.example` for the deployment side.

**Interactive docs:** with the server running, open `http://127.0.0.1:8188/api/wrapper/docs` in a browser to browse and try every endpoint (the "Try it out" button works — you can upload an image and generate right from the docs page). The raw OpenAPI spec is at `http://127.0.0.1:8188/api/wrapper/openapi.json` for code generation; swap the port if you started ComfyUI elsewhere.

The first workflow shipped is `flux2klein9b`, a FLUX.2 [klein] 9B image edit: the input image is scaled to a megapixel budget, attached to the conditioning as a reference latent, and sampled with the flux2 custom sampler stack. `ideogram4` is a text-to-image workflow (no input image needed) using the Ideogram 4 dual-model CFG stack — it natively understands rich structured JSON prompts, e.g. `curl -F "prompt=$(cat prompt.json)"` with a JSON prompt describing composition, style and elements; the full example (with bounding boxes per element) is shown in the interactive docs. `minimaxh3` is an omni-modal video workflow with three task endpoints under `/api/wrapper/minimaxh3/{task}`: `text` (text-to-video), `image` (image-to-video, first/last frame uploads) and `reference` (ref2va: up to 9 reference images, 3 videos and 3 audio clips, referenced in the prompt as `<Picture i>` / `<Video k>` / `<Audio j>`); it returns a synchronized audio+video MP4. To add another workflow (e.g. a 3D model generator), add its graph builder to the `WORKFLOWS` registry in `api_wrapper/workflows.py`, a model-setup handler in `api_wrapper/routes.py`, and it immediately gets its own `/api/wrapper/{name}/generate` endpoint — output files (video/audio/3D) are detected automatically by their node output type. All endpoints are also available without the `/api` prefix.

### Authentication

Set `WRAPPER_AUTH_TOKEN` to require `Authorization: Bearer <token>` on **every** HTTP route of the server — the wrapper endpoints and the native ComfyUI ones alike (`/prompt`, `/queue`, `/history`, `/view`, `/object_info`, `/ws`, `/internal/...`, the docs pages and the static frontend). It is meant for publicly reachable boxes such as RunPod pods, where each pod is started with its own random token (e.g. `WRAPPER_AUTH_TOKEN=$(openssl rand -hex 32)`). Unset or empty, nothing changes: the server stays open, as before.

- A missing or wrong token gets a `401` with `WWW-Authenticate: Bearer` and the body `{"error": {"type": "unauthorized", "message": "missing or invalid bearer token"}}`. The token is compared in constant time.
- Browsers cannot set headers on a WebSocket, so `/ws` (and its `/api/ws` alias) also accepts the token as a query parameter: `ws://host:8188/ws?token=<token>&clientId=...`. No other route accepts `?token=`.
- `OPTIONS` requests pass without a token: CORS preflights never carry credentials, and all they get back is an empty reply.
- The token is read once at startup, so changing it needs a restart. The stock ComfyUI web UI does not send the header, so with auth on, the server is for API clients rather than for opening the frontend in a browser. The `developer-api` compose service does not send it either.

```bash
curl -o result.png -X POST http://127.0.0.1:8188/api/wrapper/flux2klein9b/generate \
  -H "Authorization: Bearer $WRAPPER_AUTH_TOKEN" \
  -F "prompt=make it snow" -F "image=@input.png"
```

### Docker image (local GPU box and RunPod)

The worker is published to Docker Hub as **`shivanshtalwar0/comfyui`** by [`.github/workflows/docker-publish.yml`](.github/workflows/docker-publish.yml):

| Trigger | Tags pushed |
|---|---|
| push to `master` (source or Docker files changed) | `latest`, `sha-<short>` |
| tag `vX.Y.Z` | `X.Y.Z`, `X.Y`, `sha-<short>` |
| manual run (Actions → Docker image → Run workflow) | `<branch>`, `sha-<short>` |
| pull request touching Docker files | nothing: build + smoke test only |

Every build is smoke-tested before anything is pushed: torch has CUDA, `prefetch --dry-run` resolves every wrapper checkpoint, and the server boots (CPU mode) and answers `/api/wrapper/workflows` with 200 with the bearer token and 401 without it. Publishing needs the repository secret **`DOCKERHUB_TOKEN`** (a Docker Hub access token with Read & Write). `DOCKERHUB_USERNAME` / `DOCKERHUB_IMAGE` repository variables override the defaults.

The image is **CUDA only**: `python:3.12-slim` plus the PyTorch **cu128** wheels (Blackwell / RTX 5090 kernels, host driver ≥ 570). The build fails if any dependency leaves it with a CPU torch. Nothing model-sized is baked in. [`docker/entrypoint.sh`](docker/entrypoint.sh) configures everything from env, so the image runs the same with or without a compose file:

| Env | Default | Meaning |
|---|---|---|
| `WRAPPER_AUTH_TOKEN` | unset | Bearer token on every route (see Authentication). **Always set it on public boxes.** |
| `COMFYUI_DATA_DIR` | `/workspace` if present | Persistent store: `models/` is linked to `<dir>/models`, `HF_HOME` to `<dir>/huggingface`, Triton's kernel cache to `<dir>/.triton`. |
| `COMFYUI_MODELS_DIR` | `<data dir>/models` | Model store on its own. A bind mount on `/opt/ComfyUI/models` always wins. |
| `COMFYUI_PORT` | `8188` | Container port. |
| `AUTO_DOWNLOAD_MODELS` | `1` | Fetch a workflow's checkpoints on first use. |
| `VRAM_HEADROOM_GB`, `CACHE_RAM_GB`, `ASYNC_OFFLOAD_STREAMS`, `FAST_DISK` | `2`, `2 8`, `0`, `1` | Memory tuning, as in `.env.example`. |
| `COMFYUI_ARGS` | empty | Extra ComfyUI flags. |
| `MIMO_API_KEY`, `HF_TOKEN` | unset | H3 prompt rewrite; gated FLUX.2 [klein] weights. |

**Local GPU box**: `cp .env.example .env`, then `docker compose pull && docker compose up -d` (or `docker compose up -d --build` to build this tree). `COMFYUI_HOST_PORT=8282` publishes it where the FloStudio rig is reached over WireGuard.

**RunPod (FloStudio burst pods)**: voxmin-backend creates the pods itself through the RunPod API. Set System Config → FloStudio → RunPod → image to `docker.io/shivanshtalwar0/comfyui:latest` (or a `sha-` tag), and attach a network volume. RunPod mounts it at `/workspace`, which the entrypoint uses for models, the HF cache and Triton kernels. The backend gives every pod its own `WRAPPER_AUTH_TOKEN` and reaches it on port 8188 through the RunPod proxy. Put `MIMO_API_KEY` / `HF_TOKEN` in the `runpod` ExternalApi row's `podEnv`.

**Fill a network volume once** so the first pod doesn't spend its boot downloading ~40 GB. Run this on a cheap CPU pod with the volume attached, or locally against the model dir:

```bash
docker run --rm -e HF_TOKEN=... -v /workspace:/workspace shivanshtalwar0/comfyui prefetch
# specs: minimaxh3[:nvfp4|int8|fp8|bf16]  minimaxh3-ref[:quant]  flux2klein9b
docker run --rm -v "$PWD/models:/opt/ComfyUI/models" shivanshtalwar0/comfyui prefetch minimaxh3:int8 --dry-run
```

With no specs it fetches `minimaxh3:nvfp4 minimaxh3-ref:nvfp4 flux2klein9b` (override with `PREFETCH_MODELS`). That set is the FP8 H3 UNETs plus the 16 GB NVFP4 text encoder that a 32 GB Blackwell card needs, and the FLUX.2 [klein] stills model.

## Features
- A visual node graph for building and reusing image, video, audio, 3D, and text workflows without code.
- Reusable subgraphs, workflow templates, App Mode, and a local API for integrating workflows into applications.
- Efficient local execution with asynchronous queueing, partial graph re-execution, smart VRAM and RAM management, model offloading, and support for quantized models.
- Broad native model support. This is a representative list; browse the [workflow library](https://comfy.org/workflows/) for maintained, ready-to-run templates.
  - [Image generation](https://comfy.org/workflows/tag/text-to-image/): Stable Diffusion 1.5, SDXL, SD3.5, Flux.1, Flux.2, Qwen Image, Z-Image, Hunyuan Image 2.1, HiDream, Lumina Image 2.0, Chroma, Anima, LongCat Image, Ideogram 4, Krea 2, MageFlow, Microsoft Lens, PixelDiT, Kandinsky 5, and Ernie Image.
  - [Image editing](https://comfy.org/workflows/tag/image-edit/): Flux Kontext, Flux.2 Klein, Qwen Image Edit, HiDream E1.1 and O1, OmniGen2, Boogu, JoyImage Edit, MageFlow Edit, and LongCat Image Edit.
  - [Video generation](https://comfy.org/workflows/tag/video-generation/): Wan 2.1 and 2.2, LTX-Video 2 and 2.3, HunyuanVideo 1.5, Kandinsky 5 Video, CogVideoX, Cosmos Predict2, Bernini-R, SCAIL 2, and Mochi.
  - [Audio and video generation](https://comfy.org/workflows/): MiniMax H3 and LTX-AV.
  - [Audio generation](https://comfy.org/workflows/tag/text-to-audio/): ACE-Step 1.5 and Stable Audio 3.
  - [3D and vision](https://comfy.org/workflows/): Hunyuan3D 2.1, TripoSplat, SeedVR2, SUPIR, Depth Anything 3, MoGe, SAM 3 and 3.1, RT-DETRv4, and BiRefNet.
  - [Text generation](https://comfy.org/workflows/tag/text-generation/): Gemma 3 and 4, Qwen3, Qwen3.5, and Qwen3-VL, including multimodal inputs.
- Load complete checkpoints or separate diffusion models, VAEs, text encoders, LoRAs, ControlNets, adapters, and upscalers from supported model formats.
- Built-in tools for inpainting, outpainting, reference conditioning, masks and compositing, model merging, upscaling, frame interpolation, segmentation, depth estimation, and media processing.
- Save and load workflows as JSON, or recover complete workflows and seeds from supported generated media.
- Runs fully offline: core does not download anything unless you request it. Use `--disable-api-nodes` to disable the optional paid [Comfy API nodes](https://docs.comfy.org/tutorials/api-nodes/overview) and force all built-in functionality to stay offline.
- Extend ComfyUI with custom nodes
- Configure additional model locations with [`extra_model_paths.yaml`](extra_model_paths.yaml.example).


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

torch 2.5 is minimally supported but using a newer version is extremely recommended. Some features and optimizations might only work on newer versions. We generally recommend using the latest major version of pytorch with the latest cuda version unless it is less than 2 weeks old. If your pytorch is more than 6 months old, please update it.

### Instructions:

Git clone this repo.

Put your SD checkpoints (the huge ckpt/safetensors files) in: models/checkpoints

Put your VAE in: models/vae


### AMD GPUs (Linux)

AMD users can install rocm and pytorch with pip if you don't have it already installed, this is the command to install the stable version:

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2```

This is the command to install the nightly with ROCm 7.2 which might have some performance improvements:

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.2```


### AMD GPUs (Experimental: Windows and Linux), RDNA 3, 3.5 and 4 only.

These have less hardware support than the builds above but they work on windows. You also need to install the pytorch version specific to your hardware.

RDNA 3 (RX 7000 series):

```pip install --pre torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/```

RDNA 3.5 (Strix halo/Ryzen AI Max+ 365):

```pip install --pre torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx1151/```

RDNA 4 (RX 9000 series):

```pip install --pre torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/```

### Intel GPUs (Windows and Linux)

Intel Arc GPU users can install native PyTorch with torch.xpu support using pip. More information can be found [here](https://pytorch.org/docs/main/notes/get_start_xpu.html)

1. To install PyTorch xpu, use the following command:

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu```

This is the command to install the Pytorch xpu nightly which might have some performance improvements:

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu```

### NVIDIA

Nvidia users should install stable pytorch using this command:

```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130```

This is the command to install pytorch nightly instead which might have performance improvements.

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu132```

#### Troubleshooting

If you get the "Torch not compiled with CUDA enabled" error, uninstall torch with:

```pip uninstall torch```

And install it again with the command above.

### Dependencies

Install the dependencies by opening your terminal inside the ComfyUI folder and:

```pip install -r requirements.txt```

After this you should have everything installed and can proceed to running ComfyUI.

### Others:

#### Apple Mac silicon

You can install ComfyUI in Apple Mac silicon (M1, M2, M3 or M4) with any recent macOS version.

1. Install pytorch nightly. For instructions, read the [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) Apple Developer guide (make sure to install the latest pytorch nightly).
1. Follow the [ComfyUI manual installation](#manual-install-windows-linux) instructions for Windows and Linux.
1. Install the ComfyUI [dependencies](#dependencies). If you have another Stable Diffusion UI [you might be able to reuse the dependencies](#i-already-have-another-ui-for-stable-diffusion-installed-do-i-really-have-to-install-all-of-these-dependencies).
1. Launch ComfyUI by running `python main.py`

> **Note**: Remember to add your models, VAE, LoRAs etc. to the corresponding Comfy folders, as discussed in [ComfyUI manual installation](#manual-install-windows-linux).

#### Ascend NPUs

For models compatible with Ascend Extension for PyTorch (torch_npu). To get started, ensure your environment meets the prerequisites outlined on the [installation](https://ascend.github.io/docs/sources/ascend/quick_install.html) page. Here's a step-by-step guide tailored to your platform and installation method:

1. Begin by installing the recommended or newer kernel version for Linux as specified in the Installation page of torch-npu, if necessary.
2. Proceed with the installation of Ascend Basekit, which includes the driver, firmware, and CANN, following the instructions provided for your specific platform.
3. Next, install the necessary packages for torch-npu by adhering to the platform-specific instructions on the [Installation](https://ascend.github.io/docs/sources/pytorch/install.html#pytorch) page.
4. Finally, adhere to the [ComfyUI manual installation](#manual-install-windows-linux) guide for Linux. Once all components are installed, you can run ComfyUI as described earlier.

#### Cambricon MLUs

For models compatible with Cambricon Extension for PyTorch (torch_mlu). Here's a step-by-step guide tailored to your platform and installation method:

1. Install the Cambricon CNToolkit by adhering to the platform-specific instructions on the [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html)
2. Next, install the PyTorch(torch_mlu) following the instructions on the [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cambricon_pytorch_1.17.0/user_guide_1.9/index.html)
3. Launch ComfyUI by running `python main.py`

#### Iluvatar Corex

For models compatible with Iluvatar Extension for PyTorch. Here's a step-by-step guide tailored to your platform and installation method:

1. Install the Iluvatar Corex Toolkit by adhering to the platform-specific instructions on the [Installation](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536)
2. Launch ComfyUI by running `python main.py`


## [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager/tree/manager-v4)

**ComfyUI-Manager** is an extension that allows you to easily install, update, and manage custom nodes for ComfyUI.

### Setup

1. Install the manager dependencies:
   ```bash
   pip install -r manager_requirements.txt
   ```

2. Enable the manager with the `--enable-manager` flag when running ComfyUI:
   ```bash
   python main.py --enable-manager
   ```

### Command Line Options

| Flag | Description |
|------|-------------|
| `--enable-manager` | Enable ComfyUI-Manager |
| `--enable-manager-legacy-ui` | Use the legacy manager UI instead of the new UI (implies `--enable-manager`) |
| `--disable-manager-ui` | Disable the manager UI and endpoints while keeping background features like security checks and scheduled installation completion (requires `--enable-manager`) |


# Running

```python main.py```

### For AMD cards not officially supported by ROCm

Try running it with this command if you have issues:

For 6700, 6600 and maybe other RDNA2 or older: ```HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py```

For AMD 7600 and maybe other RDNA3 cards: ```HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py```

### AMD ROCm Tips

You can try setting this env variable `PYTORCH_TUNABLEOP_ENABLED=1` which might speed things up at the cost of a very slow initial run.

# Notes

Only parts of the graph that have an output with all the correct inputs will be executed.

Only parts of the graph that change from each execution to the next will be executed, if you submit the same graph twice only the first will be executed. If you change the last part of the graph only the part you changed and the part that depends on it will be executed.

Dragging a generated png on the webpage or loading one will give you the full workflow including seeds that were used to create it.

You can use () to change emphasis of a word or phrase like: (good code:1.2) or (bad code:0.8). The default emphasis for () is 1.1. To use () characters in your actual prompt escape them like \\( or \\).

You can use {day|night}, for wildcard/dynamic prompts. With this syntax "{wild|card|test}" will be randomly replaced by either "wild", "card" or "test" by the frontend every time you queue the prompt. To use {} characters in your actual prompt escape them like: \\{ or \\}.

Dynamic prompts also support C-style comments, like `// comment` or `/* comment */`.

To use a textual inversion concepts/embeddings in a text prompt put them in the models/embeddings directory and use them in the CLIPTextEncode node like this (you can omit the .pt extension):

```embedding:embedding_filename.pt```


## How to show high-quality previews?

Use ```--preview-method auto``` to enable previews.

The default installation includes a fast latent preview method that's low-resolution. To enable higher-quality previews with [TAESD](https://github.com/madebyollin/taesd), download the [taesd_decoder.pth, taesdxl_decoder.pth, taesd3_decoder.pth and taef1_decoder.pth](https://github.com/madebyollin/taesd/) and place them in the `models/vae_approx` folder. Once they're installed, restart ComfyUI and launch it with `--preview-method taesd` to enable high-quality previews.

## How to use TLS/SSL?
Generate a self-signed certificate (not appropriate for shared/production use) and key by running the command: `openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"`

Use `--tls-keyfile key.pem --tls-certfile cert.pem` to enable TLS/SSL, the app will now be accessible with `https://...` instead of `http://...`.

> Note: Windows users can use [alexisrolland/docker-openssl](https://github.com/alexisrolland/docker-openssl) or one of the [3rd party binary distributions](https://wiki.openssl.org/index.php/Binaries) to run the command example above.
<br/><br/>If you use a container, note that the volume mount `-v` can be a relative path so `... -v ".\:/openssl-certs" ...` would create the key & cert files in the current directory of your command prompt or powershell terminal.

## Support and dev channel

[Discord](https://comfy.org/discord): Try the #help or #feedback channels.

[Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org) (it's like discord but open source).

See also: [https://www.comfy.org/](https://www.comfy.org/)

> _psst — we're hiring!_ Help build ComfyUI: [comfy.org/careers](https://www.comfy.org/careers)

## Frontend Development

As of August 15, 2024, we have transitioned to a new frontend, which is now hosted in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend). The compiled JS files (from TS/Vue) are published to [pypi](https://pypi.org/project/comfyui-frontend-package) and installed as a dependency in ComfyUI.

### Reporting Issues and Requesting Features

For any bugs, issues, or feature requests related to the frontend, please use the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend). This will help us manage and address frontend-specific concerns more efficiently.

### Using the Latest Frontend

The new frontend is now the default for ComfyUI. However, please note:

1. The frontend in the main ComfyUI repository is updated fortnightly.
2. Daily releases are available in the separate frontend repository.

To use the most up-to-date frontend version:

1. For the latest daily release, launch ComfyUI with this command line argument:

   ```
   --front-end-version Comfy-Org/ComfyUI_frontend@latest
   ```

2. For a specific version, replace `latest` with the desired version number:

   ```
   --front-end-version Comfy-Org/ComfyUI_frontend@1.2.2
   ```

This approach allows you to easily switch between the stable fortnightly release and the cutting-edge daily updates, or even specific versions for testing purposes.

# QA

### Which GPU should I buy for this?

[See this page for some recommendations](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)
