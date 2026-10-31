<div align="center">

# ComfyUI
**最强大、模块化的可视化 AI 引擎和应用。**


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
[discord-url]: https://www.comfy.org/discord
[twitter-shield]: https://img.shields.io/twitter/follow/ComfyUI
[twitter-url]: https://x.com/ComfyUI

[github-release-shield]: https://img.shields.io/github/v/release/comfyanonymous/ComfyUI?style=flat&sort=semver
[github-release-link]: https://github.com/comfyanonymous/ComfyUI/releases
[github-release-date-shield]: https://img.shields.io/github/release-date/comfyanonymous/ComfyUI?style=flat
[github-downloads-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/total?style=flat
[github-downloads-latest-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest
[github-downloads-link]: https://github.com/comfyanonymous/ComfyUI/releases

![ComfyUI Screenshot](https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe)
</div>

ComfyUI 让你能够使用基于图形/节点/流程图的界面设计和执行高级的 Stable Diffusion 管道。支持 Windows、Linux 和 macOS。

## 快速开始

#### [桌面应用程序](https://www.comfy.org/download)
- 最简单的入门方式。
- 支持 Windows 和 macOS。

#### [Windows 便携包](#安装)
- 获取最新提交，完全便携。
- 支持 Windows。

#### [手动安装](#手动安装-windows-linux)
支持所有操作系统和 GPU 类型（NVIDIA、AMD、Intel、Apple Silicon、Ascend）。

## [示例](https://comfyanonymous.github.io/ComfyUI_examples/)
查看 [示例工作流](https://comfyanonymous.github.io/ComfyUI_examples/) 了解 ComfyUI 能做什么。

## 特性
- 节点/图形/流程图界面，无需编写任何代码即可实验和创建复杂的 Stable Diffusion 工作流。
- 图像模型
   - SD1.x, SD2.x ([unCLIP](https://comfyanonymous.github.io/ComfyUI_examples/unclip/))
   - [SDXL](https://comfyanonymous.github.io/ComfyUI_examples/sdxl/), [SDXL Turbo](https://comfyanonymous.github.io/ComfyUI_examples/sdturbo/)
   - [Stable Cascade](https://comfyanonymous.github.io/ComfyUI_examples/stable_cascade/)
   - [SD3 和 SD3.5](https://comfyanonymous.github.io/ComfyUI_examples/sd3/)
   - Pixart Alpha 和 Sigma
   - [AuraFlow](https://comfyanonymous.github.io/ComfyUI_examples/aura_flow/)
   - [HunyuanDiT](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_dit/)
   - [Flux](https://comfyanonymous.github.io/ComfyUI_examples/flux/)
   - [Lumina Image 2.0](https://comfyanonymous.github.io/ComfyUI_examples/lumina2/)
   - [HiDream](https://comfyanonymous.github.io/ComfyUI_examples/hidream/)
   - [Qwen Image](https://comfyanonymous.github.io/ComfyUI_examples/qwen_image/)
   - [Hunyuan Image 2.1](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_image/)
   - [Flux 2](https://comfyanonymous.github.io/ComfyUI_examples/flux2/)
   - [Z Image](https://comfyanonymous.github.io/ComfyUI_examples/z_image/)
- 图像编辑模型
   - [Omnigen 2](https://comfyanonymous.github.io/ComfyUI_examples/omnigen/)
   - [Flux Kontext](https://comfyanonymous.github.io/ComfyUI_examples/flux/#flux-kontext-image-editing-model)
   - [HiDream E1.1](https://comfyanonymous.github.io/ComfyUI_examples/hidream/#hidream-e11)
   - [Qwen Image Edit](https://comfyanonymous.github.io/ComfyUI_examples/qwen_image/#edit-model)
- 视频模型
   - [Stable Video Diffusion](https://comfyanonymous.github.io/ComfyUI_examples/video/)
   - [Mochi](https://comfyanonymous.github.io/ComfyUI_examples/mochi/)
   - [LTX-Video](https://comfyanonymous.github.io/ComfyUI_examples/ltxv/)
   - [Hunyuan Video](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_video/)
   - [Wan 2.1](https://comfyanonymous.github.io/ComfyUI_examples/wan/)
   - [Wan 2.2](https://comfyanonymous.github.io/ComfyUI_examples/wan22/)
   - [Hunyuan Video 1.5](https://docs.comfy.org/tutorials/video/hunyuan/hunyuan-video-1-5)
- 音频模型
   - [Stable Audio](https://comfyanonymous.github.io/ComfyUI_examples/audio/)
   - [ACE Step](https://comfyanonymous.github.io/ComfyUI_examples/audio/)
- 3D 模型
   - [Hunyuan3D 2.0](https://docs.comfy.org/tutorials/3d/hunyuan3D-2)
- 异步队列系统
- 许多优化：仅重新执行执行之间更改的工作流部分。
- 智能内存管理：可以通过智能卸载在低至 1GB vram 的 GPU 上自动运行大型模型。
- 即使没有 GPU 也可以工作：```--cpu```（慢）
- 可以加载 ckpt 和 safetensors：一体化检查点或独立扩散模型、VAE 和 CLIP 模型。
- 安全加载 ckpt、pt、pth 等文件。
- Embeddings/Textual inversion
- [Loras (regular, locon and loha)](https://comfyanonymous.github.io/ComfyUI_examples/lora/)
- [Hypernetworks](https://comfyanonymous.github.io/ComfyUI_examples/hypernetworks/)
- 从生成的 PNG、WebP 和 FLAC 文件加载完整的工作流（带有种子）。
- 将工作流保存/加载为 Json 文件。
- 节点界面可用于创建复杂的工作流，如 [Hires fix](https://comfyanonymous.github.io/ComfyUI_examples/2_pass_txt2img/) 或更高级的工作流。
- [区域合成](https://comfyanonymous.github.io/ComfyUI_examples/area_composition/)
- [Inpainting](https://comfyanonymous.github.io/ComfyUI_examples/inpaint/) 支持常规和 inpainting 模型。
- [ControlNet 和 T2I-Adapter](https://comfyanonymous.github.io/ComfyUI_examples/controlnet/)
- [放大模型 (ESRGAN, ESRGAN variants, SwinIR, Swin2SR, etc...)](https://comfyanonymous.github.io/ComfyUI_examples/upscale_models/)
- [GLIGEN](https://comfyanonymous.github.io/ComfyUI_examples/gligen/)
- [模型合并](https://comfyanonymous.github.io/ComfyUI_examples/model_merging/)
- [LCM 模型和 Loras](https://comfyanonymous.github.io/ComfyUI_examples/lcm/)
- 使用 [TAESD](#如何显示高质量预览) 进行潜在预览
- 完全离线工作：除非你想要，否则核心永远不会下载任何东西。
- 可选的 API 节点，通过在线 [Comfy API](https://docs.comfy.org/tutorials/api-nodes/overview) 使用外部提供商的付费模型。
- [配置文件](extra_model_paths.yaml.example) 用于设置模型的搜索路径。

工作流示例可以在 [示例页面](https://comfyanonymous.github.io/ComfyUI_examples/) 找到。

## 发布流程

ComfyUI 遵循每周发布周期，目标是周一，但由于模型发布或代码库的重大更改，这经常会发生变化。有三个相互关联的存储库：

1. **[ComfyUI Core](https://github.com/comfyanonymous/ComfyUI)**
   - 大约每周发布一个新的稳定版本（例如 v0.7.0）。
   - 稳定版本标签之外的提交可能非常不稳定，并破坏许多自定义节点。
   - 作为桌面版本的基础。

2. **[ComfyUI Desktop](https://github.com/Comfy-Org/desktop)**
   - 使用最新的稳定核心版本构建新版本。

3. **[ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend)**
   - 每周前端更新合并到核心存储库中。
   - 即将发布的核心版本的功能被冻结。
   - 下一个发布周期的开发继续进行。

## 快捷键

| 快捷键                             | 说明                                                                                                               |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `Ctrl` + `Enter`                      | 将当前图形排队生成                                                                                                 |
| `Ctrl` + `Shift` + `Enter`              | 将当前图形作为第一个排队生成                                                                                       |
| `Ctrl` + `Alt` + `Enter`                | 取消当前生成                                                                                                       |
| `Ctrl` + `Z`/`Ctrl` + `Y`                 | 撤销/重做                                                                                                          |
| `Ctrl` + `S`                          | 保存工作流                                                                                                         |
| `Ctrl` + `O`                          | 加载工作流                                                                                                         |
| `Ctrl` + `A`                          | 选择所有节点                                                                                                       |
| `Alt `+ `C`                           | 折叠/展开选定的节点                                                                                                |
| `Ctrl` + `M`                          | 静音/取消静音选定的节点                                                                                            |
| `Ctrl` + `B`                           | 绕过选定的节点（就像节点从图中移除并且连线重新连接通过一样）                                                       |
| `Delete`/`Backspace`                   | 删除选定的节点                                                                                                     |
| `Ctrl` + `Backspace`                   | 删除当前图形                                                                                                       |
| `Space`                              | 按住并移动光标时移动画布                                                                                           |
| `Ctrl`/`Shift` + `Click`                 | 将点击的节点添加到选择中                                                                                           |
| `Ctrl` + `C`/`Ctrl` + `V`                  | 复制并粘贴选定的节点（不保持与未选定节点输出的连接）                                                               |
| `Ctrl` + `C`/`Ctrl` + `Shift` + `V`          | 复制并粘贴选定的节点（保持从未选定节点的输出到粘贴节点的输入的连接）                                               |
| `Shift` + `Drag`                       | 同时移动多个选定的节点                                                                                             |
| `Ctrl` + `D`                           | 加载默认图形                                                                                                       |
| `Alt` + `+`                          | 画布放大                                                                                                           |
| `Alt` + `-`                          | 画布缩小                                                                                                           |
| `Ctrl` + `Shift` + LMB + 垂直拖动      | 画布放大/缩小                                                                                                      |
| `P`                                  | 固定/取消固定选定的节点                                                                                            |
| `Ctrl` + `G`                           | 组合选定的节点                                                                                                     |
| `Q`                                 | 切换队列的可见性                                                                                                   |
| `H`                                  | 切换历史记录的可见性                                                                                               |
| `R`                                  | 刷新图形                                                                                                           |
| `F`                                  | 显示/隐藏菜单                                                                                                      |
| `.`                                  | 适应视图到选择（未选择任何内容时适应整个图形）                                                                     |
| 双击 LMB                             | 打开节点快速搜索面板                                                                                               |
| `Shift` + 拖动                       | 一次移动多条连线                                                                                                   |
| `Ctrl` + `Alt` + LMB                   | 断开点击插槽的所有连线                                                                                             |

macOS 用户可以将 `Ctrl` 替换为 `Cmd`

# 安装

## Windows 便携版

在 [发布页面](https://github.com/comfyanonymous/ComfyUI/releases) 上有一个适用于 Windows 的便携式独立构建，应该可以在 Nvidia GPU 上运行，或者仅在 CPU 上运行。

### [直接下载链接](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)

只需下载，使用 [7-Zip](https://7-zip.org) 解压，或者在最近的 Windows 版本上使用资源管理器解压并运行。对于较小的模型，通常只需要将 checkpoints（巨大的 ckpt/safetensors 文件）放在：ComfyUI\models\checkpoints 中，但许多较大的模型有多个文件。请务必按照说明了解将它们放在 ComfyUI\models\ 的哪个子文件夹中。

如果解压有问题，请右键单击文件 -> 属性 -> 解除锁定

如果无法启动，请更新您的 Nvidia 驱动程序。

#### 替代下载：

[适用于 AMD GPU 的实验性便携版](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_amd.7z)

[带有 pytorch cuda 12.8 和 python 3.12 的便携版](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia_cu128.7z)。

[带有 pytorch cuda 12.6 和 python 3.12 的便携版](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia_cu126.7z)（支持 Nvidia 10 系列及更旧的 GPU）。

#### 如何在另一个 UI 和 ComfyUI 之间共享模型？

请参阅 [配置文件](extra_model_paths.yaml.example) 以设置模型的搜索路径。在独立的 Windows 构建中，您可以在 ComfyUI 目录中找到此文件。将此文件重命名为 extra_model_paths.yaml 并使用您喜欢的文本编辑器进行编辑。


## [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

您可以使用 comfy-cli 安装并启动 ComfyUI：
```bash
pip install comfy-cli
comfy install
```

## 手动安装 (Windows, Linux)

Python 3.14 可以工作，但您可能会遇到 torch 编译节点的问题。自由线程变体仍然缺少一些依赖项。

Python 3.13 支持得很好。如果您在 3.13 上遇到某些自定义节点依赖项的问题，可以尝试 3.12。

### 说明：

Git clone 此仓库。

将您的 SD checkpoints（巨大的 ckpt/safetensors 文件）放在：models/checkpoints

将您的 VAE 放在：models/vae


### AMD GPU (Linux)

AMD 用户如果尚未安装，可以使用 pip 安装 rocm 和 pytorch，这是安装稳定版本的命令：

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4```

这是安装带有 ROCm 7.0 的 nightly 版本的命令，可能会有一些性能改进：

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.1```


### AMD GPU（实验性：Windows 和 Linux），仅限 RDNA 3, 3.5 和 4。

这些硬件支持比上面的构建少，但它们可以在 Windows 上运行。您还需要安装特定于您的硬件的 pytorch 版本。

RDNA 3 (RX 7000 系列):

```pip install --pre torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/```

RDNA 3.5 (Strix halo/Ryzen AI Max+ 365):

```pip install --pre torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx1151/```

RDNA 4 (RX 9000 系列):

```pip install --pre torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/```

### Intel GPU (Windows 和 Linux)

Intel Arc GPU 用户可以使用 pip 安装具有 torch.xpu 支持的原生 PyTorch。更多信息可以在 [这里](https://pytorch.org/docs/main/notes/get_start_xpu.html) 找到。

1. 要安装 PyTorch xpu，请使用以下命令：

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu```

这是安装 Pytorch xpu nightly 的命令，可能会有一些性能改进：

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu```

### NVIDIA

Nvidia 用户应使用此命令安装稳定的 pytorch：

```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130```

这是安装 pytorch nightly 的命令，可能会有性能改进。

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130```

#### 故障排除

如果您收到 "Torch not compiled with CUDA enabled" 错误，请使用以下命令卸载 torch：

```pip uninstall torch```

并再次使用上面的命令安装它。

### 依赖项

通过在 ComfyUI 文件夹中打开终端并运行以下命令来安装依赖项：

```pip install -r requirements.txt```

在此之后，您应该已经安装了所有内容，并可以继续运行 ComfyUI。

### 其他：

#### Apple Mac silicon

您可以在任何最近的 macOS 版本上在 Apple Mac silicon (M1 或 M2) 上安装 ComfyUI。

1. 安装 pytorch nightly。有关说明，请阅读 [Mac 上的加速 PyTorch 训练](https://developer.apple.com/metal/pytorch/) Apple 开发者指南（确保安装最新的 pytorch nightly）。
1. 按照 Windows 和 Linux 的 [ComfyUI 手动安装](#手动安装-windows-linux) 说明进行操作。
1. 安装 ComfyUI [依赖项](#依赖项)。如果您安装了另一个 Stable Diffusion UI，[您可能能够重用依赖项](#我已经安装了另一个-stable-diffusion-ui-我真的必须安装所有这些依赖项吗)。
1. 通过运行 `python main.py` 启动 ComfyUI

> **注意**：请记住将您的模型、VAE、LoRA 等添加到相应的 Comfy 文件夹中，如 [ComfyUI 手动安装](#手动安装-windows-linux) 中所述。

#### Ascend NPU

对于与 Ascend Extension for PyTorch (torch_npu) 兼容的模型。首先，确保您的环境满足 [安装](https://ascend.github.io/docs/sources/ascend/quick_install.html) 页面上列出的先决条件。这是针对您的平台和安装方法的逐步指南：

1. 如果需要，首先安装 torch-npu 安装页面中指定的推荐或更新的 Linux 内核版本。
2. 按照针对您特定平台的说明，继续安装 Ascend Basekit，其中包括驱动程序、固件和 CANN。
3. 接下来，按照 [安装](https://ascend.github.io/docs/sources/pytorch/install.html#pytorch) 页面上的特定于平台的说明安装 torch-npu 的必要包。
4. 最后，遵循 Linux 的 [ComfyUI 手动安装](#手动安装-windows-linux) 指南。安装所有组件后，您可以如前所述运行 ComfyUI。

#### Cambricon MLU

对于与 Cambricon Extension for PyTorch (torch_mlu) 兼容的模型。这是针对您的平台和安装方法的逐步指南：

1. 按照 [安装](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html) 上的特定于平台的说明安装 Cambricon CNToolkit
2. 接下来，按照 [安装](https://www.cambricon.com/docs/sdk_1.15.0/cambricon_pytorch_1.17.0/user_guide_1.9/index.html) 上的说明安装 PyTorch(torch_mlu)
3. 通过运行 `python main.py` 启动 ComfyUI

#### Iluvatar Corex

对于与 Iluvatar Extension for PyTorch 兼容的模型。这是针对您的平台和安装方法的逐步指南：

1. 按照 [安装](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536) 上的特定于平台的说明安装 Iluvatar Corex Toolkit
2. 通过运行 `python main.py` 启动 ComfyUI


## [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager/tree/manager-v4)

**ComfyUI-Manager** 是一个扩展，允许您轻松安装、更新和管理 ComfyUI 的自定义节点。

### 设置

1. 安装管理器依赖项：
   ```bash
   pip install -r manager_requirements.txt
   ```

2. 运行 ComfyUI 时使用 `--enable-manager` 标志启用管理器：
   ```bash
   python main.py --enable-manager
   ```

### 命令行选项

| 标志 | 描述 |
|------|-------------|
| `--enable-manager` | 启用 ComfyUI-Manager |
| `--enable-manager-legacy-ui` | 使用旧版管理器 UI 而不是新 UI（需要 `--enable-manager`） |
| `--disable-manager-ui` | 禁用管理器 UI 和端点，同时保留后台功能，如安全检查和计划安装完成（需要 `--enable-manager`） |


# 运行

```python main.py```

### 对于 ROCm 未正式支持的 AMD 卡

如果有问题，请尝试使用此命令运行它：

对于 6700, 6600 和可能的其他 RDNA2 或更旧的卡：```HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py```

对于 AMD 7600 和可能的其他 RDNA3 卡：```HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py```

### AMD ROCm 提示

您可以使用此命令在某些 AMD GPU 上的 ComfyUI 中启用最近 pytorch 的实验性内存高效注意力，它应该已经在 RDNA3 上默认启用。如果这提高了您 GPU 上最新 pytorch 的速度，请报告它，以便我可以默认启用它。

```TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention```

您也可以尝试设置此环境变量 `PYTORCH_TUNABLEOP_ENABLED=1`，这可能会加快速度，但代价是非常慢的初始运行。

# 注意事项

只有具有所有正确输入的输出的图形部分才会被执行。

只有从每次执行到下一次执行发生更改的图形部分才会被执行，如果您提交相同的图形两次，只有第一次会被执行。如果您更改图形的最后一部分，只有您更改的部分和依赖于它的部分会被执行。

将生成的 png 拖到网页上或加载它将为您提供完整的工作流，包括用于创建它的种子。

您可以使用 () 来更改单词或短语的强调，例如：(good code:1.2) 或 (bad code:0.8)。() 的默认强调是 1.1。要在实际提示中使用 () 字符，请像 \\( 或 \\) 一样转义它们。

您可以使用 {day|night} 进行通配符/动态提示。使用此语法 "{wild|card|test}" 将在每次您排队提示时由前端随机替换为 "wild"、"card" 或 "test"。要在实际提示中使用 {} 字符，请像：\\{ or \\} 一样转义它们。

动态提示还支持 C 风格的注释，如 `// comment` 或 `/* comment */`。

要在文本提示中使用文本反转概念/嵌入，请将它们放在 models/embeddings 目录中，并在 CLIPTextEncode 节点中像这样使用它们（您可以省略 .pt 扩展名）：

```embedding:embedding_filename.pt```


## 如何显示高质量预览？

使用 ```--preview-method auto``` 启用预览。

默认安装包括一种快速的潜在预览方法，分辨率较低。要使用 [TAESD](https://github.com/madebyollin/taesd) 启用更高质量的预览，请下载 [taesd_decoder.pth, taesdxl_decoder.pth, taesd3_decoder.pth 和 taef1_decoder.pth](https://github.com/madebyollin/taesd/) 并将它们放在 `models/vae_approx` 文件夹中。安装完成后，重新启动 ComfyUI 并使用 `--preview-method taesd` 启动它以启用高质量预览。

## 如何使用 TLS/SSL？
通过运行以下命令生成自签名证书（不适合共享/生产使用）和密钥：`openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"`

使用 `--tls-keyfile key.pem --tls-certfile cert.pem` 启用 TLS/SSL，应用程序现在将可以通过 `https://...` 而不是 `http://...` 访问。

> 注意：Windows 用户可以使用 [alexisrolland/docker-openssl](https://github.com/alexisrolland/docker-openssl) 或 [第三方二进制发行版](https://wiki.openssl.org/index.php/Binaries) 之一来运行上面的命令示例。
<br/><br/>如果您使用容器，请注意卷挂载 `-v` 可以是相对路径，因此 `... -v ".\:/openssl-certs" ...` 将在命令提示符或 powershell 终端的当前目录中创建密钥和证书文件。

## 支持和开发频道

[Discord](https://comfy.org/discord): 尝试 #help 或 #feedback 频道。

[Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org) (它像 discord 但是开源的)。

另请参阅：[https://www.comfy.org/](https://www.comfy.org/)

## 前端开发

截至 2024 年 8 月 15 日，我们已过渡到新的前端，该前端现在托管在一个单独的存储库中：[ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend)。此存储库现在在 `web/` 目录下托管编译后的 JS（来自 TS/Vue）。

### 报告问题和请求功能

对于与前端相关的任何错误、问题或功能请求，请使用 [ComfyUI Frontend 存储库](https://github.com/Comfy-Org/ComfyUI_frontend)。这将帮助我们更有效地管理和解决前端特定的问题。

### 使用最新前端

新前端现在是 ComfyUI 的默认前端。但是，请注意：

1. 主 ComfyUI 存储库中的前端每两周更新一次。
2. 每日发布在单独的前端存储库中可用。

要使用最新的前端版本：

1. 对于最新的每日发布，使用此命令行参数启动 ComfyUI：

   ```
   --front-end-version Comfy-Org/ComfyUI_frontend@latest
   ```

2. 对于特定版本，将 `latest` 替换为所需的版本号：

   ```
   --front-end-version Comfy-Org/ComfyUI_frontend@1.2.2
   ```

这种方法允许您轻松地在稳定的每两周发布和前沿的每日更新之间切换，甚至可以切换到特定版本进行测试。

### 访问旧版前端

如果您因任何原因需要使用旧版前端，可以使用以下命令行参数访问它：

```
--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest
```

这将使用保存在 [ComfyUI Legacy Frontend 存储库](https://github.com/Comfy-Org/ComfyUI_legacy_frontend) 中的旧版前端快照。

# QA

### 我应该为此购买哪种 GPU？

[请参阅此页面以获取一些建议](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)
