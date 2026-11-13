<div align="center">

# ComfyUI
**面向内容创作的最强大、最模块化的 AI 引擎。**


[![网站][website-shield]][website-url]
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
<!-- 用于显示总用户数，参见 https://github.com/badges/shields/issues/4500#issuecomment-2060079995 -->
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

<img width="1590" height="795" alt="ComfyUI 截图" src="https://github.com/user-attachments/assets/36e065e0-bfae-4456-8c7f-8369d5ea48a2" />
<br>
</div>

ComfyUI 是为视觉专业人士打造的 AI 创作引擎，满足你对每个模型、每个参数、每个输出结果都要有完全掌控的要求。它强大且模块化的节点图界面，让创作者能够生成图像、视频、3D 模型、音频等各类内容……

- ComfyUI 原生支持最新的开源前沿模型。
- 它是目前世界上针对扩散模型优化得最好的推理引擎，并专注于本地消费级硬件。
- 借助我们业界领先的异步权重流式加载技术，即使是最庞大的开源模型，也能在低至 4GB 显存 + 8GB 内存的机器上以相当快的速度运行（充分利用你的 GPU 算力）。
- [合作伙伴节点](https://docs.comfy.org/tutorials/partner-nodes/overview#partner-nodes) 可让你使用最好的闭源模型，例如 Nano Banana、Seedance、Hunyuan3D 等。
- 它可在 Windows、Linux 和 macOS 上运行，方式包括我们的[桌面应用](https://www.comfy.org/download)、[便携版安装](#安装)或[云端版本](https://www.comfy.org/cloud)。
- 借助 App 模式（App Mode），最复杂的 workflow 也能以简单易用的界面呈现。
- 通过我们的 API 端点，它可以无缝集成到生产流程中。

## 快速开始

### 本地

#### [桌面应用](https://www.comfy.org/download)
- 最容易上手的方式。
- 支持 Windows 与 macOS。

#### [手动安装](#手动安装windowslinux)
支持所有操作系统与 GPU 类型（NVIDIA、AMD、Intel、Apple Silicon、昇腾 Ascend）。

### 云端

#### [Comfy 云服务](https://www.comfy.org/cloud)
- 我们官方的付费云版本，适合无法负担本地硬件的用户。

## 示例

通过[新版模板 workflow](https://comfy.org/workflows) 或旧的[示例 workflow](https://comfyanonymous.github.io/ComfyUI_examples/) 了解 ComfyUI 能做什么。

## 特性

- 可视化节点图，无需写代码即可构建和复用图像、视频、音频、3D 与文本 workflow。
- 可复用的子图、workflow 模板、App 模式，以及用于把 workflow 集成进应用的本地 API。
- 高效的本地执行：异步队列、部分图重执行、智能的显存与内存管理、模型卸载（offloading），并支持量化模型。
- 广泛的原生模型支持。以下仅为代表性列表；更多持续维护、开箱即用的模板请浏览 [workflow 库](https://comfy.org/workflows/)。
  - [图像生成](https://comfy.org/workflows/tag/text-to-image/)：Stable Diffusion 1.5、SDXL、SD3.5、Flux.1、Flux.2、Qwen Image、Z-Image、Hunyuan Image 2.1、HiDream、Lumina Image 2.0、Chroma、Anima、LongCat Image、Ideogram 4、Krea 2、MageFlow、Microsoft Lens、PixelDiT、Kandinsky 5、Ernie Image 和 Moss Image。
  - [图像编辑](https://comfy.org/workflows/tag/image-edit/)：Flux Kontext、Flux.2 Klein、Qwen Image Edit、HiDream E1.1 和 O1、OmniGen2、Boogu、JoyImage Edit、MageFlow Edit 和 LongCat Image Edit。
  - [视频生成](https://comfy.org/workflows/tag/video-generation/)：Wan 2.1 和 2.2、LTX-Video 2 和 2.3、HunyuanVideo 1.5、Kandinsky 5 Video、CogVideoX、Cosmos Predict2、Bernini-R、SCAIL 2 和 Mochi。
  - [音视频生成](https://comfy.org/workflows/)：MiniMax H3 和 LTX-AV。
  - [音频生成](https://comfy.org/workflows/tag/text-to-audio/)：ACE-Step 1.5、Stable Audio 3、MiniMax Music 3 和 Yue 2。
  - [3D 与视觉](https://comfy.org/workflows/)：Hunyuan3D 2.1、TripoSplat、SeedVR2、SUPIR、Depth Anything 3、MoGe、SAM 3 和 3.1、RT-DETRv4 和 BiRefNet。
  - [文本生成](https://comfy.org/workflows/tag/text-generation/)：Gemma 3 和 4、Qwen3、Qwen3.5 和 Qwen3-VL，支持多模态输入。
- 可从受支持的模型格式中加载完整 checkpoint，或分别加载扩散模型、VAE、文本编码器、LoRA、ControlNet、适配器（adapter）与放大模型。
- 内置工具涵盖：局部重绘（inpainting）、扩图（outpainting）、参考图条件控制（reference conditioning）、蒙版与合成、模型合并、放大、插帧、分割、深度估计与媒体处理。
- workflow 可保存/加载为 JSON，也能从受支持的生成媒体中恢复完整 workflow 和随机种子（seed）。
- 完全离线运行：除非你主动请求，核心不会下载任何东西。使用 `--offline` 可禁用可选的付费 [Comfy API 节点](https://docs.comfy.org/tutorials/api-nodes/overview) 并强制所有内置功能保持离线；或使用 `--disable-partner-nodes` 仅禁用 API 节点。
- 可通过自定义节点扩展 ComfyUI。
- 使用 [`extra_model_paths.yaml`](extra_model_paths.yaml.example) 配置额外的模型目录。
- 支持保存和加载高位深图像与视频：支持 16 位 PNG、32 位 EXR、10 位 AVIF 等更多格式。
- 支持以多种格式保存和加载 HDR 视频与图像。


## 发布流程

ComfyUI 采用每周发布节奏，目标日为周一，但会因模型发布或代码库的重大变更而经常调整。共有三个相互关联的仓库：

1. **[ComfyUI 核心](https://github.com/comfyanonymous/ComfyUI)**
   - 大约每 2 周发布一个新的主稳定版本（例如 v0.7.0）。
   - 从 v0.4.0 起，补丁版本将用于把修复回迁（backport）到当前稳定版。
   - 次版本号用于 master 分支上的发布。
   - 在回迁不适用的情况下，master 分支上的发布也可能使用补丁版本号。
   - 稳定版标签之外的提交可能非常不稳定，并导致许多自定义节点失效。
   - 它是桌面版发布的基础。

2. **[Comfy 桌面版](https://github.com/Comfy-Org/Comfy-Desktop)**
   - 基于最新的稳定核心版本构建新发布。

3. **[ComfyUI 前端](https://github.com/Comfy-Org/ComfyUI_frontend)**
   - 每 2 周以上，前端更新会合并进核心仓库。
   - 在即将发布的核心版本中，功能会被冻结。
   - 开发在下一个发布周期继续进行。

## 快捷键

| 按键                                 | 说明                                                                                     |
|------------------------------------|----------------------------------------------------------------------------------------|
| `Ctrl` + `Enter`                      | 将当前图加入生成队列                                                                       |
| `Ctrl` + `Shift` + `Enter`              | 将当前图作为队首加入生成队列                                                                 |
| `Ctrl` + `Alt` + `Enter`                | 取消当前生成                                                                              |
| `Ctrl` + `Z`/`Ctrl` + `Y`                 | 撤销/重做                                                                                 |
| `Ctrl` + `S`                          | 保存 workflow                                                                             |
| `Ctrl` + `O`                          | 加载 workflow                                                                             |
| `Ctrl` + `A`                          | 全选节点                                                                                  |
| `Alt `+ `C`                           | 折叠/展开选中节点                                                                          |
| `Ctrl` + `M`                          | 静音/取消静音选中节点                                                                      |
| `Ctrl` + `B`                          | 旁路选中节点（效果等同于从图中移除该节点，并将连线重新接通）                                 |
| `Delete`/`Backspace`                   | 删除选中节点                                                                              |
| `Ctrl` + `Backspace`                   | 删除当前图                                                                                |
| `Space`                              | 按住并移动光标时平移画布                                                                    |
| `Ctrl`/`Shift` + `Click`                 | 将点击的节点加入选择                                                                       |
| `Ctrl` + `C`/`Ctrl` + `V`                  | 复制粘贴选中节点（不保留与未选中节点输出之间的连接）                                          |
| `Ctrl` + `C`/`Ctrl` + `Shift` + `V`          | 复制粘贴选中节点（保留未选中节点输出到粘贴节点输入的连接）                                     |
| `Shift` + `Drag`                       | 同时移动多个选中节点                                                                       |
| `Ctrl` + `D`                           | 加载默认图                                                                                |
| `Alt` + `+`                          | 画布放大                                                                                  |
| `Alt` + `-`                          | 画布缩小                                                                                  |
| `Ctrl` + `Shift` + 左键 + 垂直拖动      | 画布放大/缩小                                                                              |
| `P`                                  | 固定/取消固定选中节点                                                                      |
| `Ctrl` + `G`                           | 为选中节点分组                                                                            |
| `Q`                                  | 切换队列可见性                                                                            |
| `H`                                  | 切换历史记录可见性                                                                         |
| `R`                                  | 刷新图                                                                                    |
| `F`                                  | 显示/隐藏菜单                                                                             |
| `.`                                  | 视图适配到选中区域（未选中任何内容时适配整张图）                                              |
| 左键双击                            | 打开节点快速搜索面板                                                                       |
| `Shift` + `Drag`                       | 同时移动多条连线                                                                          |
| `Ctrl` + `Alt` + 左键                   | 断开点击的插槽上的所有连线                                                                  |

macOS 用户可以用 `Cmd` 代替 `Ctrl`。

# 安装

## Windows 与 Mac

我们强烈建议使用[桌面应用](https://comfy.org/download)：

### [下载链接](https://comfy.org/download)

对新手而言，桌面应用是使用 ComfyUI 最简单、最好的方式。

## Windows 便携版

有一个 Windows 便携独立版本，可在 Nvidia GPU 上运行，也可以仅在 CPU 上运行。不建议普通用户使用。普通用户请使用上面的桌面应用。

[直接下载链接（nvidia）](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)

下载后，用 [7-Zip](https://7-zip.org) 或在较新版本的 Windows 资源管理器中解压并运行即可。对于较小的模型，通常只需把 checkpoint（那些巨大的 ckpt/safetensors 文件）放入：ComfyUI\models\checkpoints；但许多较大的模型包含多个文件，请务必按照说明确定它们应放入 ComfyUI\models\ 下的哪个子目录。

如果解压遇到问题，右键文件 -> 属性 -> 解除锁定。

目前上述便携版自带 Python 3.13 和 PyTorch CUDA 13.0。如果无法启动，请更新 Nvidia 驱动。

#### 所有官方便携版下载：

[适用于 AMD GPU 的便携版](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_amd.7z)

[适用于 Intel GPU 的便携版](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_intel.7z)

[适用于 Nvidia GPU 的便携版](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)（支持 20 系及以上）。

[适用于 Nvidia GPU、含 PyTorch CUDA 12.6 与 Python 3.12 的便携版](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia_cu126.7z)（支持 Nvidia 10 系及更老 GPU，请勿在新款 20 系及以上 GPU 上使用）。

#### 如何在另一个 UI 与 ComfyUI 之间共享模型？

参见[配置文件](extra_model_paths.yaml.example)设置模型的搜索路径。在 Windows 便携版中，该文件位于 ComfyUI 目录下。把该文件重命名为 extra_model_paths.yaml，并用你喜欢的文本编辑器编辑它。


## [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

你可以使用 comfy-cli 安装并启动 ComfyUI：

```bash
pip install comfy-cli
comfy install
```

## 手动安装（Windows、Linux）

Python 3.14 可用，但部分自定义节点可能会出现问题。free-threaded（无 GIL）变体可用，但某些依赖会重新启用 GIL，因此并非完全支持。

Python 3.13 支持得非常好。如果你在 3.13 上遇到某些自定义节点依赖问题，可以尝试 3.12。

torch 2.7 仅得到最低限度支持，强烈建议使用更新版本。在 Nvidia 20 系及以上 GPU 上必须使用 cu130 或更高版本的 PyTorch。某些功能和优化可能仅在新版本中有效。我们一般建议使用最新大版本的 PyTorch 搭配最新的 CUDA 版本，除非该版本发布不足 2 周。如果你的 PyTorch 已超过 6 个月未更新，请更新它。

### 安装步骤：

Git clone 本仓库。

把你的 SD checkpoint（那些巨大的 ckpt/safetensors 文件）放入：models/checkpoints

把你的 VAE 放入：models/vae


### AMD GPU（Linux）

AMD 用户如果尚未安装 rocm 和 pytorch，可以用 pip 安装。以下是安装稳定版的命令：

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2```

以下是安装 ROCm 7.2 nightly 版的命令，可能带来一些性能提升：

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.2```


### AMD GPU（Windows，ROCm 10.0）

使用 AMD 的[多架构 PyTorch 包](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)。`device-*` 附加项会自动安装你的 GPU 内核以及匹配的 ROCm 运行时，无需单独安装 HIP SDK。

请使用 Windows 11、最新的 [AMD 显卡驱动](https://www.amd.com/en/support/download/drivers.html) 和 64 位 Python 3.13。

下面的安装命令使用 `device-all` 安装所有受支持 GPU 的内核。若想减少下载体积和磁盘占用，可以把**两处** `device-all` 都替换为你 GPU 对应的目标：

| GPU | Device 附加项 |
| --- | --- |
| RX 9070 / XT、Radeon AI PRO R9700 | `device-gfx1201` |
| RX 9060 / XT | `device-gfx1200` |
| RX 7900 XT / XTX | `device-gfx1100` |
| RX 7700 XT / 7800 XT | `device-gfx1101` |
| RX 7600 / XT | `device-gfx1102` |
| Ryzen AI Max / Max+（Strix Halo） | `device-gfx1151` |

**注意：** 此表仅列出示例。表中没有的 GPU 也可能受支持：受支持的架构包括 RDNA 2、RDNA 3、RDNA 3.5 和 RDNA 4。保留 `device-all` 可安装所有受支持目标的内核。其他型号请参见 AMD 的 [GPU 目标表](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#gfx-target-lookup-table) 与 [ROCm 兼容性矩阵](https://rocm.docs.amd.com/en/docs-10.0.0/compatibility/compatibility-matrix.html)。

**ROCm 10.0.0 搭配 PyTorch 2.13：**

```bat
pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-all]==2.13.0+rocm10.0.0" "torchvision[device-all]==0.28.0+rocm10.0.0" "torchaudio==2.11.0.2+rocm10.0.0"
```

### Intel GPU（Windows 与 Linux）

Intel Arc GPU 用户可以用 pip 安装支持 torch.xpu 的原生 PyTorch。更多信息见[此处](https://pytorch.org/docs/main/notes/get_start_xpu.html)。

1. 安装 PyTorch xpu，使用以下命令：

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu```

以下是安装 PyTorch xpu nightly 版的命令，可能带来一些性能提升：

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu```

### NVIDIA

Nvidia 用户应使用以下命令安装稳定版 pytorch：

```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130```

以下命令安装的是 pytorch nightly 版，可能带来性能提升：

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu132```

#### 故障排查

如果出现 "Torch not compiled with CUDA enabled" 错误，请先卸载 torch：

```pip uninstall torch```

然后用上面的命令重新安装。

### 依赖

在 ComfyUI 文件夹内打开终端，安装依赖：

```pip install -r requirements.txt```

完成后你就安装好了一切，可以开始运行 ComfyUI 了。

### 其他平台：

#### Apple Mac silicon

你可以在任何较新的 macOS 版本的 Apple Mac silicon（M1、M2、M3 或 M4）上安装 ComfyUI。

1. 安装 pytorch nightly。安装说明请阅读 Apple 开发者指南 [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)（请确保安装最新的 pytorch nightly）。
1. 按照 Windows 与 Linux 的 [ComfyUI 手动安装](#手动安装windowslinux) 说明操作。
1. 安装 ComfyUI [依赖](#依赖)。如果你已有另一个 Stable Diffusion UI，[也许可以复用这些依赖](#我已经安装了另一个-stable-diffusion-ui-是否真的必须安装所有这些依赖)。
1. 运行 `python main.py` 启动 ComfyUI。

> **注意**：记得把你的模型、VAE、LoRA 等放入对应的 Comfy 目录，详见 [ComfyUI 手动安装](#手动安装windowslinux)。

#### 华为昇腾 NPU

适用于兼容 Huawei Ascend Extension for PyTorch（torch_npu）的型号。开始前，请确保你的环境满足[安装](https://ascend.github.io/docs/sources/ascend/quick_install.html)页面列出的前提条件。以下是根据你的平台和安装方式整理的分步指南：

1. 首先，如有必要，按 torch-npu 安装页面的说明安装推荐的或更新的 Linux 内核版本。
2. 继续安装 Ascend Basekit（包含驱动、固件和 CANN），按照针对你所在平台的说明操作。
3. 接下来，按照 [Installation](https://ascend.github.io/docs/sources/pytorch/install.html#pytorch) 页面上针对具体平台的说明，安装 torch-npu 所需的软件包。
4. 最后，按照 [ComfyUI 手动安装](#手动安装windowslinux) 指南在 Linux 上操作。所有组件安装完成后，即可按前面的说明运行 ComfyUI。

#### 寒武纪 MLU

适用于兼容 Cambricon Extension for PyTorch（torch_mlu）的型号。以下是根据你的平台和安装方式整理的分步指南：

1. 按照 [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html) 页面上针对具体平台的说明安装寒武纪 CNToolkit。
2. 接下来，按照 [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cambricon_pytorch_1.17.0/user_guide_1.9/index.html) 页面上的说明安装 PyTorch（torch_mlu）。
3. 运行 `python main.py` 启动 ComfyUI。

#### 天数智芯 Iluvatar Corex

适用于兼容 Iluvatar Extension for PyTorch 的型号。以下是根据你的平台和安装方式整理的分步指南：

1. 按照 [Installation](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536) 页面上针对具体平台的说明安装 Iluvatar Corex Toolkit。
2. 运行 `python main.py` 启动 ComfyUI。


## [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager/tree/manager-v4)

**ComfyUI-Manager** 是一个扩展，可让你轻松安装、更新和管理 ComfyUI 的自定义节点。

### 设置

1. 安装管理器依赖：
   ```bash
   pip install -r manager_requirements.txt
   ```

2. 运行 ComfyUI 时加上 `--enable-manager` 参数以启用管理器：
   ```bash
   python main.py --enable-manager
   ```

### 命令行选项

| 参数 | 说明 |
|------|-------------|
| `--enable-manager` | 启用 ComfyUI-Manager |
| `--enable-manager-legacy-ui` | 使用旧版管理器界面而非新界面（隐含 `--enable-manager`） |
| `--disable-manager-ui` | 禁用管理器界面和端点，同时保留安全检查、定时安装完成等后台功能（需要 `--enable-manager`） |


# 运行

```python main.py```

### 对于 ROCm 未官方支持的 AMD 显卡

如果遇到问题，可以尝试用这些命令运行：

对于 6700、6600 以及可能的其他 RDNA2 或更老型号：```HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py```

对于 AMD 7600 以及可能的其他 RDNA3 显卡：```HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py```

### AMD ROCm 提示

你可以尝试设置环境变量 `PYTORCH_TUNABLEOP_ENABLED=1`，它可能加快运行速度，但代价是首次运行会非常慢。

# 说明

只有图中所有输入都正确的输出部分才会被执行。

只有每次执行之间发生变化的部分才会被执行。如果你提交同一个图两次，只有第一次会执行。如果你改动了图的最后一部分，那么只有你改动的部分以及依赖它的部分会被执行。

把生成的 png 拖到网页上或加载它，你就能得到完整的 workflow，包括生成它所用的随机种子（seed）。

你可以用 () 来调整某个词或短语的权重，例如：(good code:1.2) 或 (bad code:0.8)。() 的默认权重为 1.1。如果你想在提示词中实际使用 () 字符，请转义为 \\( 或 \\)。

你可以使用 {day|night} 实现通配符/动态提示词。使用该语法时，前端每次排队提示词都会在 "wild"、"card"、"test" 中随机把 "{wild|card|test}" 替换为其一。如果你想在提示词中实际使用 {} 字符，请转义为：\\{ 或 \\}。

动态提示词还支持 C 风格注释，例如 `// comment` 或 `/* comment */`。

要在文本提示词中使用文本反转（textual inversion）概念/嵌入（embedding），请把它们放入 models/embeddings 目录，并在 CLIPTextEncode 节点中像这样使用（可以省略 .pt 扩展名）：

```embedding:embedding_filename.pt```


## 如何显示高质量预览？

使用 ```--preview-method auto``` 启用预览。

默认安装包含一种快速但低分辨率的潜空间（latent）预览方法。要使用 [TAESD](https://github.com/madebyollin/taesd) 启用更高质量的预览，请下载 [taesd_decoder.pth、taesdxl_decoder.pth、taesd3_decoder.pth 和 taef1_decoder.pth](https://github.com/madebyollin/taesd/) 并将其放入 `models/vae_approx` 文件夹。安装完成后，重启 ComfyUI，并用 `--preview-method taesd` 启动以启用高质量预览。

## 如何使用 TLS/SSL？

运行以下命令生成自签名证书（不适用于共享/生产环境）和密钥：`openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"`

使用 `--tls-keyfile key.pem --tls-certfile cert.pem` 启用 TLS/SSL，此后应用将通过 `https://...` 而不是 `http://...` 访问。

> 注意：Windows 用户可以使用 [alexisrolland/docker-openssl](https://github.com/alexisrolland/docker-openssl) 或[第三方二进制发行版](https://wiki.openssl.org/index.php/Binaries)中的一种来运行上面的命令示例。
<br/><br/>如果你使用容器，请注意卷挂载 `-v` 可以是相对路径，因此 `... -v ".\:/openssl-certs" ...` 会在你的命令提示符或 PowerShell 终端的当前目录中生成密钥和证书文件。

## 支持与开发者频道

[Discord](https://comfy.org/discord)：可尝试 #help 或 #feedback 频道。

[Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org)（类似 Discord，但是开源的）。

另见：[https://www.comfy.org/](https://www.comfy.org/)

> _嘘 —— 我们正在招人！_ 一起来构建 ComfyUI：[comfy.org/careers](https://www.comfy.org/careers)

## 前端开发

自 2024 年 8 月 15 日起，我们已切换到新的前端，现在托管在单独的仓库中：[ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend)。编译后的 JS 文件（来自 TS/Vue）发布到 [pypi](https://pypi.org/project/comfyui-frontend-package)，并作为依赖安装在 ComfyUI 中。

### 报告问题与功能请求

任何与前端相关的 bug、问题或功能请求，请使用 [ComfyUI Frontend 仓库](https://github.com/Comfy-Org/ComfyUI_frontend)。这有助于我们更高效地管理和处理前端特有的问题。

### 使用最新前端

新前端现在是 ComfyUI 的默认前端。但请注意：

1. 主 ComfyUI 仓库中的前端每两周更新一次。
2. 每日发布版本可在单独的前端仓库中获取。

要使用最新的前端版本：

1. 要使用最新的每日发布版本，请用以下命令行参数启动 ComfyUI：

   ```
   --front-end-version Comfy-Org/ComfyUI_frontend@latest
   ```

2. 要使用特定版本，请将 `latest` 替换为所需的版本号：

   ```
   --front-end-version Comfy-Org/ComfyUI_frontend@1.2.2
   ```

这样你就可以轻松地在稳定的双周发布版和前沿的每日更新之间切换，甚至可以在测试时使用特定版本。

# 常见问题（QA）

### 我应该买哪款 GPU？

[参见此页面获取一些推荐](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)
