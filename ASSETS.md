# 资产清单

核对日期：2026-09-21。扩展版本和模型状态为本机目录快照，不代表上游最新版本。

Docker Compose 将 `custom_nodes/` 和 `models/` 挂载进容器；Compose 本身不会下载资产。节点或 Manager 在启动、使用时可能另行安装依赖或下载模型。

- [自定义节点](#自定义节点)：来源、当前提交及安装方式。
- [模型](#模型)：原清单中模型的本地文件状态与下载来源，不是整个模型目录的完整索引。
- [更新记录](#更新记录)：兼容性限制、历史验证及回退资料。

## 自定义节点

### 可独立安装的扩展

目录名相对于 `custom_nodes/`，大小写应保持一致。提交号不包含本地未提交补丁。

| 扩展目录 | 来源 | 当前提交 |
| --- | --- | --- |
| `comfy_mtb` | [melMass/comfy_mtb](https://github.com/melMass/comfy_mtb) | `b35b5d8` |
| `ComfyUI-Anima-LLLite` | [kohya-ss/ComfyUI-Anima-LLLite](https://github.com/kohya-ss/ComfyUI-Anima-LLLite) | `b7495bd` |
| `ComfyUI-Autocomplete-Plus` | [newtextdoc1111/ComfyUI-Autocomplete-Plus](https://github.com/newtextdoc1111/ComfyUI-Autocomplete-Plus) | `9cfd2ac` |
| `ComfyUI-Custom-Scripts` | [pythongosssss/ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts) | `609f3af` |
| `ComfyUI-Easy-Use` | [yolain/ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use) | `32931f0` |
| `ComfyUI-Florence2` | [kijai/ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2) | `9ece3de` |
| `comfyui-frame-interpolation` | [Fannovel16/ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) | `26545cc` |
| `ComfyUI-GGUF` | [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | `6ea2651` |
| `ComfyUI-Image-Filters` | [spacepxl/ComfyUI-Image-Filters](https://github.com/spacepxl/ComfyUI-Image-Filters) | `bbb3fb0` |
| `ComfyUI-KJNodes` | [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | `d3cfe21` |
| `ComfyUI-layerdiffuse` | [huchenlei/ComfyUI-layerdiffuse](https://github.com/huchenlei/ComfyUI-layerdiffuse) | `b4f6a9e` |
| `ComfyUI-LTXVideo` | [Lightricks/ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) | `15d09ab` |
| `ComfyUI-MatAnyone` | [FuouM/ComfyUI-MatAnyone](https://github.com/FuouM/ComfyUI-MatAnyone) | `87cbce3` |
| `ComfyUI-OpenPose-Studio` | [andreszs/ComfyUI-OpenPose-Studio](https://github.com/andreszs/ComfyUI-OpenPose-Studio) | `4071e2c` |
| `comfyui-photoshop` | [NimaNzrii/comfyui-photoshop](https://github.com/NimaNzrii/comfyui-photoshop) | `d26c8f6` |
| `ComfyUI-Prompt-Assistant` | [yawiii/ComfyUI-Prompt-Assistant](https://github.com/yawiii/ComfyUI-Prompt-Assistant) | `e0587e6` |
| `ComfyUI-QwenVL` | [1038lab/ComfyUI-QwenVL](https://github.com/1038lab/ComfyUI-QwenVL) | `1b67b44` |
| `ComfyUI-RMBG` | [1038lab/ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) | `58f1947` |
| `ComfyUI-See-through` | [jtydhr88/ComfyUI-See-through](https://github.com/jtydhr88/ComfyUI-See-through) | `98d754b` |
| `ComfyUI-SeedVR2_VideoUpscaler` | [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) | `4490bd1` |
| `ComfyUI-segment-anything-2` | [kijai/ComfyUI-segment-anything-2](https://github.com/kijai/ComfyUI-segment-anything-2) | `0c35fff` |
| `ComfyUI-VideoHelperSuite` | [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | `4d907be` |
| `comfyui-WhiteRabbit` | [Artificial-Sweetener/comfyui-WhiteRabbit](https://github.com/Artificial-Sweetener/comfyui-WhiteRabbit) | `4815da4` |
| `ComfyUI_Comfyroll_CustomNodes` | [Suzie1/ComfyUI_Comfyroll_CustomNodes](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes) | `d78b780` |
| `comfyui_controlnet_aux` | [Fannovel16/comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) | `59b1fc4` |
| `ComfyUI_essentials` | [cubiq/ComfyUI_essentials](https://github.com/cubiq/ComfyUI_essentials) | `9d9f4be` |
| `ComfyUI_Fill-Nodes` | [filliptm/ComfyUI_Fill-Nodes](https://github.com/filliptm/ComfyUI_Fill-Nodes) | `a32f9b6` |
| `ComfyUI_IPAdapter_plus` | [cubiq/ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) | `a0f451a` |
| `ComfyUI_LayerStyle` | [chflame163/ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle) | `a3459a7` |
| `ComfyUI_RH_OpenAPI` | [HM-RunningHub/ComfyUI_RH_OpenAPI](https://github.com/HM-RunningHub/ComfyUI_RH_OpenAPI) | `8f9c858` |
| `ComfyUI_UltimateSDUpscale` | [ssitu/ComfyUI_UltimateSDUpscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale) | `a5547db` |
| `LanPaint` | [scraed/LanPaint](https://github.com/scraed/LanPaint) | `32cf848` |
| `Plush-for-ComfyUI` | [glibsonoran/Plush-for-ComfyUI](https://github.com/glibsonoran/Plush-for-ComfyUI) | `a9e824e` |
| `rgthree-comfy` | [rgthree/rgthree-comfy](https://github.com/rgthree/rgthree-comfy) | `2c5342a` |
| `sd-ppp-2.0` | [zombieyang/sd-ppp](https://github.com/zombieyang/sd-ppp) | `d965457` |
| `was-ns` | [ltdrdata/was-node-suite-comfyui](https://github.com/ltdrdata/was-node-suite-comfyui) | `44de705` |
| `.disabled/sources/perfectPixel` | [theamusing/perfectPixel](https://github.com/theamusing/perfectPixel) | `72096de` |

**兼容性限制：** `ComfyUI-LTXVideo` 保持在 `15d09ab`；不要直接更新到上游最新提交，详见更新记录。`sd-ppp-2.0` 当前使用官方 v2.0 安装包，克隆源码不等于获得相同的已构建前端。

### 补齐缺失扩展

以下命令仅补齐缺失的普通 Git 扩展，不更新已有目录、不恢复本地补丁，也不安装 Python 依赖。`sd-ppp-2.0` 应使用官方 v2.0 安装包；本地扩展需从备份恢复。先备份已有资产，再在仓库根目录执行：

```bash
set -e
mkdir -p custom_nodes
while read -r node_name node_url node_revision; do
  if [ -e "custom_nodes/$node_name" ] || [ -L "custom_nodes/$node_name" ]; then
    continue
  fi
  git clone "$node_url" "custom_nodes/$node_name"
  git -C "custom_nodes/$node_name" checkout --detach "$node_revision"
done <<'NODES'
comfy_mtb https://github.com/melMass/comfy_mtb.git b35b5d8
ComfyUI-Anima-LLLite https://github.com/kohya-ss/ComfyUI-Anima-LLLite.git b7495bd
ComfyUI-Autocomplete-Plus https://github.com/newtextdoc1111/ComfyUI-Autocomplete-Plus.git 9cfd2ac
ComfyUI-Custom-Scripts https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git 609f3af
ComfyUI-Easy-Use https://github.com/yolain/ComfyUI-Easy-Use.git 32931f0
ComfyUI-Florence2 https://github.com/kijai/ComfyUI-Florence2.git 9ece3de
comfyui-frame-interpolation https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git 26545cc
ComfyUI-GGUF https://github.com/city96/ComfyUI-GGUF.git 6ea2651
ComfyUI-Image-Filters https://github.com/spacepxl/ComfyUI-Image-Filters.git bbb3fb0
ComfyUI-KJNodes https://github.com/kijai/ComfyUI-KJNodes.git d3cfe21
ComfyUI-layerdiffuse https://github.com/huchenlei/ComfyUI-layerdiffuse.git b4f6a9e
ComfyUI-LTXVideo https://github.com/Lightricks/ComfyUI-LTXVideo.git 15d09ab
ComfyUI-MatAnyone https://github.com/FuouM/ComfyUI-MatAnyone.git 87cbce3
ComfyUI-OpenPose-Studio https://github.com/andreszs/ComfyUI-OpenPose-Studio.git 4071e2c
comfyui-photoshop https://github.com/NimaNzrii/comfyui-photoshop.git d26c8f6
ComfyUI-Prompt-Assistant https://github.com/yawiii/ComfyUI-Prompt-Assistant.git e0587e6
ComfyUI-QwenVL https://github.com/1038lab/ComfyUI-QwenVL.git 1b67b44
ComfyUI-RMBG https://github.com/1038lab/ComfyUI-RMBG.git 58f1947
ComfyUI-See-through https://github.com/jtydhr88/ComfyUI-See-through.git 98d754b
ComfyUI-SeedVR2_VideoUpscaler https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git 4490bd1
ComfyUI-segment-anything-2 https://github.com/kijai/ComfyUI-segment-anything-2.git 0c35fff
ComfyUI-VideoHelperSuite https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git 4d907be
comfyui-WhiteRabbit https://github.com/Artificial-Sweetener/comfyui-WhiteRabbit.git 4815da4
ComfyUI_Comfyroll_CustomNodes https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git d78b780
comfyui_controlnet_aux https://github.com/Fannovel16/comfyui_controlnet_aux.git 59b1fc4
ComfyUI_essentials https://github.com/cubiq/ComfyUI_essentials.git 9d9f4be
ComfyUI_Fill-Nodes https://github.com/filliptm/ComfyUI_Fill-Nodes.git a32f9b6
ComfyUI_IPAdapter_plus https://github.com/cubiq/ComfyUI_IPAdapter_plus.git a0f451a
ComfyUI_LayerStyle https://github.com/chflame163/ComfyUI_LayerStyle.git a3459a7
ComfyUI_RH_OpenAPI https://github.com/HM-RunningHub/ComfyUI_RH_OpenAPI.git 8f9c858
ComfyUI_UltimateSDUpscale https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git a5547db
LanPaint https://github.com/scraed/LanPaint.git 32cf848
Plush-for-ComfyUI https://github.com/glibsonoran/Plush-for-ComfyUI.git a9e824e
rgthree-comfy https://github.com/rgthree/rgthree-comfy.git 2c5342a
was-ns https://github.com/ltdrdata/was-node-suite-comfyui.git 44de705
NODES
```

命令按当前提交检出，处于 detached HEAD 状态；后续更新需明确选择分支或版本，并验证与核心的兼容性。

### PerfectPixel 集成

`perfectPixel` 的 ComfyUI 包位于上游仓库子目录。首次克隆后，在仓库根目录创建以下链接；以后只需在
`custom_nodes/.disabled/sources/perfectPixel` 中执行 `git pull`，节点代码会随上游一起更新：

```bash
mkdir -p custom_nodes/.disabled/sources
[ -d custom_nodes/.disabled/sources/perfectPixel ] || git clone https://github.com/theamusing/perfectPixel.git custom_nodes/.disabled/sources/perfectPixel
mkdir -p custom_nodes/PerfectPixelComfy
cp custom_node_overlays/PerfectPixelComfy/__init__.py custom_nodes/PerfectPixelComfy/__init__.py
ln -sfn ../.disabled/sources/perfectPixel/integrations/comfyui/PerfectPixelComfy/nodes_perfect_pixel.py custom_nodes/PerfectPixelComfy/nodes_perfect_pixel.py
ln -sfn ../.disabled/sources/perfectPixel/src/perfect_pixel/perfect_pixel.py custom_nodes/PerfectPixelComfy/perfect_pixel.py
ln -sfn ../.disabled/sources/perfectPixel/src/perfect_pixel/perfect_pixel_noCV2.py custom_nodes/PerfectPixelComfy/perfect_pixel_noCV2.py
```

### 本地扩展与适配层

以下目录不包含在批量克隆命令中；本地代码应单独备份：

| 扩展 | 说明 |
| --- | --- |
| `ComfyUI-SOS-RigTools` | SOS 本地 Rig 工具节点 |
| `ComfyUI-anima-pose-control` | Anima 姿态控制，本地版本，未确认独立来源 |
| `anima_control_lora` | Anima Control LoRA，本地版本，未确认独立来源 |
| `PerfectPixelComfy` | 链接 `.disabled/sources/perfectPixel` 上游仓库内置的 ComfyUI 集成与算法源码，保留直接 `git pull` 更新能力 |

## 模型

以下路径均相对于 `models/`。保留原有下载来源，未重新验证远程链接。`已存在` 仅表示该路径可读取且文件非空（包含有效符号链接），不代表校验和或推理验证通过；`未找到` 表示该精确路径不存在，不排除其他目录或版本已有同类模型。

### Z-Image

| 本地状态 | 模型 | 保存路径 | 来源 |
| --- | --- | --- | --- |
| 未找到 | `z_image_turbo_bf16.safetensors` | `diffusion_models/z_image_turbo_bf16.safetensors` | [Download](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors) |
| 未找到 | `qwen_3_4b.safetensors` | `text_encoders/qwen_3_4b.safetensors` | [Download](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors) |
| 未找到 | `ae.safetensors` | `vae/ae.safetensors` | [Download](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors) |
| 未找到 | `Z-Image-Turbo-Fun-Controlnet-Union.safetensors` | `model_patches/Z-Image-Turbo-Fun-Controlnet-Union.safetensors` | [Download](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union.safetensors) |

### FLUX.2

| 本地状态 | 模型 | 保存路径 | 来源 |
| --- | --- | --- | --- |
| 未找到 | `flux-2-klein-base-9b-fp8.safetensors` | `diffusion_models/flux-2-klein-base-9b-fp8.safetensors` | [Download](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8/resolve/main/flux-2-klein-base-9b-fp8.safetensors) |
| 未找到 | `qwen_3_8b_fp8mixed.safetensors` | `text_encoders/qwen_3_8b_fp8mixed.safetensors` | [Download](https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors) |
| 未找到 | `flux2-vae.safetensors` | `vae/flux2-vae.safetensors` | [Download](https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors) |
| 未找到 | `full_encoder_small_decoder.safetensors` | `vae/full_encoder_small_decoder.safetensors` | [Download](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder/resolve/main/full_encoder_small_decoder.safetensors) |

### Wan

| 本地状态 | 模型 | 保存路径 | 来源 |
| --- | --- | --- | --- |
| 未找到 | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` | [Download](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors) |
| 已存在 | `clip_vision_h.safetensors` | `clip_vision/clip_vision_h.safetensors` | [Download](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors) |
| 未找到 | `wan_2.1_vae.safetensors` | `vae/wan_2.1_vae.safetensors` | [Download](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors) |
| 未找到 | `WanAnimate_relight_lora_fp16.safetensors` | `loras/WanAnimate_relight_lora_fp16.safetensors` | [Download](https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors) |
| 未找到 | `sam2_hiera_base_plus.safetensors` | `sam2/sam2_hiera_base_plus.safetensors` | [Download](https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2_hiera_base_plus.safetensors) |

### Anima

| 本地状态 | 模型 | 保存路径 | 来源 |
| --- | --- | --- | --- |
| 已存在 | `waiANIMA_v10Base10.safetensors` | `diffusion_models/waiANIMA_v10Base10.safetensors` | 待补充链接 |
| 未找到 | `waiANIMA_v10Base10_txt.safetensors` | `text_encoders/waiANIMA_v10Base10_txt.safetensors` | 待补充链接 |
| 已存在 | `qwen_image_vae.safetensors` | `vae/qwen_image_vae.safetensors` | [Download](https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors) |

### Illustrious

| 本地状态 | 模型 | 保存路径 | 来源 |
| --- | --- | --- | --- |
| 已存在 | `illustriousXL_v01.safetensors` | `checkpoints/illustriousXL_v01.safetensors` | 待补充链接 |
| 已存在 | `waiIllustriousSDXL_v170.safetensors` | `checkpoints/waiIllustriousSDXL_v170.safetensors` | 待补充链接 |

## 更新记录

### 2026-09-21 更新记录

- 本次仅更新自定义节点，未更新 ComfyUI 主程序；16 个扩展应用了更新，另外 21 个 Git 仓库已与上游一致。
- `ComfyUI-LTXVideo` 更新到兼容提交 `15d09ab`。最新提交 `dfb2786` 依赖当前核心尚未提供的 `LTXVAddLatentGuide`，暂不采用；升级核心后再检查该扩展。
- `sd-ppp-2.0` 从官方 v2.0 安装包更新到 `d965457`，目录现保留 Git 元数据，可直接检查后续更新。
- `ComfyUI-SOS-RigTools`、`ComfyUI-anima-pose-control`、`anima_control_lora` 没有明确的独立 GitHub 来源，保留本地版本。
- 本地补丁和配置已保留。重启后节点数量从 3,220 增至 3,268，原节点无缺失；CPU 图像缩放工作流和 64×64 PNG 输出验证通过。
- 原提交、源码备份、依赖版本及详细报告位于 `/tmp/comfy-nodes-update-20260921/`；这是临时回退资料，清理该目录前应另行保存。
