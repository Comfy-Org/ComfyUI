# 资产清单

Docker Compose 会挂载仓库根目录下的 `custom_nodes/` 和 `models/`，但不会自动下载扩展或模型。以下清单根据当前运行目录整理。

## 自定义节点

### 可独立安装的扩展

| 扩展 | 来源 |
| --- | --- |
| `ComfyUI-Anima-LLLite` | [kohya-ss/ComfyUI-Anima-LLLite](https://github.com/kohya-ss/ComfyUI-Anima-LLLite) |
| `ComfyUI-Autocomplete-Plus` | [newtextdoc1111/ComfyUI-Autocomplete-Plus](https://github.com/newtextdoc1111/ComfyUI-Autocomplete-Plus) |
| `ComfyUI-Custom-Scripts` | [pythongosssss/ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts) |
| `ComfyUI-Florence2` | [kijai/ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2) |
| `ComfyUI-KJNodes` | [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) |
| `ComfyUI-MatAnyone` | [FuouM/ComfyUI-MatAnyone](https://github.com/FuouM/ComfyUI-MatAnyone) |
| `ComfyUI-RMBG` | [1038lab/ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) |
| `ComfyUI-See-through` | [jtydhr88/ComfyUI-See-through](https://github.com/jtydhr88/ComfyUI-See-through) |
| `ComfyUI-SeedVR2_VideoUpscaler` | [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) |
| `ComfyUI-WanVideoWrapper` | [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) |
| `ComfyUI-layerdiffuse` | [huchenlei/ComfyUI-layerdiffuse](https://github.com/huchenlei/ComfyUI-layerdiffuse) |
| `ComfyUI-segment-anything-2` | [kijai/ComfyUI-segment-anything-2](https://github.com/kijai/ComfyUI-segment-anything-2) |
| `ComfyUI_Comfyroll_CustomNodes` | [Suzie1/ComfyUI_Comfyroll_CustomNodes](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes) |
| `ComfyUI_IPAdapter_plus` | [cubiq/ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) |
| `ComfyUI_LayerStyle` | [chflame163/ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle) |
| `ComfyUI_UltimateSDUpscale` | [ssitu/ComfyUI_UltimateSDUpscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale) |
| `ComfyUI_essentials` | [cubiq/ComfyUI_essentials](https://github.com/cubiq/ComfyUI_essentials) |
| `LanPaint` | [scraed/LanPaint](https://github.com/scraed/LanPaint) |
| `Plush-for-ComfyUI` | [glibsonoran/Plush-for-ComfyUI](https://github.com/glibsonoran/Plush-for-ComfyUI) |
| `comfy_mtb` | [melMass/comfy_mtb](https://github.com/melMass/comfy_mtb) |
| `comfyui-frame-interpolation` | [Fannovel16/ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) |
| `comfyui-videohelpersuite` | [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) |
| `comfyui_controlnet_aux` | [Fannovel16/comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) |
| `.disabled/sources/perfectPixel` | [theamusing/perfectPixel](https://github.com/theamusing/perfectPixel)（上游源码，置于禁用区避免被 ComfyUI 重复扫描） |
| `rgthree-comfy` | [rgthree/rgthree-comfy](https://github.com/rgthree/rgthree-comfy) |
| `was-ns` | [ltdrdata/was-node-suite-comfyui](https://github.com/ltdrdata/was-node-suite-comfyui) |

在仓库根目录执行以下命令，可以补齐缺失的外部扩展；已存在的目录会跳过：

```bash
while read -r node_name node_url; do
  [ -d "custom_nodes/$node_name" ] || git clone "$node_url" "custom_nodes/$node_name"
done <<'NODES'
ComfyUI-Anima-LLLite https://github.com/kohya-ss/ComfyUI-Anima-LLLite.git
ComfyUI-Autocomplete-Plus https://github.com/newtextdoc1111/ComfyUI-Autocomplete-Plus.git
ComfyUI-Custom-Scripts https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git
ComfyUI-Florence2 https://github.com/kijai/ComfyUI-Florence2.git
ComfyUI-KJNodes https://github.com/kijai/ComfyUI-KJNodes.git
ComfyUI-MatAnyone https://github.com/FuouM/ComfyUI-MatAnyone.git
ComfyUI-RMBG https://github.com/1038lab/ComfyUI-RMBG.git
ComfyUI-See-through https://github.com/jtydhr88/ComfyUI-See-through.git
ComfyUI-SeedVR2_VideoUpscaler https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git
ComfyUI-WanVideoWrapper https://github.com/kijai/ComfyUI-WanVideoWrapper.git
ComfyUI-layerdiffuse https://github.com/huchenlei/ComfyUI-layerdiffuse.git
ComfyUI-segment-anything-2 https://github.com/kijai/ComfyUI-segment-anything-2.git
ComfyUI_Comfyroll_CustomNodes https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git
ComfyUI_IPAdapter_plus https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
ComfyUI_LayerStyle https://github.com/chflame163/ComfyUI_LayerStyle.git
ComfyUI_UltimateSDUpscale https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git
ComfyUI_essentials https://github.com/cubiq/ComfyUI_essentials.git
LanPaint https://github.com/scraed/LanPaint.git
Plush-for-ComfyUI https://github.com/glibsonoran/Plush-for-ComfyUI.git
comfy_mtb https://github.com/melMass/comfy_mtb.git
comfyui-frame-interpolation https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git
comfyui-videohelpersuite https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
comfyui_controlnet_aux https://github.com/Fannovel16/comfyui_controlnet_aux.git
rgthree-comfy https://github.com/rgthree/rgthree-comfy.git
was-ns https://github.com/ltdrdata/was-node-suite-comfyui.git
NODES
```

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

### SOS 本地扩展

以下目录当前没有可直接克隆的独立扩展仓库，不包含在一键安装命令中：

| 扩展 | 说明 |
| --- | --- |
| `ComfyUI-SOS-RigTools` | SOS 本地 Rig 工具节点 |
| `PerfectPixelComfy` | 链接 `.disabled/sources/perfectPixel` 上游仓库内置的 ComfyUI 集成与算法源码，保留直接 `git pull` 更新能力 |

## 模型

以下路径均相对于 `models/` 目录。当前模型以单独文件分发，不克隆整个 Hugging Face 仓库。

### Z-Image

| 完成 | 模型 | 保存路径 | 来源 |
| --- | --- | --- | --- |
| [ ] | `z_image_turbo_bf16.safetensors` | `diffusion_models/z_image_turbo_bf16.safetensors` | [Download](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors) |
| [ ] | `qwen_3_4b.safetensors` | `text_encoders/qwen_3_4b.safetensors` | [Download](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors) |
| [ ] | `ae.safetensors` | `vae/ae.safetensors` | [Download](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors) |
| [ ] | `Z-Image-Turbo-Fun-Controlnet-Union.safetensors` | `model_patches/Z-Image-Turbo-Fun-Controlnet-Union.safetensors` | [Download](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union.safetensors) |

### FLUX.2

| 完成 | 模型 | 保存路径 | 来源 |
| --- | --- | --- | --- |
| [ ] | `flux-2-klein-base-9b-fp8.safetensors` | `diffusion_models/flux-2-klein-base-9b-fp8.safetensors` | [Download](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8/resolve/main/flux-2-klein-base-9b-fp8.safetensors) |
| [ ] | `qwen_3_8b_fp8mixed.safetensors` | `text_encoders/qwen_3_8b_fp8mixed.safetensors` | [Download](https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors) |
| [ ] | `flux2-vae.safetensors` | `vae/flux2-vae.safetensors` | [Download](https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors) |
| [ ] | `full_encoder_small_decoder.safetensors` | `vae/full_encoder_small_decoder.safetensors` | [Download](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder/resolve/main/full_encoder_small_decoder.safetensors) |

### Wan

| 完成 | 模型 | 保存路径 | 来源 |
| --- | --- | --- | --- |
| [ ] | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` | [Download](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors) |
| [ ] | `clip_vision_h.safetensors` | `clip_vision/clip_vision_h.safetensors` | [Download](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors) |
| [ ] | `wan_2.1_vae.safetensors` | `vae/wan_2.1_vae.safetensors` | [Download](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors) |
| [ ] | `WanAnimate_relight_lora_fp16.safetensors` | `loras/WanAnimate_relight_lora_fp16.safetensors` | [Download](https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors) |
| [ ] | `sam2_hiera_base_plus.safetensors` | `sam2/sam2_hiera_base_plus.safetensors` | [Download](https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2_hiera_base_plus.safetensors) |

### Anima

| 完成 | 模型 | 保存路径 | 来源 |
| --- | --- | --- | --- |
| [ ] | `waiANIMA_v10Base10.safetensors` | `diffusion_models/waiANIMA_v10Base10.safetensors` | 待补充链接 |
| [ ] | `waiANIMA_v10Base10_txt.safetensors` | `text_encoders/waiANIMA_v10Base10_txt.safetensors` | 待补充链接 |
| [ ] | `qwen_image_vae.safetensors` | `vae/qwen_image_vae.safetensors` | [Download](https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors) |

### Illustrious

| 完成 | 模型 | 保存路径 | 来源 |
| --- | --- | --- | --- |
| [ ] | `illustriousXL_v01.safetensors` | `checkpoints/illustriousXL_v01.safetensors` | 待补充链接 |
| [ ] | `waiIllustriousSDXL_v170.safetensors` | `checkpoints/waiIllustriousSDXL_v170.safetensors` | 待补充链接 |
