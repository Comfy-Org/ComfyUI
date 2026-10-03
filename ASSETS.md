# 资产清单

Docker Compose 不会自动下载自定义节点或模型。以下命令均在仓库根目录执行。

## 自定义节点

- [ ] [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux)
- [ ] [ComfyUI-Autocomplete-Plus](https://github.com/newtextdoc1111/ComfyUI-Autocomplete-Plus)
- [ ] [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)
- [ ] [ComfyUI-See-through](https://github.com/jtydhr88/ComfyUI-See-through)
- [ ] [ComfyUI-segment-anything-2](https://github.com/kijai/ComfyUI-segment-anything-2)
- [ ] [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
- [ ] [rgthree-comfy](https://github.com/rgthree/rgthree-comfy)

PowerShell 一键克隆命令（已存在的目录会跳过）：

```powershell
@(
  "https://github.com/Fannovel16/comfyui_controlnet_aux.git",
  "https://github.com/newtextdoc1111/ComfyUI-Autocomplete-Plus.git",
  "https://github.com/kijai/ComfyUI-KJNodes.git",
  "https://github.com/jtydhr88/ComfyUI-See-through.git",
  "https://github.com/kijai/ComfyUI-segment-anything-2.git",
  "https://github.com/kijai/ComfyUI-WanVideoWrapper.git",
  "https://github.com/rgthree/rgthree-comfy.git"
) | ForEach-Object {
  $name = [IO.Path]::GetFileNameWithoutExtension($_)
  $destination = Join-Path "custom_nodes" $name
  if (-not (Test-Path $destination)) {
    git clone $_ $destination
  }
}
```

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
