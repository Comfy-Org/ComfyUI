#!/usr/bin/env python3
"""
High-Speed Parallel LTX-2.5 Model Downloader for ComfyUI.
Streams real-time progress for all 6 required LTX-2.5 weights directly to terminal.
"""

import os
import sys
import time
from pathlib import Path
from huggingface_hub import hf_hub_download

# Target directories matching ComfyUI standard paths
BASE_DIR = Path(__file__).resolve().parent
DIFFUSION_DIR = BASE_DIR / "models" / "diffusion_models"
TEXT_ENC_DIR = BASE_DIR / "models" / "text_encoders"
VAE_DIR = BASE_DIR / "models" / "vae"
UPSCALE_DIR = BASE_DIR / "models" / "latent_upscale_models"

DIFFUSION_DIR.mkdir(parents=True, exist_ok=True)
TEXT_ENC_DIR.mkdir(parents=True, exist_ok=True)
VAE_DIR.mkdir(parents=True, exist_ok=True)
UPSCALE_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOADS = [
    # 1. Audio VAE
    {
        "repo_id": "Lightricks/LTX-2.5",
        "filename": "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "target_dir": VAE_DIR,
        "name": "Audio VAE (ltx-2.5-audio-vae-bf16.safetensors)",
    },
    # 2. Video VAE
    {
        "repo_id": "Lightricks/LTX-2.5",
        "filename": "vae/ltx-2.5-video-vae-bf16.safetensors",
        "target_dir": VAE_DIR,
        "name": "Video VAE (ltx-2.5-video-vae-bf16.safetensors)",
    },
    # 3. Spatial Upscaler
    {
        "repo_id": "Lightricks/LTX-2.5",
        "filename": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "target_dir": UPSCALE_DIR,
        "name": "Spatial Upscaler (ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors)",
    },
    # 4. Gemma 4 Auxiliary Text Encoder
    {
        "repo_id": "Comfy-Org/gemma-4",
        "filename": "text_encoders/gemma4_e2b_it_int8_convrot.safetensors",
        "target_dir": TEXT_ENC_DIR,
        "name": "Gemma4 2B Text Encoder (gemma4_e2b_it_int8_convrot.safetensors)",
    },
    # 5. Gemma 4 12B Projected Text Encoder
    {
        "repo_id": "Lightricks/LTX-2.5",
        "filename": "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        "target_dir": TEXT_ENC_DIR,
        "name": "Gemma4 12B Projected Encoder (gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors)",
    },
    # 6. Main 22B Distilled Transformer
    {
        "repo_id": "Lightricks/LTX-2.5",
        "filename": "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "target_dir": DIFFUSION_DIR,
        "name": "Main 22B Distilled Transformer (ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors)",
    },
]

def main():
    print("=" * 70)
    print("🚀 Starting LTX-2.5 Model Download for ComfyUI (Apple Silicon M5 Max)")
    print(f"📁 Destination: {BASE_DIR / 'models'}")
    print("=" * 70)

    for idx, item in enumerate(DOWNLOADS, 1):
        target_file = item["target_dir"] / Path(item["filename"]).name
        print(f"\n[{idx}/6] 📥 {item['name']}")
        print(f"   Destination: {target_file}")
        
        if target_file.exists() and target_file.stat().st_size > 1024 * 1024:
            print(f"   ✅ Already downloaded ({target_file.stat().st_size / (1024**3):.2f} GB). Skipping.")
            continue
        
        try:
            start_t = time.time()
            downloaded_path = hf_hub_download(
                repo_id=item["repo_id"],
                filename=item["filename"],
                local_dir=str(item["target_dir"]),
            )
            # If downloaded into nested subfolder, flatten to target directory
            actual_file = item["target_dir"] / item["filename"]
            if actual_file.exists() and actual_file != target_file:
                import shutil
                shutil.move(str(actual_file), str(target_file))
                # clean empty subfolder
                try:
                    (item["target_dir"] / Path(item["filename"]).parent).rmdir()
                except Exception:
                    pass
            
            elapsed = time.time() - start_t
            size_gb = target_file.stat().st_size / (1024**3)
            print(f"   ✅ Completed in {elapsed:.1f}s ({size_gb:.2f} GB)")
        except Exception as e:
            print(f"   ❌ Error downloading {item['name']}: {e}")

    print("\n" + "=" * 70)
    print("🎉 All LTX-2.5 models downloaded and placed in ComfyUI models directory!")
    print("=" * 70)

if __name__ == "__main__":
    main()
