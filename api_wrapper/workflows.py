"""Workflow builders for the wrapper API.

This module is deliberately free of comfy imports so it stays unit-testable
and the graph structure is easy to audit. It builds ComfyUI API-format prompt
graphs (as accepted by POST /prompt) and declares the models a workflow needs.
"""

import os

from api_wrapper.prompt_rewriter import MODEL_OPTIONS as MINIMAX_H3_LLM_MODELS

# Models required by the FLUX.2 [klein] 9B image edit workflow, with the exact
# files expected by the loaders in the graph below. The URLs match the
# Comfy-Org/BFL repos the shipped blueprints point at.
FLUX2_KLEIN_9B_MODELS = [
    {
        "folder": "diffusion_models",
        "filename": "flux-2-klein-base-9b-fp8.safetensors",
        "url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8/resolve/main/flux-2-klein-base-9b-fp8.safetensors",
    },
    {
        "folder": "text_encoders",
        "filename": "qwen_3_8b_fp8mixed.safetensors",
        "url": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
    },
    {
        "folder": "vae",
        "filename": "flux2-vae.safetensors",
        "url": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/vae/flux2-vae.safetensors",
    },
]

FLUX2_KLEIN_9B_UNET = "flux-2-klein-base-9b-fp8.safetensors"
FLUX2_KLEIN_9B_CLIP = "qwen_3_8b_fp8mixed.safetensors"
FLUX2_KLEIN_9B_VAE = "flux2-vae.safetensors"

# Models required by the Ideogram 4 text-to-image workflow (fp8 + native nvfp4
# variants). URL layout matches the Comfy-Org repos the shipped "Text to Image
# (Ideogram v4)" blueprint points at.
IDEOGRAM4_UNET = "ideogram4_fp8_scaled.safetensors"
IDEOGRAM4_UNET_UNCONDITIONAL = "ideogram4_unconditional_fp8_scaled.safetensors"
IDEOGRAM4_UNET_FP4 = "ideogram4_nvfp4_mixed.safetensors"
IDEOGRAM4_UNET_UNCONDITIONAL_FP4 = "ideogram4_unconditional_nvfp4_mixed.safetensors"
IDEOGRAM4_CLIP = "qwen3vl_8b_fp8_scaled.safetensors"
IDEOGRAM4_VAE = "flux2-vae.safetensors"

IDEOGRAM4_MODELS = [
    {"folder": "diffusion_models", "filename": IDEOGRAM4_UNET,
     "url": "https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/diffusion_models/ideogram4_fp8_scaled.safetensors"},
    {"folder": "diffusion_models", "filename": IDEOGRAM4_UNET_UNCONDITIONAL,
     "url": "https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors"},
    {"folder": "text_encoders", "filename": IDEOGRAM4_CLIP,
     "url": "https://huggingface.co/Comfy-Org/Qwen3-VL/resolve/main/text_encoders/qwen3vl_8b_fp8_scaled.safetensors"},
    {"folder": "vae", "filename": IDEOGRAM4_VAE,
     "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors"},
]

IDEOGRAM4_FP4_MODELS = [
    {"folder": "diffusion_models", "filename": IDEOGRAM4_UNET_FP4,
     "url": "https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/diffusion_models/ideogram4_nvfp4_mixed.safetensors"},
    {"folder": "diffusion_models", "filename": IDEOGRAM4_UNET_UNCONDITIONAL_FP4,
     "url": "https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/diffusion_models/ideogram4_unconditional_nvfp4_mixed.safetensors"},
    {"folder": "text_encoders", "filename": IDEOGRAM4_CLIP,
     "url": "https://huggingface.co/Comfy-Org/Qwen3-VL/resolve/main/text_encoders/qwen3vl_8b_fp8_scaled.safetensors"},
    {"folder": "vae", "filename": IDEOGRAM4_VAE,
     "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors"},
]

# Scheduler presets from the blueprint's preset JSON; `mode` selects one and
# the optional `steps` form field overrides the preset step count.
IDEOGRAM4_PRESETS = {
    "quality": {"steps": 48, "mu": 0.0, "std": 1.5},
    "default": {"steps": 20, "mu": 0.0, "std": 1.75},
    "turbo": {"steps": 12, "mu": 0.5, "std": 1.75},
}

# Example prompt for the docs: Ideogram 4 accepts rich structured JSON prompts
# that describe composition, style and elements with bounding boxes.
IDEOGRAM4_EXAMPLE_PROMPT = """{
  "high_level_description": "A surreal streetwear mixed-media collage poster featuring a relaxed skateboarder mid-air against a vibrant blue sky, backed by giant puffy 3D letters spelling 'COMFY'. The composition blends retro magazine cutout aesthetics with grunge elements like torn paper banners and distressed red stamps, conveying an effortless, cozy vibe.",
  "style_description": {
    "aesthetics": "Retro magazine cutout style, mixed-media digital collage, high-contrast streetwear graphic, featuring rough ripped paper edges and distressed grunge textures.",
    "lighting": "High-contrast flash mixed with harsh midday sunlight on the skater cutout, contrasting with flat, bright graphic lighting on the 3D typography.",
    "photo": "Vintage grainy 35mm film with distressed halftone scan textures and subtle light leaks.",
    "medium": "Mixed-media digital collage",
    "color_palette": ["#1E73BE", "#FDFDFD", "#C82A2A", "#657C9C", "#EFEFEF"]
  },
  "compositional_deconstruction": {
    "background": "A vibrant, clear blue sky layered with a vintage grainy film texture and subtle halftone dot patterns, transitioning down to an implied pale gray concrete ramp at the very bottom edge.",
    "elements": [
      {"type": "obj", "bbox": [128, 149, 354, 810], "desc": "Massive 3D puffy, inflatable white typography spelling 'COMFY' stretching across the upper half of the canvas, acting as a surreal, soft cloud-like backdrop.", "color_palette": ["#FDFDFD", "#E0E0E0", "#D3DBE2"]},
      {"type": "obj", "bbox": [459, 37, 727, 264], "desc": "A cluster of oversized, distressed red stamped circles and dots, applied loosely to the midground like a grunge ink stamp, partially obscuring the bottom left of the text.", "color_palette": ["#C82A2A", "#A11D1D"]},
      {"type": "obj", "bbox": [23, 366, 153, 666], "desc": "A vertically oriented, torn paper side banner pinned to the left edge, displaying the bold stamped text 'STAY COZY' in high-contrast black ink.", "color_palette": ["#EFEFEF", "#1A1A1A", "#C82A2A"]},
      {"type": "obj", "bbox": [287, 210, 756, 819], "desc": "A sharp photographic cutout of a skateboarder mid-air in a relaxed pose, wearing loose-fitting washed denim jeans and a plain white tee, floating effortlessly above the concrete ramp with a distinct white cutout border.", "color_palette": ["#FDFDFD", "#657C9C", "#2B2B2B", "#DCA57D"]},
      {"type": "obj", "bbox": [773, 39, 973, 187], "desc": "A surreal, miniature floating skateboard cutout, positioned playfully in the upper right sky as if defying gravity.", "color_palette": ["#D2A679", "#2B2B2B", "#C82A2A"]},
      {"type": "obj", "bbox": [105, 830, 905, 980], "desc": "A wide, horizontal strip of heavily textured torn paper spanning the lower third of the composition, featuring the bold typographic phrase 'BEYOND THE COMFORT ZONE' intermixed with 'EFFORTLESS RIDE' alongside ripped edges that reveal the background.", "color_palette": ["#EFEFEF", "#1A1A1A", "#999999"]}
    ]
  }
}"""


def build_flux2_klein_9b_img2img(*, prompt, image=None, ref_images=(), negative_prompt="",
                                 seed=0, steps=20, cfg=5.0, megapixels=1.0,
                                 width=None, height=None,
                                 filename_prefix="wrapper/flux2_klein_9b",
                                 unet_name=FLUX2_KLEIN_9B_UNET):
    """Build the FLUX.2 [klein] 9B image edit graph (API format).

    Mirrors the shipped "Image Edit (Flux.2 Klein)" blueprint: the input image
    is scaled to a fixed megapixel budget, its latent is attached to both the
    positive and negative conditioning as a reference latent, and sampling
    runs through the flux2 custom sampler stack (CFG guider, euler,
    flux2 scheduler, empty flux2 latent at the image size).

    Several references (``image`` first, then ``ref_images``) chain one
    reference latent each — FLUX.2's multi-reference edit — so a character
    portrait and a location can shape one still. ``width``/``height`` set the
    output canvas (a video's frame); without them it takes the first
    reference's scaled size, as the blueprint does.

    File names are relative to the ComfyUI input directory. ``unet_name``
    selects the diffusion model file (fp8 by default; pass the converted
    nvfp4 file name for the fp4 path).
    """
    refs = ([image] if image else []) + list(_ref_list(ref_images))
    if not refs:
        raise ValueError("the image edit needs at least one image")
    canvas_w = ["6", 0] if width is None else width
    canvas_h = ["6", 1] if height is None else height
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": FLUX2_KLEIN_9B_CLIP, "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX2_KLEIN_9B_VAE}},
        "4": {"class_type": "LoadImage", "inputs": {"image": refs[0]}},
        # lanczos, not nearest: the budget usually upscales a still, and a
        # blocky reference is a blurred face in everything made from it.
        "5": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["4", 0], "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 1}},
        "6": {"class_type": "GetImageSize", "inputs": {"image": ["5", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": negative_prompt}},
        "9": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["3", 0]}},
        "10": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["7", 0], "latent": ["9", 0]}},
        "11": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["8", 0], "latent": ["9", 0]}},
    }
    positive, negative = ["10", 0], ["11", 0]
    next_id = 20
    for ref in refs[1:]:
        load, scale, encode, pos, neg = (str(next_id + i) for i in range(5))
        graph[load] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        graph[scale] = {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": [load, 0], "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 1}}
        graph[encode] = {"class_type": "VAEEncode", "inputs": {"pixels": [scale, 0], "vae": ["3", 0]}}
        graph[pos] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": positive, "latent": [encode, 0]}}
        graph[neg] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": negative, "latent": [encode, 0]}}
        positive, negative = [pos, 0], [neg, 0]
        next_id += 5
    graph.update({
        "12": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": positive, "negative": negative, "cfg": cfg}},
        "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "14": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": canvas_w, "height": canvas_h}},
        "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "16": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": canvas_w, "height": canvas_h, "batch_size": 1}},
        "17": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["15", 0], "guider": ["12", 0], "sampler": ["13", 0], "sigmas": ["14", 0], "latent_image": ["16", 0]}},
        "18": {"class_type": "VAEDecode", "inputs": {"samples": ["17", 0], "vae": ["3", 0]}},
        "19": {"class_type": "SaveImage", "inputs": {"images": ["18", 0], "filename_prefix": filename_prefix}},
    })
    return graph


def build_ideogram4_text2img(*, prompt, seed=0, steps=20, mu=0.0, std=1.75,
                             width=1024, height=1024,
                             filename_prefix="wrapper/ideogram4",
                             unet_name=IDEOGRAM4_UNET,
                             unconditional_unet_name=IDEOGRAM4_UNET_UNCONDITIONAL):
    """Build the Ideogram 4 text-to-image graph (API format).

    Mirrors the shipped "Text to Image (Ideogram v4)" blueprint: the positive
    conditioning is zeroed out as the negative, the main UNET runs through a
    CFG override and a dual-model guider (the unconditional UNET powers the
    negative pass), and sampling uses the ideogram4 scheduler (euler, custom
    sampler stack). Width/height are rounded up to multiples of 16 (min 256)
    exactly like the blueprint's math expressions.

    ``prompt`` is passed to the text encoder as-is: Ideogram 4 natively
    understands rich structured JSON prompts (see IDEOGRAM4_EXAMPLE_PROMPT).
    """
    width = max(((width + 15) // 16) * 16, 256)
    height = max(((height + 15) // 16) * 16, 256)
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": unconditional_unet_name, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": IDEOGRAM4_CLIP, "type": "ideogram4", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": IDEOGRAM4_VAE}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {"class_type": "CFGOverride", "inputs": {"model": ["1", 0], "cfg": 3.0, "start_percent": 0.7, "end_percent": 1.0}},
        "8": {"class_type": "DualModelGuider", "inputs": {"model": ["7", 0], "model_negative": ["2", 0], "positive": ["5", 0], "cfg": 7.0, "negative": ["6", 0]}},
        "9": {"class_type": "Ideogram4Scheduler", "inputs": {"steps": steps, "width": width, "height": height, "mu": mu, "std": std}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "12": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["8", 0], "sampler": ["11", 0], "sigmas": ["9", 0], "latent_image": ["12", 0]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["4", 0]}},
        "15": {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": filename_prefix}},
    }


def build_flux2_klein_9b_text2img(*, prompt, negative_prompt="", seed=0,
                                  steps=20, cfg=5.0, width=1024, height=1024,
                                  filename_prefix="wrapper/flux2klein9b_txt2img",
                                  unet_name=FLUX2_KLEIN_9B_UNET):
    """Build the FLUX.2 [klein] 9B text-to-image graph (API format).

    Same model stack and sampler settings as the image-edit variant (klein
    uses CFG with a real negative prompt), with an empty latent at the
    requested width/height instead of the reference-image path.
    """
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": FLUX2_KLEIN_9B_CLIP, "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX2_KLEIN_9B_VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": negative_prompt}},
        "6": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "cfg": cfg}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "8": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": width, "height": height}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "10": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["9", 0], "guider": ["6", 0], "sampler": ["7", 0], "sigmas": ["8", 0], "latent_image": ["10", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": filename_prefix}},
    }


# MiniMax H3 omni-modal video model files (Comfy-Org/MiniMax-H3). The fl2va
# family powers text-to-video and image-to-video; ref2va is the
# reference-to-video variant with its own weights. int8_convrot is the
# canonical template choice; fp8_scaled loads everywhere (bf16 fallback on
# MPS); bf16 is the full-quality option.
MINIMAX_H3_FPS = 24
# The official ComfyUI templates (video_minimax_h3_{t2v,i2v,r2v}) all ship the
# int8_convrot UNET + nvfp4_awq encoder + fp16 video VAE + fp32 audio VAE, so
# that quadruple is what the wrapper reproduces by default. int8 and fp8 UNETs
# are the same size on disk (~21 GB); int8_convrot is the one the templates and
# the model card were validated against.
MINIMAX_H3_DEFAULT_QUANT = "int8"
MINIMAX_H3_UNET_FP8 = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
MINIMAX_H3_UNET_INT8 = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
MINIMAX_H3_UNET_BF16 = "minimax_h3_fl2va_pruned_bf16.safetensors"
MINIMAX_H3_REF2VA_UNET_FP8 = "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"
MINIMAX_H3_REF2VA_UNET_INT8 = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
MINIMAX_H3_REF2VA_UNET_BF16 = "minimax_h3_ref2va_pruned_bf16.safetensors"
MINIMAX_H3_CLIP = "qwen3vl_32b_minimax_h3_bf16.safetensors"
MINIMAX_H3_CLIP_INT8 = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
MINIMAX_H3_CLIP_NVFP4 = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
MINIMAX_H3_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
MINIMAX_H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
MINIMAX_H3_BASE_URL = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"
MINIMAX_H3_QUANT_MODELS = {
    "fp8": {"unet": MINIMAX_H3_UNET_FP8, "ref2va": MINIMAX_H3_REF2VA_UNET_FP8, "clip": MINIMAX_H3_CLIP},
    "int8": {"unet": MINIMAX_H3_UNET_INT8, "ref2va": MINIMAX_H3_REF2VA_UNET_INT8, "clip": MINIMAX_H3_CLIP_INT8},
    "bf16": {"unet": MINIMAX_H3_UNET_BF16, "ref2va": MINIMAX_H3_REF2VA_UNET_BF16, "clip": MINIMAX_H3_CLIP},
    # No nvfp4 UNET variant exists; nvfp4 is a CLIP-only option. On Blackwell
    # (sm_120) the CLIP runs natively as fp4 — 16 GB vs 34 GB (int8) / 66 GB
    # (bf16) — which is what lets a 32 GB GPU fit the whole pipeline.
    "nvfp4": {"unet": MINIMAX_H3_UNET_FP8, "ref2va": MINIMAX_H3_REF2VA_UNET_FP8, "clip": MINIMAX_H3_CLIP_NVFP4},
}
MINIMAX_H3_MODEL_NAMES = (
    MINIMAX_H3_UNET_FP8, MINIMAX_H3_UNET_INT8, MINIMAX_H3_UNET_BF16,
    MINIMAX_H3_REF2VA_UNET_FP8, MINIMAX_H3_REF2VA_UNET_INT8, MINIMAX_H3_REF2VA_UNET_BF16,
    MINIMAX_H3_CLIP, MINIMAX_H3_CLIP_INT8, MINIMAX_H3_CLIP_NVFP4,
    MINIMAX_H3_VIDEO_VAE, MINIMAX_H3_AUDIO_VAE,
)


MINIMAX_H3_QUANT_ORDER = ("nvfp4", "int8", "fp8", "bf16")
# Smallest CLIP first; bf16 (66 GB) is only ever used when it is the only
# encoder present. nvfp4 (16 GB) is the canonical template default.
MINIMAX_H3_CLIP_ORDER = (MINIMAX_H3_CLIP_NVFP4, MINIMAX_H3_CLIP_INT8, MINIMAX_H3_CLIP)


def minimax_h3_model_pick(requested, ref2va, is_present):
    """Pick (unet_name, clip_name, note) for a MiniMax H3 job.

    The CLIP is chosen independently of the UNET quantization: the smallest
    encoder already on disk wins (nvfp4 16 GB -> int8 34 GB -> bf16 66 GB),
    so a previously-downloaded bf16 encoder is never preferred over a smaller
    one. The UNET follows the requested quantization when that file exists,
    otherwise the first present variant (nvfp4/int8/fp8 UNETs are ~20 GB;
    bf16 is ~40 GB). ``is_present(folder, filename)`` reports local files."""
    if requested is not None and requested not in MINIMAX_H3_QUANT_MODELS:
        raise ValueError(f"Invalid quantization: {requested}")
    q = requested or MINIMAX_H3_DEFAULT_QUANT
    quants = MINIMAX_H3_QUANT_MODELS
    unet_key = "ref2va" if ref2va else "unet"

    notes = []
    unet_order = [q] + [x for x in MINIMAX_H3_QUANT_ORDER if x != q]
    unet_quant, unet = None, None
    for qq in unet_order:
        cand = quants[qq][unet_key]
        if is_present("diffusion_models", cand):
            unet_quant, unet = qq, cand
            break
    if unet is None:
        unet_quant, unet = q, quants[q][unet_key]
    if unet_quant != q:
        notes.append(f"the quantization={q} UNET is not on disk; using {unet} already present.")

    clip = None
    for cand in MINIMAX_H3_CLIP_ORDER:
        if is_present("text_encoders", cand):
            clip = cand
            break
    if clip is None:
        clip = MINIMAX_H3_CLIP_NVFP4
    if clip != MINIMAX_H3_CLIP_ORDER[0]:
        # Only worth reporting when the preferred (smallest) encoder is absent
        # and a heavier one had to stand in -- that is the case that costs RAM.
        notes.append(f"{MINIMAX_H3_CLIP_ORDER[0]} is not on disk; using the larger {clip} "
                     "text encoder already present.")

    return unet, clip, " ".join(notes) or None


def minimax_h3_models(quantization, ref2va=False, unet_name=None, clip_name=None):
    """The 4 model files a MiniMax H3 task needs for the given quantization.

    ``unet_name``/``clip_name`` override the quantization defaults when the
    wrapper picked a different (already-downloaded) variant."""
    q = MINIMAX_H3_QUANT_MODELS[quantization]
    unet = unet_name or (q["ref2va"] if ref2va else q["unet"])
    clip = clip_name or q["clip"]
    return [
        {"folder": "diffusion_models", "filename": unet,
         "url": f"{MINIMAX_H3_BASE_URL}/diffusion_models/{unet}"},
        {"folder": "text_encoders", "filename": clip,
         "url": f"{MINIMAX_H3_BASE_URL}/text_encoders/{clip}"},
        {"folder": "vae", "filename": MINIMAX_H3_VIDEO_VAE,
         "url": f"{MINIMAX_H3_BASE_URL}/vae/{MINIMAX_H3_VIDEO_VAE}"},
        {"folder": "vae", "filename": MINIMAX_H3_AUDIO_VAE,
         "url": f"{MINIMAX_H3_BASE_URL}/vae/{MINIMAX_H3_AUDIO_VAE}"},
    ]


def minimax_h3_length(duration):
    """Duration (seconds) -> frame count at 24 fps (min 5; the nodes snap it
    to the model's 17k+5 grid)."""
    return max(5, round(duration * MINIMAX_H3_FPS))


def minimax_h3_align_frames(length):
    """Frame count the H3 nodes will actually generate for ``length``.

    Mirrors ``comfy_extras.nodes_minimax_h3.align_frame_count`` (kept local so
    this module stays comfy-import-free): the model works in 17-frame blocks
    plus a 5-frame head, so any length snaps up to the next 17k+5 value."""
    n = max(5, length)
    while n % 17 != 5:
        n += 1
    return n


# H3's native canvas: a 768 px short edge capped at 768*1344 pixels, rounded to
# a multiple of 32. Beyond that the model is out of distribution, so the wrapper
# scales an oversized request down instead of generating something unusable.
MINIMAX_H3_CANVAS_MULTIPLE = 32
MINIMAX_H3_MAX_PIXELS = 1344 * 768

# H3's four checkpoints total ~43 GB (21 GB UNET + 16 GB text encoder + 5.8 GB
# VAEs), so on a 32 GB RAM box every job already runs at 29-31 GB of host RAM
# before a single pixel is generated. What is left decides whether the run
# finishes or the kernel OOM-kills the server (exit 137), which takes the whole
# queue down with it -- so the wrapper refuses jobs it does not expect to
# survive rather than discovering the limit the hard way.
#
# Measured on the target box (RTX 5090, 32 GB VRAM / 32 GB RAM, no swap, no
# --fast-disk) with the int8 UNET + nvfp4 encoder:
#
#     canvas    frames  steps  Mpx-frames  peak RAM  result
#      864x480     124     20        51.4    30.4 GB  ok, 91 s
#      736x736     124      2        67.2    29.7 GB  ok, 33 s
#     1152x640     124     20        91.4    31.4 GB  ok, 175 s
#     1344x768     124      2       128.0    30.4 GB  ok, 57 s
#      960x544     243      2       126.9    30.6 GB  OOM-killed
#     1344x768     124     20       128.0    29.9 GB  OOM-killed after 298 s
#     1344x768     158      2       163.1    30.4 GB  OOM-killed
#     1344x768     192      2       198.3    31.1 GB  OOM-killed
#     1344x768     243      2       250.9    30.9 GB  OOM-killed
#
# Two independent limits fall out of that. Frame count is the harder one: every
# run above 124 frames died regardless of how small the canvas was, because the
# video VAE tiles spatially (256 px) but not temporally. Canvas area matters
# too, but only at a realistic step count -- 1344x768 x 124 survives 2 steps and
# dies at 20, because the longer a job sits at ~30 GB the more likely it is to
# lose the race. So both a frame cap and an area*frames budget are enforced.
#
# The budget is the largest combination that completed (1152x640 x 124), while
# the form defaults below sit at the largest one that completed with real
# margin (864x480 x 124, 1.6 GB of RAM to spare -- 1152x640 finished with only
# 0.6 GB). Both limits are env-tunable: a host with more RAM, with swap, or
# running with --fast-disk (see docker-compose.yml) can raise them without a
# code change.
MINIMAX_H3_MAX_FRAMES = max(5, int(os.environ.get("COMFY_MINIMAX_H3_MAX_FRAMES", "124")))
MINIMAX_H3_MAX_DURATION = round(MINIMAX_H3_MAX_FRAMES / MINIMAX_H3_FPS, 2)
MINIMAX_H3_MAX_PIXEL_FRAMES = max(1, int(float(
    os.environ.get("COMFY_MINIMAX_H3_MAX_PIXEL_FRAMES", str(1152 * 640 * 124)))))
# Defaults: the largest canvas/length that finished a real 20-step job with RAM
# to spare, so a request carrying nothing but a prompt gets the best output the
# host can reliably finish. 864x480 is also the resolution the official H3
# templates' own Resolution Selector ships (0.4 MP, 16:9).
MINIMAX_H3_DEFAULT_WIDTH = 864
MINIMAX_H3_DEFAULT_HEIGHT = 480
MINIMAX_H3_DEFAULT_STEPS = 20


def minimax_h3_check_budget(width, height, frames):
    """Return an error detail when a canvas/length would risk OOM-killing the
    host, or None when the job fits the configured budget."""
    if frames > MINIMAX_H3_MAX_FRAMES:
        return (f"{frames} frames at 24 fps is above this host's {MINIMAX_H3_MAX_FRAMES}-frame limit "
                f"(max duration {MINIMAX_H3_MAX_DURATION}s). Frame count is the hardest limit for "
                "MiniMax H3: the video VAE tiles spatially but not temporally, so a longer clip "
                "costs far more memory than a wider one, and no resolution is small enough to "
                "compensate.")
    budget = MINIMAX_H3_MAX_PIXEL_FRAMES
    used = width * height * frames
    if used > budget:
        max_pixels = budget // frames
        # Largest 32-aligned canvas with this aspect ratio that fits the budget.
        m = MINIMAX_H3_CANVAS_MULTIPLE
        scale = (max_pixels / (width * height)) ** 0.5
        fit_w = max(m, int(width * scale) // m * m)
        fit_h = max(m, int(height * scale) // m * m)
        return (f"{width}x{height} for {frames} frames needs {used / 1e6:.0f} megapixel-frames, "
                f"above this host's {budget / 1e6:.0f} budget. Either drop the canvas to about "
                f"{fit_w}x{fit_h} at this length, or shorten the clip.")
    return None


def minimax_h3_fit_canvas(width, height):
    """Round a requested canvas up to H3's grid, scaling it down (aspect kept)
    when it exceeds the model's 768*1344 pixel budget."""
    m = MINIMAX_H3_CANVAS_MULTIPLE
    width = max(m, -(-width // m) * m)
    height = max(m, -(-height // m) * m)
    if width * height <= MINIMAX_H3_MAX_PIXELS:
        return width, height, None
    # Round the fitted canvas *down* to the grid: rounding up here could push
    # the area back over the cap the scale factor just brought it under.
    scale = (MINIMAX_H3_MAX_PIXELS / (width * height)) ** 0.5
    fitted_w = max(m, int(width * scale) // m * m)
    fitted_h = max(m, int(height * scale) // m * m)
    note = (f"{width}x{height} exceeds MiniMax H3's {MINIMAX_H3_MAX_PIXELS}-pixel native canvas; "
            f"generating at {fitted_w}x{fitted_h} instead.")
    return fitted_w, fitted_h, note


def _minimax_h3_head(prompt, width, height, length, unet_name, clip_name,
                     video_vae_name, audio_vae_name, cond_inputs):
    """Shared model/conditioning half of every MiniMax H3 graph."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "2": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["1", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": video_vae_name}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": audio_vae_name}},
        "6": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["3", 0], "vae": ["4", 0], "prompt": prompt,
            "width": width, "height": height, "length": length, **cond_inputs}},
    }


def _minimax_h3_tail(model_ref, cond_ref, latent_ref, seed, steps, scheduler, start, filename_prefix):
    """Sampling + joint decode + mux + save half of every MiniMax H3 graph."""
    nodes = {
        str(start): {"class_type": "BasicGuider", "inputs": {"model": model_ref, "conditioning": cond_ref}},
        str(start + 1): {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        str(start + 2): {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        str(start + 3): {"class_type": "BasicScheduler", "inputs": {"model": model_ref, "scheduler": scheduler, "steps": steps, "denoise": 1.0}},
        str(start + 4): {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": [str(start + 1), 0], "guider": [str(start), 0], "sampler": [str(start + 2), 0],
            "sigmas": [str(start + 3), 0], "latent_image": latent_ref}},
        str(start + 5): {"class_type": "VAEDecode", "inputs": {"samples": [str(start + 4), 0], "vae": ["4", 0]}},
        str(start + 6): {"class_type": "VAEDecodeAudio", "inputs": {"samples": [str(start + 4), 0], "vae": ["5", 0]}},
        str(start + 7): {"class_type": "CreateVideo", "inputs": {
            "images": [str(start + 5), 0], "fps": float(MINIMAX_H3_FPS),
            "audio": [str(start + 6), 0], "bit_depth": 8}},
        str(start + 8): {"class_type": "SaveVideo", "inputs": {
            "video": [str(start + 7), 0], "filename_prefix": filename_prefix, "format": "auto", "codec": "auto"}},
    }
    return nodes


def _minimax_h3_cond_and_tail(prompt, width, height, length, seed, steps, scheduler,
                              unet_name, clip_name, cond_inputs, loader_nodes, next_id,
                              filename_prefix):
    """Head (models + conditioning) + loader nodes + sampling/decode tail."""
    head = _minimax_h3_head(prompt, width, height, length, unet_name, clip_name,
                            MINIMAX_H3_VIDEO_VAE, MINIMAX_H3_AUDIO_VAE, cond_inputs)
    tail = _minimax_h3_tail(["2", 0], ["6", 0], ["6", 1], seed, steps, scheduler,
                            next_id, filename_prefix)
    return {**head, **loader_nodes, **tail}


def build_minimax_h3_text_to_video(*, prompt, seed=0, steps=MINIMAX_H3_DEFAULT_STEPS,
                                   width=MINIMAX_H3_DEFAULT_WIDTH, height=MINIMAX_H3_DEFAULT_HEIGHT,
                                   duration=5.0, scheduler="beta",
                                   filename_prefix="wrapper/minimaxh3_t2v",
                                   unet_name=MINIMAX_H3_UNET_INT8,
                                   clip_name=MINIMAX_H3_CLIP_NVFP4):
    """MiniMax H3 text-to-video: prompt -> joint audio+video MP4."""
    return _minimax_h3_cond_and_tail(prompt, width, height, minimax_h3_length(duration),
                                     seed, steps, scheduler, unet_name, clip_name, {}, {}, 7,
                                     filename_prefix)


def build_minimax_h3_image_to_video(*, prompt, seed=0, steps=MINIMAX_H3_DEFAULT_STEPS,
                                    width=MINIMAX_H3_DEFAULT_WIDTH, height=MINIMAX_H3_DEFAULT_HEIGHT,
                                    duration=5.0, scheduler="beta",
                                    filename_prefix="wrapper/minimaxh3_i2v",
                                    first_frame=None, last_frame=None,
                                    unet_name=MINIMAX_H3_UNET_INT8,
                                    clip_name=MINIMAX_H3_CLIP_NVFP4):
    """MiniMax H3 image-to-video: first (and optional last) frame + prompt ->
    joint audio+video MP4."""
    cond_inputs = {}
    loader_nodes = {}
    next_id = 7
    loaded = []
    for input_name, ref in (("first_frame", first_frame), ("last_frame", last_frame)):
        if ref is None:
            continue
        loader_nodes[str(next_id)] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        loaded.append((input_name, str(next_id)))
        next_id += 1
    for input_name, loader in loaded:
        # Cover-crop to the canvas first. The node stretches a first frame to
        # fit (and cover-crops a last one), so an off-aspect still — a 3:4
        # photo in a 9:16 film, a 1:1 edit — squashed every face in the shot,
        # and a pair from one source warped differently at each end.
        loader_nodes[str(next_id)] = {"class_type": "ImageScale", "inputs": {
            "image": [loader, 0], "upscale_method": "lanczos",
            "width": width, "height": height, "crop": "center"}}
        cond_inputs[input_name] = [str(next_id), 0]
        next_id += 1
    return _minimax_h3_cond_and_tail(prompt, width, height, minimax_h3_length(duration),
                                     seed, steps, scheduler, unet_name, clip_name,
                                     cond_inputs, loader_nodes, next_id, filename_prefix)


def _ref_list(refs):
    """Accept a single ref filename (max=1 uploads arrive as a string) or a
    list/tuple of them."""
    if isinstance(refs, str):
        return (refs,)
    return refs or ()


def build_minimax_h3_reference_to_video(*, prompt, seed=0, steps=MINIMAX_H3_DEFAULT_STEPS,
                                        width=MINIMAX_H3_DEFAULT_WIDTH, height=MINIMAX_H3_DEFAULT_HEIGHT,
                                        duration=5.0, scheduler="beta", ref_image_size="match",
                                        filename_prefix="wrapper/minimaxh3_ref2va",
                                        ref_images=(), ref_videos=(), ref_video_audios=(),
                                        ref_audios=(),
                                        unet_name=MINIMAX_H3_REF2VA_UNET_INT8,
                                        clip_name=MINIMAX_H3_CLIP_NVFP4):
    """MiniMax H3 reference-to-video (ref2va): reference images/videos/audio +
    prompt -> joint audio+video MP4. The prompt refers to references by tag
    (<Picture i> / <Video k> / <Audio j>) in the order they were provided.
    Parity with the official ref2va workflow: up to 3 images, 1 reference video
    (with its optional soundtrack), and 2 standalone audio refs."""
    length = minimax_h3_length(duration)
    # The node's reference inputs are Autogrow groups, and ComfyUI addresses an
    # Autogrow slot by its flat dotted id ("ref_images.ref_image_0"). A nested
    # {"ref_images": {"ref_image_0": ...}} is not an input the node declares:
    # execution drops it without a validation error and the node runs with
    # ref_images={} — every reference silently ignored.
    ref_inputs = {}
    loader_nodes = {}
    next_id = 7
    for group, slot, loader, field, refs in (
            ("ref_images", "ref_image", "LoadImage", "image", ref_images),
            ("ref_videos", "ref_video", "LoadVideo", "file", ref_videos),
            ("ref_video_audios", "ref_video_audio", "LoadAudio", "audio", ref_video_audios),
            ("ref_audios", "ref_audio", "LoadAudio", "audio", ref_audios)):
        for i, ref in enumerate(_ref_list(refs)):
            loader_nodes[str(next_id)] = {"class_type": loader, "inputs": {field: ref}}
            if loader == "LoadVideo":
                # The slot takes frames (IMAGE); LoadVideo gives a VIDEO.
                loader_nodes[str(next_id + 1)] = {"class_type": "GetVideoComponents",
                                                  "inputs": {"video": [str(next_id), 0]}}
                next_id += 1
            ref_inputs[f"{group}.{slot}_{i}"] = [str(next_id), 0]
            next_id += 1
    head = _minimax_h3_head(prompt, width, height, length, unet_name, clip_name,
                            MINIMAX_H3_VIDEO_VAE, MINIMAX_H3_AUDIO_VAE, {})
    head["6"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
        "clip": ["3", 0], "vae": ["4", 0], "audio_vae": ["5", 0],
        "prompt": prompt, "width": width, "height": height, "length": length,
        "ref_image_size": ref_image_size, **ref_inputs}}
    tail = _minimax_h3_tail(["2", 0], ["6", 0], ["6", 1], seed, steps, scheduler,
                            next_id, filename_prefix)
    return {**head, **loader_nodes, **tail}

# The H3 prompt format (alignment instruction + timed shots +
# overall_soundscape + non_diegetic_music, or six labelled sections in
# reference mode) is a lot to hand-write, so `prompt` takes plain intent and an
# LLM turns it into the real thing using MiniMax's own authoring guide. See
# api_wrapper/prompt_rewriter.py. `raw_prompt` is the escape hatch for callers
# that already have an H3 prompt.
MINIMAX_H3_PROMPT_FORM_EXTRA = {
    "raw_prompt": {"type": "string",
                   "description": "A ready-made MiniMax H3 prompt, used exactly as given. Supplying "
                                  "this skips the LLM rewrite entirely -- nothing is sent to MiMo and "
                                  "no API key is needed. Use it to reproduce an earlier generation "
                                  "verbatim, or when you have written the H3 structure yourself. "
                                  "If both 'prompt' and 'raw_prompt' are sent, 'raw_prompt' wins."},
    "llm_image": {"type": "string", "format": "binary",
                  "description": "Optional image given to the prompt-writing model as visual context, "
                                 "so the H3 prompt it writes describes what is actually in your "
                                 "footage (subjects, clothing, setting, lighting, framing) instead of "
                                 "guessing from the text. Purely an input to the rewrite -- it is not "
                                 "a keyframe and never reaches H3 itself. When omitted, the task's own "
                                 "first image is used instead (the 'image' first frame, or "
                                 "'ref_images' #1), so image- and reference-driven generations get "
                                 "visual grounding for free. Ignored when 'raw_prompt' is set."},
    "llm_model": {"type": "string", "enum": list(MINIMAX_H3_LLM_MODELS), "default": "auto",
                  "description": "Which Xiaomi MiMo model writes the H3 prompt. Leave it on 'auto' "
                                 "unless you have a reason not to: mimo-v2.5-pro is text-only and the "
                                 "API rejects any request carrying an image, while mimo-v2.5 is the "
                                 "omnimodal build that can see one. 'auto' picks mimo-v2.5 whenever "
                                 "there is a context image and mimo-v2.5-pro otherwise, for its "
                                 "stronger reasoning. Pinning mimo-v2.5-pro together with an uploaded "
                                 "llm_image is refused with a 400; the context image the wrapper adds "
                                 "on its own (your keyframe or first reference) is instead skipped, "
                                 "and the response says so. Requires MIMO_API_KEY on the server."},
}

MINIMAX_H3_FORM_EXTRA = {
    "width": {"type": "integer", "minimum": 32, "maximum": 8192, "default": MINIMAX_H3_DEFAULT_WIDTH,
              "description": "Output width, rounded up to a multiple of 32. Width*height is capped at "
                             f"H3's native canvas ({MINIMAX_H3_MAX_PIXELS} px = 1344x768) and a larger "
                             "request is scaled down keeping its aspect ratio. Separately, "
                             "width*height*frames must fit this host's memory budget "
                             f"({MINIMAX_H3_MAX_PIXEL_FRAMES // 1000000} megapixel-frames, i.e. up to "
                             f"{MINIMAX_H3_MAX_PIXEL_FRAMES // MINIMAX_H3_MAX_FRAMES // 1000} kilopixels "
                             f"at the full {MINIMAX_H3_MAX_FRAMES} frames) or the request is rejected "
                             f"with a 400. The default {MINIMAX_H3_DEFAULT_WIDTH}x"
                             f"{MINIMAX_H3_DEFAULT_HEIGHT} is the largest canvas that finished a "
                             "20-step job on this box with memory to spare."},
    "height": {"type": "integer", "minimum": 32, "maximum": 8192, "default": MINIMAX_H3_DEFAULT_HEIGHT,
               "description": "Output height, rounded up to a multiple of 32. See width for the caps."},
    "duration": {"type": "number", "minimum": 0.2, "maximum": MINIMAX_H3_MAX_DURATION,
                 "default": MINIMAX_H3_MAX_DURATION,
                 "description": f"Video length in seconds at 24 fps, snapped up to the model's 17k+5 "
                                f"frame grid. This is the parameter that drives memory hardest: the "
                                f"video VAE tiles spatially but not temporally, so no resolution is "
                                f"small enough to buy back a longer clip. The maximum is the longest "
                                f"length verified not to OOM-kill this host "
                                f"({MINIMAX_H3_MAX_FRAMES} frames); raise it with the "
                                f"COMFY_MINIMAX_H3_MAX_FRAMES env var on a host with more RAM or swap. "
                                f"The model itself is trained for 124-362 frames (5-15 s)."},
    "steps": {"type": "integer", "minimum": 1, "maximum": 1000,
              "default": MINIMAX_H3_DEFAULT_STEPS,
              "description": "Sampling steps. Barely affects peak memory, but a longer run spends "
                             "longer at that peak, so very high step counts do make an OOM more "
                             "likely. 20 is what the official MiniMax H3 templates ship and is a "
                             "good quality/time balance; below ~15 motion and audio get mushy, "
                             "above ~30 returns diminish."},
    "scheduler": {"type": "string", "enum": ["beta", "normal", "simple"], "default": "beta",
                   "description": "Sigma scheduler; beta/normal outperform simple for reference-heavy prompts."},
    "quantization": {"type": "string", "default": MINIMAX_H3_DEFAULT_QUANT,
                     "description": "Weight precision for the UNET. 'int8' (default) is the "
                                    "int8_convrot checkpoint the official templates ship; 'fp8' is the "
                                    "same size (~21 GB) fp8_scaled build; 'bf16' is ~40 GB and will not "
                                    "fit this box. The text encoder is picked independently: the "
                                    "smallest one already on disk wins, which on Blackwell is the "
                                    "nvfp4_awq build (~16 GB vs 27 GB int8 / 52 GB bf16)."},
    **MINIMAX_H3_PROMPT_FORM_EXTRA,
}
MINIMAX_H3_REF_FORM_EXTRA = {
    **MINIMAX_H3_FORM_EXTRA,
    "ref_image_size": {"type": "string", "enum": ["match", "max"], "default": "match",
                        "description": "Reference image sizing: 'match' downscales refs to the generation's pixel area (faster); 'max' keeps a 2048px short edge for stronger identity fidelity (slower). Reference tokens ride through every sampling step, so 'max' with several large refs costs both time and memory on top of the canvas budget -- keep 'match' unless identity fidelity is the priority."},
}


"""The wrapper API's workflow registry.

Each entry describes one dedicated workflow API: ``build`` constructs the
API-format prompt graph, ``requires_image`` tells the generic generate
handler whether an uploaded image is mandatory, and ``uses`` lists which of
the shared form parameters the builder consumes (the rest are ignored).
Entries may declare ``tasks`` (e.g. MiniMax H3's text/image/reference
variants) to get one endpoint per task under /api/wrapper/{name}/{task}.
Route-level setup (model downloads, quantization) lives in ``api_wrapper.
routes`` keyed by the same names. Adding a new workflow = adding one entry
here plus one setup handler.
"""

WORKFLOWS = {
    "flux2klein9b": {
        "title": "FLUX.2 [klein] 9B image edit",
        "requires_image": True,
        # Either field satisfies requires_image: `image` (one) and/or up to
        # three `ref_images`, each its own reference latent.
        "image_fields": ["image", "ref_images"],
        "uploads": {"image": {"ext": "image", "max": 1}, "ref_images": {"ext": "image", "max": 3}},
        "uses": ["prompt", "negative_prompt", "seed", "steps", "cfg", "megapixels"],
        "form": ["prompt", "image", "ref_images", "negative_prompt", "seed", "steps", "cfg",
                 "megapixels", "width", "height"],
        "extra_form_properties": {
            "width": {"type": "integer", "minimum": 256, "maximum": 8192,
                      "description": "Output width. Omit to take the first image's size."},
            "height": {"type": "integer", "minimum": 256, "maximum": 8192,
                       "description": "Output height. Omit to take the first image's size."},
        },
        "build": build_flux2_klein_9b_img2img,
    },
    "flux2klein9b-txt2img": {
        "title": "FLUX.2 [klein] 9B text to image",
        "requires_image": False,
        "uses": ["prompt", "negative_prompt", "seed", "steps", "cfg"],
        "form": ["prompt", "negative_prompt", "seed", "steps", "cfg", "width", "height"],
        "extra_form_properties": {
            "width": {"type": "integer", "minimum": 256, "maximum": 8192, "default": 1024,
                      "description": "Output width."},
            "height": {"type": "integer", "minimum": 256, "maximum": 8192, "default": 1024,
                       "description": "Output height."},
        },
        "build": build_flux2_klein_9b_text2img,
    },
    "ideogram4": {
        "title": "Ideogram 4 text to image",
        "requires_image": False,
        "uses": ["prompt", "seed"],
        "form": ["prompt", "seed", "steps", "mode", "width", "height"],
        "example_prompt": IDEOGRAM4_EXAMPLE_PROMPT,
        "extra_form_properties": {
            "mode": {"type": "string", "enum": ["default", "quality", "turbo"], "default": "default",
                     "description": "Scheduler preset: default (20 steps), quality (48 steps, lower std), turbo (12 steps, mu 0.5)."},
            "width": {"type": "integer", "minimum": 256, "maximum": 8192, "default": 1024,
                      "description": "Output width; rounded up to a multiple of 16."},
            "height": {"type": "integer", "minimum": 256, "maximum": 8192, "default": 1024,
                       "description": "Output height; rounded up to a multiple of 16."},
        },
        "build": build_ideogram4_text2img,
    },
    "minimaxh3": {
        "title": "MiniMax H3 omni-modal video",
        "output_type": "video",
        "tasks": {
            "text": {
                "title": "Text to video",
                "requires_image": False,
                "uploads": {"llm_image": {"ext": "image", "max": 1}},
                # llm_image feeds the prompt rewriter only; None keeps the
                # generic handler from passing it to the graph builder.
                "upload_params": {"llm_image": None},
                "prompt_rewrite": True,
                "uses": ["prompt", "seed"],
                "form": ["prompt", "raw_prompt", "llm_image", "llm_model", "seed", "steps", "width", "height", "duration", "scheduler"],
                "extra_form_properties": MINIMAX_H3_FORM_EXTRA,
                "quantization_options": ["fp8", "int8", "bf16", "nvfp4"],
                "build": build_minimax_h3_text_to_video,
            },
            "image": {
                "title": "Image to video",
                "requires_image": True,
                "uploads": {"image": {"ext": "image", "max": 1},
                             "last_frame": {"ext": "image", "max": 1},
                             "llm_image": {"ext": "image", "max": 1}},
                "upload_params": {"image": "first_frame", "llm_image": None},
                "prompt_rewrite": True,
                "uses": ["prompt", "seed"],
                "form": ["prompt", "raw_prompt", "image", "last_frame", "llm_image", "llm_model", "seed", "steps", "width", "height", "duration", "scheduler"],
                "extra_form_properties": MINIMAX_H3_FORM_EXTRA,
                "quantization_options": ["fp8", "int8", "bf16", "nvfp4"],
                "build": build_minimax_h3_image_to_video,
            },
            "reference": {
                "title": "Reference to video (ref2va)",
                "requires_image": False,
                "uploads": {"ref_images": {"ext": "image", "max": 3},
                             "ref_videos": {"ext": "video", "max": 1},
                             "ref_video_audios": {"ext": "audio", "max": 1},
                             "ref_audios": {"ext": "audio", "max": 2},
                             "llm_image": {"ext": "image", "max": 1}},
                "upload_params": {"llm_image": None},
                "prompt_rewrite": True,
                "uses": ["prompt", "seed"],
                "form": ["prompt", "raw_prompt", "llm_image", "llm_model", "seed", "steps", "width", "height", "duration", "scheduler", "ref_image_size"],
                "extra_form_properties": MINIMAX_H3_REF_FORM_EXTRA,
                "quantization_options": ["fp8", "int8", "bf16", "nvfp4"],
                "example_prompt": (
                    "Show <Picture 1> skateboarding down a sunlit street, the camera follows "
                    "from the side, <Audio 1> with the sound of wheels rolling on asphalt, "
                    "cinematic 24fps handheld shot."),
                "build": build_minimax_h3_reference_to_video,
            },
        },
    },
}
