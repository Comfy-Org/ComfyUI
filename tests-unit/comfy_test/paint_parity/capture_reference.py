"""Author-side: capture one hooked denoise step of Tencent's reference paint UNet.

Runs in a SEPARATE pinned venv (never a ComfyUI dependency):
    python -m venv paint-parity-venv
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install diffusers==0.30.0 transformers==4.46.0 einops==0.8.0 safetensors numpy

Usage (from the reference checkout's hy3dpaint parent so the package imports):
    python capture_reference.py \
        --reference-root  <path to Hunyuan3D-2.1 checkout> \
        --unet-dir        <.../hunyuan3d-paintpbr-v2-1/unet> \
        --out             reference_v6_h64.safetensors \
        [--views 6 --height 64 --seed 7 --timestep 999 --fp16-activations]

The bundle records the shared deterministic inputs (bundle_format), the
reference model's noise prediction, and per-block activations at the module
boundaries listed by bundle_format.block_names(). Compare against the native
port with compare_reference.py (which runs in the ComfyUI environment).

Only bundle_format is imported from this repo - it depends on torch+safetensors
alone, so this script never needs comfy installed.
"""

import argparse
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bundle_format  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-root", required=True,
                    help="Hunyuan3D-2.1 checkout containing hy3dpaint/")
    ap.add_argument("--unet-dir", required=True,
                    help="hunyuan3d-paintpbr-v2-1/unet dir (config.json + diffusion_pytorch_model.bin)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--height", type=int, default=64, help="latent H (=W)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timestep", type=int, default=999)
    ap.add_argument("--fp16-activations", action="store_true")
    args = ap.parse_args()

    # Import the reference modules.py without executing hunyuanpaintpbr/__init__.py,
    # which drags in pytorch_lightning/torchvision that the pinned venv doesn't need.
    # attn_processor.py hard-codes one `.to("cuda:0")` (a pure device move; its
    # multi-GPU path is off) - patch it out IN MEMORY for CPU capture; the on-disk
    # reference stays untouched and no math changes.
    import importlib.util
    import types
    unet_dir_pkg = os.path.join(args.reference_root, "hy3dpaint", "hunyuanpaintpbr", "unet")
    for pkg, path in (("hunyuanpaintpbr", os.path.dirname(unet_dir_pkg)),
                      ("hunyuanpaintpbr.unet", unet_dir_pkg)):
        if pkg not in sys.modules:
            stub = types.ModuleType(pkg)
            stub.__path__ = [path]
            sys.modules[pkg] = stub

    def load_module(name, filename, replacements=()):
        path = os.path.join(unet_dir_pkg, filename)
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        for old, new in replacements:
            assert old in src, f"expected to patch {old!r} in {filename}"
            src = src.replace(old, new)
        # Import the patched source through the normal loader machinery from a
        # temporary file rather than exec'ing a string.
        with tempfile.NamedTemporaryFile(
            "w", suffix=f"-{filename}", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(src)
            patched_path = handle.name
        try:
            spec = importlib.util.spec_from_file_location(name, patched_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        finally:
            os.unlink(patched_path)
        return mod

    # Always strip the reference's hard-coded device move: this capture path is
    # CPU-only, and leaving it in on a CUDA machine mixes devices mid-forward.
    cpu_patch = (('.to("cuda:0")', ""),)
    load_module("hunyuanpaintpbr.unet.attn_processor", "attn_processor.py", cpu_patch)
    ref_modules = load_module("hunyuanpaintpbr.unet.modules", "modules.py")
    UNet2p5DConditionModel = ref_modules.UNet2p5DConditionModel

    with open(os.path.join(args.unet_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    try:
        model = UNet2p5DConditionModel.from_pretrained(args.unet_dir, torch_dtype=torch.float32)
    except TypeError as exc:
        # Only the known mismatch is recoverable: some diffusers versions reject
        # the raw config's private keys as unexpected keyword arguments. Anything
        # else is a real loader failure and must surface.
        if "argument" not in str(exc):
            raise
        # replicate the reference from_pretrained via from_config instead
        from diffusers import UNet2DConditionModel
        base = UNet2DConditionModel.from_config(cfg)
        model = UNet2p5DConditionModel(base)
        conv_in = base.conv_in
        model.unet.conv_in = torch.nn.Conv2d(
            12, conv_in.out_channels, kernel_size=conv_in.kernel_size,
            stride=conv_in.stride, padding=conv_in.padding,
            dilation=conv_in.dilation, groups=conv_in.groups,
            bias=conv_in.bias is not None)
        ckpt = torch.load(os.path.join(args.unet_dir, "diffusion_pytorch_model.bin"),
                          map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt, strict=True)
        model = model.float()
    model.eval()

    # the real checkpoint's dims: cross 1024, 77 learned tokens, dino-giant 1536
    tensors = bundle_format.make_parity_inputs(
        seed=args.seed, batch=1, n_pbr=len(model.pbr_setting), views=args.views,
        channels=4, height=args.height, tokens=model.pbr_token_channels,
        cross_dim=cfg["cross_attention_dim"], dino_tokens=5, dino_dim=1536,
        timestep=args.timestep)

    # use the checkpoint's own learned material tokens as encoder states so both
    # sides derive the identical conditioning from the identical weights
    learned = torch.stack([getattr(model.unet, f"learned_text_clip_{t}")
                           for t in model.pbr_setting], dim=0).unsqueeze(0)
    tensors["input/encoder_hidden_states"] = learned.detach().float()

    acts = {}
    names = bundle_format.block_names(len(model.unet.down_blocks), len(model.unet.up_blocks))

    def make_hook(name):
        def hook(_module, _args, output):
            out = output[0] if isinstance(output, tuple) else output
            out = out.detach().float()
            acts[f"act/{name}"] = out.half() if args.fp16_activations else out
        return hook

    handles = [model.get_submodule(n).register_forward_hook(make_hook(n)) for n in names]

    with torch.inference_mode():
        out = model(
            tensors["input/sample"],
            tensors["input/timestep"],
            tensors["input/encoder_hidden_states"],
            ref_latents=tensors["input/ref_latents"],
            embeds_normal=tensors["input/embeds_normal"],
            embeds_position=tensors["input/embeds_position"],
            position_maps=tensors["input/position_maps"],
            dino_hidden_states=tensors["input/dino_hidden_states"],
            mva_scale=1.0, ref_scale=1.0, cache={},
        )
    for h in handles:
        h.remove()

    if hasattr(out, "sample"):
        out = out.sample
    elif isinstance(out, (tuple, list)):
        out = out[0]
    tensors["output/noise_pred"] = out.detach().float()
    tensors.update(acts)
    bundle_format.save_bundle(args.out, tensors, {
        "source": "reference",
        "reference": "tencent/Hunyuan3D-2.1 hunyuan3d-paintpbr-v2-1",
        "input_args": {"seed": args.seed, "views": args.views, "height": args.height,
                       "timestep": args.timestep},
        "torch_version": torch.__version__,
        "note": "encoder_hidden_states are the checkpoint's learned material tokens",
        "cpu_patch": "removed hard-coded .to(\"cuda:0\") device move in attn_processor.py",
    })
    sys.stdout.write(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB, "
                     f"{len(acts)} block activations)\n")


if __name__ == "__main__":
    main()
