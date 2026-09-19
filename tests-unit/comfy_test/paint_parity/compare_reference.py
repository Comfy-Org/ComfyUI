"""Author-side: run the native paint UNet on a reference capture bundle and emit
a per-block delta markdown table.

Runs in the ComfyUI environment (needs comfy importable), against a bundle
produced by capture_reference.py in the pinned reference venv:

    python tests-unit/comfy_test/paint_parity/compare_reference.py \
        --bundle    reference_v6_h64.safetensors \
        --weights   <hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin or .safetensors> \
        --out       parity_report.md
"""

import argparse
import os
import sys

import torch
from safetensors.torch import load_file

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # comfy_test/ (for paint_parity pkg)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # repo root

from comfy.cli_args import args as comfy_args  # noqa: E402
if not torch.cuda.is_available():
    comfy_args.cpu = True

from paint_parity import bundle_format  # noqa: E402
from paint_parity import harness  # noqa: E402
from comfy.ldm.hunyuan3d.paint.loader import load_paint_unet  # noqa: E402


def _load_state_dict(path):
    if path.endswith(".safetensors"):
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="parity_report.md")
    args = ap.parse_args()

    tensors, metadata = bundle_format.load_bundle(args.bundle)
    sd = _load_state_dict(args.weights)
    patcher = load_paint_unet(sd, model_options={"dtype": torch.float32})
    model = patcher.model.diffusion_model  # the UNet2p5DConditionModel inside the BaseModel
    model.eval()

    with torch.inference_mode():
        out, acts = harness.run_model(model, tensors, capture_blocks=True)
    candidate = dict(acts)
    candidate["output/noise_pred"] = out

    keys = [k for k in tensors if k.startswith("act/") or k.startswith("output/")]
    rows = bundle_format.compare_tensors(tensors, candidate, keys)
    md = bundle_format.rows_to_markdown(
        rows, title=f"Paint UNet parity: native port vs {metadata.get('reference', 'reference')}")
    md += (f"\nbundle: `{os.path.basename(args.bundle)}`  "
           f"input_args: `{metadata.get('input_args', '?')}`  "
           f"reference torch: `{metadata.get('torch_version', '?')}`  "
           f"native torch: `{torch.__version__}`\n")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    sys.stdout.write(md + "\n")
    sys.stdout.write(f"wrote {args.out}\n")


if __name__ == "__main__":
    main()
