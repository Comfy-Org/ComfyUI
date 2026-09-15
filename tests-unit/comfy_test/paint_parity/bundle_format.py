"""Safetensors bundle format shared by the paint parity capture/compare tools.

Deliberately dependency-light (torch + safetensors only) so the author-side
reference capture can import it inside the pinned Tencent-reference venv, which
has no ComfyUI checkout.

Bundle layout (safetensors tensor keys):
    input/sample                 (B, n_pbr, V, C, H, W)  float32
    input/timestep               (1,)                    int64
    input/encoder_hidden_states  (B, n_pbr, L, cross)    float32
    input/ref_latents            (B, 1, C, H, W)         float32
    input/embeds_normal          (B, V, C, H, W)         float32
    input/embeds_position        (B, V, C, H, W)         float32
    input/position_maps          (B, V, 3, Hp, Wp)       float32 in [0, 1]
    input/dino_hidden_states     (B, Ld, Dd)             float32 (optional)
    output/noise_pred            (B*n_pbr*V, C, H, W)    float32
    act/<module path>            per-block activations   float16 (optional)

safetensors metadata (str -> str):
    format_version, source ("native-tiny" | "reference"), seed, input args (json),
    model config (json), torch version, tolerance note.
"""

import json

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

FORMAT_VERSION = "1"


def block_names(n_down, n_up):
    """Module paths of the shared per-block capture points. Both the native port and
    Tencent's reference wrapper expose the inner diffusers-style UNet as ``.unet``
    with identical child names, so the same list drives hooks on either side."""
    names = ["unet.conv_in"]
    names += [f"unet.down_blocks.{i}" for i in range(n_down)]
    names += ["unet.mid_block"]
    names += [f"unet.up_blocks.{i}" for i in range(n_up)]
    names += ["unet.conv_out"]
    return names


def make_parity_inputs(seed=7, batch=1, n_pbr=2, views=2, channels=4, height=16,
                       tokens=4, cross_dim=32, dino_tokens=5, dino_dim=16, timestep=500):
    """Deterministic random inputs for one denoise call (CPU MT19937 generator)."""
    g = torch.Generator().manual_seed(int(seed))

    def r(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32)

    return {
        "input/sample": r(batch, n_pbr, views, channels, height, height),
        "input/timestep": torch.tensor([int(timestep)], dtype=torch.int64),
        "input/encoder_hidden_states": r(batch, n_pbr, tokens, cross_dim),
        "input/ref_latents": r(batch, 1, channels, height, height),
        "input/embeds_normal": r(batch, views, channels, height, height),
        "input/embeds_position": r(batch, views, channels, height, height),
        "input/position_maps": torch.rand(batch, views, 3, height, height, generator=g),
        "input/dino_hidden_states": r(batch, dino_tokens, dino_dim),
    }


def save_bundle(path, tensors, metadata):
    meta = {"format_version": FORMAT_VERSION}
    for k, v in metadata.items():
        meta[str(k)] = v if isinstance(v, str) else json.dumps(v)
    save_file({k: v.clone().contiguous() for k, v in tensors.items()}, str(path), metadata=meta)


def load_bundle(path):
    tensors = load_file(str(path))
    with safe_open(str(path), framework="pt") as f:
        metadata = dict(f.metadata() or {})
    return tensors, metadata


def compare_tensors(reference, candidate, keys=None):
    """Per-key delta stats between two tensor dicts. Returns a list of row dicts."""
    if keys is None:
        keys = [k for k in reference if k in candidate]
    rows = []
    for k in sorted(keys):
        if k not in reference or k not in candidate:
            absent = "reference" if k not in reference else "candidate"
            rows.append({"key": k, "shape": f"missing in {absent} capture",
                         "max_abs": float("nan"), "mean_abs": float("nan"),
                         "rel": float("nan")})
            continue
        a = reference[k].float()
        b = candidate[k].float()
        if a.shape != b.shape:
            rows.append({"key": k, "shape": f"{tuple(a.shape)} vs {tuple(b.shape)}",
                         "max_abs": float("nan"), "mean_abs": float("nan"),
                         "rel": float("nan")})
            continue
        diff = (a - b).abs()
        denom = a.abs().max().clamp(min=1e-12)
        rows.append({"key": k, "shape": str(tuple(a.shape)),
                     "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
                     "rel": float(diff.max() / denom)})
    return rows


def rows_to_markdown(rows, title=""):
    lines = []
    if title:
        lines.append(f"### {title}")
        lines.append("")
    lines.append("| tensor | shape | max abs diff | mean abs diff | max rel diff |")
    lines.append("|--------|-------|--------------|---------------|--------------|")
    for r in rows:
        lines.append(f"| `{r['key']}` | {r['shape']} | {r['max_abs']:.3e} "
                     f"| {r['mean_abs']:.3e} | {r['rel']:.3e} |")
    return "\n".join(lines) + "\n"
