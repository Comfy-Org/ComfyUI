# Paint UNet parity harness

Evidence pack for "the native `UNet2p5DConditionModel` port computes what
Tencent's reference implementation computes", in two tiers:

## Tier 1 — tiny goldens (runs in CI, forever)

`goldens/tiny_golden.safetensors` (<100 KB) holds deterministic inputs and the
expected noise prediction of a seeded 2-block micro-config UNet2p5D that
exercises every attention mechanism (material, reference, multiview+PoseRoPE,
DINO). `test_hunyuan3d_paint_parity.py` rebuilds the model from its
per-parameter seeds and asserts the forward reproduces the committed output at
`atol=2e-4` fp32 CPU. This catches any silent numeric drift in the port
(refactors, torch upgrades, op changes) without weights or network access.

Regenerate (only when the model definition legitimately changes):

    python tests-unit/comfy_test/paint_parity/make_goldens.py

## Tier 2 — reference capture (author-side, never a CI/core dependency)

`capture_reference.py` runs ONE hooked denoise step of the actual Tencent
reference UNet inside a **separate pinned venv** and saves a bundle with the
shared inputs, the reference noise prediction, and per-block activations at
the boundaries both implementations share (`bundle_format.block_names`):
`unet.conv_in`, each `unet.down_blocks.i`, `unet.mid_block`, each
`unet.up_blocks.i`, `unet.conv_out`.

    # reference venv (pins from the Hunyuan3D-2.1 repo requirements)
    python -m venv paint-parity-venv
    paint-parity-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
    paint-parity-venv/bin/pip install diffusers==0.30.0 transformers==4.46.0 \
        einops==0.8.0 safetensors numpy

    paint-parity-venv/bin/python capture_reference.py \
        --reference-root /path/to/Hunyuan3D-2.1 \
        --unet-dir /path/to/hunyuan3d-paintpbr-v2-1/unet \
        --out reference_v6_h64.safetensors

`compare_reference.py` (ComfyUI environment) loads the bundle, runs the native
port with real weights and matching hooks, and writes a per-block delta
markdown table:

    python tests-unit/comfy_test/paint_parity/compare_reference.py \
        --bundle reference_v6_h64.safetensors \
        --weights /path/to/hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin \
        --out parity_report.md

Both sides consume identical inputs: `bundle_format.make_parity_inputs` is
seeded and dependency-light (torch+safetensors only), and the encoder states
are the checkpoint's own learned material tokens, so the comparison isolates
the UNet math from pipeline/preprocessing differences.

## Bundle format

See the docstring in `bundle_format.py` (`input/*`, `output/noise_pred`,
optional `act/<module path>` in fp16, metadata as safetensors str->str).
