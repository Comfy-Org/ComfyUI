# Comfy quantization format

Comfy quantized checkpoints are standard safetensors files containing quantized
weights, scale tensors, and per-module configuration. A checkpoint can mix
quantization formats and ordinary floating-point layers. ComfyUI selects native
kernels where supported and dequantizes weights for compute-dtype execution when
a format's fast path is disabled.

## Checkpoint layout

Keep the model's normal state-dict names. Each quantized module adds a
`<module>.comfy_quant` tensor containing UTF-8 JSON as a one-dimensional `uint8`
array. For example, an FP8 linear layer contains:

```text
blocks.0.proj.weight         # quantized weight
blocks.0.proj.weight_scale   # weight scale
blocks.0.proj.bias           # ordinary floating-point bias, if present
blocks.0.proj.comfy_quant    # encoded {"format": "float8_e4m3fn"}
```

To encode configuration in an existing state dict:

```python
import json
import torch

config = {"format": "float8_e4m3fn", "full_precision_matrix_mult": True}
state_dict["blocks.0.proj.comfy_quant"] = torch.tensor(
    list(json.dumps(config).encode("utf-8")), dtype=torch.uint8
)
```

`format` is required. Modules without `comfy_quant` load normally. Logical weight
dimensions come from the model's module definition; compute dtype is selected at
runtime. No file-level layer map or `format_version` field is required. The older
`_quantization_metadata` safetensors header and `scaled_fp8` checkpoints are
converted during loading; new files should use per-module entries.

## Weight formats

All tensor names below have the same `<module>.` prefix. `N` is the output width
(or vocabulary size), `K` is the input width, and `G` is the quantization group
size. Scales are FP32 unless another dtype is specified.

| `format` | `weight` storage | Scale tensors |
| --- | --- | --- |
| `float8_e4m3fn` | FP8 E4M3FN, `[N, K]` | `weight_scale`: scalar |
| `float8_e5m2` | FP8 E5M2, `[N, K]` | `weight_scale`: scalar |
| `nvfp4` | FP4 E2M1 packed into `uint8`, `[N, K/2]` before padding | `weight_scale`: FP8 E4M3FN block scales, groups of 16; `weight_scale_2`: global scalar |
| `mxfp8` | FP8 E4M3FN, `[N, K]` before padding | `weight_scale`: E8M0 power-of-two block scales, groups of 32 |
| `int8_tensorwise` | `int8`, `[N, K]` | `weight_scale`: scalar or per-row `[N, 1]` |
| `convrot_w4a4` | Signed INT4 packed into `int8`, `[N, K/2]` | `weight_scale`: per-row `[N]` |
| `asym_w4a8_int8` | 4-bit codes packed into `int8`, `[N, K/2]` | `weight_s_rel`: FP8 E4M3FN or FP32 `[N, K/G]`; `weight_s_channel`: FP32 `[N]` |
| `w6a8_int8` | Uniform 6-bit codes packed into `int8`, `[N, 3K/4]` | `weight_s_rel` and `weight_s_channel`, as above |

NVFP4 and MXFP8 use Comfy Kitchen's blocked/swizzled scale layout, with weight
dimensions padded to multiples of 16 and 32 respectively. Their block scales,
and FP8 `weight_s_rel`, can also be stored as raw `uint8` bytes. Use the layout's
quantizer and `QuantizedTensor.state_dict(prefix)` to produce the correct packing,
padding, and scale keys; add `comfy_quant` separately.

### Format options

Options belong in the module's `comfy_quant` JSON:

| Option | Applies to | Behavior |
| --- | --- | --- |
| `full_precision_matrix_mult` | Linear and MoE weights | Default `false`. `true` keeps quantized storage but dequantizes for matrix multiplication in the compute dtype; it does not force FP32. |
| `convrot` | `int8_tensorwise` | Default `false`. Enables grouped Hadamard rotation; export rotated weights with per-row scales. |
| `convrot_groupsize` | Rotated INT8 and all packed integer formats | Rotation group size, default `256`. Packed integer formats always use rotation. |
| `linear_dtype` | `convrot_w4a4` | `"int4"` (default) or `"int8"` selects the multiplication path; stored weights remain INT4. The internal `quant_group_size` is fixed at `64`. |
| `group_size` | `asym_w4a8_int8`, `w6a8_int8` | Weight scale group size, default `16`. |
| `num_experts` | MoE banks | Expert count written when saving; the model definition supplies the dimensions when loading. |

Linear/MoE loaders also accept rotation, group-size, and `linear_dtype` options
inside a nested `params` object; top-level values take precedence. Embedding
rotation options must be top-level.

`asym_w4a8_int8` optionally stores `weight_codebook`, a floating-point `[16]`
lookup table; without it, codes use uniform levels. `w6a8_int8` has no codebook.
Export these formats with symmetric weights: the ComfyUI loader does not consume
Comfy Kitchen's optional `weight_correction` tensor.

Rotation groups must divide `K` and be powers of four. W4A4 also requires `K`
divisible by 64. For grouped W4A8, `K` must be divisible by 16 and `G`; `G` must
be at least 4 and either divide 16 or be a multiple of 16. W6A8 requires `K`
divisible by 32 and `G` a multiple of 16. Accelerated backends may impose further
shape restrictions.

## Activations and smoothing

- **FP8:** optional `input_scale` supplies a calibrated scalar activation scale.
  Without it, the scale is `1.0`.
- **NVFP4:** optional `input_scale` supplies the global activation scale;
  otherwise it is computed from the current input. Block scales are dynamic.
- **MXFP8:** activation block scales are computed dynamically; a stored
  `input_scale` is not used by the quantizer.
- **Integer formats:** accept floating-point inputs and quantize activations
  inside the selected kernel. They do not load `input_scale`.
- **NVFP4 smoothing:** optional `pre_quant_scale` multiplies the input before
  the linear operation. Export weights with the matching inverse adjustment.

Calibration is needed to choose representative fixed activation scales, not to
use formats with dynamic activation scaling. Biases remain floating point.

## Embeddings and MoE

Quantized embeddings support `float8_e4m3fn`, `float8_e5m2`, and
`int8_tensorwise`, including INT8 per-row scales and optional rotation. Lookup
dequantizes the selected rows; embedding scales default to `1.0` when omitted.

Models using `MoEExperts` can store an expert bank under one module prefix.
Weights and per-expert scales carry a leading expert dimension `E`; optional
bias is `[E, N]`. Scalar scales can be shared. Grouped W4A8/W6A8 and INT8 banks
with per-row scales can also be stored with expert/output rows flattened to
`E * N`. MoE banks do not load `input_scale` or `pre_quant_scale`.

Linear weights use ComfyUI's normal offloading and LoRA patching paths. Module
state-dict saving writes quantized tensors, scales, and supported configuration.

## Attention preferences

An attention module with a `ComfyAttention` child can store
`<module>.comfy_attention.config` using the same `uint8` JSON encoding. This is
independent of weight quantization. Supply one object or an ordered list:

```json
[
  {"attention": "comfy_kitchen_sol", "tau": 1.0},
  {"attention": "comfy_kitchen_int8"}
]
```

ComfyUI selects the first supported preference for the primary device during
loading. Unknown methods are skipped with a warning; unavailable methods are
skipped. An empty list or no available preference leaves normal attention
selection in effect. Explicit attention overrides take priority. Saving
preserves the full preference list, including unsupported entries.

Only the two methods above are accepted. SOL exposes `tau` (default `1.0`);
higher values route fewer blocks through exact attention. Other SOL options use
Comfy Kitchen defaults. SOL requires matching FP16/BF16 Q/K/V inputs with head
dimension 128. Masked calls use PyTorch attention; `low_precision_attention=False`
does not disable SOL.

Implementation: [format registry](comfy/quant_ops.py),
[loaders and serialization](comfy/ops.py),
[legacy conversion](comfy/utils.py),
[attention configuration](comfy/ldm/modules/attention.py).
