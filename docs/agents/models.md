# Models, Device, and Memory

Detailed rules referenced from [AGENTS.md](../../AGENTS.md).

- Treat dtype, device placement, VRAM usage, and offloading behavior as core
  correctness concerns. Check CPU, CUDA, ROCm, MPS, DirectML, XPU, NPU, and low
  VRAM implications when touching shared execution or loading code.

## Autograd and Model Freezing

- Do not add `torch.no_grad`, `torch.inference_mode`, or inference-mode helper
  wrappers in ComfyUI code. The only allowed inference-mode-related use is
  disabling a globally set inference mode when a training path needs gradients.
- Do not add freeze, unfreeze, or trainability toggles to model classes. ComfyUI
  models are always treated as frozen for inference, so explicit freeze
  functionality is redundant and should not be added.
- Remove training-only behavior such as dropout from inference model code, but
  preserve checkpoint and state-dict compatibility when doing so. If deleting a
  module would change state-dict keys, module ordering, or checkpoint loading
  behavior, replace it with a no-op such as `nn.Identity` instead of removing the
  slot outright.

## Use the Existing Optimized Operations

- Prefer native ComfyUI formats and existing quantization/offload helpers over
  adding parallel code paths. Use `comfy.quant_ops`, `comfy.model_management`,
  `comfy.memory_management`, `comfy.pinned_memory`, `comfy_aimdo`, and
  `comfy-kitchen` helpers where they already solve the problem.
- Model implementations must use an existing optimized Comfy Kitchen or
  ComfyUI operation whenever one supports the required math and tensor layout
  without changing expected dtype, device, memory, or interface behavior. This
  is the default implementation requirement, not an optional follow-up
  optimization.
- Before implementing model math, inspect the operations already exposed by
  Comfy Kitchen, `comfy.quant_ops`, and existing ComfyUI model helpers. Check
  for optimized single, paired, fused, layout-specific, and quantized variants
  before writing a local implementation or composing lower-level torch ops.
- Use the compatible optimized operation first and adapt the model's inputs to
  its documented layout while preserving the model's exact math. If several
  optimized variants apply, benchmark representative model shapes and select
  the fastest valid path.
- Add or retain a local implementation only when no existing optimized
  operation supports the required math, layout, dtype, device, autograd, or
  patch contract. Keep differentiable or patch-compatible fallbacks when the
  optimized inference operation does not provide those contracts.
- Use the existing ComfyUI cast, offload, and cleanup helpers for parameters
  passed to optimized operations. Preserve model-specific epsilon, scaling,
  layout, dtype, device, and output-shape behavior.
- Prefer ComfyUI's shared optimized kernels and backend dispatchers over
  handwritten implementations of the same operation. Remove duplicate local
  kernels and adapt inputs to the shared operation's documented layout while
  preserving the model's original math and output contract.
- All models should use the optimized attention function selected by ComfyUI.
  Treat optimized backend functions, dispatch helpers, and capability-selected
  callables as opaque. Higher-level code must not inspect function identity,
  names, modules, or implementation details to decide behavior.
- Apply the same opacity rule to similar patterns beyond attention: callers
  should depend on the documented interface and result contract, not on which
  backend implementation was selected underneath.
- Do not use custom inference ops that only duplicate an existing op while
  upcasting to float32, such as custom RMSNorm variants. Use the generic ComfyUI
  ops and/or native torch ops instead.

## Model Classes and Constructors

- If a model class `__init__` has an `operations` parameter, assume
  `operations` is never `None`. Do not add fallback branches or default torch
  ops for a missing `operations` object.
- Do not add unnecessary parameters to model, model block, or model ops related
  classes. Constructor and forward signatures should carry only values that are
  actually needed by that object for inference.
- Reuse existing model classes, blocks, ops, and helper modules when appropriate.
  Before implementing a new version of a model component, search the existing
  model code for a class or helper that already provides the behavior.

## Model Detection

- Model detection code that inspects linear weight shapes should only use the
  first dimension. The second dimension may be half the original size for
  NVFP4 or other 4-bit quantized models.
- A model-detection signature must guard every state-dict key it dereferences.
  Do not partially match a format and then raise an incidental `KeyError` while
  extracting its configuration.
- Order model-detection checks from established or more-specific signatures to
  newer or broader signatures. Put a broad new detector near the generic
  fallback when giving it higher precedence could steal another model family.

## Tensors and Python Values

- Avoid adding `einops` usage in core inference code. Use native torch tensor
  ops such as `reshape`, `view`, `permute`, `transpose`, `flatten`, `unflatten`,
  `unsqueeze`, and `squeeze` instead.
- Do not use tensors as general-purpose Python data structures. Keep metadata,
  bookkeeping, counters, flags, shape math, padding math, index planning, memory
  estimates, and control-flow decisions in plain Python values unless the data
  must participate directly in tensor computation. Do not create tensors for
  structural metadata that is only used for Python-side control flow. Sequence
  lengths, cumulative offsets, split indices, window counts, slice boundaries,
  and repeat counts should be kept as Python ints/lists from the point they are
  computed. Do not build them as CPU/GPU tensors and then cast, move, validate,
  or convert them back to Python for `split`, `tensor_split`, indexing plans,
  loops, or cache keys. Avoid creating temporary tensors just to use tensor
  methods for scalar or structural calculations.

## Dtype and Device

- Avoid unnecessary casts and transfers. Preserve the intended compute dtype,
  storage dtype, bias dtype, and original tensor shape metadata.
- Do not cast the result of an optimized backend operation back to its input
  dtype unless that backend's documented result contract requires normalization.
  In particular, trust the selected optimized-attention implementation to honor
  its dtype contract.
- Avoid defensive shape and configuration checks that merely replace the clear
  failure from the tensor operation immediately below them. Add explicit
  validation only when it provides materially better context at a real boundary
  or prevents silent incorrect output.
- Assume inputs to the main model forward are already in the compute dtype by
  default, except integer inputs such as some model timestep tensors. Do not add
  defensive or convenience casts in model code; it is better for invalid dtype
  plumbing to error clearly than to hide it with unnecessary casts.
- Raw model parameters that are not owned by an op and may be initialized in a
  dtype different from the compute dtype should be cast at use in forward or
  inference code with `comfy.ops.cast_to_input` or
  `comfy.model_management.cast_to` to avoid dtype mismatches.
- Model code should not care what dtype it is initialized in, and model
  `__init__` methods should not contain workarounds for specific dtypes. Dtype
  workaround code, such as making a model work with fp16 compute, belongs in the
  execution or model-management layer that owns compute policy.
- Model code should not perform unnecessary device-to-CPU or CPU-to-device
  transfers. New allocations must be created on the correct device and dtype;
  never allocate on CPU and then move to GPU, or allocate in one dtype and then
  convert to another.

## Latent Layout

- Keep model-native latent layout handling inside the model or latent-format
  owner, not in helper nodes. Do not collapse, expand, pack, or unpack latent
  dimensions in nodes or other caller-side adapters just to satisfy a model
  forward; the model path should consume and return the native latent shape for
  that model family.
- DiT models should accept latent dimensions that are not exact patch-size
  multiples. Use `comfy.ldm.common_dit.pad_to_patch_size` on every patchified
  target or reference input, then crop only the target output back to its
  original dimensions.

## Memory and Caches

- Model code itself should not perform memory management. Loading, unloading,
  offloading, device movement, VRAM policy, cache lifetime, and cleanup belong
  in the relevant model-management and execution layers, not inside model
  implementations.
- Do not add global, module-level, class-level, singleton, or model-owned stores
  for tensors or other large memory that persist across executions. Temporary
  caches must be scoped to a single execution or forward/encode/decode call:
  allocate them in the owning top-level call, pass them explicitly through the
  call stack, and let them be discarded when that call returns.
- Follow the Wan VAE temporal cache pattern for temporary caches: create a local
  cache such as `feat_map` for the encode/decode operation, pass it into the
  blocks that need it, and do not retain it on the model or in global state.
- When slicing large tensors, copy the slice if the sliced tensor's lifetime
  exceeds the current function scope. Do not keep a long-lived view into a large
  backing tensor when a smaller copy would release memory sooner.
- Avoid caches that persist across different executions as much as possible.
  Persistent caches are acceptable only when they use a very minimal amount of
  memory and have a clear ownership and invalidation story.
- When condition-dependent model work would otherwise repeat on every denoising
  step and preprocessing it once materially improves performance, expose a
  model preprocessing method and call it from `BaseModel.extra_conds`, following
  patterns such as LTXAV and Anima. Pass the result through normal conditioning;
  do not add model-owned caches, sampler-option caches, or cache-management
  wrappers for this work.

## Initialization

- In model init code, prefer `torch.empty` for parameter/buffer placeholders
  that are populated from the model state dict instead of zero-initializing with
  `torch.zeros` or similar. If an allocation is not loaded from the state dict
  and is useless for inference, do not include it.
- `nn.Parameter` tensors that are stored in and populated from the model state
  dict should be initialized with `torch.empty`, not with zero, random, or
  otherwise meaningful initialization.
- Model initialization should describe module structure, not fabricate
  checkpoint-owned tensor contents. Parameters and buffers that are loaded from
  the state dict must not be manually initialized, reassigned, or filled with
  fallback values unless that value is actually used when no checkpoint key
  exists.

## Optimization

- Use fused or compound torch operations such as `addcmul` when they naturally
  match the math. Reducing Python and torch dispatch overhead is a valid
  optimization when it does not obscure the code or change dtype/device
  behavior.
- When optimizing, favor small measurable changes: fewer allocations, fewer
  device transfers, less peak memory, better batching, or use of a faster
  existing backend op.
