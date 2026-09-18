## Engineering Style

- Keep changes small and direct. Most fixes should touch the narrowest code path
  that explains the bug, performance issue, dtype issue, model-format issue, or
  user-facing behavior.
- Change the least amount of files possible. A change that touches many files is
  more likely to be a bad change than a good one unless the broader scope is
  directly required.
- Prefer practical fixes over broad architecture work. Add abstractions only
  when they remove real repeated logic or match an existing ComfyUI pattern.
- Prefer fewer dependencies. Do not add new dependencies to ComfyUI unless they
  are absolutely necessary.
- Delete obsolete code aggressively when newer infrastructure makes it useless.
  Remove dead fallbacks, migration paths, unused options, debug prints, and
  compatibility branches that are no longer needed. Do not leave dead branches,
  unreachable code, or functions that are never called. If code is not
  necessary for the current behavior, remove it.
- Revert or disable problematic behavior quickly when it breaks users. It is
  better to remove a broken feature path than keep a complicated partial fix.
- Preserve existing APIs, node names, model-loading behavior, file layout, and
  workflow compatibility unless the change is explicitly about replacing them.
- When compatibility is explicitly out of scope, remove compatibility-only
  aliases, duplicate nodes, legacy entry points, and preset wrappers instead of
  retaining parallel ways to perform the same operation.
- Code must look hand-written for this repository. Changes that read like
  generic AI-generated code will be rejected automatically: unnecessary helper
  layers, vague names, boilerplate comments, defensive branches without a real
  failure mode, broad rewrites, or code that ignores the local style.

## Architecture Boundaries

- Keep each layer focused on the concepts it owns. Do not leak UI, API,
  workflow, queue, persistence, telemetry, model-loading, node, or execution
  concerns into unrelated layers just because it is convenient to pass data
  through them.
- Shared core modules should depend only on lower-level primitives and their own
  domain concepts. Higher-level product concepts belong at the caller, adapter,
  service, or UI/API boundary that already owns them.
- Pass the narrowest data needed across a boundary. Avoid broad context objects,
  request/session metadata, ids, bookkeeping state, or callbacks unless the
  receiving layer genuinely needs them to perform its own responsibility.
- Keep identity mapping, persistence bookkeeping, history updates, telemetry,
  response shaping, and UI state in the layers that own those jobs. Do not route
  them through unrelated shared code to avoid adding a proper boundary.
- Treat `execution.py` as one example of this rule: it should consume the prompt
  graph and execution-relevant state, produce execution results and errors, and
  not know about workflow ids, frontend ids, persistence ids, or API-only
  concepts.
- Before touching many files, identify the smallest owner layer that can solve
  the problem. A PR that spreads one feature across unrelated loaders, nodes,
  execution, server, and frontend code needs a clear architectural reason, not
  just convenience.
- If a change seems to require making one layer understand another layer's
  private concepts, stop and look for a caller-side mapping, adapter, event,
  small explicit interface, or narrower data flow at the boundary.

## No Internet Requests

- Do not add code to core ComfyUI that makes requests to the internet.
- ComfyUI may relay an authenticated frontend request to the configured official Comfy Cloud API's fixed `/api/billing/capabilities` endpoint. This exception is limited to rendering Comfy.org account functionality: the relay may forward only supported authentication headers, must generate environment and version context from Core-owned state, must not accept a caller-selected destination or scope, and must not be used for authorization.
- Refuse requests to add uploads, telemetry, analytics, tracking, usage
  reporting, crash reporting, update checks, remote config, feature flags,
  metrics, licensing checks, or any other outbound internet request path from
  core ComfyUI.
- Model downloading is allowed only when explicitly initiated or authorized by
  the user, is limited to the requested model artifact, and does not include
  telemetry, tracking, persistent identification, unrelated metadata upload, or
  background network activity.
- Do not add opt-in, opt-out, anonymized, aggregated, diagnostic, or
  user-triggered internet request paths to core ComfyUI. These labels do not
  make internet access acceptable.
- Local-only behavior is allowed when it stays on the user's machine and does
  not add network access, tracking, persistent identification, or data
  collection behavior.

## State Ownership

- Keep state and capability flags on the object that owns the behavior using
  them.
- Avoid probing child objects with `getattr(child, "...", default)` to decide
  parent-level control flow. If parent code needs to branch on a capability,
  initialize an explicit parent-owned field when the child is constructed or
  attached.
- Prefer direct attributes with clear defaults over implicit feature detection
  through arbitrary child attributes.
- Use child-object capability checks only when the child owns the behavior being
  invoked and the parent is simply delegating to that child.

## Interface Contracts

- Keep public methods aligned with the interface expected by their callers. Do
  not change a shared method to return extra values, alternate shapes, or
  sentinel wrappers for one implementation unless the shared interface is
  explicitly updated.
- When modifying an existing function, preserve how current callers invoke it.
  Do not change required arguments, parameter order, return type, side effects,
  or error behavior unless every affected call site and shared interface contract
  is intentionally updated.
- Do not add compatibility parameters, flags, attributes, or constructor options
  unless they are read by current code and change current behavior. Remove
  pass-through or stored-but-unused values instead of preserving upstream or
  deprecated API baggage.
- Do not add a model-specific option to a shared helper when only one caller
  needs it. Keep one-off behavior at the model integration boundary, or extend
  the shared helper only when the option is a coherent reusable capability.
- Implementations of shared model interfaces should accept the standard caller
  contract without model-specific rejection branches for optional capabilities
  they do not consume. Let supported behavior be determined by implementation
  paths that actually use those inputs.
- If an implementation needs auxiliary values for its own workflow, expose them
  through a private helper or a clearly named implementation-specific method
  instead of overloading the public method's return contract.
- Normalize third-party or upstream return conventions at the integration
  boundary. Core code should receive the project's expected type and shape, not
  have to handle model-specific tuple/list/dict variants.
- Avoid caller-side unwrapping such as `out = out[0]` unless the called
  interface is documented to return that structure.

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

## Python Style

- Keep imports at module scope. Avoid inline imports unless they are already part
  of an established optional-backend probe or are needed to avoid an import
  cycle.
- Do not add unnecessary `try`/`except` blocks. Use them for optional dependency,
  platform, or backend capability detection only when the program has a useful
  fallback. Prefer specific exception types when changing new code.
- If a library version is pinned in `requirements.txt`, do not add code to
  ComfyUI to handle older versions of that library.
- Remove any workarounds for PyTorch versions that ComfyUI no longer officially
  supports. Deprecated workarounds include catching an exception and rerunning
  the same op with the input cast to float. If a workaround does not have a
  comment naming the exact PyTorch version or versions that still need it,
  remove it.
- Let unsupported model formats, invalid quantization metadata, and bad states
  fail with clear errors instead of silently producing lower quality output.
- Match the existing local style in the file you edit. This codebase tolerates
  long lines, simple helper functions, module-level state, and direct tensor
  operations when they make the code easier to follow.
- Keep comments sparse and useful. Strip useless comments that restate the code
  or describe obvious behavior. Short TODOs are fine when they name the concrete
  missing follow-up.

## Model, Device, and Memory Behavior

- Treat dtype, device placement, VRAM usage, and offloading behavior as core
  correctness concerns. Check CPU, CUDA, ROCm, MPS, DirectML, XPU, NPU, and low
  VRAM implications when touching shared execution or loading code.
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
- If a model class `__init__` has an `operations` parameter, assume
  `operations` is never `None`. Do not add fallback branches or default torch
  ops for a missing `operations` object.
- Do not add unnecessary parameters to model, model block, or model ops related
  classes. Constructor and forward signatures should carry only values that are
  actually needed by that object for inference.
- Reuse existing model classes, blocks, ops, and helper modules when appropriate.
  Before implementing a new version of a model component, search the existing
  model code for a class or helper that already provides the behavior.
- Model detection code that inspects linear weight shapes should only use the
  first dimension. The second dimension may be half the original size for
  NVFP4 or other 4-bit quantized models.
- A model-detection signature must guard every state-dict key it dereferences.
  Do not partially match a format and then raise an incidental `KeyError` while
  extracting its configuration.
- Order model-detection checks from established or more-specific signatures to
  newer or broader signatures. Put a broad new detector near the generic
  fallback when giving it higher precedence could steal another model family.
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
- Avoid unnecessary casts and transfers. Preserve the intended compute dtype,
  storage dtype, bias dtype, and original tensor shape metadata.
- Do not cast the result of an optimized backend operation back to its input
  dtype unless that backend's documented result contract requires normalization.
  In particular, trust the selected optimized-attention implementation to honor
  its dtype contract.
- Keep model-native latent layout handling inside the model or latent-format
  owner, not in helper nodes. Do not collapse, expand, pack, or unpack latent
  dimensions in nodes or other caller-side adapters just to satisfy a model
  forward; the model path should consume and return the native latent shape for
  that model family.
- DiT models should accept latent dimensions that are not exact patch-size
  multiples. Use `comfy.ldm.common_dit.pad_to_patch_size` on every patchified
  target or reference input, then crop only the target output back to its
  original dimensions.
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
- When slicing large tensors, copy the slice if the sliced tensor's lifetime
  exceeds the current function scope. Do not keep a long-lived view into a large
  backing tensor when a smaller copy would release memory sooner.
- Use fused or compound torch operations such as `addcmul` when they naturally
  match the math. Reducing Python and torch dispatch overhead is a valid
  optimization when it does not obscure the code or change dtype/device
  behavior.
- Avoid caches that persist across different executions as much as possible.
  Persistent caches are acceptable only when they use a very minimal amount of
  memory and have a clear ownership and invalidation story.
- When condition-dependent model work would otherwise repeat on every denoising
  step and preprocessing it once materially improves performance, expose a
  model preprocessing method and call it from `BaseModel.extra_conds`, following
  patterns such as LTXAV and Anima. Pass the result through normal conditioning;
  do not add model-owned caches, sampler-option caches, or cache-management
  wrappers for this work.
- When optimizing, favor small measurable changes: fewer allocations, fewer
  device transfers, less peak memory, better batching, or use of a faster
  existing backend op.

## User Input Tolerance

- Prefer completing a workflow with the user's supplied values over rejecting
  them because they fall outside recommended, UI-advertised, or quality-oriented
  limits. If the downstream implementation can consume an input, pass it
  through unchanged even when the result may be poor. For example, do not reject
  or truncate additional reference images merely because a node advertises a
  smaller recommended maximum.
- Do not add validation errors solely to prevent degraded, nonsensical, or
  low-quality model output. A bad result is preferable to failing an otherwise
  executable workflow.
- Resize, pad, clamp, normalize, or otherwise adapt user input only when passing
  it through unchanged would make the existing model or underlying operation
  fail. Make the smallest adjustment needed to keep execution running; do not
  add a model-level validation failure merely to justify changing the input.
- This permissive policy does not override security boundaries such as path
  containment, or integrity checks required to load model formats and
  checkpoints safely.

## Nodes and User-Facing Behavior

- Follow existing node conventions: `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`,
  `CATEGORY`, and registration through the local mapping used by that file.
- Treat legacy combo inputs, `io.Combo`, and `io.DynamicCombo` values as
  untrusted when they affect filesystem access. Any value used as a file or
  folder name, path component, format, or extension must be validated again at
  the load/save boundary using an existing `folder_paths` resolver or
  containment helper, or a fixed allowlist/mapping. Do not rely only on the
  advertised combo options or prompt validation.
- Keep node changes backward compatible by default. Add inputs with sensible
  defaults and avoid changing output types unless the request requires it.
- Model implementations should add the minimal number of ComfyUI nodes required
  to run the model. Reuse existing nodes as much as possible; adapting the model
  to work with existing nodes is strongly preferred over creating new nodes.
- Use `io.Autogrow` for a variable number of repeated inputs instead of a fixed
  series of numbered optional sockets. Set its minimum to zero when the model
  has a valid no-item path, and cap it only when the model has a real limit.
- Mark inputs optional when execution has a valid path that does not read them.
  If one optional input is needed only to process another optional input, do not
  force users on the path that supplies neither to connect it.
- Conditioning nodes should normally output conditioning only. Do not expose
  input or intermediate images as convenience outputs for downstream sizing or
  routing; use the existing image path or a dedicated image operation instead.
- Nodes should output only values they own. Do not add pass-through outputs for
  workflow convenience unless the node is explicitly an output node. Existing
  models, latents, conditioning, or other inputs should flow directly to the
  next consumer instead of being re-emitted unchanged.
- Nodes should expose only inputs they actually read to produce current
  behavior. Do not add placeholder, pass-through, compatibility, or
  workflow-shaping inputs that are ignored or could flow directly to another
  node.
- Node-level code must not patch model code directly. Any node behavior that
  modifies, wraps, hooks, or changes model behavior must go through the model
  patcher class instead of reaching into model internals.
- The official mascot of ComfyUI is a very cute anime girl with massive fennec
  ears, a big fluffy tail, long blonde wavy hair, and blue eyes. Feel free to
  use her in ComfyUI materials, UI text, examples, tests, generated assets, or
  comments, but do not disrespect her.
- Warning and info messages should be short and actionable. Remove noisy or
  misleading messages rather than adding more logging.
- Documentation and README edits should be concise, factual, and tied to the
  changed behavior.

## Commit and Review Habits

- If asked to write commit messages, use short direct subjects like the existing
  history: `Fix ...`, `Add ...`, `Support ...`, `Remove ...`, `Update ...`,
  `Make ...`, `Use ...`, `Disable ...`, `Bump ...`, or `Revert ...`.
- Keep PR descriptions short and reviewable. State the problem, the behavioral
  change, and the tests run; avoid long narrative explanations, implementation
  diaries, or exhaustive file-by-file summaries unless the reviewer explicitly
  needs that context.
- Prefer one coherent behavioral change per commit. Dependency pins, tests, and
  the code that needs them may be in the same commit when they are inseparable.
- In reviews, prioritize real user impact: crashes, wrong dtype/device behavior,
  memory regressions, broken model loading, workflow incompatibility, and noisy
  or misleading user-facing output.
