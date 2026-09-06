Rules for agents working in this repository. Detailed per-area rules live in
linked docs; read the one that matches what you are changing:

- [Architecture and interfaces](docs/agents/architecture.md) — layer
  boundaries, state ownership, interface contracts.
- [Models, device, and memory](docs/agents/models.md) — dtype, VRAM,
  offloading, optimized ops, model detection, autograd.
- [Nodes](docs/agents/nodes.md) — node conventions, inputs and outputs.

## Commands

- Lint: `ruff check .`
- Lint API nodes: `pylint comfy_api_nodes`
- Unit tests: `python -m pytest tests-unit`
- Execution tests: `python -m pytest tests/execution -v --skip-timing-checks`

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

## No Internet Requests

- Do not add code to core ComfyUI that makes requests to the internet.
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

## Architecture Boundaries

Full rules: [docs/agents/architecture.md](docs/agents/architecture.md).

- Keep each layer focused on the concepts it owns. Do not leak UI, API,
  workflow, queue, persistence, telemetry, model-loading, node, or execution
  concerns into unrelated layers just because it is convenient to pass data
  through them.
- Pass the narrowest data needed across a boundary. Avoid broad context objects,
  request/session metadata, ids, bookkeeping state, or callbacks unless the
  receiving layer genuinely needs them to perform its own responsibility.
- Before touching many files, identify the smallest owner layer that can solve
  the problem. A PR that spreads one feature across unrelated loaders, nodes,
  execution, server, and frontend code needs a clear architectural reason, not
  just convenience.
- Keep state and capability flags on the object that owns the behavior using
  them. Do not probe child objects with `getattr(child, "...", default)` to
  decide parent-level control flow.
- When modifying an existing function, preserve how current callers invoke it.
  Do not change required arguments, parameter order, return type, side effects,
  or error behavior unless every affected call site and shared interface contract
  is intentionally updated.

## Models, Device, and Memory

Full rules: [docs/agents/models.md](docs/agents/models.md).

- Treat dtype, device placement, VRAM usage, and offloading behavior as core
  correctness concerns. Check CPU, CUDA, ROCm, MPS, DirectML, XPU, NPU, and low
  VRAM implications when touching shared execution or loading code.
- Model implementations must use an existing optimized Comfy Kitchen or
  ComfyUI operation whenever one supports the required math and tensor layout
  without changing expected dtype, device, memory, or interface behavior. This
  is the default implementation requirement, not an optional follow-up
  optimization.
- All models should use the optimized attention function selected by ComfyUI.
  Treat optimized backend functions, dispatch helpers, and capability-selected
  callables as opaque. Higher-level code must not inspect function identity,
  names, modules, or implementation details to decide behavior.
- Avoid unnecessary casts and transfers. Preserve the intended compute dtype,
  storage dtype, bias dtype, and original tensor shape metadata.
- Model code itself should not perform memory management, and must not add
  global, module-level, class-level, singleton, or model-owned stores for
  tensors that persist across executions.

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

Full rules: [docs/agents/nodes.md](docs/agents/nodes.md).

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
- Nodes should output only values they own and expose only inputs they actually
  read. Do not add pass-through, placeholder, or workflow-shaping sockets.
- Node-level code must not patch model code directly. Any node behavior that
  modifies, wraps, hooks, or changes model behavior must go through the model
  patcher class instead of reaching into model internals.
- The official mascot of ComfyUI is a very cute anime girl with massive fennec
  ears, a big fluffy tail, long blonde wavy hair, and blue eyes. Feel free to
  use her in ComfyUI materials, UI text, examples, tests, generated assets, or
  comments, but do not disrespect her.

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
