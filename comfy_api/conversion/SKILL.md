---
name: convert-nodes-v4
description: >-
  Finish converting a ComfyUI custom node pack to the V4 custom node API (comfy_api.v4) after the
  automatic converter has run, so its nodes work in ComfyUI's process and under pack isolation. Use for
  V4 conversion, `# V2-MANUAL` items, ENGINE_API nodes, eng calls, pack declarations, or a node that
  fails under pack isolation.
---

# Converting a node pack to the V4 API

V4 adds one thing to the node API ComfyUI already has: a supported way into the engine. A V4 node is
an ordinary `io.ComfyNode` that sets `ENGINE_API = True` and reaches the engine only through `eng`.
Because it never reaches into ComfyUI's internals, the same node runs in ComfyUI's process, in its own
process under pack isolation, and securely under an execution provider.

Most of a conversion is automatic. Your job is the part the converter could not do, done so that the
pack's author would accept the diff upstream: keep the pack's algorithm and change only how it reaches
the engine.

## Read the API, not this file's summary

| What | Where |
|---|---|
| The public API, with every type and docstring | `comfy_api/v4.pyi` |
| Every `eng` call, its arguments and what it returns | `comfy_api/docs/reference.md` |
| How a V4 node works, the boundary rules, pack declarations | `comfy_api/docs/` (`engine.md`, `pack-isolation.md`, `packs.md`) |
| The converter and its tools | `comfy_api/conversion/` |

If a call is not in `v4.pyi`, it is not API. Don't invent one.

## Two targets

- **Every pack:** works in ComfyUI's process, and under pack isolation (`COMFY_PACK_ISOLATION=true`), where
  the pack's engine-API nodes run in the pack's own process. Most conversion problems are about crossing
  that process boundary, so test both.
- **Secure (Comfy's cloud):** the same code under an execution provider, sandboxed, with only the
  permissions the pack declares. The pack's part of this is its declarations in `pyproject.toml`.
  Sandboxing, weight handling and network policy are the platform's job, not edits to the pack.

## The workflow

1. **Run the converter** on the pack. It writes a `v2/` folder next to the pack's own code: the pack's
   files with known engine calls rewritten, a generated `_v2_nodes.py` for V1 nodes, the shim
   `_v2_engine.py`, and `V2_CONVERSION.md`, which lists what is left by file and line. Those lines
   carry `# V2-MANUAL` comments.
2. **Finish each `# V2-MANUAL` item** (below). Change nothing else.
3. **Verify** the converted pack against the original, in-process and under pack isolation.
4. **Report** in `AGENT_NOTES.md`: for each item, fixed or not, and why.

The pack's V1 code stays as it is; ComfyUI loads `v2/__init__.py` when it is there. Keep every node id,
display name, category, input, output and output order exactly as the original has them, so existing
workflows keep working.

## 0. Ask what the node is for before asking how it works

Do this first, and always before calling a node unconvertible. A node's body is one implementation of
its purpose, written when the author could reach into the whole process. Convert the purpose.

1. State the purpose in one sentence without naming any type it uses.
2. Ask what data that purpose actually needs. Usually far less than the body touches.
3. Only then read how the original does it.

| The body reaches for | The purpose usually is | V4 |
|---|---|---|
| a model's internals | that model's work: patch, decode, encode, sample | an `eng.model` / `eng.vae` / `eng.clip` call |
| `nodes.NODE_CLASS_MAPPINGS[...]()` | running another node | `eng.node.run` / `eng.node.run_output` |
| `folder_paths` | a file in one of ComfyUI's folders | `util.folders`, `eng.files.*`, `eng.models.*` |
| a file written next to the pack | a saved result or a preference | `eng.output.write` / `ui` savers, or `eng.store` |
| a module-level cache | a value kept between runs | `eng.store` (small) or `eng.cache` (large) |
| `PromptServer.instance.send_sync` | telling the user something | `eng.ui.send`, `eng.ui.progress_text` |
| `comfy.utils.ProgressBar` | progress | `eng.progress.update` |
| `model_management` | device or memory facts and actions | `eng.device.info`, `eng.memory.*` |
| a live object mutated across calls | passing a value between two nodes | plain data, or the pack's own object (it stays in the pack's process) |

## 1. Translate what the converter left

Replace each internal with the V4 call or `util` helper that does the same thing. Common ones:

| V1 | V4 |
|---|---|
| `model.clone()` | `await eng.model.clone(model)` |
| `m.set_model_sampler_cfg_function(fn)` and the other `set_model_*` patches | `await eng.model.set_cfg(m, fn)` and the matching `eng.model.set_*` |
| `comfy.sd.load_lora_for_models(...)` | `await eng.model.load_lora(model, clip, lora, ...)` |
| `vae.decode(x)`, `vae.encode(x)` | `await eng.vae.decode(vae, x)`, `await eng.vae.encode(vae, x)` |
| `clip.tokenize(t)`, `clip.encode_from_tokens(...)` | `await eng.clip.tokenize(clip, t)`, `await eng.clip.encode_from_tokens(...)` |
| `comfy.utils.load_torch_file(p)` | `util.load_torch_file(p)`, or `await eng.models.read(folder, name)` |
| `comfy.utils.common_upscale`, `comfy.sample.prepare_noise` | `util.common_upscale`, `util.prepare_noise` |
| `comfy.samplers.KSampler.SAMPLERS` / `SCHEDULERS` | `util.SAMPLER_NAMES` / `util.SCHEDULER_NAMES` |
| `nodes.MAX_RESOLUTION` | `io.MAX_RESOLUTION` (also `util.MAX_RESOLUTION`) |
| `folder_paths.get_filename_list("loras")` as dropdown options | `io.Combo.Input("lora", options=io.ModelFolder("loras"))` |
| `model_management.processing_interrupted()` | `await eng.execution.is_interrupted()` |

In synchronous code the converter wraps engine calls as `_v2e.sync(eng....)`; keep that form. In an
`async def`, use `await`.

If nothing in `v4.pyi` fits, leave the line as it is and name the capability the API would need in
`AGENT_NOTES.md` (for example "eng.sampler.calculate_sigmas(model, scheduler, steps)").

## 2. The boundary rules

Every rule here comes from a conversion that failed.

**Engine objects are handles.** A MODEL, CLIP or VAE is passed to `eng` and never reached into: no
attribute access, no `hasattr`/`getattr` (always false or failing on a handle), and no calling it.
Read facts with `eng.model.info`, `eng.vae.info`, `eng.clip.info`, `eng.model.weight_keys`.

**Functions given to sampling see only tensors and plain data.** A function passed to an
`eng.model.set_*` call runs during sampling, stays in the pack's process and can't make `eng` calls.
`args["model"]` is not there. Capture what it needs (a sigma, a percent) before handing it over.

**`define_schema` runs in ComfyUI's process at load.** No `eng` calls and no imports with side effects
there. Build the schema from literals and published constants; use `io.ModelFolder` for model
dropdowns.

**Optional inputs arrive as `None`,** often in the ordinary case. Guard them as the original does.

**Lists:** an `INPUT_IS_LIST` node gets every input as a list, hidden inputs too.

**Match the original's call.** Upstream methods are often instance methods with an argument order that
differs from the input order. Read the `def` line.

**A pack's own objects** (a namedtuple of detections, a custom class) stay in the pack's process and
come back to the pack's next node. Another pack's node can't use them.

**Files:** write with `ui` savers or `eng.output.write`, read with `eng.files.*` and `eng.models.*`.
A file written any other way may be gone after the node returns.

**State:** `eng.store` for small values that persist, `eng.cache` for large ones. Module-level state may
not survive between prompts.

**Settings:** a pack can't change ComfyUI's process-wide PyTorch settings (`torch.backends.*` flags).

**Workflow metadata:** editing `extra_pnginfo` (for example writing into a node's `widgets_values`, as
ShowText does) works in ComfyUI's process but does not reach ComfyUI under pack isolation. Note it in
`AGENT_NOTES.md`.

**Imports:** a pack that relies on a submodule something else imported (`matplotlib.pyplot` without
importing it) fails in its own process. Import what you use. A `ModuleNotFoundError` for a package the
pack never declared is a missing declaration, not a conversion defect: declare it.

## 3. Declarations

A pack declares what it needs in `pyproject.toml`, so ComfyUI and a provider can prepare without
running it. See "Declarations" in `comfy_api/docs/packs.md` for the exact form.

- `[project] dependencies`: pip requirements (or `requirements.txt`).
- `[tool.comfy] depends_on`: other packs whose nodes it runs.
- `[tool.comfy] weights`: model files it reads, with `sha256` where known.
- `[tool.comfy] permissions`: for example `network:<host>` for each host it talks to.
- `[[tool.comfy.routes]]`: its HTTP endpoints. A handler takes plain request data and returns
  `{"status", "body"}`; it is served at `/ext/<namespace>/<path>`. Update the pack's frontend calls to
  that path.

## 4. Engine extensions

Code that must touch engine internals beyond `eng` ships as `v2/engine/<name>.py` with
`class EngineOps`, called as `await eng.ext.<name>.<method>(...)`. Only allowlisted major packs may have
them. Don't write one unless you were told the pack is allowlisted. Keep it small and stable: no network,
no subprocesses, no pickle, no files outside declared folders, and no patching of ComfyUI. Every change
is a new file hash for review.

## 5. Verify

A conversion is done only when the converted node matches the original on the same inputs.

- Run the converter's verifier: original against converted, in fresh processes. Then run again with
  `COMFY_PACK_ISOLATION=true`.
- **Vary inputs** between runs; ComfyUI caches by prompt, and a cached run never executes the node.
- **Use inputs the node must change,** and check the output differs from the input where it should.
  Sharpening a solid colour passes a broken node.
- **Include a negative control** that must fail, so the comparison is shown to catch a difference.
- "Ran without error" is not "matches". Report which one you have.

## 6. Stop and report instead of improvising

Report, don't work around, when you hit:

- a licence or provenance problem;
- a node id ComfyUI already owns;
- a need for secrets or API keys (there is no V4 call for them yet);
- pickle weights;
- engine internals with no `eng` call, in a pack without engine-extension rights;
- an import with side effects (network at import, writing into ComfyUI's folders).

One probe to classify a residual as the platform's problem is enough. Write the capability it needs
and move on.

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `the converted code made an engine call outside a running node` | An `eng` call at import or in `define_schema` | Move it into `execute`, or use a published constant or `io.ModelFolder` |
| `module ... has no attribute` on `util.<name>` | The pack's own `util` module shadows the API's `util` | Import the API's as another name (the converter does this as `_v2util`) |
| `hasattr(model, ...)` always false | A MODEL is a handle under pack isolation | Use `eng.model.info` or the matching `eng` call |
| A sampling function fails reading `args["model"]` | The model is not passed to pack code | Capture what you need before handing the function over |
| `'NoneType' object is not subscriptable` on a hidden input | An `INPUT_IS_LIST` node gets hidden inputs as lists | Index them as the original does |
| A file the node wrote is missing | It wasn't written through `eng.output.write` or a `ui` saver | Use those |
| ShowText-style widget values don't stick under pack isolation | Edits to `extra_pnginfo` stay in the pack's process | Known limit; note it |
| `ModuleNotFoundError` | A dependency the pack doesn't declare | Declare it in `pyproject.toml` |
| Output differs from the original | Argument order, or a changed default | Read the original's `def` line and defaults |

## Report

`AGENT_NOTES.md` in the pack's `v2/` folder: for each `# V2-MANUAL` item, fixed or not and why; for each
node, matches / differs (with the difference) / not run (with the reason); and every capability the API
was missing. A conversion without a passing comparison is not done.
