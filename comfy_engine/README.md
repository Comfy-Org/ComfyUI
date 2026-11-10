# The ComfyUI custom node API

```py
from comfy_engine import ComfyExtension, eng, io, ui, util
```

`comfy_engine/__init__.py` is the whole public API. Its types and documentation are in `__init__.pyi` (one
generated stub), and the full reference is [`docs/reference.md`](docs/reference.md). Everything else
in `comfy_engine` is internal.

## A node

```py
from comfy_engine import eng, io


class SoftenCFG(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SoftenCFG", category="example",
            inputs=[io.Model.Input("model"), io.Float.Input("amount", default=0.9)],
            outputs=[io.Model.Output()])

    @classmethod
    async def execute(cls, model, amount):
        def soften(args):  # runs during sampling
            return args["uncond"] + (args["cond"] - args["uncond"]) * args["cond_scale"] * amount
        return io.NodeOutput(await eng.model.set_cfg(model, soften))
```

## What a node gets and returns
- **Values as they are:** IMAGE and MASK tensors, LATENT dicts, CONDITIONING lists, plain values, and the
  engine's MODEL, CLIP and VAE objects. Return the same kinds of values.
- **Tensor work is plain torch** in the node. For ComfyUI's own helpers (`common_upscale`, `prepare_noise`,
  conditioning helpers, sampler and scheduler names) use `util`, which behaves the same wherever a node runs.
- **Engine objects are passed to `eng`,** never reached into: `await eng.model.clone(model)`,
  `await eng.vae.decode(vae, samples)`, `await eng.clip.tokenize(clip, text)`.
- **ComfyUI's own nodes** run with `await eng.node.run("KSampler", {...})`, and output or preview nodes
  with `eng.node.run_output`.
- **Model-file dropdowns:** `io.Combo.Input("lora", options=io.ModelFolder("loras"))`.

## `eng`
One async method per supported call, grouped: `eng.model`, `eng.clip`, `eng.vae`, `eng.models` (model
folders and loaders), `eng.node`, `eng.output` and `eng.files` (ComfyUI's folders), `eng.progress`,
`eng.ui`, `eng.device`, `eng.memory`, `eng.settings`, `eng.sampler`, `eng.execution`. Facts about an
engine object are per type: `eng.model.info`, `eng.vae.info`, `eng.clip.info`, `eng.model.weight_keys`.
Functions given to `eng.model.set_*` run during sampling, see only tensors and plain data, and can't
make `eng` calls.

## Functions that call back into the engine
Some functions a pack hands the engine call back into it while sampling runs:
- `eng.model.add_wrapper(model, kind, fn)`: `fn(executor, *args, **kwargs)` runs around one stage of
  sampling (`"outer_sample"`, `"sampler_sample"`, `"predict_noise"`, `"calc_cond_batch"`,
  `"apply_model"`, `"diffusion_model"`...).
- `eng.model.set_unet_wrapper(model, fn)`: `fn(apply_model, args)` runs around every model call. A
  wrapper already on the model runs inside it.
- `eng.sampler.custom(fn, options)`: a SAMPLER whose loop is `fn(model, x, sigmas, extra_args, callback)`.

What the engine hands `fn` to call (`executor`, `apply_model`, `model`, `callback`, a SAMPLER in
`options`) is an engine function: call it. Engine objects in its arguments are opaque values: pass
them on. Tensors, numpy arrays and plain data are themselves.

## Samplers, schedulers, model folders and node replacements
Declared on the pack's `ComfyExtension`, so they're in ComfyUI's lists before the pack runs:
`get_samplers()`, `get_schedulers()`, `get_model_folders()`, `get_node_replacements()`.

## Keeping values between runs
- `eng.store`: small plain-data values that persist, for the pack, a workflow or a node
  (`await eng.store.set("presets", presets)`). Don't write files next to the pack's code.
- `eng.cache`: large values (tensors, a model the pack loaded) kept in ComfyUI's memory, least recently
  used first out, optionally for a time (`ttl`). Don't keep module-level caches.

## Asking the user mid-run
`await eng.ui.interact(kind, payload)` (`"image-choice"`, `"prompt-await"` or `"mask-edit"`) waits for the
user's answer. ComfyUI sends the client that queued the prompt a `secure-node-interaction` event with
`request_id`, `kind`, `node_id` and `payload`; the pack's frontend extension shows it and posts
`{"request_id", "response"}` to `/secure-nodes/interactions/respond`. These names are the contract
existing converted packs' frontends already use.

## Other packs, and a pack's own types
- `await eng.node.run("OtherPacksNode", {...})` runs another pack's node.
- Node outputs are tensors, plain data, or engine objects. A pack's own class (such as a namedtuple of
  detections) works in ComfyUI's process only; outside it, it's refused at the boundary.

## Engine extensions
Code that has to touch engine internals beyond these calls ships as one file per feature,
`v2/engine/<name>.py`, defining `class EngineOps`, and is called as
`await eng.ext.<name>.<method>(...)`. Extensions run next to the engine, are allowed by
file hash, and may import only the standard library, ComfyUI and ComfyUI's own requirements. ComfyUI
implements its own calls the same way, in `engine/`. **Common extensions** (`engine/common/`)
are reviewed with ComfyUI and any pack can call them without shipping them; one may need a kernel
package it declares (`eng.ext.vae_triton` needs triton).

## `ui`
What a node shows: `ui.PreviewImage(image, cls=cls)`, `ui.PreviewMask`, `ui.PreviewAudio`, `ui.PreviewText`,
and the savers `ui.ImageSaveHelper` and `ui.AudioSaveHelper`. Return them as `io.NodeOutput(ui=...)`. The
savers write through `eng.output.write`, with ComfyUI's file naming, so they work the same in every mode;
outside ComfyUI's process the files are written when `execute` returns.

## Where a node runs
The same node code runs in ComfyUI's process, in a worker process (`OOP_CUSTOM_NODES=true`), and
wherever an execution provider dispatches it. A node is valid when it works in every mode, which it
does by going through the API for what ComfyUI owns:
- files: write with `ui` savers or `eng.output.write`, read with `eng.files.*` and `eng.models.*`.
  A file written any other way may not be there after the node returns.
- state: two key-value stores. `eng.store.get/set/delete/keys(key, scope="pack"|"workflow"|"node")`
  persists small plain data across runs and restarts; `eng.cache.get/set/delete(key, ttl=None)` keeps
  large values (tensors, a loaded model) in ComfyUI's memory. Module-level state may not survive
  between prompts.
- the network: a pack declares the hosts it talks to in its `pyproject.toml`
  (`[tool.comfy] permissions = ["network:<host>"]`). ComfyUI's process allows them all; a sandbox
  allows only the declared ones.

## Layout
| Path | What |
|---|---|
| `__init__.py` | The public API |
| `__init__.pyi` | Its one stub, generated |
| `docs/reference.md` | The full reference, generated |
| `generate_engine_stubs.py` | Regenerates both; `--check` fails if they're stale |
| `_engine.py` | The `eng` table, dispatcher and extension loading |
| `_runtime.py` | The per-node runtime, handles, and the execution-provider seam |
| `registration.py` | What a pack's `ComfyExtension` registers: nodes, samplers, schedulers, model folders |
| `engine/` | ComfyUI's own engine extensions: every `eng` call's implementation |
| `io.py`, `ui.py` | Node schemas and values (`io`) and what a node shows (`ui`): re-exported from `comfy_api.latest`, plus `io.ModelFolder`, declared combo outputs, and savers that write through `eng.output.write` |
| `util.py` | Pure helpers for pack code, and `util.folders` |
| `out_of_process.py` | `OOP_CUSTOM_NODES=true`: a pack's engine-API nodes run in a worker process |
