# The ComfyUI custom node API (V4)

```py
from comfy_api.v4 import ComfyExtension, eng, io, ui, util
```

V4 is how a custom node pack talks to ComfyUI. It keeps the node API ComfyUI already has, where a node is an
`io.ComfyNode` that declares its inputs and outputs with `io.Schema`, and adds the part packs never had: a
supported way into the engine.

For years, packs reached into ComfyUI's internals to get their work done: `ModelPatcher` methods,
`comfy.samplers`, `model_management`, `folder_paths`, `PromptServer`. Many patched those modules outright.
That worked until ComfyUI changed, and it meant a pack could only ever run inside ComfyUI's own process, sharing
its Python environment and taking it down when it crashed. In the ComfyUI registry, 524 of about 6,100 packs
patch ComfyUI's own modules or classes.

V4 replaces that with `eng`: one documented, async call for each thing packs need from the engine, such as
`eng.model.clone`, `eng.vae.decode`, `eng.model.set_cfg`, `eng.node.run` or `eng.models.load_checkpoint`.
Engine objects are passed to `eng` and never reached into. Changes to a model are made on a copy. Nothing is
patched globally. That gives a pack three things:

- **A reliable, supported way into the engine.** The `eng` calls are the contract, so a pack keeps working when
  ComfyUI's internals change.
- **A place for what a pack owns.** Calls for a pack's own data and where it goes: saved preferences, cached
  values, output and input files, model folders, and what the pack adds to ComfyUI's lists.
- **Packs can run outside ComfyUI's process.** Because a node reaches the engine only through `eng`, pack
  isolation (`COMFY_PACK_ISOLATION=true`) can run each pack's nodes in the pack's own process. A crash doesn't
  take ComfyUI down, and running out of process is what lets a pack have its own Python environment, with
  dependencies independent of core ComfyUI's. Nodes can optionally run securely under pack isolation.

The same node code runs in ComfyUI's process and in the pack's own.

## Read this first

| If you want to | Read |
|---|---|
| Build a first pack, end to end | [Getting started](docs/getting-started.md) |
| Move an existing pack to V4 | [Converting a pack](docs/converting.md): the converter does most of it |
| See every `eng` call and its arguments | [The reference](docs/reference.md), generated from the API |

## The guide

| Topic | What it covers |
|---|---|
| [Getting started](docs/getting-started.md) | A pack from scratch: layout, entry point, a first node, running and testing it |
| [Nodes](docs/nodes.md) | Declaring a node: the schema, input and output types, widgets, lists, lazy inputs, validation, caching |
| [The engine](docs/engine.md) | `eng`: models, CLIP, VAE, loading model files, running ComfyUI's nodes, devices, memory, settings |
| [Sampling](docs/sampling.md) | Changing how a model samples: CFG and attention hooks, block patches, wrappers, custom samplers and schedulers |
| [Files and state](docs/files-and-state.md) | Saving output, reading files and model files, model folders, keeping values between runs |
| [UI](docs/ui.md) | Previews, progress, messages to the frontend, and asking the user mid-run |
| [Packs](docs/packs.md) | Pack layout, `ComfyExtension`, what a pack declares, routes, node replacements |
| [Pack isolation](docs/pack-isolation.md) | How nodes run in their own process, what changes, and how to test |
| [Engine extensions](docs/engine-extensions.md) | Shipping code that runs next to the engine, for what `eng` doesn't cover |
| [Converting a pack](docs/converting.md) | The converter: running it, its results, and finishing a conversion |
| [Migrating a pack](docs/migrating.md) | Each thing V1 packs did, monkey-patching included, and how V4 does it |
| [The reference](docs/reference.md) | Every `eng` call, generated from the API |

## What's new in V4

V3 defined how a node is declared. V4 defines how a node reaches the engine, and what that gives a pack:

1. **A reliable, supported way into the engine.** `eng` has one async method per supported engine call,
   grouped by subject: `eng.model`, `eng.clip`, `eng.vae`, `eng.models`, `eng.sampler`, `eng.node`,
   `eng.device`, `eng.memory`, `eng.settings`, `eng.execution`, `eng.progress`, `eng.ui`. `util` has the helpers
   packs used to import from `comfy.utils` and `nodes`: `load_torch_file`, `common_upscale`, `tiled_scale`,
   sampler and scheduler names, and `util.folders`. What's supported is what's documented: the reference is
   generated from the API.
2. **A place for what a pack owns.** `eng.store` (saved preferences, for the pack, a workflow or a node) and
   `eng.cache` (large values kept in memory) for state, `eng.output` and `eng.files` for files,
   `io.ModelFolder` for model dropdowns, and `ComfyExtension` registrations for the samplers, schedulers, model
   folders and node replacements a pack adds.
3. **Packs can run outside ComfyUI's process.** Pack isolation runs a pack's nodes in the pack's own process.
   That's also what lets a pack have its own Python environment, with dependencies independent of core
   ComfyUI's.
4. **Sampling through the API.** What packs did by patching `ModelPatcher` and `comfy.samplers` is a call:
   CFG and attention hooks, block patches, wrappers around sampling stages, custom samplers.
5. **ComfyUI's own nodes from pack code.** `eng.node.run("KSampler", {...})` runs any registered node and
   returns its outputs.
6. **Declarations.** A pack declares its dependencies, the packs it depends on, model weights, permissions and
   routes in `pyproject.toml`, so ComfyUI can prepare for it without running it.
7. **Engine extensions** for code that has to run next to the engine.
8. **Side by side with V1.** A pack keeps its V1 code and adds a `v2/` entry point; ComfyUI uses `v2/` when
   it's there.

A node opts in with `ENGINE_API = True`. Packs that import `comfy_api.latest`, `comfy_api.v0_0_1`,
`comfy_api.v0_0_2` or `comfy_api.index` keep working.
