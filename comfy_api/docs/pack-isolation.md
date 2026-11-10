# Pack isolation

A V4 node runs either in ComfyUI's process, like any node, or, with pack isolation
(`COMFY_PACK_ISOLATION=true`), in its pack's own process. Pack isolation keeps a crashing pack from taking
ComfyUI down and keeps packs from interfering with each other. Nodes can optionally be run securely under pack
isolation: see the last section.

The same node code runs in every mode. This page explains what pack isolation does, what changes for a node,
and how to make sure a pack works.

## How it works

With `COMFY_PACK_ISOLATION=true`, ComfyUI starts a worker process for each pack that has `ENGINE_API` nodes
when one of them first runs.

- **The pack is still imported in ComfyUI's process.** Its nodes register, its web directory is served, and
  its engine extensions load there, next to the engine. Only the nodes' own methods run in the worker:
  `execute`, and `check_lazy_status`, `validate_inputs` and `fingerprint_inputs` where a node has them.
- **Nodes without `ENGINE_API`** (V1 nodes, and V3 nodes that don't use `eng`) run in ComfyUI's process as
  usual.
- **The worker gets ComfyUI's setup:** the same Python environment, the same folders (`--output-directory`
  and the rest), the same PyTorch numeric settings, and each call runs in the caller's grad and inference
  mode.
- **On its own, pack isolation is about robustness, not security:** ComfyUI and the worker trust each other.

## What crosses the boundary

When a node runs in its worker, its inputs and outputs cross between the two processes:

| What | How it crosses |
|---|---|
| Tensors, numbers, strings, lists, dicts | Copied |
| MODEL, CLIP, VAE and other engine objects | As handles. The node passes them to `eng` and to other nodes, and they work exactly as in ComfyUI's process. The objects themselves stay in ComfyUI |
| The pack's own objects (a namedtuple of detections, a custom class) | They stay in the worker; the pack's next node gets them back. Another pack's node can't use them |
| Functions given to `eng.model.set_*`, wrappers, custom samplers | They stay in the worker, and ComfyUI calls them during sampling |
| `eng` calls | Sent to ComfyUI, which runs them next to the engine |

That's why the rules in [The engine](engine.md) matter: a node that only passes engine objects to `eng`
works identically, and a node that reaches into one fails.

## What changes for a node

Every item here was found converting real packs and running them isolated.

**Reaching into an engine object fails.** `model.model`, `clip.tokenizer` and `vae.first_stage_model` aren't
there; `hasattr(model, "x")` is always false. Use the `eng` call.

**A sampling function doesn't get the model.** Hooks receive tensors and plain data only. Get what you need in
`execute` and capture it. See [Sampling](sampling.md#how-hooks-work).

**Changes to workflow metadata stay in the pack's process.** A node that edits `extra_pnginfo`, for example
writing text into its own `widgets_values` so the saved workflow shows it, works in ComfyUI's process, but the
edit doesn't reach ComfyUI from a worker. Leave `ENGINE_API` off a node like that, so it runs in ComfyUI's
process.

**Process-wide settings are the worker's.** Setting `torch.backends.*` flags or environment variables changes
the worker, not ComfyUI.

**Module state lives in the worker.** Globals persist while the worker does, which may be shorter than
ComfyUI. Use `eng.store` and `eng.cache`: [Files and state](files-and-state.md).

**Files must go through the API.** A file written by path may be in a place ComfyUI doesn't look. Use the `ui`
savers, `eng.output.write`, and `eng.files`.

**Imports must be explicit.** In ComfyUI's process, a module some other pack imported happens to be loaded
already. A pack that uses `matplotlib.pyplot` after importing only `matplotlib` works there and fails in its
worker. Import what you use.

**Hooks cost a round trip per call.** A function ComfyUI calls on every sampling step is a call into the
worker each time. It's fine for typical hooks; keep heavy work out of them.

## Making sure a pack works

1. Run your workflows normally.
2. Run them again with `COMFY_PACK_ISOLATION=true`. The results should be the same.
3. If something differs or fails, the list above is where to look.

The converter's verifier compares a converted pack's nodes with the original's; run it in both modes. See
[Converting a pack](converting.md).

## Running securely

Nodes can optionally be run securely under pack isolation, through an execution provider
(`COMFY_EXECUTION_PROVIDER`). On top of everything above, the pack then runs in a sandbox, on its own Python
environment built from its declared `dependencies`, reaching only the network hosts declared in its
`permissions` and only the files and model weights the API gives it, and its engine extensions must be on the
allowlist.

A pack that works under pack isolation and declares what it needs is ready for that. Nothing in its code
changes. See [Packs](packs.md#declarations).
