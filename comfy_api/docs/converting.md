# Converting a pack

You don't have to move an existing pack to V4 by hand. The converter, in `comfy_api/conversion/`, takes a V1
pack, writes its V4 version, checks each node against the original, and tells you what's left. For most
packs, what's left is nothing or a few lines.

```
python comfy_api/conversion/convert_pack.py path/to/my-pack --out converted
```

This page covers running the converter, reading what it reports, and finishing a conversion yourself or with
an agent. [Migrating a pack](migrating.md) explains each change it makes, and how to write the same thing in
new code.

## Before you run it

- **Run it from a ComfyUI checkout** with V4. It starts ComfyUI on the CPU, in fresh processes, to read your
  pack and to check the result; no GPU is needed.
- **Install your pack's dependencies** in that environment. The converter imports your pack to see its
  nodes as ComfyUI lists them.
- **Your pack isn't changed.** Everything is written under `--out`.

The first run also records ComfyUI's own nodes, into `converted/core_nodes.json`; later runs reuse it.

## What it does

1. **Describe.** It imports your pack in a fresh ComfyUI process and records each node exactly as ComfyUI
   lists it: id, display name, category, inputs, outputs, and what your code uses from ComfyUI. It doesn't
   run your nodes.
2. **Convert.** It writes `converted/my-pack/`: your pack's files with the engine uses it knows rewritten,
   and a `v2/` folder with the V4 entry point.
3. **Verify.** It loads the converted pack, checks its declarations and every node's schema against the
   original's, then runs each node, original and converted, on the same made-up inputs and compares the
   outputs.
4. **Review.** It notes what a person must still convert or test, on the lines themselves and in
   `v2/V2_CONVERSION.md`.

## What it writes

```
converted/
    my-pack/
        ...                     your pack's files, with known engine uses rewritten
        v2/
            __init__.py         the V4 entry point (comfy_entrypoint)
            _v2_nodes.py        an io.ComfyNode for each V1 node, with the node's exact schema
            _v2_engine.py       the shim your existing code calls eng through
            _v2_routes.py       your pack's routes, as declared routes (if it has any)
            pyproject.toml      the declarations: dependencies, routes, model folders
            V2_CONVERSION.md    what's left, by file and line
    my-pack.work/
        conversion.json         every report and flag, in detail
```

What it rewrites:

| Your code used | Converted to |
|---|---|
| ComfyUI's helpers: `common_upscale`, `prepare_noise`, `load_torch_file`, sampler and scheduler names... | `util` |
| `folder_paths` | `util.folders` |
| Devices, memory, progress bars, interrupts, `PromptServer.send_sync` | `eng`, through the shim |
| Methods on MODEL, CLIP and VAE inputs (`clone`, `set_model_*`, `add_patches`...) | The matching `eng` call |
| `load_checkpoint_guess_config`, `load_lora_for_models` | `eng.models.load_checkpoint`, `eng.model.load_lora` |
| ComfyUI's nodes called directly (`nodes.KSampler().sample(...)`) | `eng.node.run` |
| Model-file and sampler dropdowns | `io.ModelFolder`, `util.SAMPLER_NAMES`, `util.SCHEDULER_NAMES` |
| Routes on `PromptServer.instance.routes` | Declared routes, and your frontend's URLs rewritten to match |

Each V1 node keeps its exact id, display name, category, inputs and outputs, so saved workflows keep
working.

**Converted code stays synchronous.** Your nodes' original methods run unchanged on a worker thread, and the
shim (`_v2e` in the converted files) hands each engine call to `eng` and waits. That's why converted code
still looks like V1: `_v2e.ProgressBar(total)`, `_v2e.load_checkpoint_guess_config(path)`. It's correct
V4, and it runs under pack isolation. When you next work on a node by hand, write it the
[Migrating a pack](migrating.md) way: `async def execute`, and `eng` directly.

**Your V1 code stays.** ComfyUI loads `v2/__init__.py` when it has V4, and your pack's top-level
`__init__.py` otherwise, so one install serves both.

## The result

The converter prints one of four results:

| Result | Meaning | What to do |
|---|---|---|
| `verified` | Converted with reliable rewrites only, and every node checked against the original | Run your workflows both ways (below), then publish |
| `needs-testing` | Converted, but with rewrites the checks can't fully prove, or nodes the verifier couldn't run | Test the `# V2-TEST` lines' nodes yourself |
| `needs-manual` | Some lines need a person: engine uses left, file writes, frontend hooks, routes that couldn't move | Finish the `# V2-MANUAL` lines, then test |
| `none` | No V4 pack: the pack couldn't be imported, has no nodes, or the result isn't valid | `conversion.json` says why |

A node the verifier couldn't run (it needs a real model, a network service, a file the user picks) is
reported as untested, never as passing.

## Finishing a conversion

`v2/V2_CONVERSION.md` lists each item: the file and line, what's there, and why the converter couldn't
finish it. The same lines carry a comment:

```py
blocks = model.model.diffusion_model.input_blocks   # V2-MANUAL: engine object attribute
def soften(args):                                    # V2-TEST: sampling-time function
```

- **`# V2-MANUAL`** is something to change: an engine object's attribute, a sampling function that reads
  one, a file written by path, a ComfyUI setting changed, network access to declare, a node whose output
  differs. [Migrating a pack](migrating.md) has the V4 way for each common case.
- **`# V2-TEST`** was converted, but the check couldn't prove it: a sampling-time function, a ComfyUI node
  called, output that varies run to run, a model folder added. Run that node and compare.

Change only those lines. Keep your algorithm, and every node's id, inputs and outputs, as they are. Then
re-run the verifier on the result:

```
python comfy_api/conversion/verify.py path/to/my-pack converted/my-pack \
    converted/my-pack.work/describe.json converted/my-pack.work/verify.json
```

A node that needs ComfyUI's internals directly can leave `ENGINE_API` off and run in ComfyUI's process:
see [Migrating a pack](migrating.md#comfyuis-internals-directly).

### With an agent

`comfy_api/conversion/SKILL.md` is a skill for a coding agent finishing a conversion: it fixes the
`# V2-MANUAL` items with the API, changes nothing else, writes what it did and didn't fix to
`AGENT_NOTES.md`, and re-runs the verifier. The agent works from `v4.pyi` and [the reference](reference.md),
not from memory, and doesn't invent calls. Review its diff as you would a contributor's.

## Testing and publishing

A node is done when its outputs match the original's, not just when it runs:

1. Run your workflows normally, and compare with V1.
2. Run them with `COMFY_PACK_ISOLATION=true`. They should give the same results. If not, see
   [Pack isolation](pack-isolation.md#what-changes-for-a-node).
3. Vary inputs between runs (ComfyUI caches by prompt), and use inputs the node actually changes.

Then check the declarations the converter wrote in `v2/pyproject.toml`: a `name` (permanent, since your
routes' URLs depend on it), every package you import, the weights you read and the hosts you talk to. See
[Packs](packs.md#declarations).

The converted folder is your pack. Commit it, `v2/` included, and publish as usual.

## The converter's pieces

| File | Does |
|---|---|
| `convert_pack.py` | The one command: describe, convert, verify, review |
| `describe.py` | Imports the pack in a fresh ComfyUI process and records its nodes |
| `convert.py` | The rewrites, the schema writer, the shim and the entry point |
| `verify.py` | Checks declarations, schemas, and each node's outputs against the original's |
| `review.py` | Which rewrites are reliable, which to test, which need a person |
| `packload.py` | Imports a pack with a stand-in server |
| `core_nodes.py` | Records ComfyUI's own nodes, for `eng.node.run` |
