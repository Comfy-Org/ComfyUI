# Converting a pack to the V4 API

`convert_pack.py` converts one V1 custom-node pack to the V4 API (`comfy_api.v4`), checks the result
against the original, and says what is left to do by hand:

```sh
python comfy_api/conversion/convert_pack.py path/to/some_pack --out converted
```

It needs this ComfyUI checkout (it runs ComfyUI on the CPU, in fresh processes) and the pack's own
dependencies installed. It writes `converted/some_pack/`:

- the pack's own files, with known engine uses rewritten to the API: ComfyUI helpers to `util`
  (`common_upscale`, `prepare_noise`, `load_torch_file`, sampler and scheduler names...), folders to
  `util.folders`, devices, progress, interrupts and UI events to `eng` through a small generated
  shim (`v2/_v2_engine.py`), MODEL / CLIP / VAE methods on typed node inputs to `eng` rows, and core
  nodes a pack calls directly to `eng.node.run`;
- `v2/_v2_nodes.py`: an `io.ComfyNode` per V1 node with the node's exact schema (model-folder dropdowns
  as `io.ModelFolder`, sampler and scheduler lists as `util` constants), whose `execute` runs the
  original method;
- `v2/__init__.py` (the pack's `comfy_entrypoint`), `v2/_v2_routes.py` (its routes, as declared
  routes) and `v2/pyproject.toml` (its declarations: dependencies, routes, model folders);
- `v2/V2_CONVERSION.md`: what a person or an agent must still convert (`# V2-MANUAL` lines) or test
  (`# V2-TEST` lines), and why.

The pack keeps its original code; ComfyUI loads `v2/__init__.py` when it's there, so the same install
serves V1 users and the V4 API. The converter runs none of the pack's code to convert it; verifying
runs each node on made-up inputs, in ComfyUI's process, and compares with the original.

The result is one of:

| Result | Meaning |
|---|---|
| `verified` | Converted with reliable rewrites only, and every node's behavior checked. |
| `needs-testing` | Converted, with rewrites the checks can't fully prove, or nodes the verifier couldn't run. |
| `needs-manual` | Things to convert by hand (engine uses left, file writes, frontend hooks, routes that aren't declared routes), then test. |
| `none` | No V4 pack: the pack couldn't be described, or the converted pack isn't valid. |

`converted/some_pack.work/conversion.json` has the details: the describe, conversion and verify
reports and every flag.

## Finishing by hand, or with an agent

`SKILL.md` is the skill for an agent (or a person) finishing a conversion: fix the `# V2-MANUAL` items
with the API, change nothing else, write `AGENT_NOTES.md`, re-run `verify.py`. The agent reads
`comfy_api/v4.pyi` and `comfy_api/docs/` for what exists; it must not invent calls, and must not
ship engine extensions unless the pack is allowed them.

The guide for pack authors is [Converting a pack](../docs/converting.md); what each rewrite means,
and how to write the same thing in new code, is in [Migrating a pack](../docs/migrating.md).

## The pieces

| File | Does |
|---|---|
| `convert_pack.py` | The one command: describe, convert, verify, review. |
| `describe.py` | In a fresh ComfyUI process: imports the pack with a stand-in server (`packload.LoadedPack`) and records what ComfyUI lists for each node, plus the static hints the converter needs. |
| `convert.py` | The static converter (`generate(pack_dir, describe, out_dir)`): the rewrite table, the schema writer, the generated shim and entrypoint. |
| `verify.py` | In a fresh process: declarations, each node's schema against the original's, and each node's outputs on seeded inputs against the original's. |
| `review.py` | The detectors and the notes: which rewrites are reliable, which to test, which can't be done automatically. |
| `core_nodes.py` | Writes the table of ComfyUI's own nodes the converter calls (`eng.node.run`); `convert_pack.py` runs it once into `<out>/core_nodes.json`. |
| `packload.py` | `LoadedPack`: a pack imported with a stand-in server, its nodes, routes and web directory. |
