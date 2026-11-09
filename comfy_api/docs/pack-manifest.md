# The pack manifest: `comfy-pack.json`

A V2 pack ships one file that tells ComfyUI everything about it without running its code:
`comfy-pack.json`, at the top of the pack's V2 package (next to its `__init__.py`). ComfyUI reads
it at startup and registers the pack's nodes from it, so installing a V2 pack never runs the pack.
The pack runs only when a render uses one of its nodes or the frontend calls one of its endpoints.

Authors don't write it by hand. They declare what only they know in `pyproject.toml`, and
`scripts/pack.sh manifest <pack>` writes the rest from the pack's code, once, at publish time.

## Example

```json
{
  "format": "comfy-pack/1",
  "name": "comfyui-wd14-tagger",
  "display_name": "WD14 Tagger",
  "version": "1.4.0",
  "publisher": "pythongosssss",
  "description": "Image tagging with WD14 models",
  "license": "MIT",
  "repository": "https://github.com/pythongosssss/ComfyUI-WD14-Tagger",
  "api": "2.1",

  "weights": [
    {"folder": "onnx", "filename": "wd-vit-tagger-v3.onnx", "repo_id": "SmilingWolf/wd-vit-tagger-v3",
     "revision": "7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f",
     "sha256": "35f23693620b668f4d53fd3c62bf65e40af739bc52c7eb0fbc49258b58d065b6"}
  ],
  "requirements": ["onnxruntime>=1.17", "huggingface_hub"],
  "depends_on": [{"pack": "comfyui-videohelpersuite", "version": ">=1.5"}],
  "permissions": ["network:huggingface.co"],
  "requested_permissions": ["network:huggingface.co"],

  "samplers": [], "schedulers": [], "model_folders": {"onnx": ["onnx"]}, "node_replacements": [],
  "engine_extensions": [{"file": "tagger_ops.py", "sha256": "..."}],
  "web_directory": "js",
  "routes": [{"method": "get", "path": "/tags", "module": "routes", "handler": "list_tags"}],
  "nodes": {
    "WD14Tagger": {"display_name": "WD14 Tagger", "module": "tagger", "class": "WD14Tagger", "async": true,
                   "methods": [], "schema": {"attrs": {}, "inputs": [], "outputs": [], "hidden": []}}
  }
}
```

## Fields

### Who the pack is

| Field | Required | Meaning |
|---|---|---|
| `format` | yes | `"comfy-pack/1"`. |
| `name` | yes | The pack's permanent ID: lowercase letters, digits, `.`, `_`, `-`. Its endpoint namespace (`/ext/<namespace>/`) and its environment are derived from it. |
| `version` | yes | The pack's version. |
| `display_name`, `publisher`, `description`, `license`, `repository` | no | As the registry shows them. |
| `api` | no | The `comfy_api` version the pack is written against, such as `"2.1"`. |

### What has to be in place before it runs

| Field | Meaning |
|---|---|
| `weights` | Every model file the pack uses: `folder` (a ComfyUI model folder, such as `loras` or `onnx`), `filename` (inside it; it can't leave the folder), and optionally `repo_id` and `revision` (where to download it from on Hugging Face) and `sha256`. **All of them are present and verified before the pack runs.** A missing file is downloaded when the manifest says from where; a file whose sha256 doesn't match stops the pack, with the reason. |
| `requirements` | pip requirement specifiers. They're installed into the pack's own environment, never into ComfyUI's (see [pack isolation](../_impl/isolation/README.md)). `torch`, `torchvision` and `torchaudio` stay at ComfyUI's versions. |
| `depends_on` | Other packs this one needs: `{"pack": <name>, "version": <specifier>}`. ComfyUI checks they're installed, in range; a pack whose dependency is missing doesn't run, and says what it needs. |
| `permissions` | What the pack has been **granted** beyond the plain `eng` API, such as `network:<host>`. Review grants them: `pack.sh manifest` never writes this field, and keeps it when it regenerates the file. A launcher that runs packs in a sandbox enforces it (`COMFY_PACK_PERMISSIONS`); in ComfyUI's own processes it's recorded, not enforced. |
| `requested_permissions` | What the author asks for (`[tool.comfy] permissions`), for review to grant. |

### What it has

| Field | Meaning |
|---|---|
| `nodes` | Every node: `display_name`, `module` and `class` (where its code is), `async`, `methods` (which of `validate_inputs`, `fingerprint_inputs`, `check_lazy_status` it defines), and `schema`: its `io.Schema` as JSON. ComfyUI registers each node from its schema at startup. |
| `engine_extensions` | The engine-extension files the pack ships (`v2/engine/*.py`) and their sha256, checked against ComfyUI's allowed list. |
| `samplers`, `schedulers` | Names the pack adds to ComfyUI's sampler and scheduler lists (`ComfyExtension.get_samplers` / `get_schedulers`). They're listed at startup; using one runs the pack's function in its process. |
| `model_folders` | `{folder: [extensions] or null}`: model folders the pack's files live in, registered at startup. |
| `node_replacements` | What replaces old node ids in workflows (`io.NodeReplace`, as plain data). |
| `web_directory` | The pack's frontend files, served from disk. |
| `routes` | The pack's endpoints, under `/ext/<namespace>/`: `method`, `path`, and the `module` and `handler` that answer them. |

## Writing it

In `pyproject.toml`:

```toml
[project]
name = "comfyui-wd14-tagger"
version = "1.4.0"
description = "Image tagging with WD14 models"
dependencies = ["onnxruntime>=1.17", "huggingface_hub"]   # -> requirements

[tool.comfy]
PublisherId = "pythongosssss"
DisplayName = "WD14 Tagger"
depends_on = [{pack = "comfyui-videohelpersuite", version = ">=1.5"}]
weights = [{folder = "onnx", filename = "wd-vit-tagger-v3.onnx", repo_id = "SmilingWolf/wd-vit-tagger-v3",
            revision = "7f6b584d...", sha256 = "35f23693..."}]
permissions = ["network:huggingface.co"]                    # -> requested_permissions

[[tool.comfy.routes]]                                       # -> routes: GET /ext/<namespace>/tags
method = "get"
path = "/tags"
module = "routes"                                           # v2/routes.py
handler = "tags"
```

A route's handler takes the request as plain data and answers with JSON or text (at most 1 MB):

```python
def tags(request):   # or async def
    # request: {"method": "get", "path": "/tags", "query": {"q": "cat"}, "body": ""}
    return {"status": 200, "body": json.dumps(find_tags(request["query"].get("q"))),
            "content_type": "application/json"}
```

Routes live under `/ext/<namespace>/`, where the namespace is derived from the manifest's `name` (a
readable slug and 8 hex characters of a hash), so packs can't collide and the same pack has the same
URLs wherever it's installed. A route's path is literal: no `{parameters}`; pass values in the query
or body. The frontend calls `/ext/<namespace>/tags` (or `api.fetchApi("/ext/<namespace>/tags")`).
A request starts the pack's process if it isn't running; handlers don't need the engine, and can't
make `eng` calls (those are for running nodes).

Then:

```sh
scripts/pack.sh manifest <pack>   # imports the pack once and writes comfy-pack.json
scripts/pack.sh install <pack>    # makes its environment and fetches its weights
```

Without `[project] dependencies`, the pack's `requirements.txt` is used. Every node must be an
`io.ComfyNode`; `pack.sh manifest` refuses a pack whose nodes aren't.

## Older conversions

Packs converted before this format ship `secure-nodes.json`. ComfyUI reads it as a manifest (its
schemas are the same), taking identity from `pyproject.toml` and requirements from
`requirements.txt`. Run `pack.sh manifest` to replace it.
