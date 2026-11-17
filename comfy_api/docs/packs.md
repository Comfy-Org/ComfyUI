# Packs

A pack is a folder in `custom_nodes/`: the nodes, an entry point that registers them, and a `pyproject.toml`
that says what the pack needs. This page covers the entry point, everything a pack can register, and its
declarations.

## Layout

```
my-pack/
    __init__.py          V1 code (optional: ComfyUI versions without V4 use it)
    pyproject.toml       the declarations (here, or in v2/)
    v2/
        __init__.py      comfy_entrypoint, and WEB_DIRECTORY if the pack has frontend code
        nodes.py         ...the pack's modules
        routes.py        route handlers, if any
        engine/          engine extensions, if any (allowlisted packs only)
    web/                 frontend extension code, if any
```

ComfyUI loads `v2/__init__.py` when it's there, and the top-level `__init__.py` otherwise. A pack can ship
both, so one install serves ComfyUI versions with and without V4.

## The entry point

```py
# v2/__init__.py
from comfy_api.v4 import ComfyExtension, io
from .nodes import Blur, Sharpen

WEB_DIRECTORY = "../web"         # served to the frontend, relative to this file


class MyPack(ComfyExtension):
    async def on_load(self):
        ...                       # once, when the pack loads

    async def get_node_list(self):
        return [Blur, Sharpen]


async def comfy_entrypoint():
    return MyPack()
```

`get_node_list` is required. The rest of this page is optional.

## Registrations

A `ComfyExtension` can add to ComfyUI's own lists. These are read when the pack loads, so they're in place
before any of the pack's nodes run.

```py
class MyPack(ComfyExtension):
    async def get_samplers(self):
        return {"my_sampler": my_sampler}                 # KSampler's sampler list

    async def get_schedulers(self):
        return {"my_schedule": my_schedule}               # KSampler's scheduler list

    async def get_model_folders(self):
        return {"my_pack_models": ["safetensors"]}        # a model folder; None means ComfyUI's model extensions

    async def get_node_replacements(self):
        return [io.NodeReplace(new_node_id="MyPack_Blur", old_node_id="MyPack_BlurOld")]
```

- **Samplers** are sampling loops, `fn(model, x, sigmas, extra_args, callback=None, disable=None)`;
  **schedulers** are `fn(model_sampling, steps) -> sigmas`. See [Sampling](sampling.md).
- **Model folders** appear wherever ComfyUI lists model folders, and in `io.ModelFolder` dropdowns.
- **Node replacements** let workflows saved with an old node id load with the new one. Use them when you
  rename or merge nodes, rather than keeping old classes around.

In V1, packs did all of this by patching ComfyUI's lists at import time. Registering them instead means
ComfyUI knows about them up front, they can't clash silently with another pack's, and they work when the
pack's code runs in another process.

## Declarations

A pack declares what it needs in `pyproject.toml`, at the pack's root or in `v2/`. ComfyUI and pack isolation
read it to prepare for the pack without running it.

```toml
[project]
name = "comfyui-wd14-tagger"
version = "1.4.0"
dependencies = ["onnxruntime>=1.17", "huggingface_hub"]

[tool.comfy]
depends_on = [{pack = "comfyui-videohelpersuite", version = ">=1.5"}]
weights = [{folder = "onnx", filename = "wd-vit-tagger-v3.onnx", repo_id = "SmilingWolf/wd-vit-tagger-v3", sha256 = "35f23693..."}]
permissions = ["network:huggingface.co"]

[[tool.comfy.routes]]
method = "get"
path = "/tags"
module = "routes"
handler = "tags"
```

**`name`** is the pack's permanent name. It determines where its routes live, so don't change it once
published. Without it, the folder name is used.

**`dependencies`** are the pip packages the pack imports (a `requirements.txt` works too). In ComfyUI's
process they share ComfyUI's environment; a pack that runs on its own environment gets exactly this list.
Declare everything you import that isn't one of ComfyUI's own requirements.

**`depends_on`** names other packs whose nodes this pack runs with `eng.node.run`, with a version range.

### Weights

**`weights`** lists the model files the pack reads with `eng.models.read` or `util.folders`: the `folder` and
`filename`, and where to get it (`repo_id`, optionally `revision`) and its `sha256`, so they can be fetched and
checked before the pack runs instead of the pack downloading at run time. Declaring the hash means a
changed file is caught rather than silently used.

### Permissions

**`permissions`** lists what the pack needs beyond the plain API. Today that's network access:
`"network:<host>"` for each host the pack talks to. In ComfyUI's process the network is open to every pack;
when packs run securely ([Pack isolation](pack-isolation.md#running-securely)), a pack reaches only the hosts
it declared. Declare every host, including ones a library you use calls.

### Routes

A pack that needs HTTP endpoints, usually for its own frontend, declares them:

```toml
[[tool.comfy.routes]]
method = "get"            # get, post, put, delete
path = "/tags"
module = "routes"         # v2/routes.py
handler = "tags"
```

```py
# v2/routes.py
import json


def tags(request):        # or async def
    query = request["query"].get("q", "")
    return {"status": 200, "body": json.dumps(find_tags(query)), "content_type": "application/json"}
```

- A handler gets the request as plain data, `{"method", "path", "query", "body"}`, and returns
  `{"status", "body"}`, with `"content_type"` `application/json` or `text/plain`. The body is at most 1 MB.
- Handlers run in ComfyUI's process.

Routes are served at `/ext/<namespace>/<path>`. The namespace depends only on the pack's declared
`[project] name`, so it's the same wherever the pack is installed and two packs can't collide. Compute it
once and write the path into your frontend code:

```
$ python -c "from comfy_api.latest.registration import pack_namespace; print(pack_namespace('comfyui-wd14-tagger'))"
comfyui-wd14-tagger-72e97b4a
```

```js
const tags = await api.fetchApi("/ext/comfyui-wd14-tagger-72e97b4a/tags?q=" + encodeURIComponent(q));
```

The rule: the name lowercased, with each run of characters other than `a-z` and `0-9` turned into `-`, cut
to 32 characters; then `-` and the first 8 hex characters of the SHA-256 of the exact declared name. The
hash is over the name as written, so `ComfyUI-WD14-Tagger` gives `comfyui-wd14-tagger-ca462c32`. A pack with
no `name` falls back to its folder name, and its URLs change if the user renames the folder: a pack that
declares routes should declare a name.

V1 packs added routes straight to ComfyUI's server (`PromptServer.instance.routes`). Declared routes don't
need the server object, can't shadow ComfyUI's or another pack's routes, and work when the pack runs elsewhere.

## Frontend code

`WEB_DIRECTORY` in `v2/__init__.py` points at the pack's frontend files, which ComfyUI serves to the browser,
as in V1. Frontend code talks to the pack through declared routes, events from `eng.ui.send`, and
`eng.ui.interact`: see [UI](ui.md).
