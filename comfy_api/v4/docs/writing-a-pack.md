# Writing a V4 pack

A pack is a folder in ComfyUI's `custom_nodes` folder. ComfyUI imports the folder's
`__init__.py` and calls its `comfy_entrypoint()`. A V4 pack gets everything it uses from ComfyUI
from one package, `comfy_api.v4`:

- `io` and `ui` define nodes, as in V3.
- `eng` is the API: calls into ComfyUI, in groups such as `eng.vae`, `eng.model` and
  `eng.sampler`.
- `ComfyAPISync` gives the same calls to nodes that are not async.
- `util` holds helper functions, and `util.folders` ComfyUI's folders.
- `ComfyExtension` is the base class of the object `comfy_entrypoint()` returns.

The [reference](reference.md) lists every call.

## A first pack

This `__init__.py` defines one node. It takes an image and a VAE, encodes the image to a latent and
decodes it again, and shows the node's progress after each step.

```python
from comfy_api.v4 import ComfyExtension, eng, io


class VaeRoundTrip(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ExampleVaeRoundTrip",
            display_name="VAE Round Trip",
            category="example",
            inputs=[io.Image.Input("image"), io.Vae.Input("vae")],
            outputs=[io.Image.Output()],
        )

    @classmethod
    async def execute(cls, image: io.Image.Type, vae: io.Vae.Type) -> io.NodeOutput:
        latent = await eng.vae.encode(vae, image)
        await eng.progress.update(1, 2)
        decoded = await eng.vae.decode(vae, latent)
        await eng.progress.update(2, 2)
        return io.NodeOutput(decoded)


class Extension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [VaeRoundTrip]


async def comfy_entrypoint() -> Extension:
    return Extension()
```

`execute` is `async def`, so each call is awaited. The node is defined with `io` exactly as a V3
node is; only the calls into ComfyUI go through `eng`.

## Nodes that are not async

In a node whose `execute` is a plain `def`, use `ComfyAPISync()`. It has the same groups and
calls, with the same parameters, and each call returns its result directly.

```python
from comfy_api.v4 import ComfyAPISync, ComfyExtension, io

api = ComfyAPISync()


class VaeRoundTripSync(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ExampleVaeRoundTripSync",
            display_name="VAE Round Trip (sync)",
            category="example",
            inputs=[io.Image.Input("image"), io.Vae.Input("vae")],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image: io.Image.Type, vae: io.Vae.Type) -> io.NodeOutput:
        latent = api.vae.encode(vae, image)
        api.progress.update(1, 2)
        decoded = api.vae.decode(vae, latent)
        api.progress.update(2, 2)
        return io.NodeOutput(decoded)


class Extension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [VaeRoundTripSync]


async def comfy_entrypoint() -> Extension:
    return Extension()
```

A sync call runs on the thread that makes it. Use the sync form wherever `await` cannot be
written, even in an async pack: at import, in `define_schema`, and in the functions you hand to
ComfyUI, such as a CFG function or a sampler function, which it calls as plain functions.

## Calls that need a running node

`eng.progress.update`, `eng.ui.progress_text`, `eng.ui.send`, `eng.node.run` and
`eng.node.run_output` act on the node that is running. Make them while it runs, from `execute` and
the functions it calls. Made elsewhere, for example from a route handler, a `threading.Thread` or
a thread started with `asyncio.to_thread`, they can raise `RuntimeError`: a pack that runs in its
own process cannot make any call from a thread it started. The other calls do not need a running
node.

ComfyUI runs each prompt on a thread and event loop of its own, apart from the server's, where route
handlers run. Do not share asyncio objects such as events and queues between a node and a route
handler, and do not leave tasks running when `execute` returns: they are cancelled when the prompt
ends.

## When the user cancels

When the user cancels the prompt, `eng.progress.update` raises `InterruptProcessingException`, and
so does `eng.execution.check_interrupted()`, which you can call between steps of long work. The
exception is a `BaseException`, so `except Exception` does not catch it. Let it reach ComfyUI, which
reports it as an interruption, not as an error.

## Routes

A pack can answer HTTP requests. `get_routes` returns the routes; each handler takes a
`RouteRequest` and returns a `RouteResponse`.

```python
import json

from comfy_api.v4 import ComfyExtension, Route, RouteRequest, RouteResponse, io


async def hello(request: RouteRequest) -> RouteResponse:
    name = dict(request["query"]).get("name", "world")
    return {
        "status": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"hello": name}).encode(),
    }


class Extension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return []

    async def get_routes(self) -> list[Route]:
        return [{"method": "GET", "path": "/example_pack/hello", "handler": hello}]


async def comfy_entrypoint() -> Extension:
    return Extension()
```

ComfyUI serves each route at its `path` and at `/api` followed by its `path`, here
`/example_pack/hello` and `/api/example_pack/hello`. It reads the routes once, when it loads the
pack. If a route is invalid, or a request it answers could also reach a route that ComfyUI or
another pack serves, ComfyUI disables the pack and logs a warning: none of its nodes load. When two
packs claim the same path, the one loaded later is disabled, and ComfyUI loads packs in the order
the file system lists their folders. Start each path with your pack's name, and name the events
you send with `eng.ui.send` the same way: ComfyUI does not stop a pack from sending its own event
types, such as `execution_success`.

## Helpers

`util` has functions that nodes commonly need, and `util.folders` gives ComfyUI's folders. They
are plain functions; they do not need a running node. Which folders a pack can rely on is in
[Files, folders and state](files-and-state.md).

```python
import torch

from comfy_api.v4 import util
from comfy_api.v4.util import folders

images = torch.rand(1, 64, 64, 3)
smaller = util.common_upscale(images.movedim(-1, 1), 32, 32, "bilinear", "disabled")
smaller = smaller.movedim(1, -1)
scratch_folder = folders.get_temp_directory()
```

`IMAGE` values are channels last, `(batch, height, width, channels)`; `util.common_upscale` takes
channels first, hence the `movedim` calls.

## What a pack must not do

- Open paths it worked out for itself, or keep state from one run to the next. Neither is
  supported; see [Files, folders and state](files-and-state.md).
- Import ComfyUI's own modules, such as `comfy`, `comfy_extras`, `nodes` or `folder_paths`. They
  can change in any ComfyUI release. `comfy_api.v4` only gains calls: the `eng` and `util` calls a
  pack uses keep their names, parameters and documented behaviour (see
  [Versioning](versioning.md)).
- Take `ComfyExtension` from `comfy_api.latest`. ComfyUI treats a pack whose extension is V3's as
  a V3 pack and does not read its `get_routes`. `comfy_api.v4` has `io` and `ui` too. Names V3
  keeps private, such as `comfy_api.latest._io._DynamicGroup`, are not part of V4 and can change in
  any release.
- Catch `InterruptProcessingException` and carry on.
- Expect to manage ComfyUI's model cache. ComfyUI decides which models stay loaded;
  `eng.memory.unload_all()` does nothing.
- Change ComfyUI's folders for its own use. `util.folders.add_model_folder_path` and changes to
  `util.folders.folder_names_and_paths` affect every node in ComfyUI, and nothing undoes them.
