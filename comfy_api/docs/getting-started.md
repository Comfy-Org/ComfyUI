# Getting started

This walks through a small pack from an empty folder to a working, tested node that changes how a model
samples. It covers the parts every pack has: the layout, the entry point, a node, and what it declares.

## 1. The layout

```
my-pack/
    __init__.py          V1 code, if the pack has any (ComfyUI versions without V4 use it)
    v2/
        __init__.py      the V4 entry point
        nodes.py         the nodes
        pyproject.toml   what the pack needs
```

When ComfyUI loads `custom_nodes/my-pack`, it uses `v2/__init__.py` if it's there, and the pack's own
`__init__.py` otherwise. That lets one install serve older ComfyUI versions with your V1 code and newer ones
with V4. A pack written only for V4 can leave the top-level `__init__.py` empty.

## 2. A first node

A node is a class with two parts: `define_schema`, which says what the node is, and `execute`, which does the
work.

```py
# v2/nodes.py
from comfy_api.v4 import io


class Brighten(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MyPack_Brighten",
            display_name="Brighten",
            category="my_pack/image",
            description="Adds a constant to every pixel.",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("amount", default=0.1, min=-1.0, max=1.0, step=0.01),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, amount):
        return io.NodeOutput((image + amount).clamp(0.0, 1.0))
```

Things to notice:

- **`node_id` is permanent.** Workflows save it. Prefix it with your pack's name so it can't clash with
  another pack's.
- **Inputs arrive as keyword arguments** named by their ids. An IMAGE is a plain tensor, `[batch, height,
  width, channels]`, values 0 to 1. Do the work in plain torch.
- **`execute` returns `io.NodeOutput`**, with one value per declared output, in order.
- **`ENGINE_API = True`** says the node reaches the engine only through `eng`. This node doesn't use `eng`
  at all, but setting it lets the node run under pack isolation. Set it on every V4 node.

## 3. A node that uses the engine

Engine work goes through `eng`. Here a node lowers the CFG strength of a model as it samples:

```py
from comfy_api.v4 import eng, io


class SoftenCFG(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MyPack_SoftenCFG",
            display_name="Soften CFG",
            category="my_pack/sampling",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("amount", default=0.9, min=0.0, max=2.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    async def execute(cls, model, amount):
        def soften(args):
            return args["uncond"] + (args["cond"] - args["uncond"]) * args["cond_scale"] * amount

        return io.NodeOutput(await eng.model.set_cfg(model, soften))
```

- **The MODEL is passed to `eng`, never reached into.** `eng.model.set_cfg` returns a copy that uses
  `soften` for the CFG combination; the model that came in is unchanged, so other nodes using it aren't
  affected.
- **`eng` calls are `await`ed,** so a node that makes them is `async def`.
- **`soften` runs during sampling,** on every step. It gets a dict of tensors and plain values, must not be
  `async`, and can't call `eng`. [Sampling](sampling.md) explains why, and the other hooks.

## 4. The entry point

ComfyUI finds a pack's nodes through its `ComfyExtension`:

```py
# v2/__init__.py
from comfy_api.v4 import ComfyExtension
from .nodes import Brighten, SoftenCFG


class MyPack(ComfyExtension):
    async def get_node_list(self):
        return [Brighten, SoftenCFG]


async def comfy_entrypoint():
    return MyPack()
```

`ComfyExtension` can also register samplers, schedulers, model folders and replacements for old node ids:
see [Packs](packs.md).

## 5. What the pack declares

```toml
# v2/pyproject.toml
[project]
name = "my-pack"
version = "1.0.0"
dependencies = []
```

Every pip package the pack imports goes in `dependencies`. ComfyUI's own requirements (torch, numpy, Pillow...)
are always there. When a pack runs on its own environment, only what it declares is installed, so an import
that isn't declared works on your machine and fails there. See
[Packs](packs.md) for weights, permissions and routes.

## 6. Run it

Put the folder in `ComfyUI/custom_nodes/` and start ComfyUI. The nodes appear under the `my_pack` category.

Then run it again with pack isolation:

```
COMFY_PACK_ISOLATION=true python main.py
```

Your `ENGINE_API` nodes now run in the pack's own process. A node that follows the rules behaves exactly the
same; if it doesn't, [Pack isolation](pack-isolation.md) explains what to look for. Test both ways before you
publish.

## 7. Test it

Test a node by running it on inputs it must change, and comparing with what you expect:

- **Vary the inputs between runs.** ComfyUI caches results by inputs, so an unchanged workflow doesn't run
  the node again.
- **Use inputs the node must change.** Brightening an all-white image changes nothing; a broken node passes.
- **Check both modes,** with and without `COMFY_PACK_ISOLATION`.

For an existing pack, the converter's verifier does this against your original: see
[Converting a pack](converting.md).

## Where next

- [Nodes](nodes.md): every kind of input and output, lists, lazy inputs, validation.
- [The engine](engine.md): the rest of `eng`.
- [Migrating a pack](migrating.md): if you're moving existing code.
