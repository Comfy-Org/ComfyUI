# Nodes

A V4 node is declared exactly as a V3 node is: a subclass of `io.ComfyNode` with a `define_schema` that
describes it and an `execute` that does the work. This page covers the declaration. What a node does with
the engine is in [The engine](engine.md) and [Sampling](sampling.md).

```py
from comfy_api.v4 import eng, io


class Example(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MyPack_Example", category="my_pack", inputs=[...], outputs=[...])

    @classmethod
    async def execute(cls, **inputs):
        return io.NodeOutput(...)
```

## The schema

`define_schema` returns an `io.Schema`. ComfyUI calls it when the pack loads, to list the node and validate
prompts, so it runs in ComfyUI's process even under pack isolation. **Keep engine work out of it:** don't make
`eng` calls there, and don't load models. A dropdown of model files has its own option type, `io.ModelFolder`
(below); a dropdown of input files lists the folder with `util.folders`
([Files and state](files-and-state.md#reading-files)).

| Field | What it's for |
|---|---|
| `node_id` | The id workflows save. Permanent; prefix it with your pack's name |
| `display_name` | The name users see |
| `category` | Where it sits in the Add Node menu, as a path: `"my_pack/image"` |
| `description` | The tooltip on the node |
| `inputs`, `outputs` | The sockets and widgets, in order |
| `hidden` | Hidden values the node needs, such as `io.Hidden.unique_id` (below) |
| `is_output_node` | The node is a result of the workflow (it saves or shows something), so it runs even if nothing uses its outputs |
| `is_input_list` | The node gets every input as a list (below) |
| `not_idempotent` | Run again even when inputs are unchanged |
| `search_aliases` | Other names search should find it by, such as an old name |
| `is_deprecated`, `is_experimental` | Shown to users |

## Inputs and outputs

Each type has an `Input` and an `Output`. Inputs take an id first; the id is the keyword `execute` gets.

```py
inputs=[
    io.Image.Input("image"),
    io.Mask.Input("mask", optional=True),
    io.Model.Input("model", tooltip="The model to patch"),
],
outputs=[
    io.Image.Output(display_name="image"),
    io.Mask.Output(display_name="mask"),
],
```

**Data types**, passed as the values themselves:

| Type | Value |
|---|---|
| `io.Image` | Tensor `[batch, height, width, channels]`, float 0 to 1 |
| `io.Mask` | Tensor `[batch, height, width]`, float 0 to 1 |
| `io.Latent` | Dict with `"samples"`, a tensor `[batch, channels, height, width]`, and optional keys such as `"noise_mask"` |
| `io.Conditioning` | List of `[tensor, dict]` pairs |
| `io.Audio` | Dict with `"waveform"` and `"sample_rate"` |
| `io.Sigmas` | Tensor |
| `io.Video`, `io.Mesh`, `io.Voxel`, `io.Splat`... | See `v4.pyi` |

**Engine objects**, passed to `eng` but never reached into: `io.Model`, `io.Clip`, `io.Vae`, `io.ControlNet`,
`io.ClipVision`, `io.StyleModel`, `io.UpscaleModel`, `io.Sampler`, `io.Guider`, `io.Noise`, `io.Hooks`. See
[The engine](engine.md).

**Your own types:** `io.Custom("MY_TYPE")` makes a type only your nodes produce and take. The value can be
anything, including your own class. Under pack isolation it stays in your pack's process and comes back to
your next node; see [Pack isolation](pack-isolation.md). Use exactly the same type string everywhere it
appears, including any other pack that already uses it.

**Anything:** `io.AnyType` accepts any type; `io.MultiType` accepts several.

## Widgets

Widgets are inputs with a value the user sets on the node:

```py
io.Int.Input("steps", default=20, min=1, max=10000)
io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01)
io.String.Input("prompt", multiline=True, placeholder="Describe the image")
io.Boolean.Input("enabled", default=True, label_on="on", label_off="off")
io.Combo.Input("mode", options=["fast", "accurate"], default="fast")
io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True)
```

A widget can become a socket instead: `force_input=True` makes it an input that must be linked. An
`optional=True` input that isn't connected arrives as `None`, or isn't passed at all, so give the
`execute` parameter a default.

**Model files.** For a dropdown of files in a model folder, give the combo `io.ModelFolder`:

```py
io.Combo.Input("lora_name", options=io.ModelFolder("loras"))
```

ComfyUI lists the folder each time the node list is built, so new files appear without restarting, and a pack
never touches the folder itself. The node gets the file name; load it with `eng` (see
[The engine](engine.md#loading-model-files)). A folder your pack adds is declared with `get_model_folders`
([Packs](packs.md)).

**Sampler and scheduler lists:** `io.Combo.Input("sampler_name", options=util.SAMPLER_NAMES)` and
`util.SCHEDULER_NAMES`.

**Dynamic inputs:** `io.Autogrow` (a growing list of inputs), `io.DynamicCombo` (inputs that depend on a
choice) and `io.MatchType` (inputs and outputs that share a type). See `v4.pyi`.

## Returning results

`execute` returns `io.NodeOutput` with one value per output, in order:

```py
return io.NodeOutput(image, mask)
```

To show something on the node, pass `ui=`; see [UI](ui.md):

```py
return io.NodeOutput(image, ui=ui.PreviewImage(image, cls=cls))
```

To stop the nodes that use an output, return `io.ExecutionBlocker(message)` for it. They don't run, and the
message is shown. A lazy input (below) is usually a better fit.

## Lists

By default a node runs once per value. Two flags change that:

- **`is_input_list=True`** on the schema: the node gets every input as a list, with all the values at once, and
  runs once. Use it for nodes that combine values, such as a batch builder.
- **`is_output_list=True`** on an output: that output is a list of values, and each node it connects to runs
  once per item.

With `is_input_list=True`, widget values are lists too. Hidden values aren't: `cls.hidden.unique_id` is still
the node's id.

## Lazy inputs

An input declared `lazy=True` is evaluated only if the node asks for it. That lets a switch node skip the
branch it doesn't use:

```py
@classmethod
def define_schema(cls):
    return io.Schema(
        node_id="MyPack_Switch", category="my_pack",
        inputs=[io.Boolean.Input("use_a", default=True),
                io.Image.Input("a", lazy=True), io.Image.Input("b", lazy=True)],
        outputs=[io.Image.Output()])

@classmethod
def check_lazy_status(cls, use_a, a=None, b=None):
    return ["a"] if use_a else ["b"]

@classmethod
def execute(cls, use_a, a=None, b=None):
    return io.NodeOutput(a if use_a else b)
```

`check_lazy_status` returns the ids of lazy inputs it still needs; unevaluated ones arrive as `None`.

## Validation and caching

- **`validate_inputs(cls, **inputs)`** runs before the prompt executes. Return `True`, or a message saying
  what's wrong. Only widget values are known at that point.
- **`fingerprint_inputs(cls, **inputs)`** returns a value that changes when the node must run again even
  though its inputs didn't, such as a file's modification time. Without it, ComfyUI reuses the cached
  result for unchanged inputs.

## Hidden values

A node that needs information about the prompt declares it and reads it from `cls.hidden`:

```py
io.Schema(..., hidden=[io.Hidden.unique_id, io.Hidden.prompt, io.Hidden.extra_pnginfo])

@classmethod
def execute(cls, text):
    node_id = cls.hidden.unique_id
```

| Value | What it is |
|---|---|
| `unique_id` | This node's id in the prompt |
| `prompt` | The prompt as the client sent it |
| `extra_pnginfo` | Extra data saved into images' metadata, including the workflow |
| `dynprompt` | The prompt as it's being executed |
| `auth_token_comfy_org`, `api_key_comfy_org` | For nodes that call Comfy's API services |

Under pack isolation, changes a node makes to `extra_pnginfo` stay in the pack's process: see
[Pack isolation](pack-isolation.md).

## A complete example

A node with optional inputs, a combo, a list output and a preview:

```py
from comfy_api.v4 import io, ui


class SplitTiles(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MyPack_SplitTiles", display_name="Split Into Tiles", category="my_pack/image",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("rows", default=2, min=1, max=16),
                io.Int.Input("columns", default=2, min=1, max=16),
                io.Combo.Input("order", options=["rows first", "columns first"]),
                io.Mask.Input("mask", optional=True),
            ],
            outputs=[io.Image.Output(display_name="tiles", is_output_list=True)])

    @classmethod
    def execute(cls, image, rows, columns, order, mask=None):
        if mask is not None:
            image = image * mask.unsqueeze(-1)
        height, width = image.shape[1] // rows, image.shape[2] // columns
        cells = [(r, c) for r in range(rows) for c in range(columns)]
        if order == "columns first":
            cells.sort(key=lambda rc: (rc[1], rc[0]))
        tiles = [image[:, r * height:(r + 1) * height, c * width:(c + 1) * width] for r, c in cells]
        return io.NodeOutput(tiles, ui=ui.PreviewImage(tiles[0], cls=cls))
```
