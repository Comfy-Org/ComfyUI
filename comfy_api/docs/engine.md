# The engine

`eng` is how a node uses ComfyUI's engine. Every call is `async`, documented in [the reference](reference.md),
and typed in `v4.pyi`. This page explains how the pieces fit together, with examples; the reference has every
argument.

```py
from comfy_api.v4 import eng
```

## Values and engine objects

A node deals with two kinds of things.

**Values** are data: tensors, latents, conditioning, numbers, strings, lists and dicts of them. A node reads
and computes with them directly, in plain torch.

**Engine objects** are the engine's own things: a MODEL, CLIP, VAE, CONTROL_NET, SAMPLER and the like. A node
receives them, passes them to `eng` calls and to other nodes, and returns them, but never reaches into them.

```py
# no
model.model.diffusion_model.input_blocks[0]
model.model_options["transformer_options"]
hasattr(clip, "tokenizer")

# yes
info = await eng.model.info(model)
m = await eng.model.set_option(model, ["transformer_options", "my_key"], value)
tokens = await eng.clip.tokenize(clip, text)
```

Why so strict: an engine object's internals are ComfyUI's implementation, and they change between releases.
A pack that reaches into them breaks when they do. They also can't leave ComfyUI's process, so a node that
reaches into one can't run in its own process. Under pack isolation, an engine object reaches the node as a
handle: passing it to `eng` works exactly as in ComfyUI's process, and reaching into it fails.

## Changes return copies

Every `eng` call that changes a model, CLIP or VAE returns a new one and leaves its argument unchanged:

```py
patched = await eng.model.set_cfg(model, my_cfg)      # model is unchanged
```

In V1, a node had to remember to `clone()` before patching, or it changed the model for every other node
using it. In V4 that mistake can't happen.

## Models

| Call | What it does |
|---|---|
| `eng.model.clone(model)` | A copy, for patches that shouldn't affect the original |
| `eng.model.info(model)` | Facts: model type, latent format, dtype, sigma range |
| `eng.model.weight_keys(model)` | The weight names |
| `eng.model.read_weight(model, key)` | One weight, with its patches applied, as a tensor |
| `eng.model.add_patches(model, patches, strength_patch, strength_model)` | Weight patches, in `ModelPatcher.add_patches` format |
| `eng.model.load_lora(model, clip, lora, strength_model, strength_clip)` | Apply a LoRA to a model and a CLIP |
| `eng.model.options(model)` / `set_option(model, path, value)` | Read and set `model_options`, such as `transformer_options` keys |
| `eng.model.percent_to_sigma`, `sigma`, `timestep` | The model's sampling schedule |
| `eng.model.latent_process_in` / `_out` | Scale a latent into and out of the model's range |
| `eng.model.set_attention(model, backend)` | Use one of ComfyUI's attention implementations |
| `eng.model.torch_compile(model, ...)` | Compile the model with `torch.compile` when it samples |

Hooks into sampling (`set_cfg`, attention and block patches, wrappers) are on [Sampling](sampling.md).

A LoRA, applied the way `LoraLoader` does:

```py
lora = (await eng.models.read("loras", lora_name))["sd"]
model, clip = await eng.model.load_lora(model, clip, lora, strength_model, strength_clip)
```

A start and end percentage turned into sigmas, for a hook that should act only part of the way through
sampling:

```py
sigma_start = await eng.model.percent_to_sigma(model, start_percent)
sigma_end = await eng.model.percent_to_sigma(model, end_percent)
```

## Text encoders

```py
tokens = await eng.clip.tokenize(clip, text)
conditioning = await eng.clip.encode_tokens(clip, tokens)        # what CLIPTextEncode returns
cond, pooled = await eng.clip.encode_from_tokens(clip, tokens, return_pooled=True)
```

`tokenize` returns plain token data, so a node can edit the tokens (weights, padding) before encoding. CLIP
also has `clone`, `add_patches`, `read_weight`, `info` and `weight_keys`.

## VAEs

```py
latent = {"samples": await eng.vae.encode(vae, image)}
image = await eng.vae.decode(vae, latent["samples"])
image = await eng.vae.decode_tiled(vae, latent["samples"], tile_x=512, tile_y=512)
```

`eng.vae.info` gives latent channels, dtype and compression. `eng.vae.from_state_dict(sd)` builds a VAE from
weights the pack holds.

## Loading model files

| Call | What it does |
|---|---|
| `eng.models.list(folder)` | The files in a model folder |
| `eng.models.read(folder, name)` | A file's tensors and metadata: `{"sd": {...}, "metadata": {...}}` |
| `eng.models.load_checkpoint(name)` | MODEL, CLIP and VAE from a checkpoint |
| `eng.models.load_diffusion_model(name)` | A MODEL from the `diffusion_models` folder |
| `eng.models.load_vae(name)` | A VAE |

```py
model, clip, vae = (await eng.models.load_checkpoint(ckpt_name))[:3]
```

Files are named as `models.list` and `io.ModelFolder` list them, never by path. That keeps a pack inside
ComfyUI's model folders, wherever the user has put them.

For anything ComfyUI loads with one of its own nodes, such as ControlNets, CLIP and CLIP vision models, or
upscale models, run that node (next section):

```py
(control_net,) = await eng.node.run("ControlNetLoader", {"control_net_name": name})
```

For a pack's own model architecture, read the weights with `eng.models.read` or `util.load_torch_file`, and
build the model in the pack.

## Running ComfyUI's nodes

`eng.node.run` runs any registered node and returns its outputs:

```py
(samples,) = await eng.node.run("KSampler", {
    "model": model, "positive": positive, "negative": negative, "latent_image": latent,
    "seed": seed, "steps": 20, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal",
    "denoise": 1.0,
})
```

Inputs are named as the node's schema names them; values are what nodes pass each other. This is how a pack
builds on ComfyUI's own nodes (sampling, conditioning, image operations) instead of copying their code.
`eng.node.run_output` also returns what an output or preview node showed:

```py
shown = await eng.node.run_output("PreviewImage", {"images": image})
return io.NodeOutput(ui=shown["ui"])
```

Another pack's nodes can be run the same way if it's installed; declare it in `depends_on` ([Packs](packs.md)).

## Devices, memory, settings and interrupts

```py
device = (await eng.device.info())["torch_device"]
free = await eng.memory.free_bytes()
await eng.memory.empty_cache()
await eng.memory.unload_all()                 # before loading something big
await eng.memory.load_models([model])         # load a model onto the GPU now

if await eng.execution.is_interrupted():      # the user pressed cancel
    ...
await eng.execution.check_interrupted()       # or raise if they did

disable_metadata = await eng.settings.get("disable_metadata")
```

`eng.settings.get` reads the settings a pack may depend on: `disable_metadata`, `preview_method`,
`preview_size` and `fast` (the enabled `--fast` optimizations). A pack can't change them: they're the user's.
Don't read `comfy.cli_args` directly: in a pack's own process it holds defaults, not the user's flags.

## `util`: helpers that don't need the engine

`util` has the helpers packs used to import from `comfy.utils`, `nodes` and `node_helpers`. They're plain
functions that run wherever the node runs:

| Helper | What it is |
|---|---|
| `util.common_upscale`, `lanczos`, `bislerp` | ComfyUI's image and latent resizing |
| `util.tiled_scale`, `tiled_scale_multidim`, `get_tiled_scale_steps` | Process an image in tiles |
| `util.prepare_noise` | ComfyUI's seeded noise for a latent |
| `util.conditioning_set_values`, `conditioning_set_values_with_timestep_range` | Edit conditioning |
| `util.load_torch_file`, `save_torch_file` | Read and write `.safetensors` and checkpoint files |
| `util.pillow` | Pillow calls with ComfyUI's workarounds |
| `util.SAMPLER_NAMES`, `SCHEDULER_NAMES`, `MAX_RESOLUTION` | Constants |
| `util.folders` | ComfyUI's folders, for libraries that need a path: see [Files and state](files-and-state.md) |

## Calling `eng` from synchronous code

`eng` calls are async: write a node that uses them as `async def execute` and `await` them. Converted packs
keep their synchronous code and call `eng` through a small generated helper (`_v2e.sync(...)`); new code
doesn't need it.
