# Migrating a pack

This page takes the things V1 packs did by reaching into ComfyUI, monkey-patching included, and shows how
V4 does each one: **before** you did this, **now** you do this. Every example comes from converting the
ComfyUI registry: about 6,100 packs, of which 524 patch ComfyUI's own modules or classes.

> **The auto-converter does most of this for you.** `comfy_api/conversion/convert_pack.py` converts an
> existing pack: it rewrites the uses it knows, writes the V4 entry point, checks each node against your
> original, and lists what's left. Changes it makes are marked *(automatic)* below; the ones you finish
> yourself are marked *(by hand)*. See [Converting a pack](converting.md).
>
> Converted code keeps its V1 shape: it stays synchronous and reaches `eng` through a generated shim. The
> **now** examples here are how to write the same thing directly, in new code or when you rework a node.

## What changes, in short

A V1 node reached ComfyUI's engine any way it liked: through the model's attributes, ComfyUI's modules, the
server object, file paths. A V4 node is an `io.ComfyNode` declared as in V3, with `ENGINE_API = True`, that
reaches the engine only through `eng`. Tensors and plain values are the same as before; engine objects
(MODEL, CLIP, VAE...) are passed to `eng` and never reached into; changes are made on a copy; nothing is
patched globally. In exchange, the same node runs in ComfyUI's process and in its own process under
[pack isolation](pack-isolation.md), and keeps working when ComfyUI's internals change.

Your V1 code can stay where it is. The V4 code goes in `v2/`, which ComfyUI loads when it has V4, so one
install serves both. See [Packs](packs.md#layout).

## Models

**Before**, a node cloned the model and edited its options:

```py
m = model.clone()
m.model_options["transformer_options"]["my_strength"] = strength
```

**Now:**

```py
m = await eng.model.set_option(model, ["transformer_options", "my_strength"], strength)
```

**Before**, a node reached into the model for facts:

```py
sampling = model.get_model_object("model_sampling")
sigma = sampling.percent_to_sigma(start_percent)
latent_format = model.model.latent_format
```

**Now** *(by hand: the converter can't tell what an attribute walk was for)*:

```py
sigma = await eng.model.percent_to_sigma(model, start_percent)
info = await eng.model.info(model)          # type, latent format, dtype, sigma range
```

`eng.model.sigma`, `timestep`, `latent_process_in` and `latent_process_out` cover the rest of
`model_sampling`.

**Before**, weight patches and LoRAs:

```py
m = model.clone()
m.add_patches({key: (delta,)}, 0.5)
model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora_sd, 1.0, 1.0)
```

**Now** *(automatic)*:

```py
m = await eng.model.add_patches(model, {key: (delta,)}, strength_patch=0.5)
model_lora, clip_lora = await eng.model.load_lora(model, clip, lora_sd, 1.0, 1.0)
```

There's no `clone()` to forget: every `eng.model` change returns a new model. `eng.model.read_weight(model,
key)` returns one weight with its patches applied. More in [The engine](engine.md#models).

## Sampling hooks

**Before:**

```py
m = model.clone()
m.set_model_sampler_cfg_function(my_cfg)
m.set_model_sampler_post_cfg_function(my_post_cfg)
m.set_model_attn1_patch(my_attn1)
m.set_model_unet_function_wrapper(my_wrapper)
```

**Now** *(automatic)*:

```py
m = await eng.model.set_cfg(model, my_cfg)
m = await eng.model.set_post_cfg(m, my_post_cfg)
m = await eng.model.set_attn1_patch(m, my_attn1)
m = await eng.model.set_unet_wrapper(m, my_wrapper)
```

The functions keep their V1 signatures. Two things change, and the second is *by hand*:

- A hook can't call `eng`, and it isn't `async`.
- A hook sees only tensors and plain data. **Before**, hooks often read `args["model"]`, or a model the
  node captured. **Now**, get what you need in `execute` and capture the value:

```py
# before
def my_cfg(args):
    sigma_cut = model.get_model_object("model_sampling").percent_to_sigma(0.5)
    ...

# now
sigma_cut = await eng.model.percent_to_sigma(model, 0.5)

def my_cfg(args):
    ...                                     # uses sigma_cut
```

See [Sampling](sampling.md#how-hooks-work) for every hook and its arguments.

## Monkey-patching

This is the largest change. A patch on ComfyUI's module or class changes ComfyUI for every user and every
other pack, breaks when ComfyUI changes, and under pack isolation doesn't reach ComfyUI's process at all.
Each common patch has a V4 replacement that applies to the model, sampler or node you're working with, or a
registration ComfyUI reads when the pack loads.

| Packs | Before (patched) | Now |
|---:|---|---|
| 90 | `server.PromptServer.instance`: `send_sync`, `routes`, `prompt_queue` | `eng.ui.send`, `eng.ui.progress_text`, and routes declared in `pyproject.toml` |
| 73 | `comfy.cli_args.args` (`preview_method`, `cpu`, `disable_*`...) | `eng.settings.get` for `disable_metadata`, `preview_method`, `preview_size` and `fast`. Don't read or set `cli_args` |
| 35 | `comfy.ldm.modules.attention.optimized_attention` (and `_masked`) | `eng.model.set_attention(model, "sage")` for ComfyUI's implementations; `set_attn1_patch` and `set_attn2_patch` for your own |
| 35+20 | `execution.PromptExecutor.execute`, `execution.execute`, `get_input_data`, `validate_prompt` | Node features: `check_lazy_status`, `fingerprint_inputs`, `io.ExecutionBlocker`, `eng.node.run` |
| 34+11+11 | `comfy.ldm.<family>`, `comfy.model_base.<Model>.extra_conds`, `comfy.model_detection.detect_unet_config` (adding a model family) | An [engine extension](engine-extensions.md), or contribute the model to ComfyUI |
| 28 | `comfy.model_management.unload_all_models`, `free_memory`, `EXTRA_RESERVED_VRAM` | `eng.memory.unload_all`, `free_bytes`, `empty_cache`, `load_models`. Reserved VRAM is the user's setting |
| 27+13 | `comfy.k_diffusion.sampling.<fn>`, `comfy.samplers.k_diffusion_sampling` (adding a sampler) | `ComfyExtension.get_samplers()`, and `get_schedulers()` |
| 12+12 | `comfy.samplers.KSampler.sample`, `CFGGuider.outer_sample`, `sampler_object` | `eng.model.add_wrapper(model, kind, fn)`, or `eng.sampler.custom(fn)` |
| 11+9 | `comfy.model_patcher.ModelPatcher` and `CoreModelPatcher` methods | The matching `eng.model` call, on that model |
| 13+13 | `latent_preview.prepare_callback`, `get_previewer` | `eng.progress.update(i, total, preview=image)` |
| 10+8 | `comfy.utils.PROGRESS_BAR_ENABLED`, `PROGRESS_BAR_HOOK` | `eng.progress.update` |
| 9+8 | `folder_paths.models_dir`, `get_input_directory`, `folder_names_and_paths` | `ComfyExtension.get_model_folders()`, and `util.folders` |
| 7 | `nodes.SaveImage.save_images` | `ui.ImageSaveHelper`, `ui.PreviewImage`, or `eng.output.write` |
| 6 | `comfy.utils.load_torch_file` | `util.load_torch_file`, or `eng.models.read(folder, name)` |

Counts are registry packs that assign to that target; a pack can count in several rows. The converter
removes these patches and writes the replacement where it can; where it can't, it marks the line *(by
hand)*.

### Attention

**Before**, a pack swapped ComfyUI's attention for everyone:

```py
import comfy.ldm.modules.attention as attention
attention.optimized_attention = my_attention
```

**Now**, for the model this node returns:

```py
m = await eng.model.set_attention(model, "sage")           # one of ComfyUI's implementations
m = await eng.model.set_attn1_patch(model, adjust_qkv)     # or your own change to q, k and v
```

### Adding a sampler or scheduler

**Before**, a pack added a sampler by patching ComfyUI's sampler module and list:

```py
import comfy.k_diffusion.sampling as k
k.sample_my_euler = sample_my_euler
comfy.samplers.KSampler.SAMPLERS.append("my_euler")
```

**Now** it registers it, and it's in ComfyUI's list from startup *(by hand)*:

```py
class MyPack(ComfyExtension):
    async def get_samplers(self):
        return {"my_euler": sample_my_euler}   # fn(model, x, sigmas, extra_args, callback=None, disable=None)
```

For a sampler a node builds at run time, `eng.sampler.custom(fn)` returns a SAMPLER. See
[Sampling](sampling.md).

### Wrapping the sampling loop

**Before**, a pack replaced a method on ComfyUI's class:

```py
original = comfy.samplers.CFGGuider.outer_sample

def outer_sample(self, *args, **kwargs):
    prepare()
    return original(self, *args, **kwargs)

comfy.samplers.CFGGuider.outer_sample = outer_sample
```

**Now** it wraps sampling for the model the node returns:

```py
def outer_sample(executor, *args, **kwargs):
    prepare()
    return executor(*args, **kwargs)

m = await eng.model.add_wrapper(model, "outer_sample", outer_sample)
```

The kinds are `outer_sample`, `prepare_sampling`, `sampler_sample`, `predict_noise`, `calc_cond_batch`,
`apply_model` and `diffusion_model`.

### ComfyUI's settings

**Before**, a pack changed a setting, or read one:

```py
from comfy.cli_args import args
args.preview_method = LatentPreviewMethod.TAESD
if args.disable_metadata:
    ...
```

**Now** it reads the settings it may depend on, and leaves them to the user:

```py
if await eng.settings.get("disable_metadata"):
    ...
```

`eng.settings.get` gives `disable_metadata`, `preview_method`, `preview_size` and `fast`. Reading
`comfy.cli_args` directly is wrong too, not only changing it: in a pack's own process it holds the defaults,
not the user's flags. A pack that wants its own previews sends them with `eng.progress.update(...,
preview=image)`.

### Memory

**Before:**

```py
comfy.model_management.unload_all_models()
comfy.model_management.EXTRA_RESERVED_VRAM = 2 * 1024**3
```

**Now:**

```py
await eng.memory.unload_all()
```

How much VRAM ComfyUI keeps free is the user's choice (`--reserve-vram`).

### Changing how prompts execute

**Before**, packs replaced `execution.execute` or `PromptExecutor.execute` to skip nodes, add caching, or
run something around each node. **Now** each of those is a node feature:

| To | Use |
|---|---|
| Skip work when an input isn't needed | Lazy inputs and `check_lazy_status` |
| Re-run, or not, when something outside the inputs changes | `fingerprint_inputs` |
| Stop a branch of the graph | Return `io.ExecutionBlocker` |
| Run other nodes from yours | `eng.node.run` |

See [Nodes](nodes.md).

## Running ComfyUI's nodes

**Before** *(automatic)*:

```py
samples = nodes.KSampler().sample(model, seed, steps, cfg, sampler, scheduler, pos, neg, latent)[0]
```

**Now:**

```py
(samples,) = await eng.node.run("KSampler", {
    "model": model, "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler,
    "scheduler": scheduler, "positive": pos, "negative": neg, "latent_image": latent, "denoise": 1.0})
```

Inputs are named as in the node's schema. `eng.node.run_output` also returns what an output node showed.

## Loading models

**Before** *(automatic, except the ControlNet)*:

```py
path = folder_paths.get_full_path("checkpoints", name)
model, clip, vae = comfy.sd.load_checkpoint_guess_config(path)[:3]
lora = comfy.utils.load_torch_file(folder_paths.get_full_path("loras", lora_name))
controlnet = comfy.controlnet.load_controlnet(folder_paths.get_full_path("controlnet", cn_name))
```

**Now:**

```py
model, clip, vae = (await eng.models.load_checkpoint(name))[:3]
lora = (await eng.models.read("loras", lora_name))["sd"]
(controlnet,) = await eng.node.run("ControlNetLoader", {"control_net_name": cn_name})
```

**Before**, a dropdown of model files:

```py
"lora_name": (folder_paths.get_filename_list("loras"),)
```

**Now:**

```py
io.Combo.Input("lora_name", options=io.ModelFolder("loras"))
```

A pack's own model folder is registered with `get_model_folders()`. See
[Files and state](files-and-state.md#model-files).

## Files

**Before:**

```py
out_dir = folder_paths.get_output_directory()
image.save(os.path.join(out_dir, f"{prefix}_{counter:05}.png"))

with open(os.path.join(os.path.dirname(__file__), "presets.json"), "w") as f:
    json.dump(presets, f)

image = Image.open(folder_paths.get_annotated_filepath(image_name))
```

**Now:**

```py
saved = ui.ImageSaveHelper.get_save_images_ui(images, filename_prefix=prefix, cls=cls)
await eng.output.write(png_bytes, prefix, "png")         # bytes you encoded yourself

await eng.store.set("presets", presets)                  # not next to your code

image = Image.open(bytes_io.BytesIO(await eng.files.read(image_name)))    # import io as bytes_io
```

`util.folders` still gives ComfyUI's folder paths, with `folder_paths`' names, for libraries that need a
path *(automatic)*. A file written outside the output, temp and user folders may not be there after the node
returns. See [Files and state](files-and-state.md).

## State between runs

**Before**, a module-level cache:

```py
_cache = {}

if key not in _cache:
    _cache[key] = load_big_thing()
```

**Now:**

```py
value = await eng.cache.get(key)
if value is None:
    value = load_big_thing()
    await eng.cache.set(key, value, ttl=600)
```

`eng.cache` keeps large values in ComfyUI's memory, bounded and shared fairly between packs. `eng.store`
keeps small plain data across restarts, per pack, workflow or node.

## Progress, messages and questions

**Before** *(automatic)*:

```py
pbar = comfy.utils.ProgressBar(total)
pbar.update(1)
PromptServer.instance.send_sync("my_pack.status", {"text": "done"})
```

**Now:**

```py
await eng.progress.update(i, total)
await eng.ui.send("my_pack.status", {"text": "done"})
```

**Before**, a node that waited for the user needed its own route, and a loop waiting for the frontend to
call it. **Now**, `await eng.ui.interact(kind, payload)` waits for the answer *(by hand)*. See [UI](ui.md).

## Memory and devices

**Before** *(automatic)*:

```py
device = comfy.model_management.get_torch_device()
comfy.model_management.soft_empty_cache()
```

**Now:**

```py
device = (await eng.device.info())["torch_device"]
await eng.memory.empty_cache()
```

## Routes

**Before**, a pack added routes to ComfyUI's server:

```py
@PromptServer.instance.routes.get("/my_pack/tags")
async def tags(request):
    return web.json_response(find_tags(request.query.get("q")))
```

**Now** it declares them in `pyproject.toml` and writes a plain function *(automatic)*:

```toml
[[tool.comfy.routes]]
method = "get"
path = "/tags"
module = "routes"     # v2/routes.py
handler = "tags"
```

```py
# v2/routes.py
import json


def tags(request):    # {"method", "path", "query", "body"}
    return {"status": 200, "body": json.dumps(find_tags(request["query"].get("q"))),
            "content_type": "application/json"}
```

The route is served at `/ext/<namespace>/tags`, and the converter rewrites your frontend's calls to that
path. A route that answers with a file or a stream is marked *(by hand)*. [Packs](packs.md#routes) shows how to compute the path.

## Dependencies and downloads

**Before**, a pack installed packages and downloaded weights when it ran:

```py
subprocess.run([sys.executable, "-m", "pip", "install", "onnxruntime"])
hf_hub_download("SmilingWolf/wd-vit-tagger-v3", "model.onnx", local_dir=models_dir)
```

**Now** it declares both, and reads the weights from a model folder:

```toml
[project]
dependencies = ["onnxruntime>=1.17"]

[tool.comfy]
weights = [{folder = "onnx", filename = "wd-vit-tagger-v3.onnx", repo_id = "SmilingWolf/wd-vit-tagger-v3", sha256 = "35f23693..."}]
permissions = ["network:huggingface.co"]
```

See [Packs](packs.md#declarations).

## ComfyUI's internals, directly

A node can still reach into ComfyUI the V1 way: patch a module, walk a model's attributes, import from
`comfy`. Leave `ENGINE_API` off that node, and it runs in ComfyUI's process with the same access it always
had, while the pack's other nodes use `eng`. For engine code that a major pack runs alongside its isolated
nodes, see [Engine extensions](engine-extensions.md).

## Next

- [Converting a pack](converting.md): running the converter and finishing what it leaves.
- [Pack isolation](pack-isolation.md#making-sure-a-pack-works): testing the result both ways.
