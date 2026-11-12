# Moving a pack from V3 to V4

A V3 pack imports `comfy_api.latest`. It defines its nodes with `io` and `ui`, and to do the work
it calls ComfyUI's own modules, such as `comfy.model_management`, `comfy.utils`, `nodes` and
`folder_paths`, or calls methods on the objects it receives. A V4 pack defines its nodes the same
way and makes those calls through `comfy_api.v4`.

## What stays the same

- `io` and `ui` are the same modules: `comfy_api.v4.io` is `comfy_api.latest.io`. Inputs,
  outputs, schemas and `io.NodeOutput` do not change.
- The entry point: `comfy_entrypoint()` returns a `ComfyExtension` whose `get_node_list` returns
  the node classes.
- Node ids. Keep each `node_id` as it is, and saved workflows that use the node still load.
- ComfyUI loads V3 and V4 packs side by side, and their nodes connect to each other in one
  workflow. Moving one pack does not affect another.

## What changes

1. The import: `from comfy_api.v4 import ...` instead of `from comfy_api.latest import ...`.
   Import `ComfyExtension` from `comfy_api.v4` too; it adds `get_routes`.
2. Calls into ComfyUI go through `engine`, awaited in an `async def execute`, or through
   `ComfyAPISync()` in a plain `def execute`. The objects a node receives, such as a `MODEL`, `CLIP`
   or `VAE`, are passed to the call instead of being called directly.
3. Helpers come from `util` and folders from `util.folders`, instead of `comfy.utils`,
   `node_helpers` and `folder_paths`.

A V3 node:

```python
import comfy.model_management
from comfy_api.latest import ComfyExtension, io


class EncodeImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ExampleEncodeImage",
            inputs=[io.Image.Input("image"), io.Vae.Input("vae")],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, image, vae) -> io.NodeOutput:
        comfy.model_management.throw_exception_if_processing_interrupted()
        return io.NodeOutput({"samples": vae.encode(image)})


class Extension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [EncodeImage]


async def comfy_entrypoint() -> Extension:
    return Extension()
```

The same node in V4:

```python
from comfy_api.v4 import ComfyExtension, engine, io


class EncodeImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ExampleEncodeImage",
            inputs=[io.Image.Input("image"), io.Vae.Input("vae")],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    async def execute(cls, image, vae) -> io.NodeOutput:
        await engine.execution.check_interrupted()
        return io.NodeOutput({"samples": await engine.vae.encode(vae, image)})


class Extension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [EncodeImage]


async def comfy_entrypoint() -> Extension:
    return Extension()
```

## Common calls

| V3 pack code | V4 |
|---|---|
| `vae.encode(pixels)`, `vae.decode(samples)` | `engine.vae.encode(vae, pixels)`, `engine.vae.decode(vae, samples)` |
| `model.clone()`, `clip.clone()` | `engine.model.clone(model)`, `engine.clip.clone(clip)` |
| `clip.tokenize(text)` | `engine.clip.tokenize(clip, text)` |
| `comfy.model_management.throw_exception_if_processing_interrupted()` | `engine.execution.check_interrupted()` |
| `comfy.model_management.processing_interrupted()` | `engine.execution.is_interrupted()` |
| `comfy.model_management.soft_empty_cache()` | `engine.memory.empty_cache()` |
| `nodes.common_ksampler(...)[0]` | `engine.sampler.sample(...)` |
| `comfy.utils.common_upscale(...)` | `util.image.common_upscale(...)` |
| `node_helpers.conditioning_set_values(...)` | `util.conditioning.set_values(...)` |
| `folder_paths.get_filename_list(name)` | `engine.models.list(name)` |
| `folder_paths.get_full_path_or_raise(name, file)` | `util.folders.get_full_path(name, file)`, which gives `None` where that raised |
| `comfy.utils.ProgressBar(total).update_absolute(value)` | `engine.progress.update(value, total)` |
| V3's `ComfyAPI().execution.set_progress(value, max_value)` | `engine.progress.update(value, total)`, which also raises `InterruptProcessingException` when the user cancels |
| `PromptServer.instance.send_sync(event, data, sid)` | `engine.ui.send(event, data)`, to the client that queued the prompt only |
| `comfy.model_management.get_torch_device()`, `intermediate_device()` | `(await engine.device.info())["torch_device"]`, `["intermediate_device"]`: names such as `"cuda:0"` |
| `comfy.sd.load_lora_for_models(model, clip, lora, ...)` | `engine.model.load_lora(model, clip, lora, ...)` |
| `m = model.clone()`, then `m.set_model_sampler_cfg_function(fn)` | `m = engine.model.set_cfg(model, fn)`, which returns the clone. `fn` gets V4's arguments: see below |
| `set_model_sampler_pre_cfg_function`, `set_model_sampler_post_cfg_function`, `set_model_unet_function_wrapper`, `set_model_denoise_mask_function` | `engine.model.set_pre_cfg`, `set_post_cfg`, `set_unet_wrapper`, `set_denoise_mask`, with V4's arguments: see below |
| `set_model_attn1_patch`, `set_model_attn2_patch`, `set_model_attn1_output_patch`, `set_model_attn2_output_patch`, `set_model_input_block_patch`, `set_model_output_block_patch` | `engine.model.set_attn1_patch`, `set_attn2_patch`, `set_attn1_output_patch`, `set_attn2_output_patch`, `set_input_block_patch`, `set_output_block_patch`, with V4's arguments: see below |
| V3's `ComfyAPI().node_replacement.register(...)` | `engine.node_replacement.register(...)` |
| V3's `ComfyAPI().caching.register_provider(...)` | `engine.caching.register_provider(...)`; not from a pack that runs in its own process |

The `engine` calls in the table are awaited in an `async def execute`. The [reference](reference.md)
lists every call, with what it returns and what it raises.

## Functions set on a model get V4's arguments

A function given to `engine.model.set_cfg`, `set_pre_cfg`, `set_post_cfg`, `set_unet_wrapper`,
`set_denoise_mask` or one of the patch calls is called with the arguments V4 defines for it, not
with everything ComfyUI passes the V3 function. The reference lists them under `CfgArgs`,
`PreCfgArgs`, `PostCfgArgs`, `UnetArgs`, `DenoiseMaskInfo`, `BlockInfo` and `AttentionInfo`.
A V3 function that reads another key raises `KeyError` when sampling:

- `timestep` in a CFG function is `sigma`.
- `model`, `model_options` and `conds`, which a function uses to run the model again itself,
  have no V4 form.
- `transformer_index`, `dim_head` and `activations_shape` in a patch's options have no V4 form.
