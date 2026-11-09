# The V2.1 Python API

Generated from `comfy_api/_impl/_engine.py` by `comfy_api/generate_engine_stubs.py --reference`.

A pack declares nodes with `io.ComfyNode`, the existing schema API, and reaches the engine
through one object:

```py
from comfy_api.index import eng, io
```

## Declaring an engine-API node

Set `ENGINE_API = True` on an `io.ComfyNode`. Its `execute` receives its inputs as they
are: tensors, latents, conditioning and plain values, and the engine's MODEL, CLIP and VAE
objects. It returns the same kinds of values. Nodes run in ComfyUI's process by default.
An explicitly configured external execution provider may supply another execution mode.
Persistent store and interactive UI calls require provider services; without them these calls
raise `CallNotSupported`.

## `eng`

Every row in the table below is a method: row `model.clone` is `await eng.model.clone(model)`.
Tensors and data go in and come back as themselves; engine objects come back as engine
objects.

| Member | Signature | Meaning |
|---|---|---|
| `eng.<group>.<name>` | `async (...)` | One table row, for example `eng.vae.decode(vae, samples)`. |
| `eng.ext.<file>.<method>` | `async (...)` | A method of an engine extension the pack ships, `v2/engine/<file>.py`. |

Calls are async, wherever the node runs. A function passed to a `model.set_*` row runs during
sampling and is the one exception: it's a plain function, and it sees tensors, plain data and
their containers, never engine objects. It can't make engine calls.

## Table rows

| Call | Arguments | Returns | Meaning |
|---|---|---|---|
| `eng.clip.clone` | `clip: Clip` | `Clip` | Copy a text encoder so patches don't affect the original. |
| `eng.clip.tokenize` | `clip: Clip, text: str, **kwargs: Any` | `dict` | Tokenize text with the encoder's own tokenizer. Returns plain token data. |
| `eng.clip.encode_from_tokens` | `clip: Clip, tokens: Any, return_pooled: bool = False, return_dict: bool = False` | `Any` | Encode tokens without scheduling: the conditioning tensor, or (cond, pooled) / a dict. |
| `eng.clip.encode_tokens` | `clip: Clip, tokens: Any, **kwargs: Any` | `Conditioning` | Encode tokens, including tokens the pack edited, into conditioning. |
| `eng.clip.add_patches` | `clip: Clip, patches: dict, strength_patch: float = 1.0, strength_model: float = 1.0` | `Clip` | Return a patched copy of the encoder. |
| `eng.clip.read_weight` | `clip: Clip, key: str` | `Tensor` | Read one weight with its patches applied. |
| `eng.model.clone` | `model: Model` | `Model` | Copy a model so patches don't affect the original. |
| `eng.model.add_patches` | `model: Model, patches: dict, strength_patch: float = 1.0, strength_model: float = 1.0` | `Model` | Return a patched copy of the model. |
| `eng.model.read_weight` | `model: Model, key: str` | `Tensor` | Read one weight with its patches applied. |
| `eng.model.load_lora` | `model: Model | None, clip: Clip | None, lora: dict, strength_model: float = 1.0, strength_clip: float = 1.0, lora_metadata: dict | None = None` | `tuple` | Apply a LoRA from its weights to a model and text encoder, as LoraLoader does: patched copies. |
| `eng.model.latent_process_in` | `model: Model, latent: Tensor` | `Tensor` | Scale a latent into the model's input range. |
| `eng.model.latent_process_out` | `model: Model, latent: Tensor` | `Tensor` | Scale a latent from the model's range back, as sampling output is. |
| `eng.model.options` | `model: Model` | `dict` | The model's options (model_options) as plain data. |
| `eng.model.set_option` | `model: Model, path: list[str], value: Any` | `Model` | Return a copy with one model option set, such as transformer_options keys. |
| `eng.model.percent_to_sigma` | `model: Model, percent: float` | `float` | The sigma at a fraction of the sampling schedule (start_percent / end_percent). |
| `eng.model.timestep` | `model: Model, sigma: Tensor` | `Tensor` | The model's timestep for a sigma. |
| `eng.model.sigma` | `model: Model, timestep: Tensor` | `Tensor` | The sigma for a model timestep. |
| `eng.model.set_cfg` | `model: Model, fn: Hook` | `Model` | Return a copy that runs a function on the CFG combination during sampling. |
| `eng.model.set_post_cfg` | `model: Model, fn: Hook` | `Model` | Return a copy that runs a function on the result after CFG during sampling. |
| `eng.model.set_pre_cfg` | `model: Model, fn: Hook` | `Model` | Return a copy that runs a function on the conditioning before CFG during sampling. |
| `eng.model.set_denoise_mask` | `model: Model, fn: Hook` | `Model` | Return a copy that runs a function on the denoise mask during sampling. |
| `eng.model.set_attn1_patch` | `model: Model, fn: Hook` | `Model` | Return a copy that runs a function on self-attention inputs during sampling. |
| `eng.model.set_attn2_patch` | `model: Model, fn: Hook` | `Model` | Return a copy that runs a function on cross-attention inputs during sampling. |
| `eng.model.set_attn1_output_patch` | `model: Model, fn: Hook` | `Model` | Return a copy that runs a function on self-attention outputs during sampling. |
| `eng.model.set_attn2_output_patch` | `model: Model, fn: Hook` | `Model` | Return a copy that runs a function on cross-attention outputs during sampling. |
| `eng.model.set_input_block_patch` | `model: Model, fn: Hook` | `Model` | Return a copy that runs a function on input block features during sampling. |
| `eng.model.set_input_block_patch_after_skip` | `model: Model, fn: Hook` | `Model` | Return a copy that runs a function on input block features after the skip during sampling. |
| `eng.model.set_output_block_patch` | `model: Model, fn: Hook` | `Model` | Return a copy that runs a function on output block features during sampling. |
| `eng.model.set_middle_block_after_patch` | `model: Model, fn: Hook` | `Model` | Return a copy that runs a function on middle block features during sampling. |
| `eng.model.add_wrapper` | `model: Model, kind: str, fn: Hook` | `Model` | Return a copy that runs a function around one stage of sampling; it calls the stage itself. |
| `eng.model.set_unet_wrapper` | `model: Model, fn: Hook` | `Model` | Return a copy whose every model call goes through a function; it calls the model itself. |
| `eng.vae.decode` | `vae: Vae, samples: Tensor` | `Tensor` | Decode a latent tensor, with no output clamping beyond the VAE's own. |
| `eng.vae.decode_tiled` | `vae: Vae, samples: Tensor, **kwargs: Any` | `Tensor` | Decode a latent tensor in tiles. |
| `eng.vae.encode` | `vae: Vae, pixels: Tensor` | `Tensor` | Encode an IMAGE into a latent tensor. |
| `eng.vae.encode_tiled` | `vae: Vae, pixels: Tensor, **kwargs: Any` | `Tensor` | Encode pixels in tiles, with non-square tile sizes allowed. |
| `eng.sampler.named` | `name: str, options: dict | None = None` | `Sampler` | Get a sampler by name, with extra options (for example ge_gamma). |
| `eng.sampler.custom` | `fn: Hook, options: dict | None = None, model_sampling: bool = False` | `Sampler` | A SAMPLER whose sampling loop is a pack function; it calls the model each step. |
| `eng.models.list` | `folder: str` | `list[str]` | List the model files in a model folder. |
| `eng.vae.from_state_dict` | `sd: dict` | `Vae` | Build a VAE from weights the pack holds. |
| `eng.memory.empty_cache` | `` | `None` | Release cached GPU memory. |
| `eng.execution.check_interrupted` | `` | `None` | Raise if the user interrupted the prompt. |
| `eng.execution.is_interrupted` | `` | `bool` | Read whether the user interrupted the prompt without clearing the cancellation flag. |
| `eng.node.run` | `node: str, inputs: dict` | `tuple` | Run a registered node and return its outputs as a tuple. |
| `eng.node.run_output` | `node: str, inputs: dict` | `dict` | Run a registered node and return its result and ui, for output and preview nodes. |
| `eng.progress.update` | `value: float, total: float, preview: Tensor | None = None, format: str = "JPEG"` | `None` | Report this node's progress, optionally with a preview image. |
| `eng.model.set_attention` | `model: Model, backend: str` | `Model` | Return a copy that uses one of ComfyUI's attention implementations. |
| `eng.model.torch_compile` | `model: Model, backend: str = "inductor", mode: str | None = None, fullgraph: bool = False, dynamic: bool | None = None, keys: list[str] | None = None, options: dict | None = None` | `Model` | Return a copy whose diffusion model (or listed submodules) is compiled with torch.compile at sampling. |
| `eng.models.read` | `folder: str, name: str` | `dict` | Read a model file's tensors and metadata from a model folder. |
| `eng.models.load_checkpoint` | `name: str, options: dict | None = None` | `tuple` | Load a checkpoint from the checkpoints folder: (MODEL, CLIP, VAE). |
| `eng.models.load_diffusion_model` | `name: str, options: dict | None = None` | `Model` | Load a diffusion model from the diffusion_models folder. |
| `eng.models.load_vae` | `name: str, device: str | None = None, dtype: str | None = None` | `Vae` | Load a VAE from the vae folder, optionally on a device and dtype. |
| `eng.output.write` | `data: bytes | list[bytes], filename_prefix: str, suffix: str, folder: str = "output", width: int = 0, height: int = 0, trailing_underscore: bool = True` | `dict | list[dict]` | Save bytes the node encoded (one file, or a batch) to the output or temp folder, with ComfyUI's file naming. |
| `eng.files.read` | `ref: str` | `bytes` | Read a file from the input, output or temp folder. |
| `eng.files.list` | `folder: str = "input", subfolder: str = "", recursive: bool = False, extensions: list[str] | None = None` | `list[dict]` | List files in the input, output or temp folder. |
| `eng.files.exists` | `ref: str` | `bool` | Whether a file exists in the input, output or temp folder. |
| `eng.device.info` | `` | `dict` | The devices and dtypes ComfyUI computes with. |
| `eng.memory.unload_all` | `` | `None` | Unload every model from the GPU. |
| `eng.memory.free_bytes` | `device: str | None = None` | `int` | Free memory on a device, in bytes. |
| `eng.ui.progress_text` | `text: str` | `None` | Show text on this node while it runs. |
| `eng.ui.send` | `event: str, data: dict` | `None` | Send a custom event to the client that queued the prompt. |
| `eng.ui.interact` | `kind: str, payload: dict, reuse_last: bool = False, remember: bool = False, timeout: float = 540.0` | `Any` | Ask the user something mid-run (choose images, edit a mask, confirm a prompt) and wait for the answer. |
| `eng.settings.get` | `name: str` | `Any` | Read one of ComfyUI's settings, such as disable_metadata. |
| `eng.store.get` | `key: str, scope: str = "pack", default: Any = None` | `Any` | Read a small value the pack saved, for the pack, this workflow or this node. |
| `eng.store.set` | `key: str, value: Any, scope: str = "pack"` | `None` | Save a small plain-data value that lasts across runs and restarts. |
| `eng.store.delete` | `key: str, scope: str = "pack"` | `bool` | Remove a saved value; whether there was one. |
| `eng.store.keys` | `scope: str = "pack"` | `list[str]` | The keys the pack saved in a scope. |
| `eng.cache.get` | `key: str, default: Any = None` | `Any` | A large value the pack cached (a tensor, a model it loaded), or default. |
| `eng.cache.set` | `key: str, value: Any, ttl: float | None = None` | `None` | Keep a large value in ComfyUI's memory between runs, optionally for ttl seconds. |
| `eng.cache.delete` | `key: str` | `bool` | Drop a cached value; whether there was one. |
| `eng.memory.load_models` | `models: list` | `None` | Load models onto the GPU now. |
| `eng.model.info` | `model: Model` | `dict` | Read-only facts about a diffusion model: its type, latent format, dtype and sigma range. |
| `eng.model.weight_keys` | `model: Model` | `list[str]` | List the model's weight names. |
| `eng.vae.info` | `vae: Vae` | `dict` | Read-only facts about a VAE: latent channels, dtype and compression. |
| `eng.clip.info` | `clip: Clip` | `dict` | Read-only facts about a text encoder. |
| `eng.clip.weight_keys` | `clip: Clip` | `list[str]` | List the text encoder's weight names. |

## Call details

### `eng.clip.clone(clip: Clip) -> Clip`

Copy a text encoder so patches don't affect the original.

```text
Returns:
    A new CLIP sharing weights with ``clip``; patches added to it don't affect ``clip``.
```

### `eng.clip.tokenize(clip: Clip, text: str, **kwargs: Any) -> dict`

Tokenize text with the encoder's own tokenizer. Returns plain token data.

```text
Args:
    clip: The text encoder.
    text: The prompt, including ComfyUI's ``(word:1.2)`` weighting syntax.
    **kwargs: Passed to the tokenizer, for example ``return_word_ids=True``.

Returns:
    ``{encoder_name: [[(token_id, weight, ...), ...], ...]}``: plain lists a pack may edit
    (reweight, splice) before ``encode_tokens``.
```

### `eng.clip.encode_from_tokens(clip: Clip, tokens: Any, return_pooled: bool = False, return_dict: bool = False) -> Any`

Encode tokens without scheduling: the conditioning tensor, or (cond, pooled) / a dict.

```text
Args:
    tokens: From ``clip.tokenize``.
    return_pooled: Also return the pooled output: ``(cond, pooled)``.
    return_dict: Return ComfyUI's dict form.

The unscheduled encode V1 nodes call as ``clip.encode_from_tokens``; ``clip.encode_tokens`` is the scheduled one CLIPTextEncode uses.
```

### `eng.clip.encode_tokens(clip: Clip, tokens: Any, **kwargs: Any) -> Conditioning`

Encode tokens, including tokens the pack edited, into conditioning.

```text
Args:
    clip: The text encoder.
    tokens: What ``tokenize`` returned, edited or not.
    **kwargs: ``add_dict`` (extra keys merged into each conditioning dict),
        ``unprojected``, ``show_pbar``.

Returns:
    CONDITIONING: ``[[cond_tensor, {"pooled_output": tensor, ...}], ...]``, the same value
    ``CLIPTextEncode`` produces.
```

### `eng.clip.add_patches(clip: Clip, patches: dict, strength_patch: float = 1.0, strength_model: float = 1.0) -> Clip`

Return a patched copy of the encoder.

```text
Args:
    clip: The text encoder.
    patches: ``{weight_key: patch}``, the format ``ModelPatcher.add_patches`` takes; a patch
        is a tuple such as ``(tensor,)`` for a weight difference.
    strength_patch: Scale for the patch.
    strength_model: Scale for the original weight.

Returns:
    A patched copy; ``clip`` is unchanged.
```

### `eng.clip.read_weight(clip: Clip, key: str) -> Tensor`

Read one weight with its patches applied.

```text
Returns:
    The weight tensor named ``key`` with current patches applied (see ``weight_keys``).
```

### `eng.model.clone(model: Model) -> Model`

Copy a model so patches don't affect the original.

```text
Returns:
    A new MODEL sharing weights with ``model``; patches and functions added to it don't affect ``model``.
```

### `eng.model.add_patches(model: Model, patches: dict, strength_patch: float = 1.0, strength_model: float = 1.0) -> Model`

Return a patched copy of the model.

```text
Args:
    model: The diffusion model.
    patches: ``{weight_key: patch}``, the format ``ModelPatcher.add_patches`` takes; a patch
        is a tuple such as ``(tensor,)`` for a weight difference.
    strength_patch: Scale for the patch.
    strength_model: Scale for the original weight.

Returns:
    A patched copy; ``model`` is unchanged.

Example::

    key = "diffusion_model.out.2.weight"
    model = await eng.model.add_patches(model, {key: (delta,)}, strength_patch=0.5)
```

### `eng.model.read_weight(model: Model, key: str) -> Tensor`

Read one weight with its patches applied.

```text
Returns:
    The weight tensor named ``key`` with current patches applied (see ``weight_keys``).
```

### `eng.model.load_lora(model: Model | None, clip: Clip | None, lora: dict, strength_model: float = 1.0, strength_clip: float = 1.0, lora_metadata: dict | None = None) -> tuple`

Apply a LoRA from its weights to a model and text encoder, as LoraLoader does: patched copies.

```text
Args:
    model: The diffusion model, or None to patch only the text encoder.
    clip: The text encoder, or None to patch only the model.
    lora: The LoRA's weights, ``{name: tensor}``, as ``util.load_torch_file`` or
        ``eng.models.read("loras", name)["sd"]`` gives them; any LoRA format ComfyUI reads.
    strength_model, strength_clip: As ``LoraLoader`` takes them.
    lora_metadata: The LoRA file's metadata (``eng.models.read(...)["metadata"]``), attached to the
        patched copies as ComfyUI's loader does; optional.

Returns:
    ``(model, clip)``: patched copies (None where None was given); the originals are unchanged.
```

### `eng.model.latent_process_in(model: Model, latent: Tensor) -> Tensor`

Scale a latent into the model's input range.

```text
Args:
    latent: A latent tensor (``latent["samples"]``).

Returns:
    The latent scaled into the model's input range, as sampling does.
```

### `eng.model.set_option(model: Model, path: list[str], value: Any) -> Model`

Return a copy with one model option set, such as transformer_options keys.

```text
Args:
    model: The diffusion model.
    path: Keys into ``model_options``, such as ``["transformer_options", "my_pack_strength"]``.
    value: Plain data or tensors.

Returns:
    A copy with the option set; the dicts along the path are copied, so ``model`` is unchanged.
    The V1 form ``m = model.clone(); m.model_options["transformer_options"]["k"] = v`` becomes
    ``m = await eng.model.set_option(model, ["transformer_options", "k"], v)``.
```

### `eng.model.percent_to_sigma(model: Model, percent: float) -> float`

The sigma at a fraction of the sampling schedule (start_percent / end_percent).

```text
Args:
    percent: 0.0 (start of sampling) to 1.0 (end).

Returns:
    The sigma at that point, as ComfyUI's start_percent / end_percent options compute it.
```

### `eng.model.set_cfg(model: Model, fn: Hook) -> Model`

Return a copy that runs a function on the CFG combination during sampling.

```text
Replaces the CFG combination. ``fn(args) -> Tensor`` gets a dict with ``cond`` and
``uncond`` (noise predictions, ``input - denoised``), ``cond_denoised``, ``uncond_denoised``,
``cond_scale``, ``timestep``/``sigma``, ``input`` and ``model_options``, and returns the combined
noise prediction (stock CFG: ``uncond + (cond - uncond) * cond_scale``).

Example::

    def scaled(args):
        return args["uncond"] + (args["cond"] - args["uncond"]) * args["cond_scale"] * 0.9

    model = await eng.model.set_cfg(model, scaled)

The function runs during sampling, on every step, and must not be async. It sees tensors,
plain data and their containers only: engine objects in its arguments (the model, for
one) are removed, and it can't make ``eng`` calls. It's kept with the returned copy, so
the original model is unchanged.
```

### `eng.model.set_post_cfg(model: Model, fn: Hook) -> Model`

Return a copy that runs a function on the result after CFG during sampling.

```text
Runs after CFG. ``fn(args) -> Tensor`` gets a dict with ``denoised`` (the CFG result),
``cond_denoised``, ``uncond_denoised``, ``cond_scale``, ``sigma``, ``input`` and ``model_options``,
and returns the new ``denoised``. Several can be added; they run in order.

The function runs during sampling, on every step, and must not be async. It sees tensors,
plain data and their containers only: engine objects in its arguments (the model, for
one) are removed, and it can't make ``eng`` calls. It's kept with the returned copy, so
the original model is unchanged.
```

### `eng.model.set_pre_cfg(model: Model, fn: Hook) -> Model`

Return a copy that runs a function on the conditioning before CFG during sampling.

```text
Runs before CFG. ``fn(args) -> list`` gets a dict with ``conds_out`` (the model outputs for
cond and uncond), ``cond_scale``, ``timestep``/``sigma``, ``input`` and ``model_options``, and
returns the new ``conds_out`` list.

The function runs during sampling, on every step, and must not be async. It sees tensors,
plain data and their containers only: engine objects in its arguments (the model, for
one) are removed, and it can't make ``eng`` calls. It's kept with the returned copy, so
the original model is unchanged.
```

### `eng.model.set_denoise_mask(model: Model, fn: Hook) -> Model`

Return a copy that runs a function on the denoise mask during sampling.

```text
``fn(sigma, denoise_mask, extra_options) -> Tensor`` gets the current sigma and denoise
mask (and ``extra_options["sigmas"]``), and returns the mask to use for this step.

The function runs during sampling, on every step, and must not be async. It sees tensors,
plain data and their containers only: engine objects in its arguments (the model, for
one) are removed, and it can't make ``eng`` calls. It's kept with the returned copy, so
the original model is unchanged.
```

### `eng.model.set_attn1_patch(model: Model, fn: Hook) -> Model`

Return a copy that runs a function on self-attention inputs during sampling.

```text
``fn(q, k, v, extra_options) -> (q, k, v)`` runs before each self-attention, with the
query input and the key/value context; ``extra_options`` has the block, its index, heads,
``sigmas`` and ``cond_or_uncond``.

The function runs during sampling, on every step, and must not be async. It sees tensors,
plain data and their containers only: engine objects in its arguments (the model, for
one) are removed, and it can't make ``eng`` calls. It's kept with the returned copy, so
the original model is unchanged.
```

### `eng.model.set_attn2_patch(model: Model, fn: Hook) -> Model`

Return a copy that runs a function on cross-attention inputs during sampling.

```text
``fn(q, context, value, extra_options) -> (q, context, value)`` runs before each
cross-attention; ``extra_options`` as for ``set_attn1_patch``.

The function runs during sampling, on every step, and must not be async. It sees tensors,
plain data and their containers only: engine objects in its arguments (the model, for
one) are removed, and it can't make ``eng`` calls. It's kept with the returned copy, so
the original model is unchanged.
```

### `eng.model.set_attn1_output_patch(model: Model, fn: Hook) -> Model`

Return a copy that runs a function on self-attention outputs during sampling.

```text
``fn(out, extra_options) -> Tensor`` runs on each self-attention output.

The function runs during sampling, on every step, and must not be async. It sees tensors,
plain data and their containers only: engine objects in its arguments (the model, for
one) are removed, and it can't make ``eng`` calls. It's kept with the returned copy, so
the original model is unchanged.
```

### `eng.model.set_attn2_output_patch(model: Model, fn: Hook) -> Model`

Return a copy that runs a function on cross-attention outputs during sampling.

```text
``fn(out, extra_options) -> Tensor`` runs on each cross-attention output.

The function runs during sampling, on every step, and must not be async. It sees tensors,
plain data and their containers only: engine objects in its arguments (the model, for
one) are removed, and it can't make ``eng`` calls. It's kept with the returned copy, so
the original model is unchanged.
```

### `eng.model.set_input_block_patch(model: Model, fn: Hook) -> Model`

Return a copy that runs a function on input block features during sampling.

```text
``fn(h, transformer_options) -> Tensor`` runs on each UNet input block's features.

The function runs during sampling, on every step, and must not be async. It sees tensors,
plain data and their containers only: engine objects in its arguments (the model, for
one) are removed, and it can't make ``eng`` calls. It's kept with the returned copy, so
the original model is unchanged.
```

### `eng.model.set_input_block_patch_after_skip(model: Model, fn: Hook) -> Model`

Return a copy that runs a function on input block features after the skip during sampling.

```text
``fn(h, transformer_options) -> Tensor`` runs on each UNet input block's features after the skip connection is stored.

The function runs during sampling, on every step, and must not be async. It sees tensors,
plain data and their containers only: engine objects in its arguments (the model, for
one) are removed, and it can't make ``eng`` calls. It's kept with the returned copy, so
the original model is unchanged.
```

### `eng.model.set_output_block_patch(model: Model, fn: Hook) -> Model`

Return a copy that runs a function on output block features during sampling.

```text
``fn(h, hsp, transformer_options) -> (h, hsp)`` runs on each UNet output block's features and its skip input.

The function runs during sampling, on every step, and must not be async. It sees tensors,
plain data and their containers only: engine objects in its arguments (the model, for
one) are removed, and it can't make ``eng`` calls. It's kept with the returned copy, so
the original model is unchanged.
```

### `eng.model.set_middle_block_after_patch(model: Model, fn: Hook) -> Model`

Return a copy that runs a function on middle block features during sampling.

```text
``fn(args) -> dict`` gets ``{"h", "x", "emb", "context", "y", "timesteps", "transformer_options"}`` after the UNet middle block, and returns a dict whose ``"h"`` replaces the features.

The function runs during sampling, on every step, and must not be async. It sees tensors,
plain data and their containers only: engine objects in its arguments (the model, for
one) are removed, and it can't make ``eng`` calls. It's kept with the returned copy, so
the original model is unchanged.
```

### `eng.model.add_wrapper(model: Model, kind: str, fn: Hook) -> Model`

Return a copy that runs a function around one stage of sampling; it calls the stage itself.

```text
Args:
    model: The diffusion model.
    kind: The stage: ``"outer_sample"``, ``"prepare_sampling"``, ``"sampler_sample"``,
        ``"predict_noise"``, ``"calc_cond_batch"``, ``"apply_model"`` or ``"diffusion_model"``
        (ComfyUI's ``WrappersMP``).
    fn: ``fn(executor, *args, **kwargs)``: runs instead of the stage, gets the stage as
        ``executor``, and returns what the stage returns, usually ``executor(*args, **kwargs)``
        with its own changes before or after.

Returns:
    A copy; ``model`` is unchanged.

Example::

    def double_guidance(executor, *args, **kwargs):
        out = executor(*args, **kwargs)
        return out * 1.05

    model = await eng.model.add_wrapper(model, "predict_noise", double_guidance)

The function runs during sampling and must not be async. What the engine hands it to call (the
stage, the model, the callback) is an engine function: call it like any function, with tensors,
plain data and what you were given. Engine objects in its arguments (the model, a guider, a
ControlNet) arrive as opaque values: pass them on, but don't look into them. It can't make ``eng``
calls. It runs wherever the pack runs; in a pack's own process each call to an engine function
goes to ComfyUI and back.
```

### `eng.model.set_unet_wrapper(model: Model, fn: Hook) -> Model`

Return a copy whose every model call goes through a function; it calls the model itself.

```text
``fn(apply_model, args) -> Tensor`` runs for every model call: ``args`` has ``input``,
``timestep``, ``c`` (the conditioning keywords) and ``cond_or_uncond``, and the function returns
the model output, usually ``apply_model(args["input"], args["timestep"], **args["c"])`` with its own
changes. A wrapper already on the model runs inside this one (``apply_model`` goes through it), so
wrappers from different nodes compose. The V1 form is ``model.set_model_unet_function_wrapper``.

The function runs during sampling and must not be async. What the engine hands it to call (the
stage, the model, the callback) is an engine function: call it like any function, with tensors,
plain data and what you were given. Engine objects in its arguments (the model, a guider, a
ControlNet) arrive as opaque values: pass them on, but don't look into them. It can't make ``eng``
calls. It runs wherever the pack runs; in a pack's own process each call to an engine function
goes to ComfyUI and back.
```

### `eng.vae.decode(vae: Vae, samples: Tensor) -> Tensor`

Decode a latent tensor, with no output clamping beyond the VAE's own.

```text
Args:
    vae: The VAE.
    samples: The latent tensor (``latent["samples"]``).

Returns:
    Pixels, channels last, values as the VAE produces them. Video VAEs return a time
    dimension; ``await eng.node.run("VAEDecode", {"vae": vae, "samples": latent})`` gives the
    stock IMAGE batch instead.
```

### `eng.vae.decode_tiled(vae: Vae, samples: Tensor, **kwargs: Any) -> Tensor`

Decode a latent tensor in tiles.

```text
Args:
    vae: The VAE.
    samples: The latent tensor.
    **kwargs: ``tile_x``, ``tile_y``, ``overlap``, and for video ``tile_t``, ``overlap_t``.

Returns:
    Pixels, as ``decode``.
```

### `eng.vae.encode_tiled(vae: Vae, pixels: Tensor, **kwargs: Any) -> Tensor`

Encode pixels in tiles, with non-square tile sizes allowed.

```text
Args:
    vae: The VAE.
    pixels: An IMAGE tensor ``[B, H, W, C]``.
    **kwargs: ``tile_x``, ``tile_y`` (need not be equal), ``overlap``, ``tile_t``, ``overlap_t``.

Returns:
    The latent tensor; wrap it as ``{"samples": tensor}`` for a LATENT output.
```

### `eng.sampler.named(name: str, options: dict | None = None) -> Sampler`

Get a sampler by name, with extra options (for example ge_gamma).

```text
Args:
    name: A KSampler sampler name, such as ``"euler"`` or ``"dpmpp_2m"``.
    options: Extra sampler options, such as ``{"ge_gamma": 2.0}``.

Returns:
    A SAMPLER, to return from the node or pass to ``node.run("SamplerCustom", ...)``.
```

### `eng.sampler.custom(fn: Hook, options: dict | None = None, model_sampling: bool = False) -> Sampler`

A SAMPLER whose sampling loop is a pack function; it calls the model each step.

```text
Args:
    fn: ``fn(model, x, sigmas, extra_args, callback=None, disable=None, **options) -> Tensor``, a
        sampling loop as k-diffusion samplers write it: ``model(x, sigma * s_in, **extra_args)``
        denoises, and ``callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i],
        "denoised": denoised})`` reports each step (previews and progress).
    options: Extra keyword arguments for ``fn``. A SAMPLER among them reaches ``fn`` as an engine
        function, ``sampler(model, x, sigmas, extra_args=None, callback=None, disable=None)``, so a
        sampler can wrap another one with its own model function.
    model_sampling: Also pass ``model_sampling=``, the model's sampling facts: ``sigma_min``,
        ``sigma_max``, ``sigmas``, and the engine functions ``sigma(t)``, ``timestep(s)`` and
        ``percent_to_sigma(p)``.

Returns:
    A SAMPLER, to return from the node or pass to ``node.run("SamplerCustom", ...)``.

The function runs during sampling and must not be async. What the engine hands it to call (the
stage, the model, the callback) is an engine function: call it like any function, with tensors,
plain data and what you were given. Engine objects in its arguments (the model, a guider, a
ControlNet) arrive as opaque values: pass them on, but don't look into them. It can't make ``eng``
calls. It runs wherever the pack runs; in a pack's own process each call to an engine function
goes to ComfyUI and back.
```

### `eng.models.list(folder: str) -> list[str]`

List the model files in a model folder.

```text
Args:
    folder: A model folder name: ``"checkpoints"``, ``"loras"``, ``"vae"``, ``"diffusion_models"``, ``"text_encoders"``...

Returns:
    The file names in it, as model combo inputs list them.
```

### `eng.vae.from_state_dict(sd: dict) -> Vae`

Build a VAE from weights the pack holds.

```text
Args:
    sd: ``{weight_name: tensor}`` for a VAE ComfyUI can detect from its weights.

Returns:
    A VAE.
```

### `eng.memory.empty_cache() -> None`

Release cached GPU memory.

```text
Frees cached, unused GPU memory. Models stay loaded.
```

### `eng.execution.check_interrupted() -> None`

Raise if the user interrupted the prompt.

```text
Raises the interruption ComfyUI uses when the user cancels the prompt. Call it inside long loops.
```

### `eng.execution.is_interrupted() -> bool`

Read whether the user interrupted the prompt without clearing the cancellation flag.

```text
Returns the current prompt cancellation flag as a boolean without consuming or resetting it. Use when a node must return partial results or finish a model's own stop callback. Broker failures propagate to the caller.
```

### `eng.node.run(node: str, inputs: dict) -> tuple`

Run a registered node and return its outputs as a tuple.

```text
Args:
    node: A node id as workflows name it, for example ``"KSampler"`` or ``"ConditioningSetMask"``.
    inputs: The node's inputs by name. Values are what nodes pass each other: tensors,
        latents, conditioning, and engine objects from this node's inputs or earlier calls.

Returns:
    The node's outputs as a tuple. Its hidden inputs (prompt, unique id) are filled from the
    running prompt. Nodes that ship with ComfyUI are reachable from every dispatch; another
    pack's nodes might not be.

Example::

    (samples,) = await eng.node.run("KSampler", {"model": model, "positive": pos, "negative": neg,
        "latent_image": latent, "seed": 1, "steps": 20, "cfg": 7.0, "sampler_name": "euler",
        "scheduler": "normal", "denoise": 1.0})
```

### `eng.node.run_output(node: str, inputs: dict) -> dict`

Run a registered node and return its result and ui, for output and preview nodes.

```text
Like ``node.run``, for output and preview nodes.

Returns:
    ``{"result": tuple, "ui": dict | None}``. Return the ui from your node to show it:
    ``io.NodeOutput(ui=(await eng.node.run_output("PreviewImage", {"images": image}))["ui"])``.
```

### `eng.progress.update(value: float, total: float, preview: Tensor | None = None, format: str = "JPEG") -> None`

Report this node's progress, optionally with a preview image.

```text
Args:
    value: Steps done.
    total: Steps in all.
    preview: Optional IMAGE tensor (``[H, W, C]`` or the first of ``[B, H, W, C]``, 0..1) to show
        on the node, as samplers show their previews.
    format: ``"JPEG"`` or ``"PNG"``.
```

### `eng.model.set_attention(model: Model, backend: str) -> Model`

Return a copy that uses one of ComfyUI's attention implementations.

```text
Args:
    model: The diffusion model.
    backend: An attention implementation ComfyUI has: ``"pytorch"``, ``"sage"``, ``"sage3"``,
        ``"flash"``, ``"xformers"``, ``"sub_quad"``, ``"optimized"``... Which are available
        depends on the installed kernels; an unknown name raises and lists them.

Returns:
    A copy whose attention calls go to that implementation.
```

### `eng.model.torch_compile(model: Model, backend: str = "inductor", mode: str | None = None, fullgraph: bool = False, dynamic: bool | None = None, keys: list[str] | None = None, options: dict | None = None) -> Model`

Return a copy whose diffusion model (or listed submodules) is compiled with torch.compile at sampling.

```text
Args:
    model: The diffusion model.
    backend: A torch.compile backend, such as ``"inductor"`` or ``"cudagraphs"``.
    mode, fullgraph, dynamic, options: Passed to ``torch.compile``.
    keys: Submodules of the model to compile, such as ``["diffusion_model.blocks.0"]``
        (names from ``model.weight_keys``, without the parameter suffix). Default: the
        whole diffusion model.

Returns:
    A copy that compiles when it first samples. Compilation happens in ComfyUI's process.
```

### `eng.models.read(folder: str, name: str) -> dict`

Read a model file's tensors and metadata from a model folder.

```text
Args:
    folder: A model folder: ``"loras"``, ``"checkpoints"``, ``"vae"``...
    name: A file name from ``models.list(folder)``.

Returns:
    ``{"sd": {weight_name: tensor}, "metadata": dict}``.
```

### `eng.models.load_checkpoint(name: str, options: dict | None = None) -> tuple`

Load a checkpoint from the checkpoints folder: (MODEL, CLIP, VAE).

```text
Args:
    name: A file in the checkpoints folder.
    options: ComfyUI ``model_options`` for the diffusion model.

Returns:
    ``(model, clip, vae)``, as ``CheckpointLoaderSimple`` gives.
```

### `eng.models.load_diffusion_model(name: str, options: dict | None = None) -> Model`

Load a diffusion model from the diffusion_models folder.

```text
Args:
    name: A file in the diffusion_models folder.
    options: ComfyUI ``model_options``, for example ``{"dtype": "float8_e4m3fn"}``.

Returns:
    The MODEL, as ``UNETLoader`` gives.
```

### `eng.models.load_vae(name: str, device: str | None = None, dtype: str | None = None) -> Vae`

Load a VAE from the vae folder, optionally on a device and dtype.

```text
Args:
    name: A file in the vae folder.
    device: For example ``"cpu"`` or ``"cuda:0"``; default ComfyUI's choice.
    dtype: ``"float32"``, ``"float16"`` or ``"bfloat16"``; default ComfyUI's choice.

Returns:
    The VAE. For ComfyUI's built-in names (``"taesd"``, ``"pixel_space"``...) use
    ``eng.node.run("VAELoader", ...)``.
```

### `eng.output.write(data: bytes | list[bytes], filename_prefix: str, suffix: str, folder: str = "output", width: int = 0, height: int = 0, trailing_underscore: bool = True) -> dict | list[dict]`

Save bytes the node encoded (one file, or a batch) to the output or temp folder, with ComfyUI's file naming.

```text
Saves bytes the node encoded itself (a PNG, a .safetensors file, text...).
``ui.PreviewImage``, ``ui.ImageSaveHelper`` and the other ``ui`` savers write through this call,
so they work wherever the node runs.

Args:
    data: The file's bytes, or a list of them for a batch. A batch shares one counter run, and
        ``%batch_num%`` in the prefix becomes each file's index, as ``SaveImage`` names files.
    filename_prefix: As ``SaveImage`` takes it, including subfolders and ``%date%``-style
        variables; it can't leave the folder.
    suffix: The file extension, such as ``"png"``.
    folder: ``"output"`` or ``"temp"`` (for previews).
    width, height: For ``%width%`` and ``%height%`` in the prefix.
    trailing_underscore: Name files ``<prefix>_00001_.<suffix>`` (ComfyUI's image naming, the
        default) or ``<prefix>_00001.<suffix>`` (its audio naming).

Returns:
    ``{"filename", "subfolder", "type"}`` for one file, a list of them for a batch: entries of a
    ui ``images`` list, so ``io.NodeOutput(ui={"images": [entry]})`` shows it.
```

### `eng.files.read(ref: str) -> bytes`

Read a file from the input, output or temp folder.

```text
Args:
    ref: A file ref: ``"photo.png"`` (input), ``"sub/a.png [input]"``, ``"x.png [output]"``, ``"x.png [temp]"``.

Returns:
    The file's bytes. Refs can't leave those folders.
```

### `eng.files.list(folder: str = "input", subfolder: str = "", recursive: bool = False, extensions: list[str] | None = None) -> list[dict]`

List files in the input, output or temp folder.

```text
Args:
    folder: ``"input"``, ``"output"`` or ``"temp"``.
    subfolder: Inside it.
    recursive: Include subfolders.
    extensions: Only these, such as ``["png", "jpg"]``.

Returns:
    ``[{"ref", "name", "size", "mtime"}]``, sorted; pass ``ref`` to ``files.read``.
```

### `eng.files.exists(ref: str) -> bool`

Whether a file exists in the input, output or temp folder.

```text
Returns:
    Whether ``ref`` names a file in the input, output or temp folder.
```

### `eng.device.info() -> dict`

The devices and dtypes ComfyUI computes with.

```text
Returns:
    ``{"torch_device", "intermediate_device", "intermediate_dtype", "unet_dtype",
    "offload_device", "vae_device", "vae_dtype", "text_encoder_device",
            "vae_offload_device", "text_encoder_offload_device"}``, as strings such as
    ``"cuda:0"`` and ``"float16"`` (``getattr(torch, info["unet_dtype"])``). Put tensors a node
    returns on ``intermediate_device`` in ``intermediate_dtype``, as ComfyUI's nodes do.
```

### `eng.memory.unload_all() -> None`

Unload every model from the GPU.

```text
Unloads every loaded model from the GPU and frees cached memory.
```

### `eng.memory.free_bytes(device: str | None = None) -> int`

Free memory on a device, in bytes.

```text
Args:
    device: Default ComfyUI's compute device.

Returns:
    Free memory on it, in bytes, counting memory ComfyUI can free.
```

### `eng.ui.progress_text(text: str) -> None`

Show text on this node while it runs.

```text
Args:
    text: Shown on the node, under its title, until the next update.
```

### `eng.ui.send(event: str, data: dict) -> None`

Send a custom event to the client that queued the prompt.

```text
Args:
    event: The event name a pack's frontend extension listens for.
    data: Plain data, sent as JSON.

Sent to the client that queued the running prompt, as ``PromptServer.send_sync`` does.
```

### `eng.ui.interact(kind: str, payload: dict, reuse_last: bool = False, remember: bool = False, timeout: float = 540.0) -> Any`

Ask the user something mid-run (choose images, edit a mask, confirm a prompt) and wait for the answer.

```text
Args:
    kind: ``"image-choice"``, ``"prompt-await"`` or ``"mask-edit"``.
    payload: What the pack's frontend shows (plain JSON, up to 64 KB).
    reuse_last: Answer with this node's last answer of this kind without asking.
    remember: Keep this answer for ``reuse_last``.
    timeout: Seconds to wait, 1 to 540.

Returns:
    The user's answer, checked against the kind's contract. The pack's frontend extension shows the
    request and posts the answer back (the event and route are in ``comfy_api/README.md``).
```

### `eng.settings.get(name: str) -> Any`

Read one of ComfyUI's settings, such as disable_metadata.

```text
Args:
    name: ``"disable_metadata"`` (bool: don't write prompt metadata into saved files), ``"preview_method"``, ``"preview_size"``, ``"fast"`` (the names of the enabled ``--fast`` optimizations, sorted; read-only).

Returns:
    The setting's value, as plain data.
```

### `eng.store.get(key: str, scope: str = "pack", default: Any = None) -> Any`

Read a small value the pack saved, for the pack, this workflow or this node.

```text
Args:
    key: The value's name.
    scope: ``"pack"`` (every workflow), ``"workflow"`` (this workflow, by its id) or ``"node"``
        (this node in this workflow).
    default: What to return if nothing is saved.

Returns:
    The saved plain data. Values persist in ComfyUI's user folder, the same wherever the node runs:
    keys up to 128 bytes, values up to 1 MB, 256 entries and 8 MB per scope.
```

### `eng.store.set(key: str, value: Any, scope: str = "pack") -> None`

Save a small plain-data value that lasts across runs and restarts.

```text
Args:
    key: The value's name.
    value: Plain data (JSON): strings, numbers, booleans, lists, dicts.
    scope: As ``store.get``.
```

### `eng.cache.set(key: str, value: Any, ttl: float | None = None) -> None`

Keep a large value in ComfyUI's memory between runs, optionally for ttl seconds.

```text
Args:
    key: The value's name, per pack.
    value: Tensors, plain data, or engine objects such as a model the pack loaded.
    ttl: Seconds to keep it; default until it's pushed out.

Kept in ComfyUI's memory, least recently used first out within ``COMFY_PACK_CACHE_MB``
(default 2048 MB across packs). Nothing survives a restart; use ``store`` for that.
```

### `eng.memory.load_models(models: list) -> None`

Load models onto the GPU now.

```text
Args:
    models: MODEL, CLIP or VAE objects to load onto the GPU before the node uses them.
```

### `eng.model.info(model: Model) -> dict`

Read-only facts about a diffusion model: its type, latent format, dtype and sigma range.

```text
Returns:
    ``model_type``, ``config_class``, ``unet_config``, ``latent_scale_factor``, ``latent_channels``,
    ``zsnr``, ``dtype``, ``sigma_min``, ``sigma_max``. All plain values.
```

### `eng.model.weight_keys(model: Model) -> list[str]`

List the model's weight names.

```text
Returns:
    The sorted weight names, for ``read_weight`` and ``add_patches``.
```

### `eng.vae.info(vae: Vae) -> dict`

Read-only facts about a VAE: latent channels, dtype and compression.

```text
Returns:
    ``latent_channels``, ``dtype``, ``spacial_compression_decode``, ``spacial_compression_encode``,
    ``temporal_compression_decode``, ``downscale_ratio``, ``upscale_ratio``. All plain values.
```

### `eng.clip.info(clip: Clip) -> dict`

Read-only facts about a text encoder.

```text
Returns:
    ``layer_idx``: the CLIP skip layer, or None.
```

### `eng.clip.weight_keys(clip: Clip) -> list[str]`

List the text encoder's weight names.

```text
Returns:
    The sorted weight names, for ``read_weight`` and ``add_patches``.
```

## Pure helpers: `util`

Copies of ComfyUI's tensor and data helpers, importable from pack code and identical wherever a
node runs. Each is tested against the ComfyUI original, so they can't drift.

| Name | Same as |
|---|---|
| `util.common_upscale(samples, width, height, method, crop)` | `comfy.utils.common_upscale` (with `bislerp`, `lanczos`) |
| `util.prepare_noise(latent, seed, noise_inds=None)` | `comfy.sample.prepare_noise` |
| `util.conditioning_set_values(cond, values, append=False)` | `node_helpers.conditioning_set_values` (and `_with_timestep_range`) |
| `util.pillow(fn, arg)` | `node_helpers.pillow` |
| `util.MAX_RESOLUTION`, `util.SAMPLER_NAMES`, `util.SCHEDULER_NAMES` | `nodes.MAX_RESOLUTION`, `comfy.samplers.KSampler.SAMPLERS` / `SCHEDULERS` |

## Model-file dropdowns: `io.ModelFolder`

`io.Combo.Input("lora_name", options=io.ModelFolder("loras"))` lists a model folder each time ComfyUI
builds the schema, so a pack never reads folders itself; `extensions=["safetensors"]` filters it.

## Functions that call back into the engine

`model.add_wrapper`, `model.set_unet_wrapper` and `sampler.custom` take a pack function that ComfyUI
calls during sampling and that calls back into the engine: the stage it wraps, the model, the preview
callback. Those arrive as **engine functions** (call them like any function), engine objects in the
arguments arrive as **opaque values** (pass them on, don't look into them), and tensors, numpy arrays
and plain data arrive as themselves. A SAMPLER passed to `sampler.custom` arrives as an engine
function too, so a sampler can wrap another. The same code runs in ComfyUI's process and in a pack's
own process, where each engine-function call goes to ComfyUI and back.

## Registering samplers, schedulers, model folders and node replacements

Declared on the pack's `ComfyExtension`, so ComfyUI can list them without running the pack:

```py
class Extension(ComfyExtension):
    async def get_samplers(self):          return {"my_sampler": sampling_loop}   # as sampler.custom
    async def get_schedulers(self):        return {"my_schedule": schedule}       # fn(model_sampling, steps)
    async def get_model_folders(self):     return {"onnx": ["onnx"]}
    async def get_node_replacements(self): return [io.NodeReplace(new_node_id="New", old_node_id="Old")]
```

A scheduler's `model_sampling` is `{"sigma_min", "sigma_max", "sigmas", "sigma", "timestep",
"percent_to_sigma"}`, the last three engine functions.

## Engine extensions

Code that needs live engine objects beyond these rows, or has to run next to the model on
every step, is an engine extension. Comfy writes them.

- **Form:** one file, `v2/engine/<name>.py`, with a class named `EngineOps`. It can import
  `comfy`, `torch` and the standard library, and nothing else from the pack.
- **Allowed by:** `ALLOWED_EXTENSIONS` in `comfy_api/_impl/_engine.py`, a flat list of file
  name and sha256 pairs. A file that isn't listed fails the pack's load, in every mode. Two
  packs that ship the same file use the same extension.
- **Calling it:** `await eng.ext.<name>.<method>(...)`. A call resolves to the
  file the calling pack ships, and a pack can't reach one it doesn't ship.
- **Common extensions:** ComfyUI ships reviewed extensions any pack can call without shipping them,
  in `comfy_api/_impl/engine/common/`. One may import the kernel packages it declares in `REQUIRES`
  (such as triton), which must be installed in ComfyUI's environment. `eng.ext.vae_triton` is the first.
- **Developing one:** set `COMFY_DEV_ENGINE_EXTENSIONS=1` to load unlisted files.

## Not in the API

- `comfy`, `nodes`, `server` and `folder_paths`: a pack that imports them doesn't work under
  every dispatch. Use a row, `node.run` or an extension.
- Handles, typed handle classes and `ctx`: `ImageRef`, `ModelRef`, `eng.read`, `eng.write`,
  `ctx.assets`, `ctx.models`, `ctx.output` and the rest. Use the values, rows and stock nodes.
- Named operations such as `vae.decode_video` or `model.patch`: use a stock node, pack code or
  an extension.
- `ctx.interact`: use `eng.ui.interact` to ask the user something mid-run.
- Vendor clients (Ollama, Civitai, Luma): a pack brings its own client.
