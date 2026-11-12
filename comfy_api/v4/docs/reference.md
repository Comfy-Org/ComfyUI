# V4 API reference

API version 4.0.0. Generated from the `comfy_api.v4` package by
`python -m comfy_api.v4._generate`; change the docstrings in the package, not this page.

```python
from comfy_api.v4 import ComfyAPISync, ComfyExtension, eng, io, ui, util
```

## `eng`

`eng` is the API. Its calls are in groups, such as `eng.vae` and `eng.model`, and each
call in a group is a coroutine function: in a node whose `execute` is `async def`, write
`await eng.vae.encode(vae, pixels)`.

In a node whose `execute` is a plain `def`, create `api = ComfyAPISync()`. It has the same
groups and calls with the same parameters, and each call returns its result:
`api.vae.encode(vae, pixels)`. A call runs on the thread that makes it. In the sync form,
a call that would have to wait for the event loop raises `RuntimeError`.

### `eng.get_api_version`

```python
def get_api_version() -> tuple[int, int, int]
```

Return this package's (major, minor, patch) version.

### `eng.is_api_version_at_least`

```python
def is_api_version_at_least(version: str) -> bool
```

Return True if this package is `version` or newer.

`version` is three numbers joined by dots, with or without a leading `v`, such as
`v4.0.0` or `4.0.0`; anything else raises `ValueError`.

## `eng.caching`

### `eng.caching.register_provider`

```python
async def register_provider(provider: CacheProvider) -> None
```

Register an external cache provider; does nothing if `provider` is already registered.

On a local cache miss providers are asked in registration order and the first value one
returns is used. Every provider is offered each result ComfyUI stores under a key it can
compare and serialise: not the result of a node whose `fingerprint_inputs` returns NaN, nor
of the nodes after it. It stays until unregistered or until ComfyUI restarts, also if the
pack that registered it is disabled.

### `eng.caching.unregister_provider`

```python
async def unregister_provider(provider: CacheProvider) -> None
```

Stop asking `provider` from now on; does nothing if it is not registered.

Stores already handed to it still finish.

## `eng.clip`

### `eng.clip.add_patches`

```python
async def add_patches(
    clip: io.Clip.Type,
    patches: Mapping[_PatchKey, Sequence[object]],
    strength_patch: float = 1.0,
    strength_model: float = 1.0,
) -> io.Clip.Type
```

Return a patched clone; leave the input encoder unchanged.

Keys are names from `weight_keys`, or `(name, (dim, start, length))` to patch one slice.
A one-item sequence holding a tensor of the weight's shape makes the weight
`weight * strength_model + diff * strength_patch`, after any patches the encoder already
has. Keys the encoder does not have are ignored; a difference of another shape is skipped
with a logged warning.

### `eng.clip.clone`

```python
async def clone(clip: io.Clip.Type) -> io.Clip.Type
```

Copy the encoder's settings and patches while sharing its model weights.

### `eng.clip.encode_from_tokens`

```python
async def encode_from_tokens(
    clip: io.Clip.Type,
    tokens: ClipTokens,
    *,
    return_pooled: bool | Literal['unprojected'] = False,
    return_dict: bool = False,
) -> ClipEncodingResult
```

Encode once, without applying a conditioning schedule.

`return_pooled=True` gives a (conditioning, pooled) pair; "unprojected" gives the pooled
output before projection. `return_dict=True` gives a `ClipEncoding` instead of a pair, and
"unprojected" still selects the unprojected pooled output in it.

### `eng.clip.encode_tokens`

```python
async def encode_tokens(
    clip: io.Clip.Type,
    tokens: ClipTokens,
    *,
    unprojected: bool = False,
    add_dict: dict[str, object] | None = None,
    show_pbar: bool = True,
) -> io.Conditioning.Type
```

Encode CONDITIONING entries, honoring the encoder's hook schedule.

`add_dict` is merged into each entry's options, replacing keys it shares. With a hook
schedule (from SetClipHooks) there is an entry per scheduled range, and ranges outside
`add_dict`'s `start_percent`/`end_percent` are left out. `unprojected` keeps the pooled
output before projection; `show_pbar` reports progress over the scheduled ranges.

### `eng.clip.info`

```python
async def info(clip: io.Clip.Type) -> ClipInfo
```

Describe the selected output layer; None selects the encoder's default.

`layer_idx` counts as CLIPSetLastLayer's `stop_at_clip_layer` does: -1 is the last layer.

### `eng.clip.read_weight`

```python
async def read_weight(clip: io.Clip.Type, key: str) -> torch.Tensor
```

Read a weight with this encoder's patches applied, as ComfyUI computes it.

Without patches the result shares the model's storage: clone it before changing it.
While this encoder or a clone of it is loaded, the stored weight can already hold the
loaded clone's patches, and this encoder's are applied on top of it. Raises
`AttributeError` for a name not in `weight_keys`.

### `eng.clip.tokenize`

```python
async def tokenize(
    clip: io.Clip.Type,
    text: str,
    *,
    return_word_ids: bool = False,
    images: list[torch.Tensor] | None = None,
    llama_template: str | None = None,
) -> ClipTokens
```

Return editable weighted tokens per encoder name, as a list of chunks of the prompt.

A token is `(value, weight)`, or `(value, weight, word_index)` with `return_word_ids=True`;
the word index counts the prompt's weighted pieces of text, not its words, and is 0 for
start, end and padding tokens. Values may include embeddings or
multimodal payloads. A prompt longer than one chunk gives several, which encoding joins
end to end. `images` and `llama_template` are for the text encoders that take them.

### `eng.clip.weight_keys`

```python
async def weight_keys(clip: io.Clip.Type) -> list[str]
```

List model weight names in sorted order, excluding tokenizer data.

## `eng.device`

### `eng.device.info`

```python
async def info() -> DeviceInfo
```

Return ComfyUI's device choices and dtype names without the `torch.` prefix.

These are placement defaults, not a reservation of device memory. `unet_dtype` is the
choice for a model that runs in float16, bfloat16 or float32, and `vae_dtype` is float32,
unless a launch flag such as `--fp8_e4m3fn-unet` sets them; a loaded model or VAE may use
another dtype.

## `eng.execution`

### `eng.execution.check_interrupted`

```python
async def check_interrupted() -> None
```

Raise `InterruptProcessingException` if the user cancelled the prompt.

In a running node raising clears the cancel, so let the exception reach ComfyUI.
Elsewhere (a route handler, a thread the pack started, a task left over from a node that
has finished) the cancel stays pending.

### `eng.execution.is_interrupted`

```python
async def is_interrupted() -> bool
```

Return True if the user cancelled the prompt. The cancel stays pending.

## `eng.guider`

### `eng.guider.basic`

```python
async def basic(
    model: io.Model.Type,
    conditioning: io.Conditioning.Type,
) -> io.Guider.Type
```

Return a guider for one conditioning without classifier-free guidance, as BasicGuider gives.

A guidance value in the conditioning, such as FluxGuidance sets, still applies.

### `eng.guider.cfg`

```python
async def cfg(
    model: io.Model.Type,
    positive: io.Conditioning.Type,
    negative: io.Conditioning.Type,
    cfg: float,
) -> io.Guider.Type
```

Return a classifier-free guidance guider, as the CFGGuider node gives.

## `eng.memory`

### `eng.memory.empty_cache`

```python
async def empty_cache() -> None
```

Release memory PyTorch holds for this process but does not use; keep loaded models.

Does nothing on the CPU.

### `eng.memory.free_bytes`

```python
async def free_bytes(device: str | None = None) -> int
```

Return ComfyUI's estimate of free bytes on `device`, or on its own device if None.

On CUDA it is the driver's free memory plus PyTorch's unused cached blocks; on the CPU and
MPS it is the system's available RAM. It reserves nothing
and does not count loaded models that could be evicted. Raises `RuntimeError` for a string
that is not a device name, and `KeyError` for a device this machine does not have, such as
`"cuda"` on a Mac.

### `eng.memory.load_models`

```python
async def load_models(
    models: list[io.Model.Type | io.Clip.Type | io.Vae.Type],
    *,
    options: ModelLoadOptions | None = None,
) -> None
```

Load each model onto its own load device, freeing memory as ComfyUI does to run them.

Load devices default to `torch_device`, `vae_device` and `text_encoder_device` from
`eng.device.info`. `working_memory_bytes` is the memory the work will need beside the
models. `full_load` loads each model whole rather than partly, when ComfyUI would
otherwise split it.

### `eng.memory.unload_all`

```python
async def unload_all() -> None
```

Do nothing: ComfyUI alone decides which models stay loaded.

## `eng.model`

### `eng.model.add_patches`

```python
async def add_patches(
    model: io.Model.Type,
    patches: Mapping[_PatchKey, Sequence[object]],
    strength_patch: float = 1.0,
    strength_model: float = 1.0,
) -> io.Model.Type
```

Return a patched clone; leave the input model unchanged.

Keys are names from `weight_keys`, or `(name, (dim, start, length))` to patch one slice.
A one-item sequence holding a tensor of the weight's shape makes the weight
`weight * strength_model + diff * strength_patch`, after any patches the model already
has. Keys the model does not have are ignored; a difference of another shape is skipped
with a logged warning.

### `eng.model.clone`

```python
async def clone(model: io.Model.Type) -> io.Model.Type
```

Copy model settings and patch registrations while sharing its weights.

### `eng.model.info`

```python
async def info(model: io.Model.Type) -> ModelInfo
```

Describe the model's prediction type, latent format and sigma range as plain data.

`model_type` is the prediction type the model was loaded with, such as `"EPS"` or
`"FLOW"`, not its architecture; nodes such as ModelSamplingDiscrete do not change it,
though `zsnr` and the sigma range follow them. `dtype` is the diffusion model's weight
dtype. `latent_scale_factor` is the format's scale alone; some formats also shift.

### `eng.model.latent_process_in`

```python
async def latent_process_in(model: io.Model.Type, samples: torch.Tensor) -> torch.Tensor
```

Convert a latent from its VAE's range to the range the model samples in.

Depending on the model this scales, shifts and scales, or normalizes each channel, and for a
few it also changes the latent's shape; for some models it returns `samples` itself, so
clone the result before changing it in place.

### `eng.model.latent_process_out`

```python
async def latent_process_out(
    model: io.Model.Type,
    samples: torch.Tensor,
) -> torch.Tensor
```

Convert a latent from the range the model samples in back to its VAE's range.

It undoes `latent_process_in`; for some models it returns `samples` itself, so clone the
result before changing it in place.

### `eng.model.load_lora`

```python
async def load_lora(
    model: io.Model.Type | None,
    clip: io.Clip.Type | None,
    lora: dict[str, torch.Tensor],
    strength_model: float = 1.0,
    strength_clip: float = 1.0,
    lora_metadata: Mapping[str, object] | None = None,
) -> tuple[io.Model.Type | None, io.Clip.Type | None]
```

Apply LoRA weights to clones of the model and text encoder; a None input gives None.

`lora` holds a LoRA file's tensors, in any key format LoraLoader accepts; tensors that
match neither input are skipped with a logged warning. A non-empty `lora_metadata` is
attached to the resulting copies; the caller's weight and metadata mappings are not
changed.

### `eng.model.percent_to_sigma`

```python
async def percent_to_sigma(model: io.Model.Type, percent: float) -> float
```

Return the sigma at a fraction of sampling, as conditioning's `start_percent` uses it.

0.0 is the start and gives a value at or above `sigma_max`; 1.0 is the end and gives 0.0.

### `eng.model.read_weight`

```python
async def read_weight(model: io.Model.Type, key: str) -> torch.Tensor
```

Read a weight with this model's patches applied, as ComfyUI computes it.

Without patches the result shares the model's storage: clone it before changing it.
While this model or a clone of it is loaded, the stored weight can already hold the
loaded clone's patches, and this model's are applied on top of it. Raises
`AttributeError` for a name not in `weight_keys`.

### `eng.model.set_attention_override`

```python
async def set_attention_override(
    model: io.Model.Type,
    function: AttentionOverride,
) -> io.Model.Type
```

Return a clone whose attention goes through `function(attention, q, k, v, heads, **options)`.

`options` are the options ComfyUI passed, always as keywords here, which differ by model
(such as `mask`, `attn_precision`, `skip_reshape`, `transformer_options`): pass them on to
`attention` and do not rely on any one being present. An override already on the model
runs inside, as `attention`. `function` is called only where the model's code passes ComfyUI's
transformer options to attention, which not every model does (PixArt does not); elsewhere
nothing calls it and nothing is logged.

### `eng.model.set_attn1_output_patch`

```python
async def set_attn1_output_patch(
    model: io.Model.Type,
    function: Callable[[torch.Tensor, AttentionInfo], torch.Tensor],
) -> io.Model.Type
```

Return a clone that runs `function(out, extra_options)` on each self-attention output.

Functions run in the order added, each on the previous one's result. Only UNet models
(such as SD1.x, SD2.x, SDXL and SVD) run it; on other models it is not installed,
and a warning is logged.

### `eng.model.set_attn1_patch`

```python
async def set_attn1_patch(
    model: io.Model.Type,
    function: AttentionPatch,
) -> io.Model.Type
```

Return a clone that runs `function(x, context, value, extra_options)` before each self-attention.

It returns the three inputs of the q, k and v projections. Functions run in the order
added, each on the previous one's result. Only UNet models (such as SD1.x, SD2.x, SDXL
and SVD) run it; on other models it is not installed, and a warning is logged.

### `eng.model.set_attn2_output_patch`

```python
async def set_attn2_output_patch(
    model: io.Model.Type,
    function: Callable[[torch.Tensor, AttentionInfo], torch.Tensor],
) -> io.Model.Type
```

Return a clone that runs `function(out, extra_options)` on each cross-attention output.

Functions run in the order added, each on the previous one's result. Only UNet models
(such as SD1.x, SD2.x, SDXL and SVD) run it; on other models it is not installed,
and a warning is logged.

### `eng.model.set_attn2_patch`

```python
async def set_attn2_patch(
    model: io.Model.Type,
    function: AttentionPatch,
) -> io.Model.Type
```

Return a clone that runs `function(x, context, value, extra_options)` before each cross-attention.

It returns the three inputs of the q, k and v projections. Functions run in the order
added, each on the previous one's result. Only UNet models (such as SD1.x, SD2.x, SDXL
and SVD) run it; on other models it is not installed, and a warning is logged.

### `eng.model.set_cfg`

```python
async def set_cfg(
    model: io.Model.Type,
    function: Callable[[CfgArgs], torch.Tensor],
    *,
    disable_cfg1_optimization: bool = False,
) -> io.Model.Type
```

Return a clone whose CFG step is `function(args)`, replacing any already set.

`function` returns the guided noise prediction; stock CFG is
`uncond + (cond - uncond) * cond_scale`. With KSampler or `eng.guider.cfg` at `cond_scale`
1.0 the negative prompt is not run (`uncond_denoised` is zeros) unless
`disable_cfg1_optimization` is set by this call, an earlier patch of the model, or the
sampler, as the `*_cfg_pp` samplers do. A guider that does its own guidance decides this
itself, and may not call `function`: PerpNegGuider does not.

### `eng.model.set_denoise_mask`

```python
async def set_denoise_mask(
    model: io.Model.Type,
    function: DenoiseMaskFunction,
) -> io.Model.Type
```

Return a clone whose inpaint mask is `function(sigma, denoise_mask, extra_options=...)`.

It replaces any mask function already set, and runs only when the latent has a noise
mask, at each model evaluation: more than once a step for samplers such as Heun.
`extra_options["sigmas"]` is the whole schedule.

### `eng.model.set_input_block_patch`

```python
async def set_input_block_patch(
    model: io.Model.Type,
    function: Callable[[torch.Tensor, BlockInfo], torch.Tensor],
) -> io.Model.Type
```

Return a clone that runs `function(h, extra_options)` on each UNet input block's output.

Functions run in the order added, each on the previous one's result. Only UNet models
(such as SD1.x, SD2.x, SDXL and SVD) run it; on other models it is not installed,
and a warning is logged.

### `eng.model.set_output_block_patch`

```python
async def set_output_block_patch(
    model: io.Model.Type,
    function: Callable[[torch.Tensor, torch.Tensor, BlockInfo], tuple[torch.Tensor, torch.Tensor]],
) -> io.Model.Type
```

Return a clone that runs `function(h, skip, extra_options)` before each UNet output block.

It returns `(h, skip)`; `skip` is about to be concatenated to `h`. Functions run in the
order added, each on the previous one's result. Only UNet models (such as SD1.x, SD2.x,
SDXL and SVD) run it; on other models it is not installed, and a warning is logged.

### `eng.model.set_post_cfg`

```python
async def set_post_cfg(
    model: io.Model.Type,
    function: Callable[[PostCfgArgs], torch.Tensor],
    *,
    disable_cfg1_optimization: bool = False,
) -> io.Model.Type
```

Return a clone that runs `function(args)` after CFG; it returns the new `denoised`.

Functions run in the order added. With KSampler or `eng.guider.cfg` at `cond_scale` 1.0 the
negative prompt is not run (`uncond_denoised` is zeros) unless `disable_cfg1_optimization`
is set by this call, an earlier patch of the model, or the sampler, as the `*_cfg_pp`
samplers do. A guider that does its own guidance decides this itself.

### `eng.model.set_pre_cfg`

```python
async def set_pre_cfg(
    model: io.Model.Type,
    function: Callable[[PreCfgArgs], list[torch.Tensor]],
    *,
    disable_cfg1_optimization: bool = False,
) -> io.Model.Type
```

Return a clone that runs `function(args)` before CFG; it returns the new `conds_out`.

`conds_out` holds the denoised predictions, positive first. Functions run in the order
added. With KSampler or `eng.guider.cfg` at `cond_scale` 1.0 `conds_out[1]` is zeros unless
`disable_cfg1_optimization` is set by this call, an earlier patch of the model, or the
sampler, as the `*_cfg_pp` samplers do. A guider that does its own guidance decides this
itself, and may not run `function`: DualCFGGuider does not.

### `eng.model.set_unet_wrapper`

```python
async def set_unet_wrapper(
    model: io.Model.Type,
    function: Callable[[ApplyModel, UnetArgs], torch.Tensor],
) -> io.Model.Type
```

Return a clone whose model calls go through `function(apply_model, args)`.

`function` returns the denoised prediction for the batch, usually
`apply_model(args["input"], args["timestep"], **args["c"])`; `timestep` holds sigmas. A
wrapper already on the model runs inside, as `apply_model`.

### `eng.model.sigma`

```python
async def sigma(model: io.Model.Type, timestep: torch.Tensor) -> torch.Tensor
```

Map timesteps on the model's own scale (see `timestep`) to sigma values, elementwise.

For SD1.x and SDXL, fractional steps are interpolated and steps outside 0 to 999 are
clamped.

### `eng.model.timestep`

```python
async def timestep(model: io.Model.Type, sigma: torch.Tensor) -> torch.Tensor
```

Map sigma values to the timesteps the model is conditioned on, elementwise.

The scale is the model's own: for example SD1.x and SDXL give the nearest integer step
from 0 to 999, and Flux gives the sigma itself.

### `eng.model.weight_keys`

```python
async def weight_keys(model: io.Model.Type) -> list[str]
```

List the model's state names, sorted: the names `add_patches` and `read_weight` take.

Besides weights they include buffers such as `model_sampling.sigmas`.

## `eng.models`

Model files named by folder and file name, as ComfyUI's loader nodes take them.

A name is relative to its folder: an absolute path or `..` cannot reach outside it. A name
that is not a file in the folder raises `FileNotFoundError`, except the other names
`load_vae` takes from VAELoader, such as "pixel_space".

### `eng.models.list`

```python
async def list(folder: str) -> list[str]
```

Return the sorted file names in a model folder such as "loras", subfolders included.

Only files with an extension ComfyUI accepts for the folder are listed, from every
directory configured for it. A name in a subfolder uses this machine's separator: `\` on
Windows, `/` elsewhere. Raises `KeyError` for a folder ComfyUI does not know.

### `eng.models.load_checkpoint`

```python
async def load_checkpoint(
    name: str,
    options: CheckpointOptions | None = None,
) -> tuple[io.Model.Type, io.Clip.Type | None, io.Vae.Type]
```

Load (model, clip, vae) from the checkpoints folder, as CheckpointLoaderSimple does.

`dtype` sets the stored weight type of the diffusion model only, and is ignored for a file
that holds nothing but a diffusion model. `clip` is None for a file without a text encoder.
A file without VAE weights gives a VAE that raises `RuntimeError` when used. Raises
`RuntimeError` if ComfyUI does not recognize the file as a model.

### `eng.models.load_diffusion_model`

```python
async def load_diffusion_model(
    name: str,
    options: DiffusionModelOptions | None = None,
) -> io.Model.Type
```

Load a model from the diffusion_models folder, as UNETLoader does.

`dtype` sets the stored weight type. `fp8_optimizations` multiplies in fp8, on devices that
support it, wherever the stored weights are float8_e4m3fn, set by `dtype` or already so in
the file, as UNETLoader's "fp8_e4m3fn_fast" does. Raises
`RuntimeError` if ComfyUI does not recognize the file as a diffusion model.

### `eng.models.load_vae`

```python
async def load_vae(name: str) -> io.Vae.Type
```

Load a VAE by a name the VAELoader node offers, as that node does.

Raises `FileNotFoundError` for a file that is not found (for a tiny VAE, its encoder or
decoder in `vae_approx`), and `RuntimeError` if ComfyUI does not recognize the weights as a
VAE.

### `eng.models.read`

```python
async def read(folder: str, name: str) -> ModelFile
```

Read a file's tensors without building a model.

Tensors may map the file read-only: clone one before changing it in place.
Pickle-based formats load with torch's weights-only unpickler; from a pickle, a
`state_dict` entry, or a lone entry that is a dict, is returned in place of the whole file,
and non-tensor values are kept. `metadata` is a safetensors file's header metadata, and
empty for other formats.

## `eng.node`

### `eng.node.run`

```python
async def run(node: str, inputs: dict[str, Any]) -> tuple[Any, ...]
```

Run one of ComfyUI's own nodes inside the running node and return its outputs.

`node` is a node type such as "KSampler" (a saved workflow's `class_type`), and `inputs`
its input values by name, each what the node takes in a workflow: a LATENT is
`{"samples": tensor}`, not a bare tensor. As for a workflow, ComfyUI maps a removed node to its registered
replacement (the outputs come back in the removed node's order; an output the replacement
does not map keeps its index), converts number, text and boolean inputs with `int`,
`float`, `str` or `bool`, validates the inputs (required inputs, number ranges, choice
lists, the node's own validation), fills hidden inputs and runs the node once, on this
thread. An input the node does not declare is ignored, unless the node accepts any input.
Hidden inputs: `unique_id` is the running node's and the prompt is `{}`; every other
hidden input (workflow, dynamic prompt, execution list, account tokens) is None, so a node
that needs one (the loop nodes) fails. Raises `InterruptProcessingException` if the user
cancelled the prompt.

Not done: the result is not cached and the node's change check (`IS_CHANGED`,
`fingerprint_inputs`) is not consulted, so the running node is cached on its own inputs
alone. The node's lazy-input check is not called: it gets exactly the inputs given. Each
value is one input: a node that takes lists gets one-item lists, and outputs it declares as
lists come back as lists, so a list output cannot be passed on to a node that takes lists.
The UI the node returns is not sent (see `run_output`); text and progress it sends while it
runs show on the running node.

Raises `ValueError` for an unknown node type, a node not defined by ComfyUI itself, an
asynchronous node (most partner API nodes), invalid inputs, a node that expands into other
nodes, or a removed node with an output its replacement has no counterpart for.

### `eng.node.run_output`

```python
async def run_output(node: str, inputs: dict[str, Any]) -> NodeRunOutput
```

Run an output or preview node as `run` does, and also return its UI output.

`ui` is what the node shows (for example `{"images": [{"filename": ...,
"subfolder": ..., "type": ...}]}`), `{}` if nothing. Return it from your
node to show it.

## `eng.node_replacement`

### `eng.node_replacement.register`

```python
async def register(node_replace: io.NodeReplace) -> None
```

Register a mapping from a removed node type to its replacement.

ComfyUI applies it to queued prompts and to `eng.node.run` only while no node is registered
under the old id and one is under the new id. The first mapping registered for an old id is
the one applied; registering the same pair again does nothing. It stays until ComfyUI
restarts, also if the pack that registered it is disabled.

## `eng.noise`

### `eng.noise.empty`

```python
async def empty() -> NoiseGenerator
```

Return zero noise, as the DisableNoise node gives.

Its seed is 0, so a sampler with its own randomness, such as an ancestral one, repeats the
same randomness every run.

### `eng.noise.generate`

```python
async def generate(noise: NoiseGenerator, latent: io.Latent.Type) -> torch.Tensor
```

Return the noise a NOISE value gives for `latent` as given.

Sampling first fits an all-zero latent to the model's channels, so the noise it draws for
such a latent can differ from this.

### `eng.noise.random`

```python
async def random(seed: int) -> NoiseGenerator
```

Return seeded noise, as the RandomNoise node gives; the same as KSampler's for a seed.

## `eng.progress`

### `eng.progress.update`

```python
async def update(
    value: float,
    total: float,
    preview: io.Image.Type | None = None,
) -> None
```

Show the running node's progress, and the first image of `preview` if given.

Raises `InterruptProcessingException` if the user cancelled the prompt, as ComfyUI's own
progress bar does, and `RuntimeError` unless made while the node runs, from its `execute`
or a function it calls. A call is not sent if it has no preview,
`value` is below `total`, `total` is the one last shown, and `value` has moved forward by
less than 0.5% of `total`, or not at all. What is sent goes to the clients ComfyUI's own
progress bar reports to, as it sends it: a `value` above `total` is shown as `total`.
Raises `ValueError` if `value` or `total` is not a finite number, which a browser cannot
read.

## `eng.sampler`

### `eng.sampler.custom`

```python
async def custom(
    function: SamplerFunction,
    options: Mapping[str, object] | None = None,
) -> io.Sampler.Type
```

Return a sampler whose loop is `function`, called with a copy of `options` as keywords.

### `eng.sampler.named`

```python
async def named(
    name: str,
    options: Mapping[str, bool | int | float | str] | None = None,
) -> io.Sampler.Type
```

Return one of ComfyUI's samplers by name, such as "euler", with a copy of `options`.

`options` are the sampler's keyword arguments, such as `s_noise`; one the sampler does not
take raises `TypeError` when sampling, not here. Raises `AttributeError` for a name ComfyUI
does not know.

### `eng.sampler.sample`

```python
async def sample(
    model: io.Model.Type,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive: io.Conditioning.Type,
    negative: io.Conditioning.Type,
    latent: io.Latent.Type,
    denoise: float = 1.0,
    disable_noise: bool = False,
    start_step: int | None = None,
    last_step: int | None = None,
    force_full_denoise: bool = False,
) -> io.Latent.Type
```

Sample as the KSampler nodes do; returns a shallow copy of `latent` with new `samples`.

Other entries, such as `batch_index` and `noise_mask`, are the same objects as in `latent`;
the copy has no `downscale_ratio_spacial` or `downscale_ratio_temporal`. Reports progress
and previews as KSampler does. A sampler or scheduler name ComfyUI does not know is
replaced by the first in its list, as in KSampler.

### `eng.sampler.sample_custom`

```python
async def sample_custom(
    noise: NoiseGenerator,
    guider: io.Guider.Type,
    sampler: io.Sampler.Type,
    sigmas: torch.Tensor,
    latent: io.Latent.Type,
) -> io.Latent.Type
```

Sample as SamplerCustomAdvanced does; returns a shallow copy of `latent` with new `samples`.

Other entries, such as `batch_index` and `noise_mask`, are the same objects as in `latent`;
the copy has no `downscale_ratio_spacial` or `downscale_ratio_temporal`. `noise.seed` also
seeds the sampler's own randomness. Reports progress and previews as SamplerCustomAdvanced
does.

### `eng.sampler.sigmas`

```python
async def sigmas(
    model: io.Model.Type,
    scheduler: str,
    steps: int,
    denoise: float = 1.0,
) -> torch.Tensor
```

Return the `steps + 1` noise levels a named scheduler gives this model, on the CPU.

`denoise` below 1 keeps the end of a longer schedule; 0 gives an empty tensor.
Raises ValueError for a scheduler ComfyUI does not know.

## `eng.settings`

### `eng.settings.get`

```python
async def get(name: Literal['disable_metadata', 'multi_user']) -> bool
```

Report how ComfyUI was started. Raises `ValueError` for any other name.

`disable_metadata`: files a node saves must not carry the prompt or workflow.
`multi_user`: each user has their own user data (settings, saved workflows, user files);
input, output and model folders are shared.

## `eng.ui`

### `eng.ui.progress_text`

```python
async def progress_text(text: str) -> None
```

Show `text` on the running node to the client that queued the prompt.

Nothing is sent if the prompt was queued without a client id. Raises `RuntimeError` unless
made while the node runs, from its `execute` or a function it calls.

### `eng.ui.send`

```python
async def send(event: str, data: dict) -> None
```

Send `{"type": event, "data": data}` to the client that queued the prompt.

Nothing is sent if the prompt was queued without a client id. `data` is copied as JSON now,
so later changes to it are not sent; if it has a `"prompt_id"` key, ComfyUI adds the
workflow's metadata, such as `workflow_id`, to what it sends. Raises `TypeError` if it is
not JSON-serialisable, `ValueError` for NaN, infinity or a circular reference, and
`RuntimeError` unless made while the node runs, from its `execute` or a function it
calls.

## `eng.vae`

### `eng.vae.decode`

```python
async def decode(vae: io.Vae.Type, samples: torch.Tensor) -> torch.Tensor
```

Decode a latent to pixels.

Returns channels-last pixels, as an IMAGE is. A video VAE returns
`(batch, frames, height, width, channels)`; the VAEDecode node flattens that to one image
per frame, this call does not.

### `eng.vae.decode_tiled`

```python
async def decode_tiled(
    vae: io.Vae.Type,
    samples: torch.Tensor,
    *,
    tile_x: int | None = None,
    tile_y: int | None = None,
    overlap: int | None = None,
    tile_t: int | None = None,
    overlap_t: int | None = None,
) -> torch.Tensor
```

Decode in overlapping tiles to bound memory.

Sizes are in latent units. None lets the VAE choose, except `overlap` for a video VAE
ComfyUI tiles: that is 64 pixels, and at most a quarter of a tile. Sizes are refused as
`encode_tiled` refuses them.

### `eng.vae.encode`

```python
async def encode(vae: io.Vae.Type, pixels: torch.Tensor) -> torch.Tensor
```

Encode pixels to a latent.

`pixels` is channels-last with values from 0 to 1, as an IMAGE is; channels beyond the
VAE's are dropped. Most VAEs center-crop height and width to a multiple of
`spacial_compression_encode` from `info`. A video VAE usually takes the batch as the frames
of one video.

### `eng.vae.encode_tiled`

```python
async def encode_tiled(
    vae: io.Vae.Type,
    pixels: torch.Tensor,
    *,
    tile_x: int | None = None,
    tile_y: int | None = None,
    overlap: int | None = None,
    tile_t: int | None = None,
    overlap_t: int | None = None,
) -> torch.Tensor
```

Encode in overlapping tiles to bound memory.

Sizes are in pixels, and frames for `tile_t` and `overlap_t`; for an audio VAE whose latent
has one axis besides channels, such as Stable Audio's, `tile_x` and `overlap` are in
samples. An audio VAE whose latent has two, such as ACE-Step's, cannot encode in tiles:
ComfyUI raises `IndexError`. None lets the VAE choose. A VAE that tiles itself may ignore
some sizes. Raises `ValueError` for a tile size that is not positive or a negative
overlap; keep each overlap below half its tile size.

### `eng.vae.from_state_dict`

```python
async def from_state_dict(
    sd: dict[str, torch.Tensor],
    *,
    metadata: dict[str, object] | None = None,
) -> io.Vae.Type
```

Build a VAE from weights, with the checkpoint's metadata if its format needs it.

`metadata` is a safetensors header, as `eng.models.read` returns it. Raises `RuntimeError`
if ComfyUI does not recognize the weights as a VAE.

### `eng.vae.info`

```python
async def info(vae: io.Vae.Type) -> VaeInfo
```

Describe a VAE without running it.

`temporal_compression_decode` is None for a VAE without a time axis. `downscale_ratio`
and `upscale_ratio` are None for a VAE that scales each axis differently; the `spacial_*`
fields are always numbers. `dtype` is None for a VAE without weights.

## `util`

Helpers for node code: image and latent resizing, noise, conditioning, weight files, folders.

### `util.MAX_RESOLUTION`

```python
MAX_RESOLUTION: int
```

The largest width or height most of ComfyUI's own nodes accept; a few accept more.

### `util.SAMPLER_NAMES`

```python
SAMPLER_NAMES: list[str]
```

The list of samplers ComfyUI's KSampler offers: ComfyUI's own list, not a copy.

### `util.SCHEDULER_NAMES`

```python
SCHEDULER_NAMES: list[str]
```

The list of schedulers ComfyUI's KSampler offers: ComfyUI's own list, not a copy.

### `util.bislerp`

```python
def bislerp(samples: torch.Tensor, width: int, height: int) -> torch.Tensor
```

Resize (batch, channels, height, width) by spherical interpolation of each pixel's channels.

### `util.common_upscale`

```python
def common_upscale(
    samples: torch.Tensor,
    width: int,
    height: int,
    upscale_method: str,
    crop: str,
) -> torch.Tensor
```

Resize `samples`, channels first (batch, channels, ..., height, width), to `width` x `height`.

`upscale_method` is `"bislerp"`, `"lanczos"`, or a 2-d `torch.nn.functional.interpolate`
mode such as `"nearest-exact"`, `"bilinear"`, `"bicubic"` or `"area"`. With `"lanczos"`, one
channel comes back as (batch, height, width), as `lanczos` gives it. `crop="center"` first crops
to the new aspect ratio; any other value does not crop.

### `util.conditioning_set_values`

```python
def conditioning_set_values(
    conditioning: io.Conditioning.Type,
    values: dict[str, Any],
    append: bool = False,
) -> io.Conditioning.Type
```

Return a copy of `conditioning` with `values` set in every entry's options.

With `append`, a value is added (`+`) to the one already there. Only the list, the entries and
the options dicts are new: tensors and every other option value, such as a list or dict, are
shared with `conditioning`.

### `util.conditioning_set_values_with_timestep_range`

```python
def conditioning_set_values_with_timestep_range(
    conditioning: io.Conditioning.Type,
    values: dict[str, Any],
    start_percent: float = 0.0,
    end_percent: float = 1.0,
) -> io.Conditioning.Type
```

Set `values` as `conditioning_set_values` does, from `start_percent` to `end_percent` only.

An entry's own `"start_percent"`/`"end_percent"` is respected: the part of its range outside
the requested one keeps its options, as extra entries, and an entry whose range does not
overlap is put in the result as it is, not copied. If `start_percent` is above `end_percent`,
`conditioning` itself is returned and a warning is logged.

### `util.get_tiled_scale_steps`

```python
def get_tiled_scale_steps(
    width: int,
    height: int,
    tile_x: int,
    tile_y: int,
    overlap: int,
) -> int
```

Return how many tiles `tiled_scale` runs for one image of this size; sizes are refused as
`tiled_scale` refuses them.

### `util.lanczos`

```python
def lanczos(samples: torch.Tensor, width: int, height: int) -> torch.Tensor
```

Resize (batch, channels, height, width) images with Pillow's Lanczos filter.

The images go through 8 bits: values come back clipped to 0..1 in steps of 1/255. One to four
channels; one channel comes back as (batch, height, width). With two or four channels the last is
alpha: when the size changes, the others come back 0 where the resized alpha is 0. Raises
`TypeError` for `bfloat16`.

### `util.load_torch_file`

```python
def load_torch_file(
    ckpt: str,
    safe_load: bool = False,
    device: torch.device | None = None,
    return_metadata: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, str] | None]
```

Read a `.safetensors` or `.sft` file, or a torch file, into `{name: tensor}`.

Torch files are always read with `weights_only=True`, so `safe_load` changes nothing. A
torch file's `"state_dict"` entry, or its only entry if that is a dict, is what is
returned. With `return_metadata`, returns `(weights, metadata)`; metadata is None for torch
files, and None or `{}` for a safetensors file without metadata.

### `util.pillow`

```python
def pillow(function: Callable[[Any], Any], arg: Any) -> Any
```

Return `function(arg)`; on `OSError` or `ValueError`, call it again with truncated images allowed.

Any error of these types, such as `FileNotFoundError`, causes the second call, whose error is
raised. Pillow's truncated-images setting is process-wide during that call and restored after.

### `util.prepare_noise`

```python
def prepare_noise(
    latent_image: torch.Tensor,
    seed: int,
    noise_inds: list[int] | None = None,
) -> torch.Tensor
```

Return the noise ComfyUI's KSampler adds for `seed`: made on the CPU, in the latent's dtype.

With `noise_inds` (a latent's `"batch_index"`), entry i gets the noise of batch position
`noise_inds[i]`. Seeds torch's global CPU random generator with `seed`, so later unseeded torch
random calls in the process follow from it.

### `util.save_torch_file`

```python
def save_torch_file(
    sd: dict[str, torch.Tensor],
    ckpt: str,
    metadata: dict[str, str] | None = None,
) -> None
```

Write `{name: tensor}` to `ckpt` as safetensors, whatever its extension, with `metadata`.

Raises `RuntimeError` if tensors share memory, such as tied weights, `ValueError` for a tensor
that is not contiguous, and `TypeError` for a `metadata` value that is not a string.
`load_torch_file` reads the file back only under a `.safetensors` or `.sft` name.

### `util.tiled_scale`

```python
def tiled_scale(
    samples: torch.Tensor,
    function: Callable[[torch.Tensor], torch.Tensor],
    tile_x: int = 64,
    tile_y: int = 64,
    overlap: int = 8,
    upscale_amount: float = 4,
    out_channels: int = 3,
    output_device: str | torch.device = 'cpu',
    pbar: ProgressCounter | None = None,
) -> torch.Tensor
```

Run `function` on overlapping tiles of (batch, channels, height, width) and blend the results.

`function` gets one tile of one image, (1, channels, height, width), and returns it
`upscale_amount` times the size with `out_channels` channels. The result is float32 on
`output_device`, made under `torch.inference_mode()`: clone it before changing it in place or
using it with autograd. `pbar` advances once per tile. Raises `ValueError` for a tile size
that is not positive or a negative `overlap`; keep `overlap` below the tile size.

### `util.tiled_scale_multidim`

```python
def tiled_scale_multidim(
    samples: torch.Tensor,
    function: Callable[[torch.Tensor], torch.Tensor],
    tile: Sequence[int] = (64, 64),
    overlap: int | Sequence[int] = 8,
    upscale_amount: float | Sequence[float] = 4,
    out_channels: int = 3,
    output_device: str | torch.device = 'cpu',
    downscale: bool = False,
    index_formulas: float | Sequence[float] | None = None,
    pbar: ProgressCounter | None = None,
) -> torch.Tensor
```

Tile and blend as `tiled_scale` does, over every dimension after channels.

`samples` is (batch, channels, *dims) with one `tile` size per dim; `overlap`, `upscale_amount`
and `index_formulas` are one value or one per dim. `downscale` divides by `upscale_amount` where
`tiled_scale` multiplies. A tile's start is multiplied (or divided, with `downscale`) by
`index_formulas`, by default `upscale_amount`, to place its result. Sizes are refused as
`tiled_scale` refuses them.

## `util.folders`

ComfyUI's folders: where inputs, outputs, temporary files and model files are.

The calls that return paths in the user's folders (input, output, user, models and the base
folder) are provided for compatibility with packs written against `folder_paths`. ComfyUI
decides what each answers: on the user's own machine the user's folder, but a host may give a
pack folders of its own for one run, or a path the pack cannot open. A pack that has to run
everywhere takes the user's files as node inputs, returns its results as node outputs, and writes
temporary files to `get_temp_directory()`.

A pack that runs in its own process gets the output, input, temp and user folders that process
was started with.

### `util.folders.base_path`

```python
base_path: str
```

ComfyUI's base folder: `--base-directory`, or the folder ComfyUI is installed in.

### `util.folders.models_dir`

```python
models_dir: str
```

The folder holding the model folders: `--models-directory`, or `models` in `base_path`.

### `util.folders.output_directory`

```python
output_directory: str
```

The same path as `get_output_directory()`.

### `util.folders.input_directory`

```python
input_directory: str
```

The same path as `get_input_directory()`.

### `util.folders.temp_directory`

```python
temp_directory: str
```

The same path as `get_temp_directory()`.

### `util.folders.supported_pt_extensions`

```python
supported_pt_extensions: set[str]
```

The weight file extensions of folders such as `checkpoints` and `loras`; a new set on every read.

### `util.folders.folder_names_and_paths`

```python
folder_names_and_paths: dict[str, tuple[list[str], set[str]]]
```

ComfyUI's own table of model folders: `{name: ([directories], extensions)}`.

`extensions` is a set, or a list for `configs` and `diffusers`. Most folders share one extension
set, so adding to one folder's set adds to all of them. Changes to the table are seen by every node
in this ComfyUI, and nothing removes them.

### `util.folders.add_model_folder_path`

```python
def add_model_folder_path(
    folder_name: str,
    full_folder_path: str | os.PathLike[str],
    is_default: bool = False,
) -> None
```

Add a directory to a model folder, for every node in this ComfyUI; nothing removes it.

`is_default` puts it first, moving it if it is already there; otherwise adding it again does
nothing. A folder name not in the table gets an entry with no extensions, which lists every
file. Nothing is created on disk. `"unet"` and `"clip"` name `"diffusion_models"` and
`"text_encoders"`. A path object is stored as its string. Raises `TypeError` for a folder name
that is not a string or a directory that is not a string or path: ComfyUI's table would keep
the value, and its model lists would fail.

### `util.folders.annotated_filepath`

```python
def annotated_filepath(name: str) -> tuple[str, str | None]
```

Split a mark off a name: `"a.png [output]"` gives `("a.png", the output folder)`.

A name ending in `[output]`, `[input]` or `[temp]` is marked: the mark and the one character
before it, normally a space, are removed. Any other name gives `(name, None)`.

### `util.folders.clear_filename_list_cache`

```python
def clear_filename_list_cache(folder_name: str) -> None
```

Make the next `get_filename_list(folder_name)` scan the disk again.

ComfyUI rescans by itself when a directory's modified time changes; call this after a
change it cannot see, such as new extensions for the folder. While ComfyUI builds its node
list for `/object_info`, a folder already listed for it is not scanned again.

### `util.folders.exists_annotated_filepath`

```python
def exists_annotated_filepath(name: str) -> bool
```

Return whether the path `get_annotated_filepath(name)` gives exists; False if it leads outside.

### `util.folders.filter_files_content_types`

```python
def filter_files_content_types(
    files: list[str],
    content_types: list[Literal['image', 'video', 'audio', 'model']],
) -> list[str]
```

Return the names in `files` whose type, guessed from the extension, is in `content_types`.

The type is guessed with Python's `mimetypes` from this machine's MIME table, so it can differ
between machines, except that `.webp` is always `"image"` and `.fbx` always `"model"`.
`"model"` means 3D models. Order is kept.

### `util.folders.filter_files_extensions`

```python
def filter_files_extensions(
    files: Collection[str],
    extensions: Collection[str],
) -> list[str]
```

Return the names in `files` whose lowercased extension, with its dot, is in `extensions`, sorted.

All of them if `extensions` is empty.

### `util.folders.get_annotated_filepath`

```python
def get_annotated_filepath(name: str, default_dir: str | None = None) -> str
```

Return the path of a file name as load nodes receive it, such as `"a.png [output]"`.

An unmarked name is in `default_dir`, or else the input folder. Raises `ValueError` if the
name leads outside that folder.

### `util.folders.get_directory_by_type`

```python
def get_directory_by_type(type_name: str) -> str | None
```

Return the output, input or temp folder for `"output"`, `"input"` or `"temp"`; None otherwise.

### `util.folders.get_filename_list`

```python
def get_filename_list(folder_name: str) -> list[str]
```

Return the files a model dropdown offers for this folder, sorted, relative to their directory.

Only files with one of the folder's extensions are listed, or every file if it has none.
A name in a subfolder uses this machine's separator: `\` on Windows, `/` elsewhere. Raises
`KeyError` for a folder that does not exist.

### `util.folders.get_filename_list_`

```python
def get_filename_list_(folder_name: str) -> tuple[list[str], dict[str, float], float]
```

Scan a model folder now: `(files, {each directory scanned: modified time}, scan time)`.

The scan time is a `time.perf_counter()` value. Raises `KeyError` for a folder that does
not exist.

### `util.folders.get_folder_paths`

```python
def get_folder_paths(folder_name: str) -> list[str]
```

Return the directories of a model folder such as `"loras"`, as a new list.

`"unet"` and `"clip"` name `"diffusion_models"` and `"text_encoders"`. Raises `KeyError`
for a folder that does not exist.

### `util.folders.get_full_path`

```python
def get_full_path(folder_name: str, filename: str) -> str | None
```

Return the path of `filename` in the first of the folder's directories that has it, else None.

Only files count. `filename` cannot leave the directories: a leading `/` and `..` above the
directory are dropped. An unknown folder gives None, not `KeyError`.

### `util.folders.get_full_path_or_raise`

```python
def get_full_path_or_raise(folder_name: str, filename: str) -> str
```

Return the path as `get_full_path` does, but raise `FileNotFoundError` if there is none.

### `util.folders.get_input_directory`

```python
def get_input_directory() -> str
```

Return the folder of uploaded files that ComfyUI's load nodes list.

### `util.folders.get_input_subfolders`

```python
def get_input_subfolders() -> list[str]
```

Return every folder below the input folder, relative to it with `/` separators, sorted.

Linked folders are not listed or entered; a missing input folder gives `[]`.

### `util.folders.get_output_directory`

```python
def get_output_directory() -> str
```

Return the folder ComfyUI's save nodes write to.

### `util.folders.get_save_image_path`

```python
def get_save_image_path(
    filename_prefix: str,
    output_dir: str,
    image_width: int = 0,
    image_height: int = 0,
) -> tuple[str, str, int, str, str]
```

Return where ComfyUI's save nodes would write the next file named `filename_prefix`.

Returns `(folder, filename, counter, subfolder, filename_prefix)`. `counter` is one more than
the highest N among the folder's files named `<filename>_N` where N ends at the end of the
name, a `.` or a `_` (as in ComfyUI's own `<filename>_00001_.png`), or 1.
`%width%`, `%height%`, `%year%`, `%month%`, `%day%`, `%hour%`, `%minute%` and `%second%` are
filled in, also in the returned `filename_prefix`; `%date:...%` is not. Creates the folder if
missing; raises `Exception` if the prefix leads outside `output_dir`.

### `util.folders.get_system_user_directory`

```python
def get_system_user_directory(name: str = 'system') -> str
```

Return `__<name>` in the user folder: a folder ComfyUI's user-data endpoints refuse; not created.

Raises `ValueError` unless `name` is letters, digits and underscores, not starting with an
underscore.

### `util.folders.get_temp_directory`

```python
def get_temp_directory() -> str
```

Return the folder for temporary files, which ComfyUI's preview nodes also write to.

ComfyUI empties it, so a file written there does not stay.

### `util.folders.get_user_directory`

```python
def get_user_directory() -> str
```

Return the user folder, which holds a folder for each user (`default` for a single user).

A user's folder holds their settings and saved workflows.

### `util.folders.is_within_directory`

```python
def is_within_directory(directory: str, target: str) -> bool
```

Return whether `target` is `directory` or inside it, after resolving links on both.

### `util.folders.recursive_search`

```python
def recursive_search(
    directory: str,
    excluded_dir_names: list[str] | None = None,
) -> tuple[list[str], dict[str, float]]
```

Return `(files relative to directory, {directory and each folder below it: modified time})`.

Folder keys are paths starting with `directory` as given, not relative ones. Files are in no
set order and use this machine's separator. Follows links and skips folders named in
`excluded_dir_names` at any depth; `([], {})` if `directory` is not a directory.

## `ComfyExtension`

A pack's entry point, `comfy_entrypoint()`, returns an instance of a subclass of
`comfy_api.v4.ComfyExtension`. ComfyUI calls these methods when it loads the pack.

### `ComfyExtension.get_node_list`

```python
async def get_node_list() -> list[type[io.ComfyNode]]
```

Returns a list of nodes that this extension provides.

### `ComfyExtension.get_routes`

```python
async def get_routes() -> list[Route]
```

Return the HTTP routes ComfyUI serves for this pack at `path` and at `/api` + `path`.

Read once, at load. ComfyUI disables the pack, with a warning, if a route's method is not
one `Route` names, its path is malformed or it has no handler; if the pack has routes and
loads after ComfyUI starts serving; or if a request one of its routes answers could also
reach another of its routes (a literal path beside a pattern is allowed), a route of a V4
pack loaded before it, or anything else ComfyUI's server answers: its own routes, static
folders and sub-applications, and routes of packs written to an older API. Two patterns
count as reaching one request unless a literal segment, or the number of segments,
differs before either has a `{name:regex}` part.

### `ComfyExtension.on_load`

```python
async def on_load() -> None
```

Called when an extension is loaded.
This should be used to initialize any global resources needed by the extension.

## Types

The types the calls above take and give. Import them from `comfy_api.v4`, except `ProgressCounter`, which is in `comfy_api.v4.util`.

### `ApplyModel`

```python
ApplyModel = Callable[..., torch.Tensor]
```

### `AttentionInfo`

```python
class AttentionInfo(BlockInfo):
    """`block_index` is the transformer in the block; `original_shape` is the model input's."""

    block_index: int
    original_shape: list[int]
    n_heads: int
```

### `AttentionOverride`

```python
class AttentionOverride(Protocol):
    """`options` are passed through as ComfyUI gave them; only passing them on is part of the API."""

    def __call__(
        self,
        attention: Callable[..., torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        **options: Any,
    ) -> torch.Tensor: ...
```

### `AttentionPatch`

```python
AttentionPatch = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, AttentionInfo],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]
```

### `BlockInfo`

```python
class BlockInfo(TypedDict):
    """Where a UNet patch runs: `block` is `("input" | "middle" | "output", index)`.

    `sigmas` is the current sigma; `cond_or_uncond` says which prompt each chunk of the
    batch is, as in `UnetArgs`.
    """

    block: tuple[str, int]
    cond_or_uncond: list[int]
    sigmas: torch.Tensor
```

### `CfgArgs`

```python
class CfgArgs(TypedDict):
    """What a CFG function gets. `cond` and `uncond` are noise predictions, `input - *_denoised`."""

    cond: torch.Tensor
    uncond: torch.Tensor
    cond_denoised: torch.Tensor
    uncond_denoised: torch.Tensor
    cond_scale: float
    sigma: torch.Tensor
    input: torch.Tensor
```

### `CheckpointOptions`

```python
class CheckpointOptions(TypedDict, total=False):
    dtype: WeightDtype
```

### `ClipEncoding`

```python
class ClipEncoding(TypedDict):
    """Any other keys are the encoder's extra conditioning options, passed through as they are:
    only passing them on in a conditioning entry is part of the API."""

    cond: torch.Tensor
    pooled_output: torch.Tensor | None
```

### `ClipEncodingResult`

```python
ClipEncodingResult = (
    torch.Tensor | tuple[torch.Tensor, torch.Tensor | None] | ClipEncoding
)
```

### `ClipInfo`

```python
class ClipInfo(TypedDict):
    layer_idx: int | None
```

### `ClipTokens`

```python
ClipTokens = dict[str, list[list[WeightedToken]]]
```

### `DenoiseMaskFunction`

```python
class DenoiseMaskFunction(Protocol):
    def __call__(
        self,
        sigma: torch.Tensor,
        denoise_mask: torch.Tensor,
        extra_options: DenoiseMaskInfo,
    ) -> torch.Tensor: ...
```

### `DenoiseMaskInfo`

```python
class DenoiseMaskInfo(TypedDict):
    """`sigmas` is the whole schedule of the run."""

    sigmas: torch.Tensor
```

### `Denoiser`

```python
class Denoiser(Protocol):
    """ComfyUI's model for one sampling run. Only calling it is part of the API."""

    def __call__(
        self, x: torch.Tensor, sigma: torch.Tensor, **extra_args: Any
    ) -> torch.Tensor:
        """Return the denoised prediction of `x` at noise level `sigma`, shaped like `x`.

        `sigma` holds one noise level per batch item, as `sigmas[i] * x.new_ones([x.shape[0]])`
        gives.
        """
        ...
```

### `DeviceInfo`

```python
class DeviceInfo(TypedDict):
    torch_device: str
    intermediate_device: str
    intermediate_dtype: str
    unet_dtype: str
    offload_device: str
    vae_device: str
    vae_dtype: str
    text_encoder_device: str
    vae_offload_device: str
    text_encoder_offload_device: str
```

### `DiffusionModelOptions`

```python
class DiffusionModelOptions(TypedDict, total=False):
    dtype: WeightDtype
    fp8_optimizations: bool
```

### `InterruptProcessingException`

```python
class InterruptProcessingException(BaseException):
    """Raised when the user cancels the prompt; ComfyUI reports it as an interruption, not an error.

    It is a `BaseException`, so `except Exception` lets it through to ComfyUI.
    """
```

### `ModelFile`

```python
class ModelFile(TypedDict):
    sd: dict[str, torch.Tensor]
    metadata: dict[str, str]
```

### `ModelInfo`

```python
class ModelInfo(TypedDict):
    model_type: str
    latent_scale_factor: float
    latent_channels: int
    zsnr: bool
    dtype: str
    sigma_min: float
    sigma_max: float
```

### `ModelLoadOptions`

```python
class ModelLoadOptions(TypedDict, total=False):
    working_memory_bytes: int | float
    full_load: bool
```

### `NodeRunOutput`

```python
class NodeRunOutput(TypedDict):
    """What `eng.node.run_output` returns: `result` as `eng.node.run` gives it, and `ui`."""

    result: tuple[Any, ...]
    ui: dict[str, list[Any]]
```

### `NoiseGenerator`

```python
class NoiseGenerator(Protocol):
    """What ComfyUI uses from a NOISE value; a pack's own class may provide it.

    `seed` also seeds the sampler's own randomness, such as an ancestral sampler's.
    """

    seed: int

    def generate_noise(self, input_latent: io.Latent.Type) -> torch.Tensor:
        """Return noise shaped like `input_latent["samples"]`, at unit scale.

        Sampling scales it to the first noise level.
        """
        ...
```

### `PostCfgArgs`

```python
class PostCfgArgs(TypedDict):
    """What a post-CFG function gets: `denoised` is the result so far."""

    denoised: torch.Tensor
    cond_denoised: torch.Tensor
    uncond_denoised: torch.Tensor
    cond_scale: float
    sigma: torch.Tensor
    input: torch.Tensor
```

### `PreCfgArgs`

```python
class PreCfgArgs(TypedDict):
    """What a pre-CFG function gets: the denoised prediction of each prompt, positive first."""

    conds_out: list[torch.Tensor]
    cond_scale: float
    sigma: torch.Tensor
    input: torch.Tensor
```

### `ProgressCounter`

```python
class ProgressCounter(Protocol):
    """What a `pbar` argument provides, as ComfyUI's progress bar does."""

    def update(self, value: float) -> None: ...
```

### `Route`

```python
class Route(TypedDict):
    """One HTTP route of a pack, as `ComfyExtension.get_routes` returns it."""

    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    """A "GET" route also answers HEAD requests."""
    path: str
    """Starts with "/"; `{name}` or `{name:regex}` captures a part into `match_info`."""
    handler: Callable[[RouteRequest], Awaitable[RouteResponse]]
    """Awaited on ComfyUI's event loop for each request; an exception it raises gives a 500
    response."""
```

### `RouteRequest`

```python
class RouteRequest(TypedDict):
    """An HTTP request to one of a pack's routes, as its handler receives it."""

    method: str
    """The request's method; "HEAD" for a HEAD request to a "GET" route."""
    path: str
    """The decoded path as requested, with the `/api` prefix if the request used it."""
    match_info: dict[str, str]
    """The decoded value each of the route's `{name}` parts matched."""
    query: list[tuple[str, str]]
    """The decoded query parameters in order; a repeated name appears once per value."""
    headers: dict[str, str]
    """Header names in lower case; a repeated header's values joined with ", "."""
    body: bytes
    """The whole body. A body over ComfyUI's upload limit (`--max-upload-size`) is answered 413
    before the handler runs."""
    remote: str | None
    """The client's address as the connection reports it (a proxy's, behind one)."""
```

### `RouteResponse`

```python
class RouteResponse(TypedDict):
    """What a route's handler returns; a missing key gives the client a 500 response."""

    status: int
    headers: dict[str, str]
    """Without "Content-Type", a body is sent as "application/octet-stream"."""
    body: bytes
```

### `SamplerArgs`

```python
class SamplerArgs(TypedDict, total=False):
    """What a sampler function passes back whole: `model(x, sigma, **extra_args)`.

    ComfyUI owns this dict and may add keys; only the keys below are part of the API.
    """

    seed: int | None
    denoise_mask: torch.Tensor | None
```

### `SamplerFunction`

```python
class SamplerFunction(Protocol):
    """A sampling loop written as k-diffusion writes them; returns the sampled `x`.

    `model`, `x` and `sigmas` come by position, the rest by keyword, with each of
    the options given to `eng.sampler.custom` as one more keyword. `x` is the input
    latent in the model's range, already noised to `sigmas[0]` the way the model
    noises a latent, which differs between models: a flow model mixes noise and
    latent instead of adding them. `disable` is True when the loop should not draw
    its own console progress bar.
    """

    def __call__(
        self,
        model: Denoiser,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        extra_args: SamplerArgs,
        callback: Callable[[SamplerStep], None] | None,
        disable: bool | None,
    ) -> torch.Tensor: ...
```

### `SamplerStep`

```python
class SamplerStep(TypedDict):
    """What a sampler function passes to `callback` once per step.

    `i` is the 0-based step index; progress shows `i + 1` of `len(sigmas) - 1`. `denoised` is the
    step's prediction, shown as the preview.
    """

    x: torch.Tensor
    i: int
    sigma: torch.Tensor
    sigma_hat: torch.Tensor
    denoised: torch.Tensor
```

### `UnetArgs`

```python
class UnetArgs(TypedDict):
    """What a model wrapper gets. `timestep` holds sigmas.

    `cond_or_uncond` says which prompt each chunk of the batch is, by its place in the guider's
    list of prompts: with KSampler, `eng.guider.cfg` and `eng.guider.basic`, 0 is positive and 1
    negative; other guiders, such as DualCFGGuider and PerpNegGuider, number their prompts their
    own way. `c` is ComfyUI's conditioning for the model, passed through as it is: only passing
    it on to `apply_model` as `**c` is part of the API.
    """

    input: torch.Tensor
    timestep: torch.Tensor
    c: dict[str, Any]
    cond_or_uncond: list[int]
```

### `VaeInfo`

```python
class VaeInfo(TypedDict):
    latent_channels: int
    dtype: str | None
    spacial_compression_decode: int | float
    spacial_compression_encode: int | float
    temporal_compression_decode: int | None
    downscale_ratio: int | float | None
    upscale_ratio: int | float | None
```

### `WeightDtype`

```python
WeightDtype = Literal["float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"]
```

### `WeightedToken`

```python
WeightedToken = tuple[object, float] | tuple[object, float, int]
```

### `_PatchKey`

```python
_PatchKey = TypeVar("_PatchKey", bound=str | tuple[object, ...])
```

## From V3

These names are V3's objects, unchanged:

- `CacheContext`: `comfy_api.latest.Caching.CacheContext`
- `CacheProvider`: `comfy_api.latest.Caching.CacheProvider`
- `CacheValue`: `comfy_api.latest.Caching.CacheValue`
- `IO`: `comfy_api.latest.IO`
- `Input`: `comfy_api.latest.Input`
- `InputImpl`: `comfy_api.latest.InputImpl`
- `Types`: `comfy_api.latest.Types`
- `UI`: `comfy_api.latest.UI`
- `io`: `comfy_api.latest.io`
- `ui`: `comfy_api.latest.ui`
