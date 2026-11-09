"""The engine object a pack calls.

    from comfy_api.index import eng

    latent = await eng.vae.encode_tiled(vae, pixels, tile_x=512)
    model = await eng.model.set_post_cfg(model, my_hook)
    outputs = await eng.node.run("ConditioningCombine", inputs)
    model = await eng.ext.video_attention.enhance_a_video(model, latent, "wan", 2.0)

``CALLS`` is the supported surface: one row per call, read by one dispatcher.
Each row is also a method on ``eng`` (``"model.clone"`` is ``eng.model.clone``).
A row's name is the stable promise. Its target is an engine method or function,
so core can rename a method by editing one row. Packs that need engine objects
beyond these rows use an engine extension: one file in ``v2/engine/``, listed
in ``ALLOWED_EXTENSIONS``, called as ``eng.ext.<file>.<method>(...)``.

A pack works with the values themselves: tensors, latents, conditioning, and the
engine's MODEL, CLIP and VAE objects, which it passes to ``eng`` calls. Nothing
here hands a pack a handle. A provider that runs pack code in another process
turns engine objects into handles at its own boundary, and gives the pack the
data behind data handles, so pack code is the same wherever it runs.
Every call is async, including ones that return immediately.
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import importlib
import inspect
import logging
import os
import types
from typing import Any, NamedTuple, Optional

from . import _runtime
from ._runtime import CallNotSupported

logger = logging.getLogger(__name__)


class Call(NamedTuple):
    kind: Optional[str]  # handle kind of the first argument; None for functions
    target: str          # dotted path on the first argument, or "module:name"
    args: str = ""       # argument list for the generated stub
    returns: str = "Any"
    clone: bool = False  # clone the first argument, call on it, return the clone
    hook: bool = False   # the argument after the handle runs during sampling
    defaults: tuple = ()  # (keyword, value) pairs the row always passes
    doc: str = ""        # one line for the API reference


def _hook(method: str, runs_on: str) -> Call:
    return Call("MODEL", f"engine:model.{method}", "fn: Hook", "Model", hook=True,
                doc=f"Return a copy that runs a function on {runs_on} during sampling.")


CALLS: dict[str, Call] = {
    # CLIP
    "clip.clone": Call("CLIP", "engine:clip.clone", "", "Clip", doc="Copy a text encoder so patches don't affect the original."),
    "clip.tokenize": Call("CLIP", "engine:clip.tokenize", "text: str, **kwargs: Any", "dict", doc="Tokenize text with the encoder's own tokenizer. Returns plain token data."),
    "clip.encode_from_tokens": Call("CLIP", "engine:clip.encode_from_tokens", "tokens: Any, return_pooled: bool = False, return_dict: bool = False", "Any", doc="Encode tokens without scheduling: the conditioning tensor, or (cond, pooled) / a dict."),
    "clip.encode_tokens": Call("CLIP", "engine:clip.encode_tokens", "tokens: Any, **kwargs: Any", "Conditioning", doc="Encode tokens, including tokens the pack edited, into conditioning."),
    "clip.add_patches": Call(None, "engine:clip.add_patches", "clip: Clip, patches: dict, strength_patch: float = 1.0, strength_model: float = 1.0", "Clip", doc="Return a patched copy of the encoder."),
    "clip.read_weight": Call("CLIP", "engine:clip.read_weight", "key: str", "Tensor", doc="Read one weight with its patches applied."),
    # MODEL: weights and objects
    "model.clone": Call("MODEL", "engine:model.clone", "", "Model", doc="Copy a model so patches don't affect the original."),
    "model.add_patches": Call(None, "engine:model.add_patches", "model: Model, patches: dict, strength_patch: float = 1.0, strength_model: float = 1.0", "Model", doc="Return a patched copy of the model."),
    "model.read_weight": Call("MODEL", "engine:model.read_weight", "key: str", "Tensor", doc="Read one weight with its patches applied."),
    "model.load_lora": Call(None, "engine:model.load_lora", "model: Model | None, clip: Clip | None, lora: dict, strength_model: float = 1.0, strength_clip: float = 1.0, lora_metadata: dict | None = None", "tuple", doc="Apply a LoRA from its weights to a model and text encoder, as LoraLoader does: patched copies."),
    "model.latent_process_in": Call("MODEL", "engine:model.latent_process_in", "latent: Tensor", "Tensor", doc="Scale a latent into the model's input range."),
    "model.latent_process_out": Call("MODEL", "engine:model.latent_process_out", "latent: Tensor", "Tensor", doc="Scale a latent from the model's range back, as sampling output is."),
    "model.options": Call("MODEL", "engine:model.options", "", "dict", doc="The model's options (model_options) as plain data."),
    "model.set_option": Call("MODEL", "engine:model.set_option", "path: list[str], value: Any", "Model", doc="Return a copy with one model option set, such as transformer_options keys."),
    "model.percent_to_sigma": Call("MODEL", "engine:model.percent_to_sigma", "percent: float", "float", doc="The sigma at a fraction of the sampling schedule (start_percent / end_percent)."),
    "model.timestep": Call("MODEL", "engine:model.timestep", "sigma: Tensor", "Tensor", doc="The model's timestep for a sigma."),
    "model.sigma": Call("MODEL", "engine:model.sigma", "timestep: Tensor", "Tensor", doc="The sigma for a model timestep."),
    # MODEL: functions that run during sampling
    "model.set_cfg": _hook("set_cfg", "the CFG combination"),
    "model.set_post_cfg": _hook("set_post_cfg", "the result after CFG"),
    "model.set_pre_cfg": _hook("set_pre_cfg", "the conditioning before CFG"),
    "model.set_denoise_mask": _hook("set_denoise_mask", "the denoise mask"),
    "model.set_attn1_patch": _hook("set_attn1_patch", "self-attention inputs"),
    "model.set_attn2_patch": _hook("set_attn2_patch", "cross-attention inputs"),
    "model.set_attn1_output_patch": _hook("set_attn1_output_patch", "self-attention outputs"),
    "model.set_attn2_output_patch": _hook("set_attn2_output_patch", "cross-attention outputs"),
    "model.set_input_block_patch": _hook("set_input_block_patch", "input block features"),
    "model.set_input_block_patch_after_skip": _hook("set_input_block_patch_after_skip", "input block features after the skip"),
    "model.set_output_block_patch": _hook("set_output_block_patch", "output block features"),
    "model.set_middle_block_after_patch": _hook("set_middle_block_after_patch", "middle block features"),
    # MODEL: functions that run during sampling and call back into the engine
    "model.add_wrapper": Call("MODEL", "engine:model.add_wrapper", "kind: str, fn: Hook", "Model", doc="Return a copy that runs a function around one stage of sampling; it calls the stage itself."),
    "model.set_unet_wrapper": Call("MODEL", "engine:model.set_unet_wrapper", "fn: Hook", "Model", doc="Return a copy whose every model call goes through a function; it calls the model itself."),
    # VAE: rows that keep behaviour the stock nodes don't expose
    "vae.decode": Call("VAE", "engine:vae.decode", "samples: Tensor", "Tensor", doc="Decode a latent tensor, with no output clamping beyond the VAE's own."),
    "vae.decode_tiled": Call("VAE", "engine:vae.decode_tiled", "samples: Tensor, **kwargs: Any", "Tensor", doc="Decode a latent tensor in tiles."),
    "vae.encode": Call("VAE", "engine:vae.encode", "pixels: Tensor", "Tensor", doc="Encode an IMAGE into a latent tensor."),
    "vae.encode_tiled": Call("VAE", "engine:vae.encode_tiled", "pixels: Tensor, **kwargs: Any", "Tensor", doc="Encode pixels in tiles, with non-square tile sizes allowed."),
    # Engine functions
    "sampler.named": Call(None, "comfy.samplers:ksampler", "name: str, options: dict | None = None", "Sampler", doc="Get a sampler by name, with extra options (for example ge_gamma)."),
    "sampler.custom": Call(None, "engine:sampler.custom", "fn: Hook, options: dict | None = None, model_sampling: bool = False", "Sampler", doc="A SAMPLER whose sampling loop is a pack function; it calls the model each step."),
    "models.list": Call(None, "engine:models.list", "folder: str", "list[str]", doc="List the model files in a model folder."),
    "vae.from_state_dict": Call(None, "engine:vae.from_state_dict", "sd: dict", "Vae", doc="Build a VAE from weights the pack holds."),
    "memory.empty_cache": Call(None, "engine:memory.empty_cache", "", "None", doc="Release cached GPU memory."),
    "execution.check_interrupted": Call(None, "comfy.model_management:throw_exception_if_processing_interrupted", "", "None", doc="Raise if the user interrupted the prompt."),
    "execution.is_interrupted": Call(None, "comfy.model_management:processing_interrupted", "", "bool", doc="Read whether the user interrupted the prompt without clearing the cancellation flag."),
    # Helpers below
    "node.run": Call(None, "engine:node.run", "node: str, inputs: dict", "tuple", doc="Run a registered node and return its outputs as a tuple."),
    "node.run_output": Call(None, "engine:node.run_output", "node: str, inputs: dict", "dict", doc="Run a registered node and return its result and ui, for output and preview nodes."),
    "progress.update": Call(None, "engine:progress.update", "value: float, total: float, preview: Tensor | None = None, format: str = \"JPEG\"", "None", doc="Report this node's progress, optionally with a preview image."),
    # Rows implemented by ComfyUI's own engine extensions (comfy_api/_impl/engine/*.py)
    "model.set_attention": Call("MODEL", "engine:model.set_attention", "backend: str", "Model", doc="Return a copy that uses one of ComfyUI's attention implementations."),
    "model.torch_compile": Call("MODEL", "engine:model.torch_compile", "backend: str = \"inductor\", mode: str | None = None, fullgraph: bool = False, dynamic: bool | None = None, keys: list[str] | None = None, options: dict | None = None", "Model", doc="Return a copy whose diffusion model (or listed submodules) is compiled with torch.compile at sampling."),
    "models.read": Call(None, "engine:models.read", "folder: str, name: str", "dict", doc="Read a model file's tensors and metadata from a model folder."),
    "models.load_checkpoint": Call(None, "engine:models.load_checkpoint", "name: str, options: dict | None = None", "tuple", doc="Load a checkpoint from the checkpoints folder: (MODEL, CLIP, VAE)."),
    "models.load_diffusion_model": Call(None, "engine:models.load_diffusion_model", "name: str, options: dict | None = None", "Model", doc="Load a diffusion model from the diffusion_models folder."),
    "models.load_vae": Call(None, "engine:models.load_vae", "name: str, device: str | None = None, dtype: str | None = None", "Vae", doc="Load a VAE from the vae folder, optionally on a device and dtype."),
    "output.write": Call(None, "engine:output.write", "data: bytes | list[bytes], filename_prefix: str, suffix: str, folder: str = \"output\", width: int = 0, height: int = 0, trailing_underscore: bool = True", "dict | list[dict]", doc="Save bytes the node encoded (one file, or a batch) to the output or temp folder, with ComfyUI's file naming."),
    "files.read": Call(None, "engine:files.read", "ref: str", "bytes", doc="Read a file from the input, output or temp folder."),
    "files.list": Call(None, "engine:files.list", "folder: str = \"input\", subfolder: str = \"\", recursive: bool = False, extensions: list[str] | None = None", "list[dict]", doc="List files in the input, output or temp folder."),
    "files.exists": Call(None, "engine:files.exists", "ref: str", "bool", doc="Whether a file exists in the input, output or temp folder."),
    "device.info": Call(None, "engine:device.info", "", "dict", doc="The devices and dtypes ComfyUI computes with."),
    "memory.unload_all": Call(None, "engine:memory.unload_all", "", "None", doc="Unload every model from the GPU."),
    "memory.free_bytes": Call(None, "engine:memory.free_bytes", "device: str | None = None", "int", doc="Free memory on a device, in bytes."),
    "ui.progress_text": Call(None, "engine:ui.progress_text", "text: str", "None", doc="Show text on this node while it runs."),
    "ui.send": Call(None, "engine:ui.send", "event: str, data: dict", "None", doc="Send a custom event to the client that queued the prompt."),
    "ui.interact": Call(None, "engine:ui.interact", "kind: str, payload: dict, reuse_last: bool = False, remember: bool = False, timeout: float = 540.0", "Any", doc="Ask the user something mid-run (choose images, edit a mask, confirm a prompt) and wait for the answer."),
    "settings.get": Call(None, "engine:settings.get", "name: str", "Any", doc="Read one of ComfyUI's settings, such as disable_metadata."),
    "store.get": Call(None, "engine:store.get", "key: str, scope: str = \"pack\", default: Any = None", "Any", doc="Read a small value the pack saved, for the pack, this workflow or this node."),
    "store.set": Call(None, "engine:store.set", "key: str, value: Any, scope: str = \"pack\"", "None", doc="Save a small plain-data value that lasts across runs and restarts."),
    "store.delete": Call(None, "engine:store.delete", "key: str, scope: str = \"pack\"", "bool", doc="Remove a saved value; whether there was one."),
    "store.keys": Call(None, "engine:store.keys", "scope: str = \"pack\"", "list[str]", doc="The keys the pack saved in a scope."),
    "cache.get": Call(None, "engine:cache.get", "key: str, default: Any = None", "Any", doc="A large value the pack cached (a tensor, a model it loaded), or default."),
    "cache.set": Call(None, "engine:cache.set", "key: str, value: Any, ttl: float | None = None", "None", doc="Keep a large value in ComfyUI's memory between runs, optionally for ttl seconds."),
    "cache.delete": Call(None, "engine:cache.delete", "key: str", "bool", doc="Drop a cached value; whether there was one."),
    "memory.load_models": Call(None, "engine:memory.load_models", "models: list", "None", doc="Load models onto the GPU now."),
    "model.info": Call("MODEL", "engine:info.info", "", "dict", doc="Read-only facts about a diffusion model: its type, latent format, dtype and sigma range."),
    "model.weight_keys": Call("MODEL", "engine:info.weight_keys", "", "list[str]", doc="List the model's weight names."),
    "vae.info": Call("VAE", "engine:info.info", "", "dict", doc="Read-only facts about a VAE: latent channels, dtype and compression."),
    "clip.info": Call("CLIP", "engine:info.info", "", "dict", doc="Read-only facts about a text encoder."),
    "clip.weight_keys": Call("CLIP", "engine:info.weight_keys", "", "list[str]", doc="List the text encoder's weight names."),
}


# One line per group of rows, for the generated reference and stubs.
GROUPS: dict[str, str] = {
    "clip": "Text encoders (CLIP inputs): tokenize, encode, patch and read weights.",
    "model": "Diffusion models (MODEL inputs): copy, patch weights, and add functions that run during sampling.",
    "vae": "VAEs: decode and encode latents, and build a VAE from weights.",
    "sampler": "Samplers, for nodes that take a SAMPLER.",
    "models": "The model folders ComfyUI is configured with.",
    "memory": "GPU memory.",
    "execution": "The running prompt.",
    "node": "Running ComfyUI's own nodes from pack code.",
    "progress": "This node's progress bar.",
    "output": "Files a node saves to ComfyUI's output or temp folder.",
    "files": "Files in ComfyUI's input, output and temp folders.",
    "device": "Where ComfyUI computes.",
    "ui": "What the running node shows in the UI.",
    "settings": "ComfyUI settings a node may read.",
    "store": "Small values a pack keeps between runs and restarts: for the pack, a workflow or a node.",
    "cache": "Large values a pack keeps in ComfyUI's memory between runs.",
}

_ENGINE_FUNCTION_COMMON = """
The function runs during sampling and must not be async. What the engine hands it to call (the
stage, the model, the callback) is an engine function: call it like any function, with tensors,
plain data and what you were given. Engine objects in its arguments (the model, a guider, a
ControlNet) arrive as opaque values: pass them on, but don't look into them. It can't make ``eng``
calls. It runs wherever the pack runs; in a pack's own process each call to an engine function
goes to ComfyUI and back."""

_HOOK_COMMON = """
The function runs during sampling, on every step, and must not be async. It sees tensors,
plain data and their containers only: engine objects in its arguments (the model, for
one) are removed, and it can't make ``eng`` calls. It's kept with the returned copy, so
the original model is unchanged."""

# Longer documentation per row: arguments, returns, the function contract for hooks.
NOTES: dict[str, str] = {
    "clip.clone": "Returns:\n    A new CLIP sharing weights with ``clip``; patches added to it don't affect ``clip``.",
    "clip.tokenize": """Args:
    clip: The text encoder.
    text: The prompt, including ComfyUI's ``(word:1.2)`` weighting syntax.
    **kwargs: Passed to the tokenizer, for example ``return_word_ids=True``.

Returns:
    ``{encoder_name: [[(token_id, weight, ...), ...], ...]}``: plain lists a pack may edit
    (reweight, splice) before ``encode_tokens``.""",
    "clip.encode_tokens": """Args:
    clip: The text encoder.
    tokens: What ``tokenize`` returned, edited or not.
    **kwargs: ``add_dict`` (extra keys merged into each conditioning dict),
        ``unprojected``, ``show_pbar``.

Returns:
    CONDITIONING: ``[[cond_tensor, {"pooled_output": tensor, ...}], ...]``, the same value
    ``CLIPTextEncode`` produces.""",
    "clip.add_patches": """Args:
    clip: The text encoder.
    patches: ``{weight_key: patch}``, the format ``ModelPatcher.add_patches`` takes; a patch
        is a tuple such as ``(tensor,)`` for a weight difference.
    strength_patch: Scale for the patch.
    strength_model: Scale for the original weight.

Returns:
    A patched copy; ``clip`` is unchanged.""",
    "clip.read_weight": "Returns:\n    The weight tensor named ``key`` with current patches applied (see ``weight_keys``).",
    "model.clone": "Returns:\n    A new MODEL sharing weights with ``model``; patches and functions added to it don't affect ``model``.",
    "model.load_lora": """Args:
    model: The diffusion model, or None to patch only the text encoder.
    clip: The text encoder, or None to patch only the model.
    lora: The LoRA's weights, ``{name: tensor}``, as ``util.load_torch_file`` or
        ``eng.models.read("loras", name)["sd"]`` gives them; any LoRA format ComfyUI reads.
    strength_model, strength_clip: As ``LoraLoader`` takes them.
    lora_metadata: The LoRA file's metadata (``eng.models.read(...)["metadata"]``), attached to the
        patched copies as ComfyUI's loader does; optional.

Returns:
    ``(model, clip)``: patched copies (None where None was given); the originals are unchanged.""",
    "model.add_patches": """Args:
    model: The diffusion model.
    patches: ``{weight_key: patch}``, the format ``ModelPatcher.add_patches`` takes; a patch
        is a tuple such as ``(tensor,)`` for a weight difference.
    strength_patch: Scale for the patch.
    strength_model: Scale for the original weight.

Returns:
    A patched copy; ``model`` is unchanged.

Example::

    key = "diffusion_model.out.2.weight"
    model = await eng.model.add_patches(model, {key: (delta,)}, strength_patch=0.5)""",
    "model.read_weight": "Returns:\n    The weight tensor named ``key`` with current patches applied (see ``weight_keys``).",
    "model.latent_process_in": "Args:\n    latent: A latent tensor (``latent[\"samples\"]``).\n\nReturns:\n    The latent scaled into the model's input range, as sampling does.",
    "model.set_cfg": """Replaces the CFG combination. ``fn(args) -> Tensor`` gets a dict with ``cond`` and
``uncond`` (noise predictions, ``input - denoised``), ``cond_denoised``, ``uncond_denoised``,
``cond_scale``, ``timestep``/``sigma``, ``input`` and ``model_options``, and returns the combined
noise prediction (stock CFG: ``uncond + (cond - uncond) * cond_scale``).

Example::

    def scaled(args):
        return args["uncond"] + (args["cond"] - args["uncond"]) * args["cond_scale"] * 0.9

    model = await eng.model.set_cfg(model, scaled)
""" + _HOOK_COMMON,
    "model.set_post_cfg": """Runs after CFG. ``fn(args) -> Tensor`` gets a dict with ``denoised`` (the CFG result),
``cond_denoised``, ``uncond_denoised``, ``cond_scale``, ``sigma``, ``input`` and ``model_options``,
and returns the new ``denoised``. Several can be added; they run in order.
""" + _HOOK_COMMON,
    "model.set_pre_cfg": """Runs before CFG. ``fn(args) -> list`` gets a dict with ``conds_out`` (the model outputs for
cond and uncond), ``cond_scale``, ``timestep``/``sigma``, ``input`` and ``model_options``, and
returns the new ``conds_out`` list.
""" + _HOOK_COMMON,
    "model.set_denoise_mask": """``fn(sigma, denoise_mask, extra_options) -> Tensor`` gets the current sigma and denoise
mask (and ``extra_options["sigmas"]``), and returns the mask to use for this step.
""" + _HOOK_COMMON,
    "model.set_attn1_patch": """``fn(q, k, v, extra_options) -> (q, k, v)`` runs before each self-attention, with the
query input and the key/value context; ``extra_options`` has the block, its index, heads,
``sigmas`` and ``cond_or_uncond``.
""" + _HOOK_COMMON,
    "model.set_attn2_patch": """``fn(q, context, value, extra_options) -> (q, context, value)`` runs before each
cross-attention; ``extra_options`` as for ``set_attn1_patch``.
""" + _HOOK_COMMON,
    "model.set_attn1_output_patch": "``fn(out, extra_options) -> Tensor`` runs on each self-attention output.\n" + _HOOK_COMMON,
    "model.set_attn2_output_patch": "``fn(out, extra_options) -> Tensor`` runs on each cross-attention output.\n" + _HOOK_COMMON,
    "model.set_input_block_patch": "``fn(h, transformer_options) -> Tensor`` runs on each UNet input block's features.\n" + _HOOK_COMMON,
    "model.set_input_block_patch_after_skip": "``fn(h, transformer_options) -> Tensor`` runs on each UNet input block's features after the skip connection is stored.\n" + _HOOK_COMMON,
    "model.set_output_block_patch": "``fn(h, hsp, transformer_options) -> (h, hsp)`` runs on each UNet output block's features and its skip input.\n" + _HOOK_COMMON,
    "model.set_middle_block_after_patch": "``fn(args) -> dict`` gets ``{\"h\", \"x\", \"emb\", \"context\", \"y\", \"timesteps\", \"transformer_options\"}`` after the UNet middle block, and returns a dict whose ``\"h\"`` replaces the features.\n" + _HOOK_COMMON,
    "model.add_wrapper": """Args:
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
""" + _ENGINE_FUNCTION_COMMON,
    "model.set_unet_wrapper": """``fn(apply_model, args) -> Tensor`` runs for every model call: ``args`` has ``input``,
``timestep``, ``c`` (the conditioning keywords) and ``cond_or_uncond``, and the function returns
the model output, usually ``apply_model(args["input"], args["timestep"], **args["c"])`` with its own
changes. A wrapper already on the model runs inside this one (``apply_model`` goes through it), so
wrappers from different nodes compose. The V1 form is ``model.set_model_unet_function_wrapper``.
""" + _ENGINE_FUNCTION_COMMON,
    "sampler.custom": """Args:
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
""" + _ENGINE_FUNCTION_COMMON,
    "vae.decode": """Args:
    vae: The VAE.
    samples: The latent tensor (``latent["samples"]``).

Returns:
    Pixels, channels last, values as the VAE produces them. Video VAEs return a time
    dimension; ``await eng.node.run("VAEDecode", {"vae": vae, "samples": latent})`` gives the
    stock IMAGE batch instead.""",
    "vae.decode_tiled": "Args:\n    vae: The VAE.\n    samples: The latent tensor.\n    **kwargs: ``tile_x``, ``tile_y``, ``overlap``, and for video ``tile_t``, ``overlap_t``.\n\nReturns:\n    Pixels, as ``decode``.",
    "vae.encode_tiled": "Args:\n    vae: The VAE.\n    pixels: An IMAGE tensor ``[B, H, W, C]``.\n    **kwargs: ``tile_x``, ``tile_y`` (need not be equal), ``overlap``, ``tile_t``, ``overlap_t``.\n\nReturns:\n    The latent tensor; wrap it as ``{\"samples\": tensor}`` for a LATENT output.",
    "sampler.named": "Args:\n    name: A KSampler sampler name, such as ``\"euler\"`` or ``\"dpmpp_2m\"``.\n    options: Extra sampler options, such as ``{\"ge_gamma\": 2.0}``.\n\nReturns:\n    A SAMPLER, to return from the node or pass to ``node.run(\"SamplerCustom\", ...)``.",
    "models.list": "Args:\n    folder: A model folder name: ``\"checkpoints\"``, ``\"loras\"``, ``\"vae\"``, ``\"diffusion_models\"``, ``\"text_encoders\"``...\n\nReturns:\n    The file names in it, as model combo inputs list them.",
    "vae.from_state_dict": "Args:\n    sd: ``{weight_name: tensor}`` for a VAE ComfyUI can detect from its weights.\n\nReturns:\n    A VAE.",
    "memory.empty_cache": "Frees cached, unused GPU memory. Models stay loaded.",
    "execution.check_interrupted": "Raises the interruption ComfyUI uses when the user cancels the prompt. Call it inside long loops.",
    "execution.is_interrupted": "Returns the current prompt cancellation flag as a boolean without consuming or resetting it. Use when a node must return partial results or finish a model's own stop callback. Broker failures propagate to the caller.",
    "node.run": """Args:
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
        "scheduler": "normal", "denoise": 1.0})""",
    "node.run_output": """Like ``node.run``, for output and preview nodes.

Returns:
    ``{"result": tuple, "ui": dict | None}``. Return the ui from your node to show it:
    ``io.NodeOutput(ui=(await eng.node.run_output("PreviewImage", {"images": image}))["ui"])``.""",
    "progress.update": """Args:
    value: Steps done.
    total: Steps in all.
    preview: Optional IMAGE tensor (``[H, W, C]`` or the first of ``[B, H, W, C]``, 0..1) to show
        on the node, as samplers show their previews.
    format: ``"JPEG"`` or ``"PNG"``.""",
    "model.set_attention": """Args:
    model: The diffusion model.
    backend: An attention implementation ComfyUI has: ``"pytorch"``, ``"sage"``, ``"sage3"``,
        ``"flash"``, ``"xformers"``, ``"sub_quad"``, ``"optimized"``... Which are available
        depends on the installed kernels; an unknown name raises and lists them.

Returns:
    A copy whose attention calls go to that implementation.""",
    "model.torch_compile": """Args:
    model: The diffusion model.
    backend: A torch.compile backend, such as ``"inductor"`` or ``"cudagraphs"``.
    mode, fullgraph, dynamic, options: Passed to ``torch.compile``.
    keys: Submodules of the model to compile, such as ``["diffusion_model.blocks.0"]``
        (names from ``model.weight_keys``, without the parameter suffix). Default: the
        whole diffusion model.

Returns:
    A copy that compiles when it first samples. Compilation happens in ComfyUI's process.""",
    "models.read": """Args:
    folder: A model folder: ``"loras"``, ``"checkpoints"``, ``"vae"``...
    name: A file name from ``models.list(folder)``.

Returns:
    ``{"sd": {weight_name: tensor}, "metadata": dict}``.""",
    "models.load_checkpoint": """Args:
    name: A file in the checkpoints folder.
    options: ComfyUI ``model_options`` for the diffusion model.

Returns:
    ``(model, clip, vae)``, as ``CheckpointLoaderSimple`` gives.""",
    "models.load_diffusion_model": """Args:
    name: A file in the diffusion_models folder.
    options: ComfyUI ``model_options``, for example ``{"dtype": "float8_e4m3fn"}``.

Returns:
    The MODEL, as ``UNETLoader`` gives.""",
    "models.load_vae": """Args:
    name: A file in the vae folder.
    device: For example ``"cpu"`` or ``"cuda:0"``; default ComfyUI's choice.
    dtype: ``"float32"``, ``"float16"`` or ``"bfloat16"``; default ComfyUI's choice.

Returns:
    The VAE. For ComfyUI's built-in names (``"taesd"``, ``"pixel_space"``...) use
    ``eng.node.run("VAELoader", ...)``.""",
    "output.write": """Saves bytes the node encoded itself (a PNG, a .safetensors file, text...).
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
    ui ``images`` list, so ``io.NodeOutput(ui={"images": [entry]})`` shows it.""",
    "files.read": "Args:\n    ref: A file ref: ``\"photo.png\"`` (input), ``\"sub/a.png [input]\"``, ``\"x.png [output]\"``, ``\"x.png [temp]\"``.\n\nReturns:\n    The file's bytes. Refs can't leave those folders.",
    "files.list": "Args:\n    folder: ``\"input\"``, ``\"output\"`` or ``\"temp\"``.\n    subfolder: Inside it.\n    recursive: Include subfolders.\n    extensions: Only these, such as ``[\"png\", \"jpg\"]``.\n\nReturns:\n    ``[{\"ref\", \"name\", \"size\", \"mtime\"}]``, sorted; pass ``ref`` to ``files.read``.",
    "files.exists": "Returns:\n    Whether ``ref`` names a file in the input, output or temp folder.",
    "device.info": """Returns:
    ``{"torch_device", "intermediate_device", "intermediate_dtype", "unet_dtype",
    "offload_device", "vae_device", "vae_dtype", "text_encoder_device",
            "vae_offload_device", "text_encoder_offload_device"}``, as strings such as
    ``"cuda:0"`` and ``"float16"`` (``getattr(torch, info["unet_dtype"])``). Put tensors a node
    returns on ``intermediate_device`` in ``intermediate_dtype``, as ComfyUI's nodes do.""",
    "memory.unload_all": "Unloads every loaded model from the GPU and frees cached memory.",
    "memory.free_bytes": "Args:\n    device: Default ComfyUI's compute device.\n\nReturns:\n    Free memory on it, in bytes, counting memory ComfyUI can free.",
    "ui.progress_text": "Args:\n    text: Shown on the node, under its title, until the next update.",
    "ui.send": "Args:\n    event: The event name a pack's frontend extension listens for.\n    data: Plain data, sent as JSON.\n\nSent to the client that queued the running prompt, as ``PromptServer.send_sync`` does.",
    "store.get": """Args:
    key: The value's name.
    scope: ``"pack"`` (every workflow), ``"workflow"`` (this workflow, by its id) or ``"node"``
        (this node in this workflow).
    default: What to return if nothing is saved.

Returns:
    The saved plain data. Values persist in ComfyUI's user folder, the same wherever the node runs:
    keys up to 128 bytes, values up to 1 MB, 256 entries and 8 MB per scope.""",
    "store.set": "Args:\n    key: The value's name.\n    value: Plain data (JSON): strings, numbers, booleans, lists, dicts.\n    scope: As ``store.get``.",
    "cache.set": """Args:
    key: The value's name, per pack.
    value: Tensors, plain data, or engine objects such as a model the pack loaded.
    ttl: Seconds to keep it; default until it's pushed out.

Kept in ComfyUI's memory, least recently used first out within ``COMFY_PACK_CACHE_MB``
(default 2048 MB across packs). Nothing survives a restart; use ``store`` for that.""",
    "ui.interact": """Args:
    kind: ``"image-choice"``, ``"prompt-await"`` or ``"mask-edit"``.
    payload: What the pack's frontend shows (plain JSON, up to 64 KB).
    reuse_last: Answer with this node's last answer of this kind without asking.
    remember: Keep this answer for ``reuse_last``.
    timeout: Seconds to wait, 1 to 540.

Returns:
    The user's answer, checked against the kind's contract. The pack's frontend extension shows the
    request and posts the answer back (the event and route are in ``comfy_api/README.md``).""",
    "settings.get": "Args:\n    name: ``\"disable_metadata\"`` (bool: don't write prompt metadata into saved files), ``\"preview_method\"``, ``\"preview_size\"``, ``\"fast\"`` (the names of the enabled ``--fast`` optimizations, sorted; read-only).\n\nReturns:\n    The setting's value, as plain data.",
    "memory.load_models": "Args:\n    models: MODEL, CLIP or VAE objects to load onto the GPU before the node uses them.",
    "model.info": """Returns:
    ``model_type``, ``config_class``, ``unet_config``, ``latent_scale_factor``, ``latent_channels``,
    ``zsnr``, ``dtype``, ``sigma_min``, ``sigma_max``. All plain values.""",
    "vae.info": """Returns:
    ``latent_channels``, ``dtype``, ``spacial_compression_decode``, ``spacial_compression_encode``,
    ``temporal_compression_decode``, ``downscale_ratio``, ``upscale_ratio``. All plain values.""",
    "clip.info": "Returns:\n    ``layer_idx``: the CLIP skip layer, or None.",
    "model.set_option": """Args:
    model: The diffusion model.
    path: Keys into ``model_options``, such as ``["transformer_options", "my_pack_strength"]``.
    value: Plain data or tensors.

Returns:
    A copy with the option set; the dicts along the path are copied, so ``model`` is unchanged.
    The V1 form ``m = model.clone(); m.model_options["transformer_options"]["k"] = v`` becomes
    ``m = await eng.model.set_option(model, ["transformer_options", "k"], v)``.""",
    "model.percent_to_sigma": "Args:\n    percent: 0.0 (start of sampling) to 1.0 (end).\n\nReturns:\n    The sigma at that point, as ComfyUI's start_percent / end_percent options compute it.",
    "clip.encode_from_tokens": "Args:\n    tokens: From ``clip.tokenize``.\n    return_pooled: Also return the pooled output: ``(cond, pooled)``.\n    return_dict: Return ComfyUI's dict form.\n\nThe unscheduled encode V1 nodes call as ``clip.encode_from_tokens``; ``clip.encode_tokens`` is the scheduled one CLIPTextEncode uses.",
    "model.weight_keys": "Returns:\n    The sorted weight names, for ``read_weight`` and ``add_patches``.",
    "clip.weight_keys": "Returns:\n    The sorted weight names, for ``read_weight`` and ``add_patches``.",
}


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
async def _resolve(value: Any, refs: Any) -> Any:
    if isinstance(value, _runtime.Ref):
        value = await refs.resolve(value)
    if type(value) is dict:
        return {k: await _resolve(v, refs) for k, v in value.items()}
    if type(value) in (list, tuple):
        return type(value)([await _resolve(v, refs) for v in value])
    return value


CORE_EXTENSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine")
COMMON_EXTENSIONS_DIR = os.path.join(CORE_EXTENSIONS_DIR, "common")
_core_extensions: dict[str, Any] = {}
_common_extensions: dict[str, Any] = {}


def core_extension(stem: str) -> Any:
    """ComfyUI's own engine extension ``engine/<stem>.py``: the same form packs ship, part of core."""
    ops = _core_extensions.get(stem)
    if ops is None:
        path = os.path.join(CORE_EXTENSIONS_DIR, stem + ".py")
        with open(path, "rb") as file:
            source = file.read()
        ops = _core_extensions[stem] = _instantiate((stem + ".py", "core"), path, source)
    return ops


def _lookup(path: str, subject: Any = None) -> Any:
    if path.startswith("engine:"):
        stem, _, method = path[len("engine:"):].partition(".")
        return getattr(core_extension(stem), method)
    if ":" in path:
        module, name = path.split(":")
        return getattr(importlib.import_module(module), name)
    target = subject
    for part in path.split("."):
        target = getattr(target, part)
    return target


def hook_view(value: Any) -> Any:
    """What a sampling function sees: tensors, plain data and their containers.

    The same view applies wherever the function runs, so a function can't come
    to depend on an engine object that isn't always reachable.
    """
    if value is None or isinstance(value, (str, bool, int, float)) or _runtime._looks_like_tensor(value):
        return value
    if type(value) is dict:
        return {k: hook_view(v) for k, v in value.items() if isinstance(k, str) and _viewable(v)}
    if type(value) in (list, tuple):
        return type(value)(hook_view(v) for v in value if _viewable(v))
    return None


def _viewable(value: Any) -> bool:
    return (value is None or type(value) in (str, bool, int, float, dict, list, tuple)
            or _runtime._looks_like_tensor(value))


def _sampling_function(fn: Any) -> Any:
    @functools.wraps(fn)
    def run(*args: Any, **kwargs: Any) -> Any:
        return fn(*[hook_view(a) for a in args], **{k: hook_view(v) for k, v in kwargs.items()})
    return run


# --------------------------------------------------------------------------- #
# Pack functions that call back into the engine (wrappers, custom samplers)
# --------------------------------------------------------------------------- #
class Opaque:
    """An engine value inside what a pack function is given (a model, a guider, a ControlNet...).

    The function can pass it back to an ``EngineFunction`` it was given, and can't look into it,
    so it works the same wherever the function runs."""

    __slots__ = ("_value",)

    def __init__(self, value: Any) -> None:
        self._value = value

    def __repr__(self) -> str:
        return f"<engine value: {type(self._value).__name__}>"


class EngineFunction:
    """An engine callable a pack function is given (a wrapper's ``executor``, a sampler's ``model``,
    the preview ``callback``). The function calls it, synchronously, while it runs; arguments and
    results are tensors, plain data and ``Opaque`` values."""

    __slots__ = ("_fn", "_name")

    def __init__(self, fn: Any, name: str = "engine function") -> None:
        self._fn = fn
        self._name = name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return call_view(self._fn(*unview(list(args)), **unview(dict(kwargs))))

    def __repr__(self) -> str:
        return f"<{self._name}>"


def call_view(value: Any) -> Any:
    """What a pack function that calls back into the engine sees: tensors, numpy arrays, plain
    data and their containers as they are; engine functions as themselves; anything else as an
    ``Opaque`` value it can pass back."""
    if value is None or isinstance(value, (str, bool, int, float, Opaque, EngineFunction)):
        return value
    if _runtime._looks_like_tensor(value) or type(value).__module__ == "numpy":
        return value
    if type(value) is dict and all(isinstance(k, str) for k in value):
        return {k: call_view(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        return type(value)(call_view(v) for v in value)
    return Opaque(value)


def unview(value: Any) -> Any:
    """The engine's values back from a pack function's view. A pack function in them (a model
    wrapper handed to a sampler) gets the same view when the engine calls it."""
    if isinstance(value, Opaque):
        return value._value
    if isinstance(value, EngineFunction):
        return value._fn
    if isinstance(value, (types.FunctionType, types.MethodType, functools.partial)):
        return pack_function(value)
    if type(value) is dict:
        return {k: unview(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        return type(value)(unview(v) for v in value)
    return value


def model_sampling_view(model_sampling: Any) -> dict:
    """A model's sampling facts for pack functions: plain values and engine functions."""
    return {"sigma_min": float(model_sampling.sigma_min), "sigma_max": float(model_sampling.sigma_max),
            "sigmas": getattr(model_sampling, "sigmas", None),
            "sigma": EngineFunction(model_sampling.sigma, "sigma"),
            "timestep": EngineFunction(model_sampling.timestep, "timestep"),
            "percent_to_sigma": EngineFunction(model_sampling.percent_to_sigma, "percent_to_sigma")}


def sampler_function(sampler: Any) -> EngineFunction:
    """A SAMPLER as an engine function a pack sampler can run:
    ``sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, **options)``."""
    def run(model: Any, x: Any, sigmas: Any, extra_args: Any = None, callback: Any = None, disable: Any = None,
            **options: Any) -> Any:
        return sampler.sampler_function(model, x, sigmas, extra_args=extra_args or {}, callback=callback,
                                        disable=disable, **{**sampler.extra_options, **options})
    return EngineFunction(run, "sampler")


def check_sigmas(sigmas: Any) -> None:
    """Custom-sampler sigma checks (a 1-D, finite, non-negative, non-increasing
    schedule of 2..4097 values), without its "ends at 0": ComfyUI's SplitSigmas makes partial ones."""
    import torch

    if not _runtime._looks_like_tensor(sigmas) or sigmas.ndim != 1 or not 2 <= len(sigmas) <= 4097:
        raise ValueError("a sampler's sigmas are a 1-D tensor of 2 to 4097 values")
    values = sigmas.detach().float().cpu()
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()) or bool((values[1:] > values[:-1]).any()):
        raise ValueError("a sampler's sigmas must be finite, non-negative and non-increasing")


def pack_function(fn: Any, engine_args: tuple = ()) -> Any:
    """``fn`` as the engine calls it: its arguments in ``call_view`` (those at the positions in
    ``engine_args``, and keywords named there, as ``EngineFunction``), its result unviewed."""
    if not callable(fn):
        raise TypeError("needs a function")

    @functools.wraps(fn)
    def run(*args: Any, **kwargs: Any) -> Any:
        viewed = [EngineFunction(a, f"engine function {i}") if (i in engine_args and callable(a)) else call_view(a)
                  for i, a in enumerate(args)]
        named = {k: EngineFunction(v, k) if (k in engine_args and callable(v)) else call_view(v)
                 for k, v in kwargs.items()}
        return unview(fn(*viewed, **named))
    return run


def _check_subject(name: str, row: Call, value: Any) -> None:
    if row.kind is None:
        return
    if value is None:
        raise TypeError(f"eng.{name} needs a {row.kind}, got None")
    kind = value.kind if isinstance(value, _runtime.Ref) else _runtime.handle_kind(value)
    if kind != row.kind and kind != "OPAQUE":
        raise TypeError(f"eng.{name} needs a {row.kind}, got {kind}")


async def dispatch(_subject: Any, name: str, args: Any = (), kwargs: Optional[dict] = None) -> Any:
    """``ops.apply("engine.call", ...)``: one row or extension method per call.

    Takes and returns engine values. A provider that received handles from
    another process resolves them here against its own table (``rt.refs``), and
    turns the result back into handles itself, at its boundary.
    """
    rt = _runtime.current_runtime()
    row = CALLS.get(name)
    call_args = await _resolve(list(args), rt.refs)
    call_kwargs = await _resolve(dict(kwargs or {}), rt.refs)
    if row is None:
        fn = _extension_method(name, rt.pack)
    else:
        _check_subject(name, row, args[0] if args else None)
        call_kwargs = {**dict(row.defaults), **call_kwargs}
        if row.hook:
            if len(call_args) < 2 or not callable(call_args[1]):
                raise TypeError(f"eng.{name} needs a function to run during sampling")
            call_args[1] = _sampling_function(call_args[1])
        if row.kind is None or row.target.startswith("engine:"):
            fn = _lookup(row.target)  # a core extension method takes its subject as the first argument
        else:
            subject = call_args.pop(0)
            if row.clone:
                subject = subject.clone()
            fn = _lookup(row.target, subject)
    result = fn(*call_args, **call_kwargs)
    if inspect.isawaitable(result):
        result = await result
    if row is not None and row.clone:
        result = subject
    return result


# --------------------------------------------------------------------------- #
# Engine extensions: one file each, allowed by file name and hash
# --------------------------------------------------------------------------- #
ALLOWED_EXTENSIONS: list[tuple[str, str]] = [
    ("video_attention.py", "sha256:20d6f85ece9c1824aba90b3870acf5814b0f9d96c099f17c3a1f28d2a978fffd"),
]
DEV_EXTENSIONS_ENV = "COMFY_DEV_ENGINE_EXTENSIONS"

_extensions: dict[tuple[str, str], Any] = {}  # (file name, digest) -> EngineOps
_pack_extensions: dict[str, dict[str, tuple[str, str]]] = {}  # pack -> stem -> key


class ExtensionNotAllowed(ImportError):
    """A pack ships an engine extension that isn't on the allowed list."""


def _extension_dir(pack_root: str) -> str:
    nested = os.path.join(pack_root, "v2", "engine")
    if os.path.basename(os.path.normpath(pack_root)) == "v2" and not os.path.isdir(nested):
        return os.path.join(pack_root, "engine")  # loaded from the pack's v2/ conversion
    return nested


def load_pack_extensions(pack: str, pack_root: str) -> list[str]:
    """Load the extensions a pack ships, or fail the pack's load."""
    directory = _extension_dir(pack_root)
    if not os.path.isdir(directory):
        return []
    entries = sorted(os.listdir(directory))
    loaded: dict[str, tuple[str, str]] = {}
    for file_name in entries:
        path = os.path.join(directory, file_name)
        if not file_name.endswith(".py"):
            if os.path.isfile(path) or (os.path.isdir(path) and file_name != "__pycache__"):
                raise ExtensionNotAllowed(f"an engine extension is one .py file; found {path}")
            continue
        with open(path, "rb") as file:
            source = file.read()  # the bytes that are hashed are the bytes that run
        key = (file_name, "sha256:" + hashlib.sha256(source).hexdigest())
        if file_name[:-3] in _table_groups():
            raise ExtensionNotAllowed(f"engine extension {file_name} has the name of an eng group; rename the file")
        if key not in ALLOWED_EXTENSIONS and not os.environ.get(DEV_EXTENSIONS_ENV):
            raise ExtensionNotAllowed(f"engine extension {file_name} ({key[1]}) is not allowed")
        check_extension_imports(file_name, source, path)  # the developer override doesn't skip this
        if key not in _extensions:
            _extensions[key] = _instantiate(key, path, source)
        loaded[file_name[:-3]] = key
    _pack_extensions[pack] = loaded
    return sorted(loaded)


def _table_groups() -> set:
    return {name.partition(".")[0] for name in CALLS}


_COMFY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_NOT_ENGINE = frozenset({"custom_nodes", "tests", "tests_unit", "script_examples", "input", "output", "temp",
                         "user", "models", "notebooks", "alembic_db", "app"})
_allowed_modules: Optional[frozenset] = None


def _requirement_names(path: str) -> set:
    import re

    names = set()
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as file:
            for line in file:
                line = line.split("#", 1)[0].strip()
                match = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)", line)
                if match and not line.startswith("-"):
                    names.add(re.sub(r"[-_.]+", "-", match.group(1)).lower())
    return names


def extension_allowed_modules() -> frozenset:
    """What an engine extension may import: the standard library, ComfyUI's own modules, and the
    packages ComfyUI itself requires (``requirements.txt``). An extension adds nothing else."""
    global _allowed_modules
    if _allowed_modules is None:
        import importlib.metadata
        import re
        import sys

        allowed = set(sys.stdlib_module_names)
        for entry in os.listdir(_COMFY_ROOT):
            name = entry[:-3] if entry.endswith(".py") else entry
            path = os.path.join(_COMFY_ROOT, entry)
            if name.isidentifier() and name not in _NOT_ENGINE and (entry.endswith(".py") or (
                    os.path.isdir(path) and any(f.endswith(".py") for f in os.listdir(path)))):
                allowed.add(name)  # ComfyUI's own modules and packages (some are namespace packages)
        required = _requirement_names(os.path.join(_COMFY_ROOT, "requirements.txt"))
        for module, distributions in importlib.metadata.packages_distributions().items():
            if any(re.sub(r"[-_.]+", "-", d).lower() in required for d in distributions):
                allowed.add(module)
        _allowed_modules = frozenset(allowed)
    return _allowed_modules


def extension_imports(source: bytes, path: str) -> list:
    """Every module an extension imports, read statically: import statements and literal
    ``importlib.import_module`` / ``__import__`` calls. A name it can't read is ``"?"``."""
    import ast

    found = []
    for node in ast.walk(ast.parse(source, path)):
        if isinstance(node, ast.Import):
            found += [(alias.name, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            found.append(("." if node.level else node.module or "?", node.lineno))
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name in ("import_module", "__import__"):
                arg = node.args[0] if node.args else None
                literal = isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                found.append((arg.value if literal else "?", node.lineno))
    return found


def declared_requires(source: bytes, path: str) -> list:
    """A common extension's ``REQUIRES = [...]``: the extra packages it may import."""
    import ast

    for node in ast.parse(source, path).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "REQUIRES" for t in node.targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ExtensionNotAllowed(f"{path}: REQUIRES is a list of module names")
            return value
    return []


def common_extension(stem: str) -> Any:
    """``eng.ext.<stem>`` for a pack that doesn't ship ``<stem>.py``: ComfyUI's common extension, if
    it has one. Common extensions are reviewed with ComfyUI and may import the packages they declare
    in ``REQUIRES`` (kernel packages such as triton), which must be installed in ComfyUI's environment."""
    ops = _common_extensions.get(stem)
    if ops is not None:
        return ops
    path = os.path.join(COMMON_EXTENSIONS_DIR, stem + ".py")
    if not stem.isidentifier() or not os.path.isfile(path):
        return None
    with open(path, "rb") as file:
        source = file.read()
    requires = declared_requires(source, path)
    check_extension_imports(stem + ".py", source, path, extra=requires)
    import importlib.util
    missing = [m for m in requires if importlib.util.find_spec(m) is None]
    if missing:
        raise CallNotSupported(f"eng.ext.{stem} is a common extension that needs {', '.join(missing)}; "
                               "install it into ComfyUI's environment")
    ops = _common_extensions[stem] = _instantiate((stem + ".py", "common"), path, source)
    return ops


def check_extension_imports(file_name: str, source: bytes, path: str, extra: Any = ()) -> None:
    """Refuse an extension that would bring a dependency ComfyUI doesn't have into its process."""
    allowed = extension_allowed_modules() | set(extra)
    refused = sorted({f"{name} (line {line})" for name, line in extension_imports(source, path)
                      if name in (".", "?") or name.split(".")[0] not in allowed})
    if refused:
        raise ExtensionNotAllowed(
            f"engine extension {file_name} imports what ComfyUI doesn't ship: {', '.join(refused)}. "
            "Extensions use the standard library, ComfyUI and ComfyUI's own requirements only")


def _instantiate(key: tuple[str, str], path: str, source: bytes) -> Any:
    module = types.ModuleType(f"comfy_engine_extension_{key[0][:-3]}_{key[1][7:19]}")
    module.__file__ = path
    exec(compile(source, path, "exec"), module.__dict__)
    ops = module.__dict__.get("EngineOps")
    if not inspect.isclass(ops):
        raise ExtensionNotAllowed(f"{key[0]} has no EngineOps class")
    return ops()


def _extension_method(name: str, pack: str) -> Any:
    stem, _, method = name.rpartition(".")
    files = None
    owner = pack
    while owner and files is None:
        files = _pack_extensions.get(owner)
        owner = owner.rpartition(".")[0]
    key = (files or {}).get(stem)
    if method.startswith("_"):
        raise CallNotSupported(f"unknown engine call {name!r}")
    ops = _extensions[key] if key is not None else common_extension(stem)  # the pack's own first
    if ops is None:
        raise CallNotSupported(f"unknown engine call {name!r}")
    fn = getattr(ops, method, None)
    if not callable(fn):
        raise CallNotSupported(f"unknown engine call {name!r}")
    return fn


# --------------------------------------------------------------------------- #
# The object packs use
# --------------------------------------------------------------------------- #
def _is_function(value: Any) -> bool:
    return isinstance(value, (types.FunctionType, types.MethodType, functools.partial))


async def _send(name: str, args: tuple, kwargs: dict) -> Any:
    """Every ``eng`` call goes through the runtime's ops provider.

    In process that is ``EngineCalls``, which calls ``dispatch`` with the values
    as they are. A provider that runs the pack elsewhere converts at its boundary.
    """
    rt = _runtime.current_runtime()
    return await rt.ops.apply("engine.call", None, {"name": name, "args": list(args), "kwargs": dict(kwargs)})


def _row_method(name: str, row: Call) -> Any:
    async def method(*args: Any, **kwargs: Any) -> Any:
        return await _send(name, args, kwargs)
    method.__name__ = name.rpartition(".")[2]
    method.__qualname__ = f"eng.{name}"
    method.__doc__ = row.doc
    return method


class _Group:
    """``eng.model``, ``eng.vae``...: one method per table row in the group."""

    def __init__(self, prefix: str, methods: dict) -> None:
        self._prefix = prefix
        self.__dict__.update(methods)

    def __repr__(self) -> str:
        return f"<eng.{self._prefix}: {', '.join(sorted(k for k in self.__dict__ if not k.startswith('_')))}>"


class _ExtensionFile:
    """``eng.ext.<file>``: the methods of one engine extension the pack ships."""

    def __init__(self, stem: str) -> None:
        self._stem = stem

    def __getattr__(self, method: str) -> Any:
        if method.startswith("_"):
            raise AttributeError(method)
        name = f"{self._stem}.{method}"

        async def call(*args: Any, **kwargs: Any) -> Any:
            return await _send(name, args, kwargs)
        call.__name__ = method
        call.__qualname__ = f"eng.ext.{name}"
        return call

    def __repr__(self) -> str:
        return f"<eng.ext.{self._stem}>"


class _Extensions:
    """``eng.ext``: the engine extensions the calling pack ships, by file name.

    ``await eng.ext.video_attention.enhance_a_video(model, ...)`` calls ``EngineOps.enhance_a_video``
    in the pack's ``v2/engine/video_attention.py``. Names resolve when called, against the calling
    pack's own extensions, so a pack can't reach a file it doesn't ship.
    """

    def __getattr__(self, stem: str) -> _ExtensionFile:
        if stem.startswith("_"):
            raise AttributeError(stem)
        if stem in _table_groups():
            raise AttributeError(f"eng.ext.{stem}: {stem!r} is a table group; call eng.{stem}.<name>(...)")
        return _ExtensionFile(stem)

    def __repr__(self) -> str:
        return "<eng.ext>"


class Engine:
    def __init__(self) -> None:
        groups: dict[str, dict] = {}
        for name, row in CALLS.items():
            group, _, method = name.partition(".")
            groups.setdefault(group, {})[method] = _row_method(name, row)
        for group, methods in groups.items():
            setattr(self, group, _Group(group, methods))
        self.ext = _Extensions()


eng = Engine()
