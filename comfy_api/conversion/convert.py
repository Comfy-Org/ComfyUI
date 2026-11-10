"""Convert one V1 custom-node pack into a V2 pack (``v2/`` + ``pyproject.toml`` declarations), statically.

    python convert.py <pack dir> <describe.json> <out dir>

Reads the pack's source and what ``describe.py`` recorded; runs none of the pack's code. Writes
``<out>/<pack>/__init__.py`` (a stub) and ``<out>/<pack>/v2/``:

- the pack's own files, with known engine calls rewritten to the V2 API (``REWRITES``: ComfyUI
  helpers to ``util``; devices, progress, model lists, UI events and interrupts to ``eng`` through
  ``_v2_engine``; MODEL / CLIP / VAE methods on typed node inputs to ``eng`` rows);
- ``_v2_nodes.py``: an ``io.ComfyNode`` per V1 node, with the node's exact schema (V1 options carried
  in ``extra_dict``; model-folder dropdowns as ``io.ModelFolder``; sampler / scheduler lists as
  ``util`` constants), whose ``execute`` runs the original method;
- ``__init__.py``: the pack's ``comfy_entrypoint``.

Prints a JSON report: what was rewritten, and every engine use left (``leftovers``).
"""
import ast
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import normalize  # noqa: E402

ENGINE_ROOTS = {"comfy", "folder_paths", "nodes", "server", "node_helpers", "comfy_extras", "latent_preview",
                "execution", "comfy_execution", "comfy_api_nodes", "app", "comfy_config"}

# dotted engine name -> replacement expression (``util.*`` is pure; ``_v2e.*`` is generated, over eng)
REWRITES = {
    "comfy.utils.common_upscale": "util.common_upscale",
    "comfy.utils.ProgressBar": "_v2e.ProgressBar",
    "comfy.sample.prepare_noise": "util.prepare_noise",
    "nodes.MAX_RESOLUTION": "util.MAX_RESOLUTION",
    "node_helpers.conditioning_set_values": "util.conditioning_set_values",
    "node_helpers.pillow": "util.pillow",
    "comfy.samplers.KSampler.SAMPLERS": "util.SAMPLER_NAMES",
    "comfy.samplers.KSampler.SCHEDULERS": "util.SCHEDULER_NAMES",
    "comfy.samplers.SAMPLER_NAMES": "util.SAMPLER_NAMES",
    "comfy.samplers.SCHEDULER_NAMES": "util.SCHEDULER_NAMES",
    "comfy.samplers.KSAMPLER_NAMES": "util.SAMPLER_NAMES",
    "nodes.KSampler.SAMPLERS": "util.SAMPLER_NAMES",
    "nodes.KSampler.SCHEDULERS": "util.SCHEDULER_NAMES",
    "nodes.KSamplerAdvanced.SAMPLERS": "util.SAMPLER_NAMES",
    "nodes.KSamplerAdvanced.SCHEDULERS": "util.SCHEDULER_NAMES",
    "comfy.model_management.get_torch_device": "_v2e.get_torch_device",
    "comfy.model_management.intermediate_device": "_v2e.intermediate_device",
    "comfy.model_management.unet_offload_device": "_v2e.offload_device",
    "comfy.model_management.soft_empty_cache": "_v2e.soft_empty_cache",
    "comfy.model_management.throw_exception_if_processing_interrupted": "_v2e.throw_exception_if_processing_interrupted",
    "comfy.model_management.processing_interrupted": "_v2e.processing_interrupted",
    "comfy.model_management.InterruptProcessingException": "_v2e.InterruptProcessingException",
    "comfy.model_management.get_free_memory": "_v2e.get_free_memory",
    # ComfyUI's folders, by path (util.folders: folder_paths' names, usable anywhere)
    "folder_paths.models_dir": "util.folders.models_dir",
    "folder_paths.get_input_directory": "util.folders.get_input_directory",
    "folder_paths.get_output_directory": "util.folders.get_output_directory",
    "folder_paths.get_temp_directory": "util.folders.get_temp_directory",
    "folder_paths.get_user_directory": "util.folders.get_user_directory",
    "folder_paths.get_directory_by_type": "util.folders.get_directory_by_type",
    "folder_paths.get_folder_paths": "util.folders.get_folder_paths",
    "folder_paths.get_filename_list": "_v2e.get_filename_list",
    "folder_paths.get_full_path": "util.folders.get_full_path",
    "folder_paths.get_full_path_or_raise": "util.folders.get_full_path_or_raise",
    "folder_paths.get_annotated_filepath": "util.folders.get_annotated_filepath",
    "folder_paths.exists_annotated_filepath": "util.folders.exists_annotated_filepath",
    "folder_paths.get_save_image_path": "util.folders.get_save_image_path",
    "folder_paths.add_model_folder_path": "util.folders.add_model_folder_path",
    "server.PromptServer.instance.send_sync": "_v2e.send_sync",
    "server.PromptServer.instance.app.add_routes": "_v2e.add_routes",
    "server.PromptServer.instance.send_progress_text": "_v2e.send_progress_text",
    "server.PromptServer.instance.send": "_v2e.send",
    "server.PromptServer.instance.send_json": "_v2e.send",
    "server.web": "_v2e.web",  # aiohttp's web, which ComfyUI's server module re-exports
    # pure helpers (util: copies of ComfyUI's, tested against them)
    "comfy.utils.lanczos": "util.lanczos",
    "comfy.utils.bislerp": "util.bislerp",
    "comfy.utils.tiled_scale": "util.tiled_scale",
    "comfy.utils.tiled_scale_multidim": "util.tiled_scale_multidim",
    "comfy.utils.get_tiled_scale_steps": "util.get_tiled_scale_steps",
    "comfy.utils.load_torch_file": "util.load_torch_file",
    "comfy.utils.save_torch_file": "util.save_torch_file",
    "node_helpers.conditioning_set_values_with_timestep_range": "util.conditioning_set_values_with_timestep_range",
    # devices and memory, through eng
    "comfy.model_management.intermediate_dtype": "_v2e.intermediate_dtype",
    "comfy.model_management.unet_dtype": "_v2e.unet_dtype",
    "comfy.model_management.vae_dtype": "_v2e.vae_dtype",
    "comfy.model_management.vae_device": "_v2e.vae_device",
    "comfy.model_management.text_encoder_device": "_v2e.text_encoder_device",
    "comfy.model_management.is_device_mps": "_v2e.is_device_mps",
    "comfy.model_management.is_device_cpu": "_v2e.is_device_cpu",
    "comfy.model_management.is_device_cuda": "_v2e.is_device_cuda",
    "comfy.model_management.get_autocast_device": "_v2e.get_autocast_device",
    "comfy.model_management.unload_all_models": "_v2e.unload_all_models",
    "comfy.model_management.load_model_gpu": "_v2e.load_model_gpu",
    "comfy.model_management.load_models_gpu": "_v2e.load_models_gpu",
    "comfy.sd.load_lora_for_models": "_v2e.load_lora_for_models",
    # model loading by path, through eng.models (files in ComfyUI's model folders; elsewhere raises)
    "comfy.sd.load_checkpoint_guess_config": "_v2e.load_checkpoint_guess_config",
    "comfy.sd.load_diffusion_model": "_v2e.load_diffusion_model",
    "comfy.sd.load_unet": "_v2e.load_unet",
    "comfy.utils.PROGRESS_BAR_ENABLED": "_v2e.PROGRESS_BAR_ENABLED",
    "comfy.model_management.cleanup_models": "_v2e.cleanup_models",
    "comfy.model_management.cleanup_models_gc": "_v2e.cleanup_models",
    # ComfyUI's KSampler, as nodes.py's helper
    "nodes.common_ksampler": "_v2e.common_ksampler",
    # ComfyUI's samplers called as its own nodes call them: run as those nodes (KSampler, SamplerCustom)
    "comfy.sample.sample": "_v2e.sample",
    "comfy.sample.sample_custom": "_v2e.sample_custom",
    "comfy.sample.fix_empty_latent_channels": "_v2e.fix_empty_latent_channels",
    "comfy.sample.prepare_empty_noise": "_v2e.prepare_empty_noise",
    "latent_preview.prepare_callback": "_v2e.prepare_callback",
    "comfy_execution.graph.ExecutionBlocker": "_v2io.ExecutionBlocker",
    "comfy_execution.graph_utils.ExecutionBlocker": "_v2io.ExecutionBlocker",
    # the pack's own HTTP routes: its table, registered through ComfyExtension.get_routes
    "server.PromptServer.instance.routes": "_v2e.routes",
    "comfy.cli_args.args": "_v2e.args",
    # typing helpers: names only, no engine
    "comfy.comfy_types.node_typing.ComfyNodeABC": "object",
    "comfy.comfy_types.ComfyNodeABC": "object",
    "comfy.comfy_types.node_typing.InputTypeDict": "dict",
    "comfy.comfy_types.InputTypeDict": "dict",
    "comfy.comfy_types.node_typing.IO": "_v2e.IO",
    "comfy.comfy_types.IO": "_v2e.IO",
    # output nodes' savers, through ``ui`` (which writes wherever the node runs)
    "nodes.PreviewImage": "_v2e.PreviewImage",
    "nodes.SaveImage": "_v2e.SaveImage",
}
# what the shim's PromptServer.instance stand-in has (a file that asks it for more keeps its leftover)
PROMPT_SERVER_FACADE = {"routes", "app", "send_sync", "send", "send_json", "send_progress_text", "client_id",
                        "last_node_id", "last_prompt_id"}
# ...and how far below each a chain may go: None is "anything" (an aiohttp RouteTableDef, a plain value)
PROMPT_SERVER_CHAINS = {
    "routes": None, "client_id": None, "last_node_id": None, "last_prompt_id": None,
    "send_sync": {}, "send": {}, "send_json": {}, "send_progress_text": {},
    "app": {"add_routes": {}, "frozen": {}, "on_startup": None, "on_shutdown": None, "on_cleanup": None, "middlewares": None,
            "router": {"add_route": {}, "add_get": {}, "add_post": {}, "add_put": {}, "add_delete": {}, "add_static": {},
                       "add_routes": {}}},
}


def _facade_has(parts):
    """Whether the shim's server stand-in has the attribute chain ``parts`` (below ``PromptServer.instance``)."""
    level = PROMPT_SERVER_CHAINS
    for part in parts:
        if level is None:
            return True
        if part not in level:
            return False
        level = level[part]
    return True
# ComfyUI's own model folders, read from its folder_paths.py (a pack's others are reported to ComfyUI)
_COMFY_ROOT = os.environ.get("COMFY_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMFY_FOLDERS = sorted(set(re.findall(r'^folder_names_and_paths\["([^"]+)"\]\s*=', open(os.path.join(_COMFY_ROOT, "folder_paths.py")).read(), re.M)))
# ComfyUI's own nodes: "module.Class" -> {"id", "inputs", "function"} (core_nodes.py makes the table)
_CORE_PATH = os.environ.get("CONVERTER_CORE_NODES") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "core_nodes.json")
CORE_NODES = json.load(open(_CORE_PATH)) if os.path.exists(_CORE_PATH) else {}
# ComfyUI loads comfy_extras' files by path, so a class there reports its file as its module
# ("/…/ComfyUI/comfy_extras/nodes_mask.GrowMask"): keyed here as packs import it (comfy_extras.nodes_mask.GrowMask)
for _key in [k for k in CORE_NODES if os.sep in k or "/" in k]:
    _module_path, _, _class = _key.rpartition(".")
    _parts = _module_path.replace(os.sep, "/").split("/")
    for _root in ("comfy_extras", "comfy_api_nodes"):
        if _root in _parts:
            CORE_NODES.setdefault(".".join(_parts[len(_parts) - 1 - _parts[::-1].index(_root):] + [_class]), CORE_NODES[_key])
CORE_NODE_IDS = {entry["id"]: entry for entry in CORE_NODES.values()}
COMFY_INDEX = normalize.ComfyIndex(_COMFY_ROOT)  # what this ComfyUI's modules define, for the pack's probes of it
# a text encoder loaded from files in ComfyUI's text_encoders folder, as ComfyUI's CLIPLoader nodes load it
REWRITES["comfy.sd.load_clip"] = "_v2e.load_clip"
for _name in COMFY_INDEX.class_values("comfy.sd.CLIPType"):
    REWRITES[f"comfy.sd.CLIPType.{_name}"] = f"_v2e.CLIPType.{_name}"


def _core_node(core):
    """The shim expression that stands for one of ComfyUI's node classes."""
    return f"_v2e.CoreNode({core['id']!r}, {core['inputs']!r}, {core['function']!r}" + \
        (f", {core['params']!r})" if core.get("params") else ")")

# ComfyUI's pure helpers, copied verbatim from the running ComfyUI's source into each converted pack's
# shim: a module-level function qualifies when every global it reads is a builtin, one of these modules,
# or another qualifying function of the same file (so it can't reach ComfyUI's state or the engine).
_PURE_MODULES = {"torch", "math", "logging", "itertools", "struct", "np", "numpy", "av"}
_NOT_COPIED = {"use_more_memory", "offloaded_memory"}  # take ComfyUI's loaded-model list, which a pack process lacks
_PURE_HEADER = ("import torch, math, logging, itertools, struct\nimport numpy as np\nimport numpy\n"
                "try:\n    import av  # ComfyUI's own requirement, for its audio and video file helpers\n"
                "except ImportError:\n    av = None\n")


def _pure_functions(path):
    import builtins
    import symtable
    source = open(path, encoding="utf-8").read()
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and not node.decorator_list}
    reads = {}
    for table in symtable.symtable(source, path, "exec").get_children():
        if table.get_type() != "function" or table.get_name() not in functions:
            continue
        names, stack = set(), [table]
        while stack:
            current = stack.pop()
            for symbol in current.get_symbols():
                if symbol.is_declared_global():  # sets ComfyUI's module state: a copy would set its own
                    names.add("<global " + symbol.get_name() + ">")
                elif symbol.is_global() and symbol.is_referenced():
                    names.add(symbol.get_name())
            stack += current.get_children()
        node = functions[table.get_name()]  # defaults and annotations run where the function is defined
        for part in [*node.args.defaults, *[d for d in node.args.kw_defaults if d is not None],
                     *[a.annotation for a in node.args.args + node.args.kwonlyargs if a.annotation], node.returns]:
            if part is not None:
                names |= {n.id for n in ast.walk(part) if isinstance(n, ast.Name)}
        reads[table.get_name()] = names
    allowed = set(dir(builtins)) | _PURE_MODULES
    pure = set(reads) - _NOT_COPIED
    while True:
        bad = {name for name in pure if reads[name] - allowed - pure}
        if not bad:
            break
        pure -= bad
    return {name: ast.get_source_segment(source, functions[name]) for name in sorted(pure)}


# module -> (shim namespace, {name: source}); a missing file (an older ComfyUI) gives no copies
PURE_COPIES = {}
for _module, _namespace, _file in (("comfy.utils", "comfy_utils", "comfy/utils.py"),
                                   ("comfy.model_management", "comfy_mm", "comfy/model_management.py"),
                                   ("comfy.sd1_clip", "comfy_sd1_clip", "comfy/sd1_clip.py")):
    try:
        PURE_COPIES[_module] = (_namespace, _pure_functions(os.path.join(_COMFY_ROOT, _file)))
    except (OSError, SyntaxError):
        PURE_COPIES[_module] = (_namespace, {})
# More of ComfyUI's helpers that compute on tensors and plain data only, named one by one: a function that
# takes one of ComfyUI's objects (a model, its sampling) reads no global either, so the test above can't tell
PURE_NAMED = {
    "comfy.ldm.common_dit": ("comfy_common_dit", "comfy/ldm/common_dit.py", {"pad_to_patch_size"}),
    "comfy_execution.graph_utils": ("comfy_graph_utils", "comfy_execution/graph_utils.py", {"is_link", "add_graph_prefix"}),
    "execution": ("comfy_execution_", "comfy_execution/graph_utils.py", {"is_link"}),  # re-exported there
    "comfy.k_diffusion.sampling": ("comfy_k_sampling", "comfy/k_diffusion/sampling.py",
                                   {"append_zero", "default_noise_sampler", "get_ancestral_step", "get_sigmas_exponential",
                                    "get_sigmas_karras", "get_sigmas_laplace", "get_sigmas_polyexponential", "get_sigmas_vp"}),
    "comfy.k_diffusion.utils": ("comfy_k_utils", "comfy/k_diffusion/utils.py", {"append_dims"}),
    "comfy.model_sampling": ("comfy_model_sampling", "comfy/model_sampling.py",
                             {"flux_time_shift", "time_snr_shift", "rescale_zero_terminal_snr_sigmas"}),
    "comfy.samplers": ("comfy_samplers", "comfy/samplers.py", {"kl_optimal_scheduler"}),
    "node_helpers": ("comfy_node_helpers", "node_helpers.py", {"image_alpha_fix", "string_to_torch_dtype"}),
    "comfy.ldm.modules.diffusionmodules.util": ("comfy_dm_util", "comfy/ldm/modules/diffusionmodules/util.py",
                                                {"betas_for_alpha_bar", "extract_into_tensor", "make_beta_schedule",
                                                 "make_ddim_sampling_parameters", "make_ddim_timesteps", "mean_flat",
                                                 "noise_like", "zero_module", "scale_module"}),
    "comfy.ldm.modules.attention": ("comfy_attention", "comfy/ldm/modules/attention.py", {"default", "exists"}),
    "comfy.ldm.util": ("comfy_ldm_util", "comfy/ldm/util.py", {"exists", "mean_flat", "count_params"}),
    "comfy.lora": ("comfy_lora", "comfy/lora.py", {"pad_tensor_to_shape"}),
    "comfy_extras.nodes_audio": ("comfy_nodes_audio", "comfy_extras/nodes_audio.py", {"load", "f32_pcm"}),
}
for _module, (_namespace, _file, _names) in PURE_NAMED.items():
    try:
        _found = _pure_functions(os.path.join(_COMFY_ROOT, _file))
    except (OSError, SyntaxError):
        _found = {}
    PURE_COPIES[_module] = (_namespace, {n: src for n, src in _found.items() if n in _names})
for _module, (_namespace, _functions) in PURE_COPIES.items():
    for _name in _functions:
        REWRITES.setdefault(f"{_module}.{_name}", f"_v2e.{_namespace}.{_name}")  # existing util.* rows win
REWRITES.update({
    "comfy.model_management.vae_offload_device": "_v2e.vae_offload_device",
    "comfy.model_management.text_encoder_offload_device": "_v2e.text_encoder_offload_device",
    "comfy.model_management.OOM_EXCEPTION": "_v2e.OOM_EXCEPTION",
})


def _pure_shim():
    """The shim's namespaces of verbatim pure copies (each run in a module of its own)."""
    parts = ["\n\n# ComfyUI's pure helpers, verbatim from its source (converter: PURE_COPIES)\nimport types as _v2types\n"]
    for module, (namespace, functions) in PURE_COPIES.items():
        body = _PURE_HEADER + "\n\n".join(functions.values()) + "\n"
        parts.append(f"{namespace} = _v2types.ModuleType({module!r} + ' (V2 copy)')\n"
                     f"exec(compile({body!r}, __file__ + ':{namespace}', 'exec'), {namespace}.__dict__)\n")
    return "".join(parts)
# typed_functions may carry the types the pack's own nodes create under this key (not a class name)
OWNED_TYPES_KEY = " owned types"


def owned_types(nodes):
    """Types the pack's own nodes create: a node outputs the type without taking it as an input (a TTS loader
    returning its model as "MODEL"). Not a patch node's type (MODEL in, MODEL out: SeamlessTile): that's ComfyUI's."""
    owned = set()
    for entry in (nodes or {}).values():
        info = entry.get("info") or {}
        inputs = set()
        for section in ("required", "optional"):
            for spec in ((info.get("input") or {}).get(section) or {}).values():
                kind = spec[0] if isinstance(spec, list) and spec else spec
                if isinstance(kind, str):  # a combo's choices are a list, not a type
                    inputs.add(kind)
        owned |= {t for t in (info.get("output") or []) if isinstance(t, str) and t not in inputs}
    return frozenset(owned)
# PyTorch calls that change a process-wide numeric setting
TORCH_SETTING_CALLS = {"torch.set_float32_matmul_precision", "torch.use_deterministic_algorithms",
                       "torch.backends.cuda.matmul.allow_tf32", "torch.set_default_dtype"}
# a ComfyUI module imported in a class body (a class attribute): what the class attribute becomes in V2.1
CLASS_IMPORT_EQUIVALENTS = {"folder_paths": "util.folders"}
# engine objects: in V2.1 a pack gets an opaque handle (Ref) for each, so an attribute that isn't a V2.1 call fails
# when the node runs (WhiteRabbit's ``upscale_model.to(device)``). verify.MODEL_KINDS lists the same types.
ENGINE_KINDS = {"MODEL", "CLIP", "VAE", "CONTROL_NET", "CLIP_VISION", "CLIP_VISION_OUTPUT", "UPSCALE_MODEL",
                "STYLE_MODEL", "GLIGEN", "PHOTOMAKER", "HOOKS"}
# methods on engine objects -> eng rows (receiver first); "assign" rows return a copy the V1 call mutated
METHODS = {
    "CLIP": {"tokenize": "clip.tokenize", "encode_from_tokens": "clip.encode_from_tokens",
             "encode_from_tokens_scheduled": "clip.encode_tokens", "clone": "clip.clone"},
    "VAE": {"encode": "vae.encode", "decode": "vae.decode", "decode_tiled": "vae.decode_tiled",
            "encode_tiled": "vae.encode_tiled"},
    "MODEL": {"clone": "model.clone"},
}
# comfy.patcher_extension.WrappersMP members -> eng.model.add_wrapper kinds
WRAPPER_KINDS = {"OUTER_SAMPLE": "outer_sample", "PREPARE_SAMPLING": "prepare_sampling", "SAMPLER_SAMPLE": "sampler_sample",
                 "PREDICT_NOISE": "predict_noise", "CALC_COND_BATCH": "calc_cond_batch", "APPLY_MODEL": "apply_model",
                 "DIFFUSION_MODEL": "diffusion_model"}
# model_options keys that hold functions: set through their own eng rows (set_cfg, add_wrapper...), not set_option
FUNCTION_OPTIONS = {"patches", "patches_replace", "wrappers", "callbacks", "sampler_cfg_function", "sampler_post_cfg_function",
                    "sampler_pre_cfg_function", "model_function_wrapper", "denoise_mask_function", "sampler_calc_cond_batch_function"}
# what the shim's stand-in for a model's sampling object has (``model.get_model_object("model_sampling")``)
FACADES = {"MODEL_SAMPLING": {"percent_to_sigma", "timestep", "sigma", "sigma_min", "sigma_max"}}
MODEL_SETTERS = {
    "set_model_sampler_cfg_function": "model.set_cfg", "set_model_sampler_post_cfg_function": "model.set_post_cfg",
    "set_model_sampler_pre_cfg_function": "model.set_pre_cfg", "set_model_unet_function_wrapper": "model.set_unet_wrapper",
    "set_model_denoise_mask_function": "model.set_denoise_mask", "set_model_attn1_patch": "model.set_attn1_patch",
    "set_model_attn2_patch": "model.set_attn2_patch", "set_model_attn1_output_patch": "model.set_attn1_output_patch",
    "set_model_attn2_output_patch": "model.set_attn2_output_patch", "set_model_input_block_patch": "model.set_input_block_patch",
    "set_model_input_block_patch_after_skip": "model.set_input_block_patch_after_skip",
    "set_model_output_block_patch": "model.set_output_block_patch",
}
MODEL_SETTER_PARAMETERS = {
    name: (({"set_model_sampler_cfg_function": "sampler_cfg_function",
             "set_model_sampler_post_cfg_function": "post_cfg_function",
             "set_model_sampler_pre_cfg_function": "pre_cfg_function"}[name], "disable_cfg1_optimization")
           if name.startswith("set_model_sampler_") else
           (("unet_wrapper_function",) if name == "set_model_unet_function_wrapper" else
            ("denoise_mask_function",) if name == "set_model_denoise_mask_function" else ("patch",)))
    for name in MODEL_SETTERS
}
HIDDEN = {"PROMPT": "prompt", "EXTRA_PNGINFO": "extra_pnginfo", "UNIQUE_ID": "unique_id", "DYNPROMPT": "dynprompt",
          "AUTH_TOKEN_COMFY_ORG": "auth_token_comfy_org", "API_KEY_COMFY_ORG": "api_key_comfy_org"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", "v2", ".github", "tests", "test"}


# --------------------------------------------------------------------------- #
# Rewriting engine calls in the pack's own code
# --------------------------------------------------------------------------- #
def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def _module_level(body):
    """A module's statements, with those inside module-level ``try``/``if`` blocks (``try: from comfy...
    import IO`` / ``except ImportError: class IO: ...``) as they also bind module names."""
    for node in body:
        yield node
        if isinstance(node, ast.Try) or (hasattr(ast, "TryStar") and isinstance(node, ast.TryStar)):
            yield from _module_level(node.body + node.orelse + node.finalbody + [s for h in node.handlers for s in h.body])
        elif isinstance(node, ast.If):
            yield from _module_level(node.body + node.orelse)


def _prune_all(source, path, dropped):
    """``__all__`` naming an engine import the converter removed: the name goes from ``__all__`` too
    (``from it import *`` would fail on it; importers get the rewritten name another way)."""
    try:
        tree = ast.parse(source, path)
    except SyntaxError:
        return source
    bound = set()
    for node in _module_level(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound |= {a.asname or a.name.split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                bound |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
    lines = source.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets) \
                and isinstance(node.value, (ast.List, ast.Tuple)) \
                and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.value.elts):
            names = [e.value for e in node.value.elts]
            kept = [n for n in names if not (n in dropped and n not in bound)]
            if kept != names and lines[node.lineno - 1][:node.col_offset].strip() == "" \
                    and lines[node.end_lineno - 1][node.end_col_offset:].strip() == "":
                for index in range(node.lineno - 1, node.end_lineno):
                    lines[index] = ""
                lines[node.lineno - 1] = "__all__ = [\n" + "".join(f"    {n!r},\n" for n in kept) + "]\n"
    candidate = "".join(lines)
    try:
        compile(candidate, path, "exec")
        return candidate
    except SyntaxError:
        return source


def module_engine_aliases(source, path, include_private=False):
    """A module's module-level engine names, as ``from it import *`` hands them on: {name: dotted}."""
    try:
        tree = ast.parse(source, path)
    except SyntaxError:
        return {}
    found = {}
    for node in _module_level(tree.body):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in ENGINE_ROOTS:
                    found[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module \
                and node.module.split(".")[0] in ENGINE_ROOTS:
            for alias in node.names:
                if alias.name != "*":
                    found[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return {k: v for k, v in found.items() if include_private or not k.startswith("_")}


def star_imports(source, path):
    """``from X import *`` in a module: [(level, X)]."""
    try:
        tree = ast.parse(source, path)
    except SyntaxError:
        return []
    return [(n.level, n.module or "") for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names)]


_PRELOADED = {"PIL"}  # packages whose submodules ComfyUI has imported before a pack loads (PIL.Image...)


def _explicit_submodules(source, path):
    """``import PIL`` then ``PIL.Image.open(...)`` works in ComfyUI only because ComfyUI imported PIL.Image
    first; a pack process may not have, so the submodules used get imported beside ``import PIL``."""
    try:
        tree = ast.parse(source, path)
    except SyntaxError:
        return source
    plain = [n for n in tree.body if isinstance(n, ast.Import) and any(a.name in _PRELOADED and a.asname is None for a in n.names)]
    if not plain:
        return source
    have = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    have |= {f"{n.module}.{a.name}" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module for a in n.names}
    have |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    used = sorted({f"{n.value.id}.{n.attr}" for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                   and isinstance(n.value, ast.Name) and n.value.id in _PRELOADED and n.attr[:1].isupper()} - have)
    if not used:
        return source
    lines = source.splitlines(keepends=True)
    first = plain[0]
    if first.end_lineno != first.lineno:
        return source
    line = lines[first.lineno - 1]
    lines[first.lineno - 1] = line.rstrip("\r\n") + "\n" + "".join(f"import {u}\n" for u in used)
    candidate = "".join(lines)
    try:
        compile(candidate, path, "exec")
        return candidate
    except SyntaxError:
        return source


def named_imports(source, path):
    """``from X import a, b`` in a module (not ``*``): [(level, X, [a, b])]; X may be one of the pack's modules."""
    try:
        tree = ast.parse(source, path)
    except SyntaxError:
        return []
    return [(n.level, n.module or "", [a.name for a in n.names if a.asname is None])
            for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and not any(a.name == "*" for a in n.names)
            and not (n.level == 0 and n.module and n.module.split(".")[0] in ENGINE_ROOTS)]


def _drop_pack_imports(source, path, unused):
    """Take ``unused`` names out of ``from <pack module> import ...`` statements (the engine name they
    handed on was rewritten away); a statement left empty goes, or becomes ``pass``."""
    tree = ast.parse(source, path)
    lines = source.splitlines(keepends=True)
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and any(a.name in unused and a.asname is None for a in n.names)
             and not (n.level == 0 and n.module and n.module.split(".")[0] in ENGINE_ROOTS)]
    for node in sorted(nodes, key=lambda n: -n.lineno):
        if lines[node.end_lineno - 1][node.end_col_offset:].strip() not in ("", "\\") \
                or lines[node.lineno - 1][:node.col_offset].strip() != "":
            continue  # shares its line with other code: leave it
        kept = [a for a in node.names if not (a.name in unused and a.asname is None)]
        for index in range(node.lineno - 1, node.end_lineno):
            lines[index] = ""
        if kept:
            names = ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in kept)
            lines[node.lineno - 1] = " " * node.col_offset + f"from {'.' * node.level}{node.module or ''} import {names}\n"
        elif _needs_pass(tree, node):
            lines[node.lineno - 1] = " " * node.col_offset + "pass  # import removed by the V2 converter\n"
    candidate = "".join(lines)
    try:
        compile(candidate, path, "exec")
        return candidate
    except SyntaxError:
        return source


class _Module:
    def __init__(self, source, path):
        self.source, self.path = source, path
        self.tree = ast.parse(source, path)
        self.lines = source.splitlines(keepends=True)
        # Python's source extractor treats these as characters, unlike
        # splitlines. Keep its existing sort order for those uncommon sources.
        self._simple_source_lines = not any(c in source for c in "\v\f\x1c\x1d\x1e\x85\u2028\u2029")
        self.offsets = [0]
        for line in self.lines:
            self.offsets.append(self.offsets[-1] + len(line))
        self.aliases = {}  # local name -> dotted engine name
        self.imports = []  # (node, engine?) for engine imports
        # A name ComfyUI doesn't have isn't ComfyUI's: ``from nodes.batch_filename import BatchFilename`` is the
        # pack's own ``nodes`` package, imported by its top-level name in a script or behind a fallback.
        def comfys(dotted):
            return COMFY_INDEX.kind(dotted) != normalize.ABSENT

        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in ENGINE_ROOTS and comfys(alias.name):
                        local = alias.asname or alias.name.split(".")[0]
                        self.aliases[local] = alias.name if alias.asname else alias.name.split(".")[0]
                        self.imports.append(node)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module \
                    and node.module.split(".")[0] in ENGINE_ROOTS and comfys(node.module):
                names = [a for a in node.names if a.name == "*" or comfys(f"{node.module}.{a.name}")]
                for alias in names:
                    self.aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
                if names:
                    self.imports.append(node)
        self.shadowed = _shadowed_loads(self.tree)  # ids of Name loads bound by an enclosing function

    def pos(self, line, col):
        return self.offsets[line - 1] + len(self.lines[line - 1].encode("utf-8")[:col].decode("utf-8", "ignore"))

    def span(self, node):
        return self.pos(node.lineno, node.col_offset), self.pos(node.end_lineno, node.end_col_offset)

    def length(self, node):
        if not self._simple_source_lines:
            return len(ast.get_source_segment(self.source, node) or "")
        start, end = self.span(node)
        return end - start

    def resolve(self, node):
        parts = _dotted(node)
        if not parts or parts[0] not in self.aliases:
            return None
        root = node
        while isinstance(root, ast.Attribute):
            root = root.value
        if id(root) in self.shadowed:  # ``def cb(args): args["cond"]`` is the parameter, not cli_args.args
            return None
        return ".".join([self.aliases[parts[0]]] + parts[1:])


CHANGERS = {"add", "discard", "remove", "append", "extend", "insert", "update", "clear", "pop", "setdefault"}

# Attributes only ComfyUI's ModelPatcher / CLIP / VAE have (read from comfy/model_patcher.py and comfy/sd.py),
# less names a torch module, a diffusers or HF object also has (to, eval, encode, decode, device, size, model...).
# Used on helper functions' parameters named like an engine object, which aren't typed node inputs.
ENGINE_ONLY_ATTRS = {
    "MODEL": {"add_callback", "add_callback_with_key", "add_hook_patches", "add_object_patch", "add_patches",
              "add_weight_wrapper", "add_wrapper", "add_wrapper_with_key", "additional_models", "apply_hooks",
              "calculate_weight", "clone", "clone_has_same_weights", "current_loaded_device", "get_additional_models",
              "get_attachment", "get_key_patches", "get_model_object", "get_wrappers", "inject_model", "load_device",
              "model_options", "model_patches_to", "model_size", "model_state_dict", "object_patches",
              "offload_device", "partially_load", "partially_unload", "patch_model", "patch_weight_to_device",
              "set_model_attn1_output_patch", "set_model_attn1_patch", "set_model_attn1_replace",
              "set_model_attn2_output_patch", "set_model_attn2_patch", "set_model_attn2_replace",
              "set_model_compute_dtype", "set_model_denoise_mask_function", "set_model_double_block_patch",
              "set_model_emb_patch", "set_model_forward_timestep_embed_patch", "set_model_input_block_patch",
              "set_model_input_block_patch_after_skip", "set_model_middle_block_after_patch",
              "set_model_noise_refiner_patch", "set_model_optimized_attention", "set_model_output_block_patch",
              "set_model_patch", "set_model_patch_replace", "set_model_post_input_patch", "set_model_rope_options",
              "set_model_sampler_calc_cond_batch_function", "set_model_sampler_cfg_function",
              "set_model_sampler_post_cfg_function", "set_model_sampler_pre_cfg_function",
              "set_model_unet_function_wrapper", "unpatch_model", "weight_inplace_update"},
    "CLIP": {"add_hooks_to_dict", "add_patches", "apply_hooks_to_conds", "clip_layer", "clone", "cond_stage_model",
             "encode_from_tokens", "encode_from_tokens_scheduled", "get_key_patches", "get_sd", "layer_idx", "load_sd",
             "patcher", "set_tokenizer_option", "tokenize", "tokenizer_options", "use_clip_schedule"},
    "VAE": {"decode_tiled", "decode_tiled_", "decode_tiled_1d", "decode_tiled_3d", "downscale_index_formula",
            "downscale_ratio", "encode_tiled", "encode_tiled_", "encode_tiled_1d", "encode_tiled_3d",
            "first_stage_model", "get_sd", "latent_channels", "memory_used_decode", "memory_used_encode",
            "output_device", "patcher", "process_input", "process_output", "spacial_compression_decode",
            "spacial_compression_encode", "temporal_compression_decode", "upscale_index_formula", "upscale_ratio",
            "vae_dtype"},
}


def _guess_kind(name):
    """MODEL / CLIP / VAE for a parameter named like one (``model``, ``base_model``, ``clip``, ``vae_in``)."""
    words = name.lower().split("_")
    for kind, word in (("CLIP", "clip"), ("VAE", "vae"), ("MODEL", "model"), ("MODEL", "patcher")):
        if word in words:
            return kind
    return None
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _local_names(scope):
    """Names a function (or comprehension) binds itself: parameters, assignment/loop/with targets, except
    names, nested defs and classes, imports of non-engine modules; less its ``global``/``nonlocal`` names.
    An engine import inside a function isn't counted: the converter rewrites its uses like a module's."""
    if isinstance(scope, _COMPREHENSIONS):
        return {n.id for g in scope.generators for n in ast.walk(g.target) if isinstance(n, ast.Name)}
    a = scope.args
    names = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg] if x is not None}
    declared = set()
    todo = list(scope.body) if isinstance(scope.body, list) else [scope.body]
    while todo:
        n = todo.pop()
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            declared.update(n.names)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)  # its body is its own scope
        elif isinstance(n, (ast.Lambda,) + _COMPREHENSIONS):
            pass
        elif isinstance(n, ast.Import):
            names |= {(x.asname or x.name).split(".")[0] for x in n.names if x.name.split(".")[0] not in ENGINE_ROOTS}
        elif isinstance(n, ast.ImportFrom):
            if not (n.level == 0 and n.module and n.module.split(".")[0] in ENGINE_ROOTS):
                names |= {x.asname or x.name for x in n.names}
        else:
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                names.add(n.id)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                names.add(n.name)
            todo.extend(ast.iter_child_nodes(n))
    return names - declared


def _shadowed_loads(tree):
    """ids of the Name loads that refer to a function's own binding (or an enclosing function's), not to
    the module's name of the same spelling. Decorators and defaults belong to the enclosing scope; argument
    annotations aren't visited (they're handled as annotations)."""
    shadowed = set()
    todo = [(tree, ())]
    while todo:
        node, scopes = todo.pop()
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and any(node.id in s for s in scopes):
                shadowed.add(id(node))
        elif isinstance(node, _FUNCTIONS):
            outside = list(getattr(node, "decorator_list", [])) + node.args.defaults + \
                      [d for d in node.args.kw_defaults if d is not None]
            todo += [(part, scopes) for part in outside]
            inner = scopes + (_local_names(node),)
            todo += [(part, inner) for part in (node.body if isinstance(node.body, list) else [node.body])]
        elif isinstance(node, _COMPREHENSIONS):
            inner = scopes + (_local_names(node),)
            todo += [(child, inner) for child in ast.iter_child_nodes(node)]
        else:
            todo += [(child, scopes) for child in ast.iter_child_nodes(node)]
    return shadowed


# keys of the dicts ComfyUI hands sampling functions that hold engine objects, not tensors or plain data
ENGINE_OBJECT_KEYS = ("model", "inner_model")
# keys only ComfyUI's sampling-function args dicts have: a function reading one of them is a sampling hook
SAMPLING_KEYS = {"denoised", "cond_denoised", "uncond_denoised", "sigma", "sigmas", "cond_scale", "model_options",
                 "timestep", "input_cond", "input_uncond",
                 # attention / block patches' extra_options (DepthDiff reads extra_options["model"], Danny r4)
                 "cond_or_uncond", "transformer_index", "original_shape", "activations_shape", "n_heads", "dim_head"}


def _dict_keys_read(function, param):
    """The constant keys ``function`` reads from its parameter ``param`` (``param["k"]`` / ``param.get("k")``)."""
    keys = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == param \
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            keys.add(node.slice.value)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == param and node.args \
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            keys.add(node.args[0].value)
    return keys


def _call_method(tree, function, name):
    """The ``__call__`` of the module's own class whose instance ``name`` is assigned in ``function``."""
    classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets) \
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) \
                and node.value.func.id in classes:
            for item in classes[node.value.func.id].body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__call__":
                    return item
    return None


def _engine_object_reads(hook, param):
    """``param["model"]`` / ``param.get("model")`` reads in a sampling function (its args dict)."""
    found = []
    for node in ast.walk(hook):
        key = None
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == param \
                and isinstance(node.slice, ast.Constant):
            key = node.slice.value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == param and node.args \
                and isinstance(node.args[0], ast.Constant):
            key = node.args[0].value
        if key in ENGINE_OBJECT_KEYS and key not in found:
            found.append(key)
    return found


def _direct_body_ids(function):
    """ids of the nodes in ``function``'s own body: not in a function, lambda or class nested in it."""
    found, stack = set(), list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        found.add(id(node))
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(node))
    return found


def _apply_within(source, start, end, edits):
    """``source[start:end]`` with the edits that lie inside it applied."""
    out, last = [], start
    for s, e, t in sorted(x for x in edits if start <= x[0] and x[1] <= end):
        if s < last:
            continue
        out.append(source[last:s])
        out.append(t)
        last = e
    out.append(source[last:end])
    return "".join(out)


# typed_functions may carry, under this key, types that reach this module's functions from the pack's other
# modules: {"function": {param: TYPE}, "Class.method": {param: TYPE}} (see _pack_types)
HELPER_TYPES_KEY = " helper types"


def _method(cls, name, classes, depth=0):
    """The class's method ``name``: its own, or the one it inherits from a base class of the same module."""
    for item in cls.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
            return item
    for base in cls.bases if depth < 4 else ():
        if isinstance(base, ast.Name) and base.id in classes and classes[base.id] is not cls:
            found = _method(classes[base.id], name, classes, depth + 1)
            if found is not None:
                return found
    return None


def _typed_params(tree, typed_functions):
    """``{function node: {param: TYPE}}`` for node methods whose params have engine-object types."""
    found, conflicts = {}, set()
    classes = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.setdefault(node.name, node)

    def add(function, types):
        accepted = {x.arg for x in function.args.posonlyargs + function.args.args + function.args.kwonlyargs}
        for param, kind in types.items():
            if param not in accepted or not (kind in METHODS or kind in ENGINE_KINDS):
                continue
            if found.setdefault(function, {}).setdefault(param, kind) != kind:
                conflicts.add((id(function), param))  # two nodes share the method with different types

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in typed_functions and node.name != HELPER_TYPES_KEY:
            fname, types = typed_functions[node.name]
            method = _method(node, fname, classes) if fname else None
            if method is not None:
                add(method, types)
    handed = typed_functions.get(HELPER_TYPES_KEY, (None, {}))[1] if isinstance(typed_functions, dict) else {}
    module_level = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name, types in handed.items():
        owner, _, method = name.rpartition(".")
        function = _method(classes[owner], method, classes) if owner in classes else module_level.get(method) if not owner else None
        if function is not None:
            add(function, types)
    found = {f: {p: t for p, t in params.items() if (id(f), p) not in conflicts} for f, params in found.items()}
    return _through_helpers(tree, {f: p for f, p in found.items() if p})


def _pack_types(trees, typed, target):
    """Engine-object types followed across the pack's modules: a node method that hands its ``clip`` to
    ``encode(clip, text)`` imported from another file makes that function's ``clip`` a CLIP; a node class
    whose method is its imported base class's types that method. ``{rel: {name: {param: TYPE}}}``."""
    functions = (ast.FunctionDef, ast.AsyncFunctionDef)
    handed = {rel: {} for rel in trees}
    module_level = {rel: {n.name: n for n in tree.body if isinstance(n, functions)} for rel, tree in trees.items()}
    class_names = {rel: {n.name for n in tree.body if isinstance(n, ast.ClassDef)} for rel, tree in trees.items()}
    imports = {}
    for rel, tree in trees.items():
        table = {}  # local name -> (module file, name in it; None: the module itself)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    as_module = target(rel, node.level, ".".join(x for x in (node.module, alias.name) if x))
                    if as_module:
                        table[alias.asname or alias.name] = (as_module, None)
                    elif node.module or node.level:
                        source = target(rel, node.level, node.module or "")
                        if source and source != rel:
                            table[alias.asname or alias.name] = (source, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    source = target(rel, 0, alias.name)
                    if source and (alias.asname or "." not in alias.name):
                        table[alias.asname or alias.name] = (source, None)
        imports[rel] = table

    def give(rel, name, param, kind):
        if handed[rel].setdefault(name, {}).get(param) is None:
            handed[rel][name][param] = kind
            return True
        return False

    for _round in range(4):
        grew = False
        for rel, tree in trees.items():
            table = imports[rel]
            if not table:
                continue
            found = _typed_params(tree, {**typed, HELPER_TYPES_KEY: (None, handed[rel])})
            for function, params in found.items():
                local = dict(params)
                for node in ast.walk(function):  # ``m = model.clone()``: the same type
                    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                            and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) \
                            and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id in local \
                            and node.value.func.attr == "clone":
                        local.setdefault(node.targets[0].id, local[node.value.func.value.id])
                for call in ast.walk(function):
                    if not isinstance(call, ast.Call) or any(isinstance(a, ast.Starred) for a in call.args):
                        continue
                    f, dest = call.func, None
                    if isinstance(f, ast.Name) and table.get(f.id, (None, None))[1]:
                        dest = table[f.id]
                    elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in table \
                            and table[f.value.id][1] is None:
                        dest = (table[f.value.id][0], f.attr)
                    callee = module_level.get(dest[0], {}).get(dest[1]) if dest else None
                    if callee is None:
                        continue
                    positional = [x.arg for x in callee.args.posonlyargs + callee.args.args]
                    named = set(positional) | {x.arg for x in callee.args.kwonlyargs}
                    taken = {n.id for n in ast.walk(callee) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
                    passed = list(zip(positional, call.args)) + [(k.arg, k.value) for k in call.keywords if k.arg in named]
                    for param, value in passed:
                        kind = local.get(value.id) if isinstance(value, ast.Name) else None
                        if kind and param not in taken:
                            grew = give(dest[0], dest[1], param, kind) or grew
            # a node class whose method is its imported base class's
            classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
            for cls in classes.values():
                if cls.name not in typed or cls.name == HELPER_TYPES_KEY or not typed[cls.name][0]:
                    continue
                fname, types = typed[cls.name]
                if _method(cls, fname, classes) is not None:
                    continue
                for base in cls.bases:
                    source, name = table.get(base.id, (None, None)) if isinstance(base, ast.Name) else (None, None)
                    if source and name and name in class_names.get(source, ()):
                        for param, kind in types.items():
                            if kind in METHODS or kind in ENGINE_KINDS:
                                grew = give(source, f"{name}.{fname}", param, kind) or grew
        if not grew:
            break
    return {rel: names for rel, names in handed.items() if names}


def _through_helpers(tree, found):
    """The types followed into the module's own helpers: ``encode(self, clip, text)`` calling
    ``self.encode_text(clip, text)`` makes ``encode_text``'s ``clip`` a CLIP. Only calls this module defines
    (``self.f``/``cls.f`` in the same class, ``f`` at module level), only arguments that are a typed name, and
    never a parameter the helper assigns to (then its type isn't certain)."""
    if not found:
        return {}
    functions = (ast.FunctionDef, ast.AsyncFunctionDef)
    module_level = {n.name: n for n in tree.body if isinstance(n, functions)}
    owner, methods = {}, {}
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef):
            methods[id(cls)] = {n.name: n for n in cls.body if isinstance(n, functions)}
            for item in cls.body:
                if isinstance(item, functions):
                    owner[id(item)] = cls

    def assigned(function):
        """Names the function assigns to, less ``model = model.clone()`` (a clone has the same type)."""
        clones = {id(n.targets[0]) for n in ast.walk(function)
                  if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                  and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                  and n.value.func.attr == "clone" and not n.value.args
                  and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == n.targets[0].id}
        return {n.id for n in ast.walk(function) if isinstance(n, ast.Name)
                and isinstance(n.ctx, (ast.Store, ast.Del)) and id(n) not in clones}

    def params(function, bound):
        a = function.args
        names = [x.arg for x in a.posonlyargs + a.args]
        return (names[1:] if bound else names), {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}

    # Class-qualified calls are proved only for plain, unmodified local namespaces.
    # Do not infer a type from a parameter spelling or an annotation.
    shadowed = _shadowed_loads(tree)
    static_helpers = {}
    parents = {id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    dynamic = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id in {"exec", "eval", "globals", "locals", "setattr", "delattr"}
                  for n in ast.walk(tree))
    if not dynamic and not _binds_name(tree, "staticmethod"):
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef) or cls.bases or cls.keywords or cls.decorator_list:
                continue
            if not any(isinstance(m, functions) and len(m.decorator_list) == 1
                       and isinstance(m.decorator_list[0], ast.Name) and m.decorator_list[0].id == "staticmethod"
                       for m in cls.body):
                continue
            if any(n is not cls and isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == cls.name for n in ast.walk(tree)):
                continue
            if any((isinstance(n, ast.Name) and n.id == cls.name and isinstance(n.ctx, (ast.Store, ast.Del)))
                   or (isinstance(n, ast.arg) and n.arg == cls.name)
                   or (isinstance(n, (ast.Import, ast.ImportFrom)) and any((a.asname or a.name.split(".")[0]) == cls.name for a in n.names))
                   or (isinstance(n, ast.Attribute) and isinstance(n.ctx, (ast.Store, ast.Del))
                       and isinstance(n.value, ast.Name) and n.value.id == cls.name)
                   for n in ast.walk(tree)):
                continue
            if any(isinstance(n, ast.Name) and n.id == cls.name and isinstance(n.ctx, ast.Load)
                   and not (isinstance(parents.get(id(n)), ast.Attribute) and parents[id(n)].value is n)
                   for n in ast.walk(tree)):
                continue
            for method in cls.body:
                if not isinstance(method, functions) or len(method.decorator_list) != 1:
                    continue
                dec = method.decorator_list[0]
                if not isinstance(dec, ast.Name) or dec.id != "staticmethod":
                    continue
                if sum(getattr(n, "name", None) == method.name or
                       isinstance(n, (ast.Assign, ast.AnnAssign)) and any(isinstance(t, ast.Name) and t.id == method.name
                       for t in (n.targets if isinstance(n, ast.Assign) else [n.target])) for n in cls.body) != 1:
                    continue
                static_helpers[(cls.name, method.name)] = method

    def static_callee(call):
        f = call.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and id(f.value) not in shadowed:
            return static_helpers.get((f.value.id, f.attr))
        return None

    # Every use must be a direct call in a known typed function; escaping a helper
    # as a callback or invoking it with an untyped argument remains a manual case.
    call_owner = {}
    for fn in (n for n in ast.walk(tree) if isinstance(n, functions)):
        for n in ast.walk(fn):
            if isinstance(n, ast.Call):
                call_owner[id(n)] = fn
    found = dict(found)
    for _round in range(4):  # helpers of helpers, a few levels deep
        grew = False
        for function, typed in list(found.items()):
            local = dict(typed)
            for node in ast.walk(function):
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) \
                        and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id in local \
                        and node.value.func.attr == "clone":
                    local[node.targets[0].id] = local[node.value.func.value.id]
            for call in ast.walk(function):
                if not isinstance(call, ast.Call):
                    continue
                callee, bound = None, False
                qualified_static = False
                if isinstance(call.func, ast.Name):
                    callee = module_level.get(call.func.id)
                elif isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) \
                        and call.func.value.id in ("self", "cls") and id(function) in owner:
                    callee, bound = methods[id(owner[id(function)])].get(call.func.attr), True
                    if callee is not None and any(isinstance(d, ast.Name) and d.id == "staticmethod"
                                                  for d in callee.decorator_list):
                        bound = False
                if callee is None:
                    candidate = static_callee(call)
                    if candidate is not None:
                        uses = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                                and isinstance(n.value, ast.Name)
                                and static_helpers.get((n.value.id, n.attr)) is candidate]
                        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.func in uses]
                        positional, named = params(candidate, False)
                        # Reject escapes, *args/**kwargs, unknown callers, rebinding and
                        # nested parameter shadowing before adding any new type facts.
                        safe = len(uses) == len(calls) and all(
                            id(c) in call_owner and call_owner[id(c)] in found
                            and not any(isinstance(a, ast.Starred) for a in c.args)
                            and not any(k.arg is None for k in c.keywords)
                            for c in calls)
                        if safe:
                            callee = candidate
                            qualified_static = True
                if callee is None or callee is function or any(isinstance(a, ast.Starred) for a in call.args):
                    continue
                positional, named = params(callee, bound)
                passed = list(zip(positional, call.args)) + [(k.arg, k.value) for k in call.keywords if k.arg in named]
                taken = assigned(callee)
                for param, value in passed:
                    if isinstance(value, ast.Name) and value.id in local and param not in taken:
                        if qualified_static:
                            kind = local[value.id]
                            valid = True
                            for invocation in (n for n in ast.walk(tree) if isinstance(n, ast.Call) and static_callee(n) is callee):
                                caller = call_owner[id(invocation)]
                                args = dict(zip(positional, invocation.args))
                                args.update({k.arg: k.value for k in invocation.keywords})
                                actual = args.get(param)
                                if not isinstance(actual, ast.Name) or found[caller].get(actual.id) != kind or actual.id in assigned(caller):
                                    valid = False
                            if any(_binds_name(n, param) for n in ast.walk(callee)
                                   if n is not callee and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef))):
                                valid = False
                            if not valid:
                                continue
                        if found.setdefault(callee, {}).get(param) is None:
                            found[callee][param] = local[value.id]
                            grew = True
        if not grew:
            break
    return found


def _binds_name(tree, name):
    """Whether this module binds ``name`` itself, at any scope."""
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            if any((a.asname or a.name).split(".")[0] == name for a in n.names):
                return True
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)) and n.id == name:
            return True
        elif isinstance(n, ast.arg) and n.arg == name:
            return True
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name:
            return True
        elif isinstance(n, ast.ExceptHandler) and n.name == name:
            return True
    return False


def _safe_optional_alias(tree, import_node, name, guard_only=False):
    """Recognize a single optional import binding with an except-None fallback.

    Use the generated engine facade for the probe, and refuse
    ambiguous rebinding or any use that treats the alias as a module.
    """
    parents = {id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    loads = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)]
    if not loads or (guard_only and any(not (isinstance(parents.get(id(n)), ast.Compare)
        and len(parents[id(n)].ops) == 1 and isinstance(parents[id(n)].ops[0], (ast.Is, ast.IsNot))
        and any(isinstance(x, ast.Constant) and x.value is None for x in
                [parents[id(n)].left, *parents[id(n)].comparators])) for n in loads)):
        return False
    parent = parents.get(id(import_node))
    if not isinstance(parent, ast.Try) or import_node not in parent.body:
        return False
    def scope(n):
        while id(n) in parents:
            n = parents[id(n)]
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.Module)):
                return n
        return None
    if any(scope(n) is not scope(import_node) for n in loads):
        return False
    fallbacks = []
    for n in ast.walk(tree):
        if isinstance(n, ast.arg) and n.arg == name:
            return False
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name:
            return False
        if isinstance(n, (ast.Import, ast.ImportFrom)) and n is not import_node and any(
                (a.asname or a.name.split('.')[0]) == name for a in n.names):
            return False
        if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, (ast.Store, ast.Del)):
            assign = parents.get(id(n))
            handler = parents.get(id(assign))
            if not (isinstance(assign, ast.Assign) and isinstance(assign.value, ast.Constant)
                    and assign.value.value is None and handler in parent.handlers
                    and len(assign.targets) == 1 and assign.targets[0] is n):
                return False
            fallbacks.append(n)
    return bool(fallbacks)


def _node_class_only_run(node, parents):
    """Whether a node class looked up in ComfyUI's table is only instantiated and run: the shim's stand-in
    runs the node, it can't answer ``INPUT_TYPES()`` or ``define_schema()`` or be subclassed."""
    def run_use(use):
        owner = parents.get(id(use))
        if isinstance(owner, ast.Call) and owner.func is use:
            return run_use(owner) or True  # cls(): the instance; then its method
        if isinstance(owner, ast.Attribute) and owner.value is use:
            return owner.attr == "FUNCTION" or (owner.attr[:1].islower() and not owner.attr.startswith("define_")
                                                and owner.attr not in ("mro",))
        if isinstance(owner, ast.Compare) and len(owner.ops) == 1 and isinstance(owner.ops[0], (ast.Is, ast.IsNot)):
            return True
        if isinstance(owner, (ast.If, ast.While, ast.IfExp)) and owner.test is use:
            return True
        if isinstance(owner, ast.UnaryOp) and isinstance(owner.op, ast.Not):
            return True
        if isinstance(owner, ast.Call) and isinstance(owner.func, ast.Name) and owner.func.id == "getattr" \
                and owner.args and owner.args[0] is use:
            return True  # getattr(obj, cls.FUNCTION)
        return None

    owner = parents.get(id(node))
    if isinstance(owner, ast.Assign) and len(owner.targets) == 1 and isinstance(owner.targets[0], ast.Name):
        name, scope = owner.targets[0].id, owner
        while id(scope) in parents and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            scope = parents[id(scope)]
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False  # a module-level name: other code may read the class
        reads = [n for n in ast.walk(scope) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)]
        writes = [n for n in ast.walk(scope) if isinstance(n, ast.Name) and n.id == name and not isinstance(n.ctx, ast.Load)]
        # an instance made from it (``obj = cls()``) is followed one step, by the same rule
        return len(writes) == 1 and bool(reads) and all(run_use(r) for r in reads)
    return bool(run_use(node))


def _chain(node, parents):
    """The whole attribute chain an engine-object attribute starts, as the worklist names it:
    ``model.model.model_sampling.percent_to_sigma(x)`` is ``model.model_sampling.percent_to_sigma()``."""
    parts = [node.attr]
    while True:
        owner = parents.get(id(node))
        if isinstance(owner, ast.Attribute) and owner.value is node:
            parts.append(owner.attr)
            node = owner
        elif isinstance(owner, ast.Call) and owner.func is node:
            key = owner.args[0].value if len(owner.args) == 1 and isinstance(owner.args[0], ast.Constant) \
                and isinstance(owner.args[0].value, str) and parts[-1] in ("get_model_object", "get_attachment") else None
            parts[-1] += f"({key!r})" if key is not None else "()"
            node = owner
            if key is None:
                break
        elif isinstance(owner, ast.Subscript) and owner.value is node and isinstance(owner.slice, ast.Constant) \
                and isinstance(owner.slice.value, str):
            parts[-1] += f"[{owner.slice.value!r}]"
            node = owner
        else:
            break
    return ".".join(parts)


def _calls_model_only(function):
    """Whether a sampling loop uses its model (its first parameter) only by calling it: eng.sampler.custom
    hands the loop a function, so ``model.inner_model`` or passing the model on wouldn't work."""
    a = function.args
    params = a.posonlyargs + a.args
    if params and params[0].arg in ("self", "cls"):
        params = params[1:]
    if not params or a.vararg is not None:
        return False
    model = params[0].arg
    parents = {id(child): node for node in ast.walk(function) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and node.id == model:
            owner = parents.get(id(node))
            if not (isinstance(node.ctx, ast.Load) and isinstance(owner, ast.Call) and owner.func is node):
                return False
    return True


def _weights_file(module, scope, sd, metadata, use):
    """The source text of the file path when ``sd`` and ``metadata`` are exactly what one
    ``load_torch_file(path, return_metadata=True)`` gave, unchanged up to ``use``; else None."""
    if not (isinstance(sd, ast.Name) and isinstance(metadata, ast.Name)):
        return None
    names = {sd.id, metadata.id}
    stores = [n for n in ast.walk(scope) if isinstance(n, ast.Name) and n.id in names and isinstance(n.ctx, (ast.Store, ast.Del))]
    loaded = [n for n in ast.walk(scope) if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Tuple) and [getattr(e, "id", None) for e in n.targets[0].elts] == [sd.id, metadata.id]
              and isinstance(n.value, ast.Call) and module.resolve(n.value.func) == "comfy.utils.load_torch_file"]
    if len(loaded) != 1 or len(stores) != 2 or loaded[0].lineno >= use.lineno:
        return None
    call = loaded[0].value
    flags = {k.arg: k.value for k in call.keywords}
    if len(call.args) != 1 or not isinstance(call.args[0], ast.Name) or set(flags) - {"return_metadata", "safe_load"} \
            or not (isinstance(flags.get("return_metadata"), ast.Constant) and flags["return_metadata"].value is True):
        return None
    path = call.args[0].id
    if sum(1 for n in ast.walk(scope) if isinstance(n, ast.Name) and n.id == path and isinstance(n.ctx, (ast.Store, ast.Del))) > 1:
        return None
    parents = {id(child): node for node in ast.walk(scope) for child in ast.iter_child_nodes(node)}
    for read in ast.walk(scope):  # the weights must reach the VAE as the file had them
        if isinstance(read, ast.Name) and read.id == sd.id and isinstance(read.ctx, ast.Load) and read is not sd:
            owner = parents.get(id(read))
            if isinstance(owner, ast.Subscript) and isinstance(owner.ctx, ast.Load):
                continue
            if isinstance(owner, ast.Compare) or (isinstance(owner, ast.Attribute) and owner.attr in ("keys", "get", "items", "values")):
                continue
            if isinstance(owner, ast.Call) and isinstance(owner.func, ast.Name) and owner.func.id == "len":
                continue
            return None
    return path


def rewrite(source, path, typed_functions, star_aliases=None, keep=frozenset(), facade_exports=frozenset()):
    """``star_aliases``: engine names this module gets from the pack's own modules by ``import *``;
    ``keep``: engine names another module gets from this one that way and still uses (not removed)."""
    module = _Module(source, path)
    for name, dotted in (star_aliases or {}).items():
        module.aliases.setdefault(name, dotted)
    edits, done, left = [], [], []
    replaced = set()
    # type annotations that name engine types (``model: ModelPatcher``) aren't uses: they become strings,
    # and the import goes if nothing else needs it
    annotations = []
    for node in ast.walk(module.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            annotations += [x.annotation for x in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]
                            if x is not None and x.annotation is not None]
            if node.returns is not None:
                annotations.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
    for annotation in annotations:
        if isinstance(annotation, ast.Constant):
            continue
        if any(isinstance(n, (ast.Name, ast.Attribute)) and module.resolve(n) for n in ast.walk(annotation)):
            start, end = module.span(annotation)
            text = ast.get_source_segment(source, annotation) or ""
            if "\n" not in text:
                edits.append((start, end, repr(text)))
                done.append("annotation")
    # a model folder's files as a V3 Combo's options: ComfyUI lists them (io.ModelFolder) each time
    for node in ast.walk(module.tree):
        if isinstance(node, ast.keyword) and node.arg == "options" and isinstance(node.value, ast.Call) \
                and module.resolve(node.value.func) == "folder_paths.get_filename_list" and len(node.value.args) == 1:
            start, end = module.span(node.value)
            edits.append((start, end, f"_v2io.ModelFolder({ast.get_source_segment(source, node.value.args[0])})"))
            done.append("folder_paths.get_filename_list")
    # ComfyUI's VAE name list, used at schema time (a pack's INPUT_TYPES runs outside any node, so no engine call)
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Call) and module.resolve(node.func) == "nodes.VAELoader.vae_list":
            start, end = module.span(node)
            edits.append((start, end, "_v2e.vae_list()"))
            done.append("nodes.VAELoader.vae_list")
    # dotted engine names (a core node as a base class stays: it's ComfyUI's class, not a call)
    # core nodes are rewritten where they're called: nodes.X() or nodes.X.method(...)
    core_calls = set()
    for n in ast.walk(module.tree):
        if isinstance(n, ast.Call):
            if isinstance(n.func, (ast.Attribute, ast.Name)):
                core_calls.add(id(n.func))
            # a method that runs the node; ``X.define_schema()`` reads the class, which the stand-in can't answer
            if isinstance(n.func, ast.Attribute) and n.func.attr[:1].islower() and not n.func.attr.startswith("define_"):
                core_calls.add(id(n.func.value))
    parents = {id(child): node for node in ast.walk(module.tree) for child in ast.iter_child_nodes(node)}
    server_vars = {t.id for n in ast.walk(module.tree) if isinstance(n, ast.Assign) and module.resolve(n.value) == "server.PromptServer.instance"
                   for t in n.targets if isinstance(t, ast.Name)}
    server_attrs = {n.attr for n in ast.walk(module.tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                    and n.value.id in server_vars}
    facade_ok = server_attrs <= PROMPT_SERVER_FACADE
    # a module that also uses a part of the server the stand-in lacks (its queue, say) keeps the server itself
    # flagged: handing the stand-in on would fail where that part is read
    for n in ast.walk(module.tree):
        if isinstance(n, ast.Attribute) and not isinstance(parents.get(id(n)), ast.Attribute):
            chain = module.resolve(n) or ""
            if chain.startswith("server.PromptServer.instance.") and not chain.split(".")[3].startswith("_") \
                    and not _facade_has(chain.split(".")[3:]):
                facade_ok = False
    # Stock server-instance lookup: use only the existing bounded pack facade.
    # A raw instance must not escape, and unknown attributes keep their blocker.
    if not _binds_name(module.tree, "getattr"):
        for call in ast.walk(module.tree):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == "getattr" and len(call.args) in (2, 3) and not call.keywords
                    and module.resolve(call.args[0]) == "server.PromptServer"
                    and isinstance(call.args[1], ast.Constant) and call.args[1].value == "instance"
                    and (len(call.args) == 2 or isinstance(call.args[2], ast.Constant) and call.args[2].value is None)):
                continue
            receiver = call.args[0]
            if not isinstance(receiver, ast.Name):
                continue
            conflicting_import = any(
                isinstance(n, ast.ImportFrom) and any((a.asname or a.name) == receiver.id
                    and not (n.level == 0 and n.module == "server" and a.name == "PromptServer") for a in n.names)
                or isinstance(n, ast.Import) and any((a.asname or a.name.split(".")[0]) == receiver.id for a in n.names)
                for n in ast.walk(module.tree))
            if conflicting_import:
                continue
            # No guessed class identity after rebinding or an escaping class
            # reference. The supported case is the stock imported class only.
            class_reads = [n for n in ast.walk(module.tree) if isinstance(n, ast.Name) and n.id == receiver.id]
            if any(isinstance(n.ctx, (ast.Store, ast.Del)) for n in class_reads):
                continue
            if any(isinstance(n, ast.arg) and n.arg == receiver.id for n in ast.walk(module.tree)):
                continue
            if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == receiver.id for n in ast.walk(module.tree)):
                continue
            stock_reads = True
            for read in class_reads:
                context = parents.get(id(read))
                if isinstance(context, ast.Attribute) and context.value is read and context.attr == "instance" and isinstance(context.ctx, ast.Load):
                    continue
                if (isinstance(context, ast.Call) and isinstance(context.func, ast.Name) and context.func.id == "getattr"
                        and context.args and context.args[0] is read):
                    continue
                stock_reads = False
            if not stock_reads:
                continue
            owner = parents.get(id(call))
            safe = isinstance(owner, ast.Attribute) and owner.attr in PROMPT_SERVER_FACADE
            if isinstance(owner, ast.Assign) and len(owner.targets) == 1 and isinstance(owner.targets[0], ast.Name):
                bound = owner.targets[0].id
                scope = owner
                while id(scope) in parents and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    scope = parents[id(scope)]
                reads = [n for n in ast.walk(scope) if isinstance(n, ast.Name) and n.id == bound and isinstance(n.ctx, ast.Load)]
                writes = [n for n in ast.walk(scope) if isinstance(n, ast.Name) and n.id == bound and isinstance(n.ctx, (ast.Store, ast.Del))]
                safe = len(writes) == 1 and bool(reads)
                for read in reads:
                    context = parents.get(id(read))
                    attr = isinstance(context, ast.Attribute) and context.value is read and context.attr in PROMPT_SERVER_FACADE
                    null_test = (isinstance(context, ast.Compare) and len(context.ops) == 1
                                 and isinstance(context.ops[0], (ast.Is, ast.IsNot))
                                 and any(isinstance(n, ast.Constant) and n.value is None for n in [context.left, *context.comparators]))
                    if not (attr or null_test):
                        safe = False
            if safe:
                start, end = module.span(call)
                edits.append((start, end, "_v2e.prompt_server"))
                done.append("server.PromptServer.instance (stock getattr)")
    own_util = _binds_name(module.tree, "util")
    # ComfyUI's node table read for one of ComfyUI's own nodes, by a constant id:
    # ``nodes.NODE_CLASS_MAPPINGS["KSamplerSelect"]``, ``.get("VAEDecode")``, ``"VAEDecode" in ...``
    for node in ast.walk(module.tree):
        key = table = None
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            key, table = node.slice, node.value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" \
                and len(node.args) in (1, 2) and not node.keywords:
            key, table = node.args[0], node.func.value
        elif isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)):
            key, table = node.left, node.comparators[0]
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value in CORE_NODE_IDS
                and isinstance(table, (ast.Name, ast.Attribute)) and module.resolve(table) == "nodes.NODE_CLASS_MAPPINGS"):
            continue
        if isinstance(node, ast.Compare):
            replacement = "True" if isinstance(node.ops[0], ast.In) else "False"
        elif not _node_class_only_run(node, parents):
            continue
        else:
            replacement = _core_node(CORE_NODE_IDS[key.value])
        edits.append((*module.span(node), replacement))
        done.append("nodes.NODE_CLASS_MAPPINGS (one of ComfyUI's nodes)")
    for node in ast.walk(module.tree):
        if isinstance(node, (ast.Attribute, ast.Name)) and isinstance(getattr(node, "ctx", None), ast.Load):
            full = module.resolve(node)
            replacement = REWRITES.get(full)
            # the server's attributes the shim's stand-in has, read straight off the instance
            top = node
            while isinstance(parents.get(id(top)), ast.Attribute) and parents[id(top)].value is top:
                top = parents[id(top)]
            if replacement is None and full and full.startswith("server.PromptServer.instance.") \
                    and not any(".".join(full.split(".")[:i]) in REWRITES for i in range(4, full.count(".") + 2)) \
                    and _facade_has((module.resolve(top) or full).split(".")[3:]):
                replacement = "_v2e.prompt_server" + full[len("server.PromptServer.instance"):]
            if full == "server.PromptServer.instance" and replacement is None and facade_ok \
                    and not isinstance(parents.get(id(node)), ast.Attribute):
                replacement = "_v2e.prompt_server"
            # a pack's own mark on the server (``PromptServer.instance._my_routes_registered = True``, read back
            # with getattr): kept on the stand-in. A public attribute set there patches ComfyUI, and stays flagged.
            owner = parents.get(id(node))
            if full == "server.PromptServer.instance" and replacement is None and isinstance(owner, ast.Attribute) \
                    and owner.attr.startswith("_") and not owner.attr.startswith("__"):
                replacement = "_v2e.prompt_server"
            if replacement is None and full and (full == "folder_paths" or full.startswith("folder_paths.")):
                replacement = "util.folders" + full[len("folder_paths"):]  # folder_paths' names, as util.folders
            if replacement is not None and own_util and replacement.startswith("util."):
                replacement = "_v2" + replacement  # the module's own ``util`` would hide V2's (D2: all 71 nodes dead)
            if replacement is None and full in CORE_NODES and id(node) in core_calls:
                replacement = _core_node(CORE_NODES[full])
            if replacement is not None:
                start, end = module.span(node)
                if not any(s <= start and end <= e for s, e, _ in edits):
                    edits = [(s, e, t) for s, e, t in edits if not (start <= s and e <= end)]
                    edits.append((start, end, replacement))
                    done.append(full)
    # A named sampler is already a bounded engine operation. Preserve sync
    # helpers, and await it directly in an async function's own body.
    for node in sorted((n for n in ast.walk(module.tree) if isinstance(n, ast.Call)),
                       key=module.length):
        if module.resolve(node.func) != "comfy.samplers.sampler_object":
            continue
        if len(node.args) == 1 and not isinstance(node.args[0], ast.Starred) and not node.keywords:
            argument = node.args[0]
        elif not node.args and len(node.keywords) == 1 and node.keywords[0].arg == "name":
            argument = node.keywords[0].value
        else:
            continue
        start, end = module.span(node)
        if any(s <= start and end <= e for s, e, _ in edits):
            continue
        argument = _apply_within(source, *module.span(argument), edits)
        owner = parents.get(id(node))
        while owner is not None and not isinstance(owner, _FUNCTIONS):
            owner = parents.get(id(owner))
        # KSamplerSelect uses sampler_object, including uni_pc, uni_pc_bh2 and
        # ddim. eng.sampler.named calls ksampler, which treats those differently.
        call = f"eng.node.run('KSamplerSelect', {{'sampler_name': {argument}}})"
        replacement = (f"(await {call})" if isinstance(owner, ast.AsyncFunctionDef) else f"_v2e.sync({call})") + "[0]"
        edits[:] = [(s, e, t) for s, e, t in edits if not (start <= s and e <= end)]
        edits.append((start, end, replacement))
        done.append("comfy.samplers.sampler_object")
    # a VAE built from weights (``comfy.sd.VAE(sd=sd)``) is eng.vae.from_state_dict; built from a file's weights
    # and metadata as ComfyUI's VAELoader does, it's that file's VAE (eng.models.load_vae)
    built_vaes = {}  # id(the call) -> True
    for node in sorted((n for n in ast.walk(module.tree) if isinstance(n, ast.Call)), key=module.length):
        if module.resolve(node.func) != "comfy.sd.VAE" or len(node.args) > 1 \
                or any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
            continue
        given = {k.arg: k.value for k in node.keywords}
        if node.args:
            if "sd" in given:
                continue
            given["sd"] = node.args[0]
        start, end = module.span(node)
        if any(s <= start and end <= e for s, e, _ in edits):
            continue
        owner = parents.get(id(node))
        while owner is not None and not isinstance(owner, _FUNCTIONS):
            owner = parents.get(id(owner))
        if set(given) == {"sd"}:
            call = f"eng.vae.from_state_dict({_apply_within(source, *module.span(given['sd']), edits)})"
        elif set(given) == {"sd", "metadata"}:
            weights_path = _weights_file(module, owner or module.tree, given["sd"], given["metadata"], node)
            if weights_path is None:
                continue
            call = f"eng.models.load_vae(_v2e.model_file('vae', {weights_path}))"
        else:
            continue
        replacement = f"(await {call})" if isinstance(owner, ast.AsyncFunctionDef) else f"_v2e.sync({call})"
        edits[:] = [(s, e, t) for s, e, t in edits if not (start <= s and e <= end)]
        edits.append((start, end, replacement))
        done.append("comfy.sd.VAE")
        built_vaes[id(node)] = True
    # samplers: ``comfy.samplers.ksampler(name, options)`` is eng.sampler.named; ``KSAMPLER(fn, extra_options)``
    # around the pack's own sampling loop is eng.sampler.custom, when the loop only calls its model
    definitions = {}
    for n in ast.walk(module.tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.setdefault(n.name, []).append(n)
    for node in sorted((n for n in ast.walk(module.tree) if isinstance(n, ast.Call)), key=module.length):
        full = module.resolve(node.func)
        if full not in ("comfy.samplers.ksampler", "comfy.samplers.KSAMPLER") \
                or any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
            continue
        names = ("sampler_name" if full.endswith("ksampler") else "sampler_function", "extra_options", "inpaint_options")
        given = dict(zip(names, node.args))
        if len(node.args) > 3 or any(k.arg in given or k.arg not in names for k in node.keywords):
            continue
        given.update({k.arg: k.value for k in node.keywords})
        if names[0] not in given or "inpaint_options" in given:
            continue
        start, end = module.span(node)
        if any(s <= start and end <= e for s, e, _ in edits):
            continue
        if full.endswith("KSAMPLER"):
            fn = given[names[0]]
            found = definitions.get(fn.id, []) if isinstance(fn, ast.Name) and id(fn) not in module.shadowed else []
            if len(found) != 1 or not _calls_model_only(found[0]):
                continue
        arguments = [_apply_within(source, *module.span(given[names[0]]), edits)]
        if "extra_options" in given:
            arguments.append(_apply_within(source, *module.span(given["extra_options"]), edits))
        call = f"eng.sampler.{'named' if full.endswith('ksampler') else 'custom'}({', '.join(arguments)})"
        owner = parents.get(id(node))
        while owner is not None and not isinstance(owner, _FUNCTIONS):
            owner = parents.get(id(owner))
        edits[:] = [(s, e, t) for s, e, t in edits if not (start <= s and e <= end)]
        edits.append((start, end, f"(await {call})" if isinstance(owner, ast.AsyncFunctionDef) else f"_v2e.sync({call})"))
        done.append(full)
    # engine-object methods on typed node inputs (and on what their clone() returns)
    # The AST is unchanged while edits are collected. Reuse this analysis in
    # the rewrite and both later detector passes, especially for large packs.
    typed_params = _typed_params(module.tree, typed_functions)
    # a variable that holds a VAE built above is a VAE for the rest of its function
    built_names = {}
    for function in ast.walk(module.tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and id(node.value) in built_vaes:
                name = node.targets[0].id
                stores = sum(1 for n in ast.walk(function) if isinstance(n, ast.Name) and n.id == name
                             and isinstance(n.ctx, (ast.Store, ast.Del)))
                if stores == 1 and not any(a.arg == name for a in ast.walk(function) if isinstance(a, ast.arg)):
                    typed_params.setdefault(function, {})[name] = "VAE"
                    built_names.setdefault(id(function), set()).add(name)
    for function, params in typed_params.items():
        typed = dict(params)
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) \
                    and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id in typed \
                    and node.value.func.attr == "clone":
                typed[node.targets[0].id] = typed[node.value.func.value.id]
        # innermost first, and each call's arguments taken with the edits already made inside them:
        # ``clip.encode_from_tokens_scheduled(clip.tokenize(text))`` rewrites both calls
        def text(a):
            return _apply_within(source, *module.span(a), edits)

        def replace(start, end, new):
            edits[:] = [(s, e, t) for s, e, t in edits if not (start <= s and e <= end)]
            edits.append((start, end, new))

        # in an ``async def`` (and not in a sync function inside it) an engine call is awaited: _v2e.sync there
        # would block the loop it waits on
        awaited = _direct_body_ids(function) if isinstance(function, ast.AsyncFunctionDef) else set()

        def engine_call(node, call):
            return f"(await {call})" if id(node) in awaited else f"_v2e.sync({call})"

        def model_name(node):
            """The typed MODEL variable ``node`` is, or None."""
            return node.id if isinstance(node, ast.Name) and typed.get(node.id) == "MODEL" else None

        def model_object(node, key):
            """The MODEL whose ``get_model_object(key)`` / ``.model.<key>`` this expression is, or None."""
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get_model_object" \
                    and len(node.args) == 1 and not node.keywords and isinstance(node.args[0], ast.Constant) \
                    and node.args[0].value == key:
                return model_name(node.func.value)
            if isinstance(node, ast.Attribute) and node.attr == key and isinstance(node.value, ast.Attribute) \
                    and node.value.attr == "model":
                return model_name(node.value.value)
            return None

        # ``m.model_options[...][...] = value``: eng.model.set_option (a copy with the option set)
        for node in ast.walk(function):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript)):
                continue
            keys, target = [], node.targets[0]
            while isinstance(target, ast.Subscript):
                keys.append(target.slice)
                target = target.value
            if isinstance(target, ast.Attribute) and target.attr == "model_options" and model_name(target.value) \
                    and all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in keys) \
                    and not {k.value for k in keys} & FUNCTION_OPTIONS and not isinstance(node.value, ast.Lambda):
                receiver = target.value.id
                path = ", ".join(repr(k.value) for k in reversed(keys))
                replace(*module.span(node), f"{receiver} = " + engine_call(
                    node, f"eng.model.set_option({receiver}, [{path}], {text(node.value)})"))
                done.append("MODEL.model_options (set)")
        # ``m.model_options`` read (``.get("transformer_options", {})``, ``["x"]``, ``"x" in``): eng.model.options, a
        # copy. Only where nothing could be written through it: not kept in a variable, not handed on.
        for node in ast.walk(function):
            if not (isinstance(node, ast.Attribute) and node.attr == "model_options" and isinstance(node.ctx, ast.Load)
                    and model_name(node.value)):
                continue
            top, read_only = node, True
            while True:
                owner = parents.get(id(top))
                if isinstance(owner, ast.Subscript) and owner.value is top and isinstance(owner.ctx, ast.Load):
                    top = owner
                elif isinstance(owner, ast.Attribute) and owner.value is top and owner.attr in ("get", "keys", "items", "values") \
                        and isinstance(parents.get(id(owner)), ast.Call) and parents[id(owner)].func is owner:
                    top = parents[id(owner)]
                else:
                    break
            owner = parents.get(id(top))
            if isinstance(owner, ast.Compare) and top in owner.comparators and isinstance(owner.ops[0], (ast.In, ast.NotIn)):
                pass
            elif isinstance(owner, (ast.If, ast.While, ast.IfExp, ast.BoolOp, ast.UnaryOp, ast.Compare, ast.JoinedStr, ast.FormattedValue)):
                pass
            elif isinstance(owner, (ast.For, ast.comprehension)) and owner.iter is top:
                pass
            elif top is node:
                read_only = False  # the options themselves, kept or handed on
            elif isinstance(owner, (ast.Assign, ast.AnnAssign, ast.Return, ast.Call, ast.keyword, ast.Tuple, ast.List, ast.Dict,
                                    ast.Attribute, ast.Subscript, ast.Starred, ast.NamedExpr, ast.AugAssign, ast.Delete, ast.Yield)):
                read_only = False  # a dict or list from inside them may be changed where it lands
            if read_only:
                replace(*module.span(node), engine_call(node, f"eng.model.options({node.value.id})"))
                done.append("MODEL.model_options (read)")
        # a model's sampling object: the shim's stand-in, which answers through eng.model
        sampling_ids = set()
        for node in ast.walk(function):
            name = model_object(node, "model_sampling") if isinstance(getattr(node, "ctx", ast.Load()), ast.Load) else None
            if name is None:
                continue
            replace(*module.span(node), f"_v2e.ModelSampling({name})")
            done.append("MODEL.model_sampling")
            sampling_ids.add(id(node))
            owner = parents.get(id(node))
            if isinstance(owner, ast.Attribute) and owner.value is node and owner.attr not in FACADES["MODEL_SAMPLING"]:
                left.append(f"MODEL_SAMPLING.{_chain(owner, parents)}")
            elif isinstance(owner, ast.Assign) and len(owner.targets) == 1 and isinstance(owner.targets[0], ast.Name):
                bound = owner.targets[0].id
                if bound not in typed and sum(1 for n in ast.walk(function) if isinstance(n, ast.Name) and n.id == bound
                                              and isinstance(n.ctx, (ast.Store, ast.Del))) == 1:
                    typed[bound] = "MODEL_SAMPLING"
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            # the model's latent scaling and dtype
            if isinstance(node.func, ast.Attribute) and len(node.args) == 1 and not node.keywords:
                receiver, name, row = node.func.value, None, None
                if node.func.attr in ("process_latent_in", "process_latent_out") and isinstance(receiver, ast.Attribute) \
                        and receiver.attr == "model":
                    name, row = model_name(receiver.value), "latent_" + node.func.attr.replace("_latent", "")
                elif node.func.attr in ("process_in", "process_out"):
                    name, row = model_object(receiver, "latent_format"), "latent_" + node.func.attr
                if name is not None:
                    replace(*module.span(node), engine_call(node, f"eng.model.{row}({name}, {text(node.args[0])})"))
                    done.append(f"MODEL.{row}")
                    continue
            if isinstance(node.func, ast.Attribute) and node.func.attr == "model_dtype" and not node.args \
                    and not node.keywords and model_name(node.func.value):
                replace(*module.span(node), f"_v2e.model_dtype({node.func.value.id})")
                done.append("MODEL.model_dtype")
                continue
            # sigmas for a scheduler, from a model's sampling object
            if module.resolve(node.func) == "comfy.samplers.calculate_sigmas" and len(node.args) == 3 and not node.keywords \
                    and (id(node.args[0]) in sampling_ids or (isinstance(node.args[0], ast.Name)
                                                              and typed.get(node.args[0].id) == "MODEL_SAMPLING")):
                replace(*module.span(node.func), "_v2e.calculate_sigmas")
                done.append("comfy.samplers.calculate_sigmas")

        # a stable sort by length: an inner call comes before the call it's in; a statement before its own call
        candidates = sorted((n for n in ast.walk(function) if isinstance(n, (ast.Expr, ast.Call))),
                            key=module.length)
        for node in candidates:
            if isinstance(node, ast.Expr):
                call = node.value
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) \
                        and isinstance(call.func.value, ast.Name) and call.func.attr == "throw_exception_if_invalid" \
                        and call.func.value.id in built_names.get(id(function), ()) and not call.args and not call.keywords:
                    replace(*module.span(node), "pass  # V2: the engine checks a VAE it builds")
                    continue
                # ``m.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "key", fn)``: eng.model.add_wrapper (ComfyUI
                # keeps a list per key, so a key only matters to code that removes by it, which stays flagged)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) \
                        and call.func.attr in ("add_wrapper", "add_wrapper_with_key") \
                        and isinstance(call.func.value, ast.Name) and typed.get(call.func.value.id) == "MODEL" \
                        and not call.keywords and len(call.args) == (2 if call.func.attr == "add_wrapper" else 3) \
                        and not any(isinstance(a, ast.Starred) for a in call.args):
                    kind = (module.resolve(call.args[0]) or "") if isinstance(call.args[0], (ast.Name, ast.Attribute)) else ""
                    kind = WRAPPER_KINDS.get(kind.rpartition(".")[2]) if kind.startswith("comfy.patcher_extension.WrappersMP.") else None
                    if kind:
                        receiver = call.func.value.id
                        replace(*module.span(node), f"{receiver} = " + engine_call(
                            node, f"eng.model.add_wrapper({receiver}, {kind!r}, {text(call.args[-1])})"))
                        done.append(f"MODEL.{call.func.attr}")
                        continue
                # ``m.add_patches(patches, strength)`` changes m; the engine row returns the patched copy
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "add_patches" \
                        and isinstance(call.func.value, ast.Name) and typed.get(call.func.value.id) in ("MODEL", "CLIP") \
                        and not any(isinstance(a, ast.Starred) for a in call.args) and not any(k.arg is None for k in call.keywords):
                    receiver = call.func.value.id
                    row = "model" if typed[receiver] == "MODEL" else "clip"
                    arguments = [receiver] + [text(a) for a in call.args] + [text(k) for k in call.keywords]
                    replace(*module.span(node), f"{receiver} = " + engine_call(node, f"eng.{row}.add_patches({', '.join(arguments)})"))
                    done.append(f"{typed[receiver]}.add_patches")
                    continue
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) \
                        and isinstance(call.func.value, ast.Name) and typed.get(call.func.value.id) == "MODEL" \
                        and call.func.attr in MODEL_SETTERS:
                    receiver = call.func.value.id
                    parameters = MODEL_SETTER_PARAMETERS[call.func.attr]
                    # Bind the legacy signature before translating: engine hook rows accept only fn.
                    # Never erase an unknown keyword or a dynamic argument expansion.
                    values = [(parameters[i] if i < len(parameters) else None, a)
                              for i, a in enumerate(call.args)] + [(k.arg, k.value) for k in call.keywords]
                    names = [name for name, _ in values]
                    if (any(isinstance(a, ast.Starred) for a in call.args)
                            or any(name not in parameters for name in names)
                            or len(names) != len(set(names)) or parameters[0] not in names):
                        left.append(f"MODEL.{call.func.attr} (unsupported arguments)")
                        continue
                    start, end = module.span(node)
                    if "disable_cfg1_optimization" in names:
                        # Evaluate arguments once, in Python's positional/keyword evaluation order,
                        # before invoking the hook. Fresh names cannot shadow pack locals/globals.
                        prefix = f"_v2_setter_{start}_"
                        while any(prefix in token for token in source.split()):
                            prefix += "_"
                        temps = [prefix + str(i) for i in range(len(values) + 1)]
                        bound = dict(zip(names, temps[1:]))
                        statements = [f"{', '.join(temps)} = {receiver}, {', '.join(text(a) for _, a in values)}",
                                      f"{receiver} = " + engine_call(node, f"eng.{MODEL_SETTERS[call.func.attr]}({temps[0]}, {bound[parameters[0]]})")]
                        option = engine_call(node, f"eng.model.set_option({receiver}, ['disable_cfg1_optimization'], True)")
                        statements += [f"{receiver} = {option} if {bound['disable_cfg1_optimization']} else {receiver}",
                                       f"del {', '.join(temps)}"]
                        replace(start, end, "; ".join(statements))
                    else:
                        fn = text(values[0][1])
                        replace(start, end, f"{receiver} = " + engine_call(node, f"eng.{MODEL_SETTERS[call.func.attr]}({receiver}, {fn})"))
                    done.append(f"MODEL.{call.func.attr}")
                continue
            if module.resolve(node.func) == "comfy.samplers.calculate_sigmas" \
                    and len(node.args) == 3 and not node.keywords:
                sampling = node.args[0]
                if isinstance(sampling, ast.Call) and isinstance(sampling.func, ast.Attribute) \
                        and sampling.func.attr == "get_model_object" \
                        and isinstance(sampling.func.value, ast.Name) \
                        and typed.get(sampling.func.value.id) == "MODEL" \
                        and len(sampling.args) == 1 and not sampling.keywords \
                        and isinstance(sampling.args[0], ast.Constant) and sampling.args[0].value == "model_sampling":
                    start, end = module.span(node)
                    call = (f"eng.node.run('BasicScheduler', {{'model': {text(sampling.func.value)}, "
                            f"'scheduler': {text(node.args[1])}, 'steps': {text(node.args[2])}, 'denoise': 1.0}})")
                    replace(start, end, engine_call(node, call) + "[0]")
                    done.append("comfy.samplers.calculate_sigmas")
                    continue
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                kind = typed.get(node.func.value.id)
                row = METHODS.get(kind, {}).get(node.func.attr)
                if row:
                    start, end = module.span(node)
                    # inside an edit already made (a statement above, or a replaced ComfyUI call): left to it
                    if not any(s <= start and end <= e for s, e, _ in edits):
                        args = [node.func.value.id] + [text(a) for a in node.args] + [text(k) for k in node.keywords]
                        replace(start, end, engine_call(node, f"eng.{row}({', '.join(args)})"))
                        done.append(f"{kind}.{node.func.attr}")
        # any other attribute of a typed engine object: in V2.1 a MODEL/CLIP/VAE is an opaque handle (Ref), so
        # ``model.model_options``, ``model.get_model_object(...)``, ``vae.get_sd()`` fail when the node runs
        for node in ast.walk(function):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in typed:
                start, end = module.span(node)
                if any(s <= start and end <= e for s, e, _ in edits):
                    continue
                kind = typed[node.value.id]
                if kind == "MODEL" and node.attr in MODEL_SETTERS:
                    left.append(f"MODEL.{node.attr} (unsupported arguments or call context)")
                    continue
                if node.attr in METHODS.get(kind, {}) or node.attr in FACADES.get(kind, ()) or node.attr.startswith("__"):
                    continue
                left.append(f"{kind}.{_chain(node, parents)}")
        # a sampling function handed to a MODEL setter sees tensors and plain data only (the API's hook_view), so
        # its ``args["model"]`` / ``args.get("model")`` isn't there (sd-perturbed-attention's post_cfg_function)
        local_defs = {n.name: n for n in ast.walk(module.tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and isinstance(node.func.value, ast.Name) and typed.get(node.func.value.id) == "MODEL" \
                    and node.func.attr in MODEL_SETTERS:
                argument = node.args[0] if node.args else next(
                    (k.value for k in node.keywords if k.arg == MODEL_SETTER_PARAMETERS[node.func.attr][0]), None)
                if not isinstance(argument, ast.Name):
                    continue
                hook = local_defs.get(argument.id)
                skip_self = False
                if hook is None:  # an object with __call__ (``calc = FDG_APG_Combined_Function(params)``)
                    hook = _call_method(module.tree, function, argument.id)
                    skip_self = hook is not None
                params = (hook.args.posonlyargs + hook.args.args)[1 if skip_self else 0:] if hook is not None else []
                if params:
                    param = params[0].arg
                    for read in _engine_object_reads(hook, param):
                        left.append(f"sampling function args[{read!r}] (not passed to pack code)")
    # a sampling hook defined anywhere in the module (returned by a factory, registered from another file:
    # DiversityBoost's diversity_hook, Danny r3): its args dict read for "model" along with sampling keys
    for function in ast.walk(module.tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # any parameter: a sampler hook's args dict comes first, an attention patch's extra_options last
        for argument in function.args.posonlyargs + function.args.args:
            if argument.arg in ("self", "cls"):
                continue
            if _dict_keys_read(function, argument.arg) & SAMPLING_KEYS:
                for read in _engine_object_reads(function, argument.arg):
                    flag = f"sampling function args[{read!r}] (not passed to pack code)"
                    if flag not in left:
                        left.append(flag)
    # an engine object handed to a helper function (``build_sigmas(model, ...)`` then ``model.get_model_object``):
    # not a typed node input, so nothing above rewrote it; an attribute only ComfyUI's object has fails on a Ref
    typed_ids = {id(f) for f in typed_params}
    for function in ast.walk(module.tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) or id(function) in typed_ids:
            continue
        a = function.args
        guessed = {x.arg: _guess_kind(x.arg) for x in a.posonlyargs + a.args + a.kwonlyargs}
        guessed = {k: v for k, v in guessed.items() if v}
        for node in ast.walk(function) if guessed else ():
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in guessed \
                    and node.attr in ENGINE_ONLY_ATTRS[guessed[node.value.id]]:
                left.append(f"{guessed[node.value.id]}.{node.attr}")
    # ``hasattr(clip, "tokenize")`` / ``getattr(model, "x", None)`` on an engine object: always false / the default on
    # a V2.1 handle, so the code silently takes another branch (AK-XZ-Axis Batch Prompts, Danny r3: the agent kept
    # the checks, the else branch built malformed conditioning). Only real types: node inputs and the helpers they're
    # passed to (no guessing by name: TTS/VLM packs keep their own models in a ``model`` param), and not a type the
    # pack's own nodes output (its own "MODEL" objects stay in its process and have their attributes).
    owned = typed_functions.get(OWNED_TYPES_KEY, (None, frozenset()))[1] if isinstance(typed_functions, dict) else ()
    kinds_by_function = {id(f): params for f, params in typed_params.items()}
    for function in ast.walk(module.tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        kinds = {k: v for k, v in (kinds_by_function.get(id(function)) or {}).items()
                 if v in ENGINE_KINDS and v not in owned}
        for node in ast.walk(function) if kinds else ():
            # calling the handle itself (``upscale_model(tile)`` in a tiling lambda: MambaIR, Danny r4): a V2.1
            # handle isn't callable
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in kinds:
                left.append(f"{kinds[node.func.id]} called directly (a V2.1 handle isn't callable)")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("hasattr", "getattr") \
                    and len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and node.args[0].id in kinds \
                    and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                kind = kinds[node.args[0].id]
                if node.func.id == "hasattr" and (node.args[1].value in METHODS.get(kind, {})
                                                  or (kind == "MODEL" and node.args[1].value in MODEL_SETTERS)):
                    # a method the converter calls through eng for this object: ComfyUI's object has it
                    start, end = module.span(node)
                    if not any(s <= start and end <= e for s, e, _ in edits):
                        edits.append((start, end, "True"))
                        done.append(f"{kind}.{node.args[1].value} (hasattr)")
                    continue
                left.append(f"{kinds[node.args[0].id]}.{node.args[1].value} ({node.func.id} on a V2.1 handle)")
    # PyTorch's process-wide numeric settings changed by the pack (``torch.backends.cuda.matmul.allow_tf32 = True``,
    # ``torch.set_float32_matmul_precision(...)``): in ComfyUI that changed how every node computes; an isolated
    # pack can't change ComfyUI's, so its engine calls compute with ComfyUI's settings (CachingCLIPTextEncode, Danny r5)
    settings_changed = []
    torch_names = {a.asname or a.name for n in ast.walk(module.tree) if isinstance(n, ast.Import)
                   for a in n.names if a.name == "torch"}

    def torch_dotted(expr):
        parts = _dotted(expr)
        return ".".join(["torch"] + parts[1:]) if parts and parts[0] in torch_names else None

    for node in ast.walk(module.tree):
        target = None
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            for t in getattr(node, "targets", None) or [node.target]:
                if isinstance(t, ast.Attribute):
                    target = torch_dotted(t)
        elif isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name)):
            name = torch_dotted(node.func)
            if name in TORCH_SETTING_CALLS:
                target = name
        if target and (target.startswith("torch.backends.") or target in TORCH_SETTING_CALLS):
            settings_changed.append(target)
    # only a problem when the module also has the engine compute for it (CachingCLIPTextEncode sets TF32 and encodes
    # with eng.clip): a pack's own models run in its own process, where its setting still applies as it did in
    # ComfyUI (SAM2, BiRefNet...); isolated, it just no longer changes every other node's math, which is the point
    engine_compute = [d for d in done if d.split(".")[0] in ENGINE_KINDS or d.startswith(("nodes.", "comfy.sd.", "comfy.sample"))]
    if engine_compute:
        for target in sorted(set(settings_changed)):
            left.append(f"{target} (a process-wide PyTorch setting the engine's calls for this pack won't follow)")
    # a node of the saved workflow changed in place through the hidden extra_pnginfo: in ComfyUI that edits the
    # workflow saved with later images (ShowText writes its text into a node's widgets_values, through
    # ``workflow = extra_pnginfo[0]["workflow"]`` and ``node = next(x for x in workflow["nodes"] ...)``); a pack
    # gets a copy, so the change never reaches ComfyUI (Danny r7). Only real aliases are followed: a subscript or
    # attribute chain, ``.get()``, ``next()`` over one, or a loop over one. A new object built from the metadata
    # (a saver's exif or save options) is the pack's own. Only widgets_values writes are flagged: a write to any
    # other key may be read back by the same node, which works on a copy (2026-11-07: the broad version flagged 42
    # packs, mostly savers changing their own objects).
    def _root(t):
        while True:
            if isinstance(t, (ast.Subscript, ast.Attribute)):
                t = t.value
            elif isinstance(t, ast.Call) and isinstance(t.func, ast.Attribute) and t.func.attr in ("get", "values", "items"):
                t = t.func.value
            elif isinstance(t, ast.Call) and isinstance(t.func, ast.Name) and t.func.id in ("next", "enumerate", "reversed", "iter") and t.args:
                t = t.args[0]
            elif isinstance(t, (ast.GeneratorExp, ast.ListComp)) and len(t.generators) == 1:
                t = t.generators[0].iter
            else:
                break
        return t.id if isinstance(t, ast.Name) else None
    def _widgets(t):
        return any(isinstance(n, ast.Constant) and n.value == "widgets_values" for n in ast.walk(t))
    for function in ast.walk(module.tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        tainted = {a.arg for a in function.args.posonlyargs + function.args.args + function.args.kwonlyargs
                   if a.arg.endswith("pnginfo")}
        if not tainted:
            continue
        hit = False
        for node in sorted((n for n in ast.walk(function) if hasattr(n, "lineno") or isinstance(n, ast.comprehension)),
                           key=lambda n: (getattr(n, "lineno", 0) or getattr(n.iter, "lineno", 0), getattr(n, "col_offset", 0))):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.For, ast.comprehension, ast.NamedExpr)):
                value = getattr(node, "value", None) or getattr(node, "iter", None)
                targets = getattr(node, "targets", None) or [getattr(node, "target", None)]
                hit = hit or any(isinstance(t, ast.Subscript) and _root(t) in tainted and _widgets(t) for t in targets if t)
                if value is not None and _root(value) in tainted and not isinstance(node, ast.AugAssign):
                    for t in targets:
                        if isinstance(t, ast.Name):
                            tainted.add(t.id)
                        elif isinstance(t, ast.Tuple):  # for i, node in enumerate(...)
                            tainted.update(e.id for e in t.elts if isinstance(e, ast.Name))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in CHANGERS \
                    and isinstance(node.func.value, ast.Subscript) and _root(node.func.value) in tainted \
                    and _widgets(node.func.value):
                hit = True  # node["widgets_values"].append(...)
        if hit:
            left.append("a workflow node changed in place via extra_pnginfo (the saved workflow won't see it)")
    # a ComfyUI module imported at run time (``importlib.import_module("comfy.sd")``, ``__import__("comfy.sd")``):
    # nothing above sees its calls, so they'd run ComfyUI's code in the pack (SodaLoadStoryboardLoRA, Danny r4)
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str) and node.args[0].value.split(".")[0] in ENGINE_ROOTS:
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if name in ("import_module", "__import__"):
                left.append(f"{node.args[0].value} (imported at run time)")
    # ComfyUI's command-line settings changed by the pack (``args.fast.add("cublas_ops")``, ``args.x = y``):
    # in V2.1 they're read-only settings of the host, so a change can't reach ComfyUI. Left for hand conversion.
    for node in ast.walk(module.tree):
        target = None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in CHANGERS:
            target = node.func.value
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
            for t in getattr(node, "targets", None) or [getattr(node, "target", None)]:
                if isinstance(t, (ast.Attribute, ast.Subscript)):
                    target = t.value if isinstance(t, ast.Subscript) else t
        full = module.resolve(target) if isinstance(target, (ast.Attribute, ast.Name)) else None
        if full and full.startswith("comfy.cli_args.args"):
            left.append(full + " (changed)")
    edits.sort()
    out, last = [], 0
    for start, end, text in edits:
        if start < last:
            continue
        out.append(source[last:start])
        out.append(text)
        last = end
        replaced.add((start, end))
    out.append(source[last:])
    new = "".join(out)
    # what still uses the engine, read from the rewritten code
    m2 = _Module(new, path)
    for name, dotted in (star_aliases or {}).items():
        m2.aliases.setdefault(name, dotted)
    for node in ast.walk(m2.tree):
        if isinstance(node, (ast.Attribute, ast.Name)) and isinstance(getattr(node, "ctx", None), ast.Load):
            full = m2.resolve(node)
            if full:
                left.append(full)
    left = _outermost(left)
    # engine imports whose names nothing uses any more go
    loaded = {n.id for n in ast.walk(m2.tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
              and id(n) not in m2.shadowed}
    lines = new.splitlines(keepends=True)
    seen_imports = set()
    # an import in a class body binds a class attribute, used as ``self.folder_paths`` / ``cls.folder_paths``
    # (MakkiTools): an attribute load of the name counts as a use, and the import becomes its V2 equivalent
    class_imports = {id(item) for cls in ast.walk(m2.tree) if isinstance(cls, ast.ClassDef)
                     for item in cls.body if isinstance(item, (ast.Import, ast.ImportFrom))}
    attr_loads = {n.attr for n in ast.walk(m2.tree) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load)}
    for node in sorted(m2.imports, key=lambda n: -n.lineno):
        if id(node) in seen_imports:
            continue
        seen_imports.add(id(node))
        bound = [a.asname or (a.name if isinstance(node, ast.ImportFrom) else a.name.split(".")[0]) for a in node.names]
        # ``from comfy.samplers import *``: which of the module's names the file uses can't be told from the
        # import (LCMSampler takes ``torch`` and ``Sampler`` from it), so it stays, and is reported
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            left.append(f"{node.module}.* (imported with *)")
            continue
        # A pack module may expose this already-supported metadata namespace to
        # another pack module. Keep the bounded facade, never the Core class.
        if (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == 'comfy.comfy_types'
                and len(node.names) == 1 and node.names[0].name == 'IO' and bound[0] in facade_exports and bound[0] not in keep
                and node.end_lineno == node.lineno and lines[node.lineno - 1][:node.col_offset].strip() == ''):
            lines[node.lineno - 1] = ' ' * node.col_offset + f'{bound[0]} = _v2e.IO  # V2 metadata export\n'
            done.append('comfy.comfy_types.IO (metadata export)')
            continue
        # A local fallback assignment makes the scope analyzer treat the load as
        # shadowed. Preserve the binding for the known module availability probe.
        if isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0].name == 'nodes' \
                and _safe_optional_alias(m2.tree, node, bound[0], guard_only=True) and node.end_lineno == node.lineno \
                and lines[node.lineno - 1][:node.col_offset].strip() == "":
            lines[node.lineno - 1] = ' ' * node.col_offset + f'{bound[0]} = _v2e  # V2 availability probe\n'
            done.append('nodes availability probe')
            continue
        # A shadowed load can still refer to this import in its own scope. Do
        # not erase the binding or conceal unresolved engine use in that case.
        shadow_loads = {n.id for n in ast.walk(m2.tree) if isinstance(n, ast.Name)
                        and isinstance(n.ctx, ast.Load) and id(n) in m2.shadowed}
        if isinstance(node, ast.ImportFrom) and len(node.names) == 1 and bound[0] in shadow_loads \
                and _safe_optional_alias(m2.tree, node, bound[0]) and node.end_lineno == node.lineno \
                and lines[node.lineno - 1][:node.col_offset].strip() == "":
            full = f'{node.module}.{node.names[0].name}'
            equivalent = REWRITES.get(full)
            if equivalent and (equivalent.startswith('_v2e.') or equivalent.startswith('util.')):
                if own_util and equivalent.startswith('util.'):
                    equivalent = '_v2' + equivalent
                lines[node.lineno - 1] = ' ' * node.col_offset + f'{bound[0]} = {equivalent}  # V2 optional import\n'
                done.append(full)
                continue
        if any(b in shadow_loads for b in bound):
            left.extend(m2.aliases.get(b, b) + ' (potential local import use)' for b in bound if b in shadow_loads)
            continue
        if id(node) in class_imports and isinstance(node, ast.Import) and len(node.names) == 1 \
                and bound[0] in attr_loads and bound[0] not in loaded:
            equivalent = CLASS_IMPORT_EQUIVALENTS.get(node.names[0].name)
            if equivalent and lines[node.lineno - 1][:node.col_offset].strip() == "" and node.end_lineno == node.lineno:
                module, attr = equivalent.rsplit(".", 1)  # its own import: the file may have no V2 import line
                lines[node.lineno - 1] = (" " * node.col_offset + f"from comfy_api.v4 import {module} as _v2_{module}; "
                                          f"{bound[0]} = _v2_{module}.{attr}  # V2: was import {node.names[0].name}\n")
                done.append(f"class import {node.names[0].name}")
            elif not equivalent:
                left.append(f"{node.names[0].name} (imported in a class body, used as an attribute)")
            continue
        if any(b in loaded or b in keep for b in bound):
            # a mixed statement (``import os, folder_paths``): the unused engine names go, the rest stay
            if isinstance(node, ast.Import):
                kept = [a for a, b in zip(node.names, bound) if b in loaded or b in keep or a.name.split(".")[0] not in ENGINE_ROOTS]
            else:
                kept = [a for a, b in zip(node.names, bound) if b in loaded or b in keep]
            last = lines[node.end_lineno - 1]
            if len(kept) < len(node.names) and last[node.end_col_offset:].strip() in ("", "\\") \
                    and lines[node.lineno - 1][:node.col_offset].strip() == "":
                names = ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in kept)
                text = f"import {names}" if isinstance(node, ast.Import) else f"from {node.module} import {names}"
                for index in range(node.lineno - 1, node.end_lineno):
                    lines[index] = ""
                lines[node.lineno - 1] = " " * node.col_offset + text + "\n"
            continue
        for index in range(node.lineno - 1, node.end_lineno):
            lines[index] = ""
        if _needs_pass(m2.tree, node):
            lines[node.lineno - 1] = " " * node.col_offset + "pass  # engine import removed by the V2 converter\n"
    candidate = "".join(lines)
    try:
        compile(candidate, path, "exec")
        new = candidate
    except SyntaxError:
        pass  # keep the imports rather than break the file
    engine_bound = {a.asname or (a.name if isinstance(n, ast.ImportFrom) else a.name.split(".")[0])
                    for n in m2.imports for a in n.names}
    left.extend(m2.aliases[name] + ' (retained for pack import)' for name in keep if name in m2.aliases)
    new = _prune_all(new, path, engine_bound - set(keep) - set(facade_exports))
    if done:
        new = _add_imports(new, path)
    return new, sorted(set(done)), sorted(set(left))


def _names_used(tree, name):
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)}


def _needs_pass(tree, import_node):
    """An import that is the only statement of its block can't just vanish."""
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody", "handlers"):
            block = getattr(node, field, None)
            if isinstance(block, list) and import_node in block and len(block) == 1:
                return True
    return False


def _outermost(names):
    return [n for n in names if not any(o != n and o.startswith(n + ".") for o in names)] if names else []


_V2_IMPORT_MARK = "# --- V2 converter imports ---"


def _add_imports(source, path):
    depth = path.count("/")  # path is relative to v2/
    dots = "." * (depth + 1)
    up = ", ".join(['".."'] * depth)
    header = (f"{_V2_IMPORT_MARK}\nfrom comfy_api.v4 import eng, ui, util, io as _v2io  # noqa: F401\n"
              f"try:\n    from {dots} import _v2_engine as _v2e  # noqa: F401\n"
              "except ImportError:  # the pack imports this file by a top-level name (it put its folder on sys.path)\n"
              "    import importlib.util as _v2u, os as _v2o, sys as _v2s\n"
              f"    _v2p = _v2o.path.realpath(_v2o.path.join(_v2o.path.dirname(_v2o.path.abspath(__file__)), {up + ', ' if up else ''}'_v2_engine.py'))\n"
              "    _v2e = _v2s.modules.get('_v2_engine@' + _v2p)\n"
              "    if _v2e is None:\n"
              "        _v2spec = _v2u.spec_from_file_location('_v2_engine@' + _v2p, _v2p)\n"
              "        _v2e = _v2s.modules[_v2spec.name] = _v2u.module_from_spec(_v2spec)\n"
              "        _v2spec.loader.exec_module(_v2e)\n")
    if re.search(r"(?<![\w.])_v2util\.", source):
        header = header.replace("io as _v2io  # noqa: F401\n", "io as _v2io  # noqa: F401\n"
                                "from comfy_api.v4 import util as _v2util  # noqa: F401\n", 1)
    tree = ast.parse(source)
    insert_line = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insert_line = node.end_lineno
        elif isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and insert_line == 0 \
                and node is tree.body[0]:
            insert_line = node.end_lineno
    lines = source.splitlines(keepends=True)
    return "".join(lines[:insert_line]) + header + "".join(lines[insert_line:])


# --------------------------------------------------------------------------- #
# Generated node classes
# --------------------------------------------------------------------------- #
IO_TYPES = {"INT": "io.Int", "FLOAT": "io.Float", "STRING": "io.String", "BOOLEAN": "io.Boolean", "IMAGE": "io.Image",
            "MASK": "io.Mask", "LATENT": "io.Latent", "CONDITIONING": "io.Conditioning", "MODEL": "io.Model",
            "CLIP": "io.Clip", "VAE": "io.Vae", "CONTROL_NET": "io.ControlNet", "SAMPLER": "io.Sampler",
            "SIGMAS": "io.Sigmas", "NOISE": "io.Noise", "GUIDER": "io.Guider", "CLIP_VISION": "io.ClipVision",
            "CLIP_VISION_OUTPUT": "io.ClipVisionOutput", "UPSCALE_MODEL": "io.UpscaleModel", "AUDIO": "io.Audio",
            "VIDEO": "io.Video", "*": "io.AnyType"}


def _io_class(io_type):
    return IO_TYPES.get(io_type, f"io.Custom({io_type!r})")


def _literal(value):
    return repr(value)


def _input(name, spec, section, hints):
    io_type = spec[0] if spec else "*"
    options = dict(spec[1]) if len(spec) > 1 and isinstance(spec[1], dict) else {}
    optional = ", optional=True" if section == "optional" else ""
    typed = ""
    if io_type == "STRING":  # V3's String always lists multiline (default False), over extra_dict
        typed = f", multiline={bool(options.pop('multiline', False))!r}"
    extra = f"{typed}, extra_dict={_literal(options)}" if options else typed
    if isinstance(io_type, list):
        hint = hints.get(name, {})
        if "model_folder" in hint:
            opts = f"io.ModelFolder({hint['model_folder']!r})"
        elif "util_constant" in hint:
            opts = f"util.{hint['util_constant']}"
        else:
            opts = _literal(io_type)
        # Combo.Input overwrites multiselect=False; preserve list-valued widgets.
        combo = "io.MultiCombo" if options.get("multiselect") is True else "io.Combo"
        return f"{combo}.Input({name!r}, options={opts}{optional}{extra})"
    if isinstance(io_type, dict):
        raise ValueError(f"input {name!r}: an unplain type")
    return f"{_io_class(io_type)}.Input({name!r}{optional}{extra})"


def _output(index, io_type, info):
    names = info.get("output_name") or []
    lists = info.get("output_is_list") or []
    tips = info.get("output_tooltips") or []
    name = names[index] if index < len(names) else None
    kwargs = [f"display_name={name!r}"] if name is not None else []
    if index < len(lists) and lists[index]:
        kwargs.append("is_output_list=True")
    if index < len(tips) and tips[index]:
        kwargs.append(f"tooltip={tips[index]!r}")
    if isinstance(io_type, list):
        return f"io.Combo.Output(options={_literal(io_type)}{', ' if kwargs else ''}{', '.join(kwargs)})"
    return f"{_io_class(io_type)}.Output({', '.join(kwargs)})"


def node_source(node_id, entry):
    info = entry["info"]
    inputs = []
    hidden = []
    hidden_args = {}
    seen = set()
    for section in ("required", "optional"):
        for name, spec in (info["input"].get(section) or {}).items():
            if name in seen:  # in both required and optional: V1 takes the required one, a V2 schema can't have both
                continue
            seen.add(name)
            spec = spec if isinstance(spec, list) else [spec]
            inputs.append(_input(name, spec, section, entry.get("hints", {})))
    for name, spec in (info["input"].get("hidden") or {}).items():
        kind = spec[0] if isinstance(spec, list) else spec
        if isinstance(kind, str) and kind in HIDDEN:  # others are set by the pack's frontend, if at all
            hidden.append(f"io.Hidden.{HIDDEN[kind]}")
            hidden_args[name] = HIDDEN[kind]
    outputs = [_output(i, t, info) for i, t in enumerate(info.get("output") or [])]
    flags = []
    for key, arg in (("output_node", "is_output_node"), ("is_input_list", "is_input_list"), ("deprecated", "is_deprecated"),
                     ("experimental", "is_experimental"), ("dev_only", "is_dev_only"), ("api_node", "is_api_node"),
                     ("has_intermediate_output", "has_intermediate_output")):
        if info.get(key):
            flags.append(f"{arg}=True")
    if info.get("search_aliases"):
        flags.append(f"search_aliases={_literal(info['search_aliases'])}")
    if info.get("essentials_category"):
        flags.append(f"essentials_category={info['essentials_category']!r}")
    class_name = "V2_" + re.sub(r"\W", "_", node_id)
    methods = entry.get("methods", [])
    lines = [
        f"class {class_name}(io.ComfyNode):",
        f"    \"\"\"{node_id}, converted from V1: runs the original ``{entry['class']}.{entry['function']}``.\"\"\"",
        "    ENGINE_API = True",
        "",
        "    @classmethod",
        "    def define_schema(cls):",
        f"        return io.Schema(node_id={node_id!r}, display_name={info.get('display_name', node_id)!r},",
        f"                         category={info.get('category', 'sd')!r}, description={info.get('description') or ''!r},",
        f"                         inputs=[{', '.join(inputs)}],",
        f"                         outputs=[{', '.join(outputs)}],",
        f"                         hidden=[{', '.join(hidden)}]{', ' if flags else ''}{', '.join(flags)})",
        "",
        "    @classmethod",
        "    async def execute(cls, **inputs):",
        f"        return await _v2e.run_v1(_V1[{node_id!r}], {entry['function']!r}, cls, inputs, {hidden_args!r})",
    ]
    if "VALIDATE_INPUTS" in methods:
        lines += ["", "    @classmethod", "    def validate_inputs(cls, **inputs):",
                  f"        return _V1[{node_id!r}].VALIDATE_INPUTS(**inputs)"]
    if "IS_CHANGED" in methods:
        lines += ["", "    @classmethod", "    def fingerprint_inputs(cls, **inputs):",
                  f"        return _V1[{node_id!r}].IS_CHANGED(**inputs)"]
    if "check_lazy_status" in methods:
        lines += ["", "    @classmethod", "    def check_lazy_status(cls, **inputs):",
                  f"        return _V1[{node_id!r}]().check_lazy_status(**inputs)"]
    return class_name, "\n".join(lines)


V3_ENTRY = '''import inspect as _v2_inspect


def _v2_engine_api(cls):
    """A node the pack wrote for the V3 schema API, as an engine-API node (values in, values out). A sync
    ``execute`` runs on a worker thread with its engine calls routed back (``_v2e.sync``), as V1 methods do."""
    attrs = {"ENGINE_API": True, "__module__": cls.__module__}
    original = _v2_inspect.getattr_static(cls, "execute", None)
    function = getattr(original, "__func__", None)
    if function is not None and not _v2_inspect.iscoroutinefunction(function):
        async def execute(c, *args, **kwargs):
            return await _v2_shim.run_v3(function, c, *args, **kwargs)
        execute.__name__, execute.__doc__ = "execute", function.__doc__
        attrs["execute"] = classmethod(execute)
    return type(cls.__name__, (cls,), attrs)


def _v2_is_v3(cls):
    from comfy_api.v4 import io
    return isinstance(cls, type) and issubclass(cls, io.ComfyNode)


class _V2Extension(_V2ComfyExtension):
    _original = None

    async def _from_pack(self):
        if self._original is None and _V2_ORIGINAL_ENTRY is not None:
            original = _V2_ORIGINAL_ENTRY()
            self._original = await original if _v2_inspect.isawaitable(original) else original
        return self._original

    async def on_load(self):
        original = await self._from_pack()
        if original is not None:
            await original.on_load()

    async def get_node_list(self):
        original = await self._from_pack()
        v3 = list(await original.get_node_list()) if original is not None else []
        v3 += [_V2_ORIGINAL_MAPPINGS[i] for i in _V2_V3_IDS if i in _V2_ORIGINAL_MAPPINGS and _V2_ORIGINAL_MAPPINGS[i] not in v3]
        return _V2_NODES + [_v2_engine_api(cls) for cls in v3 if _v2_is_v3(cls)]

    async def _registered(self, name):
        original = await self._from_pack()
        method = getattr(original, name, None) if original is not None else None
        return await method() if method is not None else None

    async def get_samplers(self):
        return await self._registered("get_samplers")

    async def get_schedulers(self):
        return await self._registered("get_schedulers")

    async def get_model_folders(self):
        own = await self._registered("get_model_folders") or {}
        return {**_v2_shim.pack_model_folders(), **own}

    async def get_node_replacements(self):
        return await self._registered("get_node_replacements")

    async def get_routes(self):
        own = await self._registered("get_routes")
        return list(own or []) + list(_v2_shim.routes)


async def comfy_entrypoint():
    return _V2Extension()
'''


ENGINE_SHIM = '''"""Generated by the V2 converter: sync access to ``eng`` for the pack's original (sync) code.

The original method runs on a worker thread (``run_v1``); ``sync`` hands an ``eng`` call to the node's
event loop and waits. Everything here uses only ``comfy_api.v4``.
"""
import asyncio
import contextvars
import os
import sys

from comfy_api.v4 import eng, io

# One copy per pack: a pack that imports its modules by top-level name loads this file by path too
# (see the import the converter adds to each file); both names then share this module.
_KEY = "_v2_engine@" + os.path.realpath(__file__)
if sys.modules.setdefault(_KEY, sys.modules[__name__]) is not sys.modules[__name__]:
    sys.modules[__name__] = sys.modules[_KEY]

_LOOP = contextvars.ContextVar("v2_converter_loop", default=None)

from aiohttp import web as _web  # noqa: E402

#: The pack's HTTP routes (its code registered them on PromptServer.instance.routes); the pack's
#: extension gives them to ComfyUI (get_routes), which serves them at the same paths.
routes = getattr(sys.modules.get(__name__), "routes", None) or _web.RouteTableDef()


def sync(coro):
    loop = _LOOP.get()
    if loop is None:
        coro.close()
        raise RuntimeError("the converted code made an engine call outside a running node")
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


def running():
    """Whether a node is running: outside one (at import, say) the helpers below give defaults."""
    return _LOOP.get() is not None


def _device(key):
    import torch
    if not running():  # at import: this process's best device; ComfyUI's is asked for while a node runs
        return torch.device("cuda" if key == "torch_device" and torch.cuda.is_available() else "cpu")
    return torch.device(sync(eng.device.info())[key])


def get_torch_device():
    return _device("torch_device")


def intermediate_device():
    return _device("intermediate_device")


def offload_device():
    return _device("offload_device")


def _dtype(key):
    import torch
    if not running():
        return torch.float32
    return getattr(torch, sync(eng.device.info())[key])


def intermediate_dtype(*args, **kwargs):
    return _dtype("intermediate_dtype")


def unet_dtype(*args, **kwargs):
    return _dtype("unet_dtype")


def vae_dtype(*args, **kwargs):
    return _dtype("vae_dtype")


def vae_device(*args, **kwargs):
    return _device("vae_device")


def text_encoder_device(*args, **kwargs):
    return _device("text_encoder_device")


def vae_offload_device(*args, **kwargs):
    return _device("vae_offload_device")


def text_encoder_offload_device(*args, **kwargs):
    return _device("text_encoder_offload_device")


try:  # as comfy.model_management defines it
    import torch as _v2torch
    OOM_EXCEPTION = _v2torch.cuda.OutOfMemoryError
except Exception:
    OOM_EXCEPTION = Exception


def _device_type(device):
    return getattr(device, "type", str(device).split(":")[0])


def is_device_mps(device):
    return _device_type(device) == "mps"


def is_device_cpu(device):
    return _device_type(device) == "cpu"


def is_device_cuda(device):
    return _device_type(device) == "cuda"


def get_autocast_device(device):
    return device.type if hasattr(device, "type") else "cuda"  # as comfy.model_management has it


def unload_all_models(*args, **kwargs):
    if running():
        sync(eng.memory.unload_all())


def load_models_gpu(models, *args, **kwargs):
    if running():
        sync(eng.memory.load_models(list(models)))


def load_model_gpu(model, *args, **kwargs):
    load_models_gpu([model])


def add_routes(definitions):
    """``PromptServer.instance.app.add_routes(...)``: they join the pack's routes."""
    for item in definitions:
        if hasattr(item, "method"):
            routes.route(item.method, item.path, **dict(getattr(item, "kwargs", {}) or {}))(item.handler)
        else:
            routes.static(item.prefix, item.path, **dict(getattr(item, "kwargs", {}) or {}))


class _Outputs(tuple):
    """A node's outputs: a tuple, as a V1 node returns, that also reads as a V3 node's ``NodeOutput``."""
    result = args = property(lambda self: tuple(self))
    ui = None


class CoreNode:
    """One of ComfyUI's nodes, as V1 code calls it (``nodes.LoraLoader().load_lora(model, clip, ...)``):
    the call runs the node with ``eng.node.run``, positional arguments named in the order of the node's
    method (``params``, when it isn't the schema's input order: ``VAEEncode.encode(self, vae, pixels)``).
    It returns the node's outputs as a tuple, as the V1 method does."""

    def __init__(self, node_id, inputs, function, params=None):
        self._node_id, self._function, self._known = node_id, function, set(inputs)
        self._inputs = list(params) if params else list(inputs)

    def __call__(self, *args, **kwargs):
        return self  # nodes.X(): the instance the method is called on

    def __getattr__(self, name):
        if name == "FUNCTION":
            return self._function
        if name.startswith("_") or name.isupper() or name in ("INPUT_TYPES", "RETURN_TYPES", "define_schema"):
            raise AttributeError(f"{self._node_id}.{name}: the converter runs ComfyUI's nodes, it doesn't read them")

        def run(*args, **kwargs):
            if args and args[0] is self:
                args = args[1:]
            # a method parameter that isn't one of the node's inputs (SaveAudio's ``format``) isn't the node's
            inputs = {k: v for k, v in zip(self._inputs, args) if k in self._known}
            inputs.update(kwargs)
            return _Outputs(sync(eng.node.run(self._node_id, inputs)))
        run.__name__ = name
        # the method's parameters, for packs that read them (Impact: ``'overlap' in inspect.signature(...)``)
        import inspect
        run.__signature__ = inspect.Signature(
            [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD) for n in self._inputs])
        return run


web = _web


def send_progress_text(text, node_id=None, sid=None):
    """``PromptServer.instance.send_progress_text``: text on the running node."""
    if running():
        sync(eng.ui.progress_text(str(text)))


async def send(event, data, sid=None):
    """``PromptServer.instance.send`` / ``send_json``: a UI event, from a running node."""
    try:
        await eng.ui.send(event, data)
    except RuntimeError:
        pass  # outside a running node there's no prompt to send it to


class _Router:
    """``PromptServer.instance.app.router``: routes added here join the pack's routes."""

    def add_route(self, method, path, handler, **kwargs):
        routes.route(method, path, **kwargs)(handler)

    def add_get(self, path, handler, **kwargs):
        self.add_route("GET", path, handler, **kwargs)

    def add_post(self, path, handler, **kwargs):
        self.add_route("POST", path, handler, **kwargs)

    def add_put(self, path, handler, **kwargs):
        self.add_route("PUT", path, handler, **kwargs)

    def add_delete(self, path, handler, **kwargs):
        self.add_route("DELETE", path, handler, **kwargs)

    def add_static(self, prefix, path, **kwargs):
        routes.static(prefix, path, **kwargs)

    def add_routes(self, definitions):
        add_routes(definitions)


class _App(dict):
    """``PromptServer.instance.app``: routes, and the aiohttp app's hook lists (a pack process has no
    aiohttp app, so startup and shutdown hooks are kept but never called)."""

    def __init__(self):
        super().__init__()
        self.router = _Router()
        self.on_startup, self.on_shutdown, self.on_cleanup, self.middlewares = [], [], [], []
        self.frozen = False  # routes can always be added: they're the pack's own table

    def add_routes(self, definitions):
        add_routes(definitions)


class _PromptServer:
    """``PromptServer.instance`` for converted code: the pack's routes and UI events, through the API."""
    routes = routes
    app = _App()
    last_node_id = last_prompt_id = None

    @property
    def client_id(self):
        """Set while a prompt runs, as ComfyUI's is: events go to the client that queued it, whatever this says."""
        return "v2-client" if running() else None

    def send_sync(self, event, data, sid=None):
        send_sync(event, data, sid)

    async def send(self, event, data, sid=None):
        await send(event, data, sid)

    async def send_json(self, event, data, sid=None):
        await send(event, data, sid)

    def send_progress_text(self, text, node_id=None, sid=None):
        send_progress_text(text, node_id, sid)


prompt_server = _PromptServer()


def pack_model_folders():
    """Model folders the pack added when it loaded (util.folders), for ComfyUI to list:
    ``{folder: [extensions] or None}``."""
    from comfy_api.v4 import util
    found = {}
    for name, (paths, extensions) in util.folders.folder_names_and_paths.items():
        if name not in _COMFY_FOLDERS:
            extensions = sorted(e for e in extensions if e) if not isinstance(extensions, str) else [extensions]
            found[name] = extensions or None
    return found


def load_lora_for_models(model, clip, lora, strength_model, strength_clip, lora_metadata=None):
    """``comfy.sd.load_lora_for_models``: eng.model.load_lora (with ComfyUI's lora_metadata, LoRABlockFilter)."""
    if lora_metadata:
        return sync(eng.model.load_lora(model, clip, lora, strength_model, strength_clip, lora_metadata=lora_metadata))
    return sync(eng.model.load_lora(model, clip, lora, strength_model, strength_clip))


def _contained(path, bases):
    """The name of ``path`` under one of ``bases``, or None. By the path as written (a model file may be a
    symlink into a cache elsewhere, as ComfyUI allows), then by both resolved, for a base reached by a link."""
    import os as _os
    written = _os.path.normpath(_os.path.abspath(str(path)))
    resolved = _os.path.realpath(str(path))
    for base in bases:
        for root, candidate in ((_os.path.normpath(_os.path.abspath(base)), written),
                                (_os.path.realpath(base), resolved)):
            if candidate.startswith(root.rstrip(_os.sep) + _os.sep):
                return _os.path.relpath(candidate, root).replace(_os.sep, "/")
    return None


def model_file(folder, path):
    """The name in ComfyUI's ``folder`` of a file the pack has by path (as util.folders.get_full_path gave it).
    The engine resolves the name in its own model folders: that, not this, is where access is decided."""
    from comfy_api.v4 import util
    name = _contained(path, util.folders.get_folder_paths(folder))
    if name is None:
        raise NotImplementedError(f"{path} isn't in ComfyUI's {folder} folder: the engine loads models only from "
                                  "its model folders")
    return name


def load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=False,
                                 embedding_directory=None, output_model=True, model_options={}, te_model_options={},
                                 **kwargs):
    """``comfy.sd.load_checkpoint_guess_config``: eng.models.load_checkpoint, for a file in the checkpoints folder.
    Returns ComfyUI's 4-tuple (model, clip, vae, clipvision); a CLIP vision model isn't loaded here."""
    if output_clipvision or model_options or te_model_options:
        raise NotImplementedError("load_checkpoint_guess_config with a CLIP vision output or model options")
    model, clip, vae = sync(eng.models.load_checkpoint(model_file("checkpoints", ckpt_path)))
    return (model if output_model else None, clip if output_clip else None, vae if output_vae else None, None)


def load_diffusion_model(unet_path, model_options={}, **kwargs):
    """``comfy.sd.load_diffusion_model``: eng.models.load_diffusion_model, for a file in diffusion_models."""
    options = dict(model_options or {})
    if "dtype" in options and options["dtype"] is not None:
        options["dtype"] = str(options["dtype"]).removeprefix("torch.")
    return sync(eng.models.load_diffusion_model(model_file("diffusion_models", unet_path), options or None))


def load_unet(unet_path, dtype=None):
    """``comfy.sd.load_unet``: load_diffusion_model with a dtype."""
    return load_diffusion_model(unet_path, {"dtype": dtype} if dtype is not None else {})


#: ComfyUI shows progress bars (comfy.utils.PROGRESS_BAR_ENABLED); ComfyUI decides, so always on here.
PROGRESS_BAR_ENABLED = True


def cleanup_models(*args, **kwargs):
    """``comfy.model_management.cleanup_models``: ComfyUI cleans up its models itself."""


def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0,
                    disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    """``nodes.common_ksampler``: ComfyUI's KSampler (or KSamplerAdvanced, for the step options)."""
    common = {"model": model, "steps": steps, "cfg": cfg, "sampler_name": sampler_name, "scheduler": scheduler,
              "positive": positive, "negative": negative, "latent_image": latent}
    if not disable_noise and start_step is None and last_step is None and not force_full_denoise:
        return sync(eng.node.run("KSampler", {**common, "seed": seed, "denoise": denoise}))
    if denoise != 1.0:
        raise NotImplementedError("common_ksampler with both denoise and step options")
    return sync(eng.node.run("KSamplerAdvanced", {
        **common, "add_noise": "disable" if disable_noise else "enable", "noise_seed": seed,
        "start_at_step": start_step or 0, "end_at_step": 10000 if last_step is None else last_step,
        "return_with_leftover_noise": "disable" if force_full_denoise else "enable"}))


class ModelSampling:
    """A model's sampling object (``model.get_model_object("model_sampling")``), as far as the API has it."""

    def __init__(self, model):
        self._model = model

    def percent_to_sigma(self, percent):
        return sync(eng.model.percent_to_sigma(self._model, percent))

    def timestep(self, sigma):
        return sync(eng.model.timestep(self._model, sigma))

    def sigma(self, timestep):
        return sync(eng.model.sigma(self._model, timestep))

    def _fact(self, name):
        import torch
        return torch.tensor(sync(eng.model.info(self._model))[name])

    sigma_min = property(lambda self: self._fact("sigma_min"))
    sigma_max = property(lambda self: self._fact("sigma_max"))


def calculate_sigmas(model_sampling, scheduler_name, steps):
    """``comfy.samplers.calculate_sigmas``: ComfyUI's BasicScheduler for the model."""
    return sync(eng.node.run("BasicScheduler", {"model": model_sampling._model, "scheduler": scheduler_name,
                                                "steps": steps, "denoise": 1.0}))[0]


def model_dtype(model):
    import torch
    return getattr(torch, sync(eng.model.info(model))["dtype"])


class _ClipTypes:
    """``comfy.sd.CLIPType``: the names, which is what ComfyUI's CLIP loader nodes take."""

    def __getattr__(self, name):
        if name.isupper():
            return name
        raise AttributeError(name)


CLIPType = _ClipTypes()


def load_clip(ckpt_paths, embedding_directory=None, clip_type="STABLE_DIFFUSION", model_options={}, **kwargs):
    """``comfy.sd.load_clip`` for files in ComfyUI's text_encoders folder: ComfyUI's CLIPLoader / DualCLIPLoader."""
    if model_options or kwargs:
        raise NotImplementedError("comfy.sd.load_clip with model options")
    names = [model_file("text_encoders", path) for path in ckpt_paths]
    kind = str(clip_type).rsplit(".", 1)[-1].lower()
    if len(names) == 1:
        return sync(eng.node.run("CLIPLoader", {"clip_name": names[0], "type": kind}))[0]
    if len(names) == 2:  # the dual loader's name for the default type
        return sync(eng.node.run("DualCLIPLoader", {"clip_name1": names[0], "clip_name2": names[1],
                                                    "type": "sdxl" if kind == "stable_diffusion" else kind}))[0]
    raise NotImplementedError(f"comfy.sd.load_clip with {len(names)} files")


def _noise_added(noise, latent_image, seed):
    """Whether ``noise`` is what ComfyUI's sampler nodes add for ``seed`` (True) or none (False). The nodes
    make their own noise, so any other noise can't be sampled through them."""
    import torch
    from comfy_api.v4 import util
    if getattr(noise, "is_nested", False) or getattr(latent_image, "is_nested", False):
        raise NotImplementedError("sampling a nested (video + audio) latent through ComfyUI's sampler nodes")
    if torch.count_nonzero(noise) == 0:
        return False
    if seed is not None and noise.shape == latent_image.shape \
            and torch.equal(noise.cpu(), util.prepare_noise(latent_image, seed).to(noise.dtype)):
        return True
    raise NotImplementedError("sampling with noise that isn't ComfyUI's for the seed (prepare_noise(latent, seed)) "
                              "or zeros: ComfyUI's sampler nodes make their own noise")


def _node_callback(callback):
    """A sampler node reports progress and previews itself; a pack's own callback would never be called."""
    if callback is not None and not hasattr(callback, "x0_output"):
        raise NotImplementedError("sampling through ComfyUI's sampler nodes with the pack's own step callback")


def sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0,
           disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None,
           sigmas=None, callback=None, disable_pbar=False, seed=None):
    """``comfy.sample.sample``, as ComfyUI's KSampler nodes call it: run as that node, which reports progress
    and previews itself (``callback`` isn't called)."""
    if sigmas is not None:
        raise NotImplementedError("comfy.sample.sample with its own sigmas")
    latent = {"samples": latent_image}
    if noise_mask is not None:
        latent["noise_mask"] = noise_mask
    added = _noise_added(noise, latent_image, seed)
    _node_callback(callback)
    return common_ksampler(model, seed if seed is not None else 0, steps, cfg, sampler_name, scheduler, positive,
                           negative, latent, denoise=denoise, disable_noise=not added, start_step=start_step,
                           last_step=last_step, force_full_denoise=force_full_denoise)[0]["samples"]


def sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask=None,
                  callback=None, disable_pbar=False, seed=None):
    """``comfy.sample.sample_custom``, as ComfyUI's SamplerCustom calls it: run as that node. A callback from
    ``prepare_callback(model, steps, x0_output)`` gets its ``x0_output["x0"]``, as the node's does."""
    latent = {"samples": latent_image}
    if noise_mask is not None:
        latent["noise_mask"] = noise_mask
    added = _noise_added(noise, latent_image, seed)
    _node_callback(callback)
    output, denoised = sync(eng.node.run("SamplerCustom", {
        "model": model, "add_noise": added, "noise_seed": seed if seed is not None else 0, "cfg": cfg,
        "positive": positive, "negative": negative, "sampler": sampler, "sigmas": sigmas, "latent_image": latent}))
    x0_output = getattr(callback, "x0_output", None)
    if x0_output is not None and denoised is not output:
        x0_output["x0"] = sync(eng.model.latent_process_in(model, denoised["samples"]))
    return output["samples"]


def fix_empty_latent_channels(model, latent_image, downscale_ratio_spacial=None, downscale_ratio_temporal=None):
    """``comfy.sample.fix_empty_latent_channels``: an empty latent takes the model's channel count. ComfyUI's
    sampler nodes do this again, and the rest of it (a video model's extra dimension), when they run."""
    import torch
    if getattr(latent_image, "is_nested", False) or torch.count_nonzero(latent_image) != 0:
        return latent_image
    if downscale_ratio_spacial is not None or downscale_ratio_temporal is not None:
        raise NotImplementedError("fix_empty_latent_channels with a downscale ratio")
    channels = sync(eng.model.info(model)).get("latent_channels")
    if channels is not None and channels != latent_image.shape[1]:
        latent_image = comfy_utils.repeat_to_batch_size(latent_image, channels, dim=1)
    return latent_image


def prepare_empty_noise(latent_image):
    import torch
    if getattr(latent_image, "is_nested", False):
        raise NotImplementedError("empty noise for a nested (video + audio) latent")
    return torch.zeros_like(latent_image, device="cpu")


def prepare_callback(model, steps, x0_output_dict=None):
    """``latent_preview.prepare_callback``: the sampling callback. Called by the pack's own loop it reports
    progress; handed to ``sample`` / ``sample_custom`` above, the node that runs reports progress and previews."""
    def callback(step, x0, x, total_steps):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0
        if running():
            sync(eng.progress.update(step + 1, total_steps))
    callback.x0_output = x0_output_dict
    return callback


def soft_empty_cache(*args, **kwargs):
    if running():
        sync(eng.memory.empty_cache())


def get_free_memory(device=None, *args, **kwargs):
    if not running():
        return 0
    return sync(eng.memory.free_bytes(str(device) if device is not None else None))


class InterruptProcessingException(Exception):
    pass


def throw_exception_if_processing_interrupted():
    if running():
        sync(eng.execution.check_interrupted())


def processing_interrupted():
    if not running():
        return False
    try:
        sync(eng.execution.check_interrupted())
    except Exception:
        return True
    return False


def get_filename_list(folder_name):
    if not running():
        # Native host/import-time behavior remains native; missing support fails explicitly.
        from comfy_api.v4 import util
        return util.folders.get_filename_list(folder_name)
    return sync(eng.models.list(folder_name))


def send_sync(event, data, sid=None):
    if running():
        sync(eng.ui.send(event, data))


class _Args:
    def __getattr__(self, name):
        return sync(eng.settings.get(name)) if running() else None


args = _Args()


class ProgressBar:
    def __init__(self, total, node_id=None):
        self.total, self.current = total, 0

    def update_absolute(self, value, total=None, preview=None):
        if total is not None:
            self.total = total
        self.current = value
        if running():
            sync(eng.progress.update(value, self.total))

    def update(self, value):
        self.update_absolute(self.current + value)


class _IO(str):
    """``comfy.comfy_types.IO``: the type names as strings (``IO.INT`` is ``"INT"``, ``IO.ANY`` is ``"*"``)."""

    def __getattr__(self, name):
        return "*" if name == "ANY" else name


IO = _IO()


def _plain_ui(saved, images=None):
    """ComfyUI's savers return the ui and pass the images through as the node's result (``"result": (images,)``):
    a pack subclassing SaveImage / PreviewImage gets the same output (SaveImageWithoutMetadata lost it)."""
    out = {"ui": saved.as_dict()} if hasattr(saved, "as_dict") else {"ui": saved}
    if images is not None:
        out["result"] = (images,)
    return out


def vae_list():
    """``nodes.VAELoader.vae_list``: the names ComfyUI offers for a VAE, from the model folders (no engine call)."""
    from comfy_api.v4 import util
    video_taes = ["taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5", "taeltx_2", "taeh3"]
    image_taes = ["taesd", "taesdxl", "taesd3", "taef1", "taef2", "taeqi2_1"]
    vaes = list(util.folders.get_filename_list("vae"))
    have_encoder, have_decoder = set(), set()
    for name in util.folders.get_filename_list("vae_approx"):
        parts = name.rsplit("_", 1)
        if len(parts) != 2 or parts[0] not in image_taes:
            vaes += [name] if any(name.startswith(tae) for tae in video_taes) else []
            continue
        if parts[1].startswith("encoder."):
            have_encoder.add(parts[0])
        elif parts[1].startswith("decoder."):
            have_decoder.add(parts[0])
    return vaes + [k for k in have_decoder if k in have_encoder] + ["pixel_space"]


def _hidden_as_cls(prompt, extra_pnginfo):
    """What ``ImageSaveHelper`` reads the PNG metadata from (``cls.hidden.prompt`` / ``.extra_pnginfo``)."""
    import types
    if isinstance(prompt, list):  # an INPUT_IS_LIST node's hidden inputs are lists
        prompt = prompt[0] if prompt else None
    if isinstance(extra_pnginfo, list):
        extra_pnginfo = extra_pnginfo[0] if extra_pnginfo else None
    return types.SimpleNamespace(hidden=types.SimpleNamespace(prompt=prompt, extra_pnginfo=extra_pnginfo))


class SaveImage:
    """``nodes.SaveImage``: the same attributes (``type``, ``prefix_append``, ``compress_level``), which subclasses
    set or change, and the prompt and workflow written into the PNG (Danny r7: PromptLoop's PNGs had none)."""

    type = "output"
    prefix_append = ""
    compress_level = 4

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        from comfy_api.v4 import io, ui
        folder = io.FolderType.temp if str(getattr(self, "type", "output")) == "temp" else io.FolderType.output
        saved = ui.ImageSaveHelper.save_images(
            images, filename_prefix=filename_prefix + (getattr(self, "prefix_append", "") or ""), folder_type=folder,
            cls=_hidden_as_cls(prompt, extra_pnginfo), compress_level=getattr(self, "compress_level", 4))
        return _plain_ui(ui.SavedImages(saved), images)


class PreviewImage(SaveImage):
    """``nodes.PreviewImage``: a SaveImage into the temp folder (Danny r7: MaskPreview's went to output)."""

    type = "temp"
    prefix_append = "_temp_"
    compress_level = 1

    def __init__(self):
        import random
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
        self.compress_level = 1


def _output(result):
    if isinstance(result, io.NodeOutput):
        return result
    if isinstance(result, dict):
        return io.NodeOutput(*(result.get("result") or ()), ui=result.get("ui"), expand=result.get("expand"))
    if result is None:
        return io.NodeOutput()
    return io.NodeOutput(*tuple(result))


async def run_v1(v1_class, function, cls, inputs, hidden_args):
    """Run the original V1 method on a worker thread, its engine calls routed back to this loop."""
    hidden = getattr(cls, "hidden", None)
    # ComfyUI hands an INPUT_IS_LIST node every input as a list, hidden ones too (ShowText's extra_pnginfo[0])
    as_list = bool(getattr(v1_class, "INPUT_IS_LIST", False))
    for arg, field in hidden_args.items():
        value = getattr(hidden, field, None) if hidden is not None else None
        inputs[arg] = [value] if as_list else value
    def invoke():
        import inspect
        method = getattr(v1_class(), function)
        result = method(**inputs)
        if inspect.isawaitable(result):
            async def finish():
                return await result
            return asyncio.run(finish())
        return result

    token = _LOOP.set(asyncio.get_running_loop())
    try:
        result = await asyncio.to_thread(invoke)
    finally:
        _LOOP.reset(token)
    return _output(result)


async def run_v3(execute, cls, *args, **kwargs):
    """Run a V3 node's sync ``execute`` on a worker thread, its engine calls routed back to this loop."""
    token = _LOOP.set(asyncio.get_running_loop())
    try:
        return await asyncio.to_thread(execute, cls, *args, **kwargs)
    finally:
        _LOOP.reset(token)
'''


ROUTES_MODULE = """\"\"\"Generated by the V2 converter: the pack's endpoints as V2 routes.

The pack's code registers aiohttp handlers (on the table that was PromptServer.instance.routes);
ComfyUI serves them at /ext/<namespace>/<path> (comfy-pack.json lists them), and ``handle`` answers
each request with the handler the pack registered for its path. V2 routes take and give JSON or text.
\"\"\"
import asyncio
import inspect
import json
from urllib.parse import parse_qsl

from . import _v2_engine as _v2e


class _MultiDict(dict):
    \"\"\"Query and form values, as aiohttp's MultiDict gives them (one value per key).\"\"\"

    def getone(self, key, default=None):
        return self.get(key, default)

    def getall(self, key, default=None):
        return [self[key]] if key in self else (default if default is not None else [])


class _URL:
    def __init__(self, path, query):
        self.path, self.query = path, query


class Request:
    \"\"\"What an aiohttp handler reads from a request: method, path, query, and the body as JSON or text.\"\"\"

    def __init__(self, message):
        self.method = message["method"].upper()
        self.path = message["path"]
        self.query = _MultiDict(message.get("query") or {})
        self.rel_url = _URL(self.path, self.query)
        self.match_info = {}
        self.headers = {}
        self._body = message.get("body") or ""
        self.can_read_body = self.body_exists = bool(self._body)
        self.content_type = "application/json" if self._body.lstrip()[:1] in ("{", "[") else "text/plain"

    async def json(self, *, loads=json.loads):
        return loads(self._body) if self._body else None

    async def text(self):
        return self._body

    async def read(self):
        return self._body.encode("utf-8")

    async def post(self):
        return _MultiDict(parse_qsl(self._body))


def _reply(response):
    from aiohttp import web
    if isinstance(response, dict):
        return {"status": 200, "body": json.dumps(response), "content_type": "application/json"}
    if not isinstance(response, web.Response):
        raise TypeError(f"a V2 route answers JSON or text, not {type(response).__name__}")
    content_type = "application/json" if response.content_type == "application/json" else "text/plain"
    return {"status": response.status, "body": response.text or "", "content_type": content_type}


async def handle(message):
    \"\"\"One request to one of the pack's routes.\"\"\"
    from aiohttp import web
    method = message["method"].upper()
    for route in _v2e.routes:
        if getattr(route, "path", None) == message["path"] and getattr(route, "method", "").upper() in (method, "*"):
            def invoke():
                result = route.handler(Request(message))
                if inspect.isawaitable(result):
                    async def finish():
                        return await result
                    return asyncio.run(finish())
                return result

            token = _v2e._LOOP.set(asyncio.get_running_loop())
            try:
                return _reply(await asyncio.to_thread(invoke))
            except web.HTTPException as error:
                return {"status": error.status, "body": error.text or "", "content_type": "text/plain"}
            finally:
                _v2e._LOOP.reset(token)
    return {"status": 404, "body": json.dumps({"error": "no such route"}), "content_type": "application/json"}
"""
ROUTE_METHODS = ("get", "post", "put", "patch", "delete")
FRONTEND_SUFFIXES = (".js", ".mjs", ".ts", ".vue", ".html", ".jsx", ".tsx")


def route_called(v2, path):
    """Whether the pack's own code calls a route: its frontend mentions the path, or its Python does other than
    as the exact quoted path of the route's declaration (``@routes.get("/x")``), as in
    ``requests.post("http://localhost:8188/x")``. No caller in the pack: an outside app or script calls it."""
    for root, _dirs, files in os.walk(v2):
        for file_name in files:
            frontend = file_name.endswith(FRONTEND_SUFFIXES)
            if not (frontend or file_name.endswith(".py")) or file_name.startswith("_v2_"):
                continue
            try:
                text = open(os.path.join(root, file_name), encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                continue
            if not frontend:
                text = text.replace(f'"{path}"', "").replace(f"'{path}'", "")
            if path in text:
                return True
    return False


def _normal_name(name):
    """The pack's permanent name, as ``comfy_api.latest.registration.pack_name`` makes it."""
    return re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-")[:100] or "pack"


def pack_namespace(canonical_id):
    """Where the pack's routes live, as ``comfy_api.latest.registration.pack_namespace`` says: /ext/<namespace>/."""
    import hashlib
    slug = re.sub(r"[^a-z0-9]+", "-", canonical_id.lower()).strip("-")[:32].strip("-") or "pack"
    return f"{slug}-{hashlib.sha256(canonical_id.encode('utf-8')).hexdigest()[:8]}"


def convert_routes(pack_dir, pack_out, v2, describe, report):
    """The pack's routes as V2 routes: declared in pyproject.toml (served at /ext/<namespace>/<path>),
    answered by _v2_routes.handle; the pack's frontend URLs rewritten to match. What can't move is
    reported."""
    import tomllib
    pyproject = os.path.join(pack_dir, "pyproject.toml")
    text = open(pyproject, encoding="utf-8").read() if os.path.exists(pyproject) else ""
    try:
        project_name = (tomllib.loads(text).get("project") or {}).get("name") if text else None
    except Exception:
        project_name, text = None, ""
    name = _normal_name(project_name or os.path.basename(pack_out))
    namespace = pack_namespace(name)
    report["manifest_name"], report["namespace"] = name, namespace
    declared, unmoved = [], []
    for method, path in describe.get("routes") or []:
        method = method.lower()
        if method == "head":
            continue
        if method not in ROUTE_METHODS or any(c in path for c in "{}?#") or not path.startswith("/"):
            unmoved.append(f"{method.upper()} {path}: a V2 route is a literal path (get, post, put, patch, delete)")
            continue
        declared.append((method, path))
    if declared:
        with open(os.path.join(v2, "_v2_routes.py"), "w") as file:
            file.write(ROUTES_MODULE)
        text = text.rstrip() + "\n" if text else f'[project]\nname = "{name}"\n'
        for method, path in declared:
            text += f'\n[[tool.comfy.routes]]\nmethod = "{method}"\npath = {json.dumps(path)}\nmodule = "_v2_routes"\nhandler = "handle"\n'
    if text:
        with open(os.path.join(pack_out, "pyproject.toml"), "w", encoding="utf-8") as file:
            file.write(text)
        if declared:
            # Registration reads v2/ first; the parent copy is retained for existing consumers.
            with open(os.path.join(v2, "pyproject.toml"), "w", encoding="utf-8") as file:
                file.write(text)
    # the frontend calls them at /ext/<namespace>/<path> now
    paths = sorted({p for _, p in declared}, key=len, reverse=True)
    found = {p: 0 for p in paths}
    # a route declared as /api/x the frontend calls as ``api.fetchApi("/x")`` (fetchApi adds /api): the same route
    via_api = {p[len("/api"):]: p for p in paths if p.startswith("/api/") and p[len("/api"):] not in found}
    api_pattern = re.compile(r"""(fetchApi\(\s*)(["'`])(""" + "|".join(re.escape(p) for p in sorted(via_api, key=len, reverse=True))
                             + r""")(?=["'`?])""") if via_api else None
    if paths:
        pattern = re.compile(r"""(["'`])(""" + "|".join(re.escape(p) for p in paths) + r""")(?=["'`?])""")
        for root, _dirs, files in os.walk(v2):
            for file_name in files:
                if not file_name.endswith(FRONTEND_SUFFIXES):
                    continue
                path = os.path.join(root, file_name)
                try:
                    source = open(path, encoding="utf-8").read()
                except (UnicodeDecodeError, OSError):
                    continue

                def moved(match):
                    found[match.group(2)] += 1
                    return f"{match.group(1)}/ext/{namespace}{match.group(2)}"
                new = pattern.sub(moved, source)
                # A literal route-family prefix in a template wrapper, e.g.
                # `/fal_api${path}`. Preserve the dynamic suffix verbatim.
                prefixes = {p.rsplit("/", 1)[0] for p in paths if "/" in p[1:]}
                template_pattern = re.compile(r"`(" + "|".join(re.escape(p) for p in sorted(prefixes, key=len, reverse=True)) + r")(?=\$\{)") if prefixes else None
                if template_pattern is not None:
                    def moved_template(match):
                        prefix = match.group(1)
                        for route in paths:
                            if route.startswith(prefix + "/"):
                                found[route] += 1
                        return "`/ext/" + namespace + prefix
                    new = template_pattern.sub(moved_template, new)

                if api_pattern is not None:
                    def moved_api(match):
                        full = via_api[match.group(3)]
                        found[full] += 1
                        return f"{match.group(1)}{match.group(2)}/ext/{namespace}{full}"
                    new = api_pattern.sub(moved_api, new)
                if new != source:
                    open(path, "w", encoding="utf-8").write(new)
    missing = [p for p, n in found.items() if n == 0]
    # a route nothing in the pack calls (an outside app or script does, or nothing): a note, not a manual item
    uncalled = [p for p in missing if not route_called(v2, p)]
    report["routes"] = {"declared": [f"{m.upper()} {p}" for m, p in declared], "not_moved": unmoved,
                        "not_in_frontend": [p for p in missing if p not in uncalled], "uncalled": uncalled}
    # handlers that answer with files or streams: V2 routes answer JSON or text
    streams = []
    for root, _dirs, files in os.walk(v2):
        for file_name in files:
            if file_name.endswith(".py") and not file_name.startswith("_v2_"):
                source = open(os.path.join(root, file_name), encoding="utf-8", errors="ignore").read()
                if re.search(r"\b(FileResponse|StreamResponse|WebSocketResponse)\b", source):
                    streams.append(os.path.relpath(os.path.join(root, file_name), v2))
    if declared and streams:
        report["routes"]["files_or_streams"] = streams


def _v1_fallback(pack_dir, nodes):
    """Code for ``_v2_nodes.py`` finding V1 classes a pack lists only through its V3 entry point
    (sd-perturbed-attention: ``v3_schema_stub(tpg_nodes)``, no NODE_CLASS_MAPPINGS in its __init__): from the
    module describe found each one in. Empty for every pack whose __init__ has the table, so their code is as it was."""
    try:
        init = open(os.path.join(pack_dir, "__init__.py"), encoding="utf-8", errors="replace").read()
    except OSError:
        init = ""
    if "NODE_CLASS_MAPPINGS" in init:
        return ""
    found = {}
    for node_id, entry in sorted(nodes.items()):
        module, qualname = entry.get("module") or "", entry.get("qualname") or ""
        if entry.get("kind") == "v1" and qualname and (module == "/pack" or module.startswith("/pack.")):
            found[node_id] = ("." + module[len("/pack."):] if module != "/pack" else ".", qualname)
    if not found:
        return ""
    return ("\n# V1 classes the pack lists only through its V3 entry point: from their own modules\n"
            "import importlib as _v2_importlib\n"
            f"for _v2_id, (_v2_module, _v2_name) in {found!r}.items():\n"
            "    if _v2_id not in _V1:\n"
            "        _v2_obj = _v2_importlib.import_module(_v2_module, __package__)\n"
            "        for _v2_part in _v2_name.split('.'):\n"
            "            _v2_obj = getattr(_v2_obj, _v2_part)\n"
            "        _V1[_v2_id] = _v2_obj\n")


def shim_text():
    """The generated ``_v2_engine.py``: the same for every pack."""
    return (ENGINE_SHIM + f"\n\n# ComfyUI's own model folders (the pack's others are reported by pack_model_folders)\n"
            f"_COMFY_FOLDERS = frozenset({COMFY_FOLDERS!r})\n" + _pure_shim())


ENTRY_V1_TABLE = "\n\nNODE_CLASS_MAPPINGS = None\nNODE_DISPLAY_NAME_MAPPINGS = None\n"
_OLD_ENTRY_TEXT = (
    ("\n\nNODE_CLASS_MAPPINGS = {}\nNODE_DISPLAY_NAME_MAPPINGS = {}\n", ENTRY_V1_TABLE),
    ("# Added by the V2 converter: the pack's V2 entry point. Its V1 table is cleared after, so\n"
     "# ComfyUI lists the V2 nodes only.\n",
     "# Added by the V2 converter: the pack's V2 entry point. Its V1 table is set to None after,\n"
     "# so ComfyUI lists the V2 nodes only: ComfyUI's loader reads a V1 table that isn't None and\n"
     "# then never calls comfy_entrypoint (an empty {} registered no node in process, 2026-11-07).\n"),
)


def entry_updated(text):
    """An entry written by an older converter, with its V1-table lines as this one writes them."""
    for old, new in _OLD_ENTRY_TEXT:
        text = text.replace(old, new)
    return text


# files a pack ships beside its nodes: its tests, its install and example scripts
_SCRIPT_FILE = re.compile(r"(^|/)(test_[^/]*|[^/]*_test|conftest|setup|install|uninstall|benchmark[^/]*|example[^/]*|demo[^/]*)\.py$"
                          r"|(^|/)(tests?|examples?|scripts?|tools|docs|benchmarks?|standalone|dev|demo)/")
_DYNAMIC_IMPORTS = {"import_module", "__import__", "spec_from_file_location", "exec_module", "load_module",
                    "walk_packages", "iter_modules"}


def _not_loaded(sources, describe):
    """The pack's files its nodes never load: scripts and tests that nothing reachable from ``__init__.py``,
    ``prestartup_script.py`` or a node class's module imports. They stay as the pack has them (ComfyUI never
    runs them), and aren't engine use to convert. A pack that imports by computed name gets no such list:
    what it loads can't be told from its source."""
    def module_file(parts):
        stem = "/".join(parts)
        for candidate in (stem + "/__init__.py", stem + ".py"):
            if candidate in sources:
                return candidate
        return None

    roots = {r for r in ("__init__.py", "prestartup_script.py") if r in sources}
    for entry in (describe.get("nodes") or {}).values():
        # where describe found the node's class: "/pack.sub.module" (the pack is imported as /pack)
        parts = [x for x in (entry.get("module") or "").split(".")[1:] if x]
        found = module_file(parts) if parts else "__init__.py"
        if found is None:
            return set()  # a node defined where this can't follow (a generated module): nothing is left out
        roots.add(found)
    seen, todo = set(), list(roots)
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        try:
            tree = ast.parse(sources[rel], rel)
        except SyntaxError:
            return set()
        base = [x for x in os.path.dirname(rel).split("/") if x]
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.ImportFrom):
                module = node.module.split(".") if node.module else []
                starts = [base[:len(base) - (node.level - 1)]] if node.level else [[], base]
                for start in starts:
                    targets += [start + module] + [start + module + [a.name] for a in node.names]
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    targets += [alias.name.split("."), base + alias.name.split(".")]
            elif isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                if name in _DYNAMIC_IMPORTS:
                    return set()
            for parts in targets:
                for end in range(len(parts), 0, -1):
                    found = module_file(parts[:end])
                    if found:
                        todo += [found] + [f for f in (module_file(parts[:i]) for i in range(1, end)) if f]
                        break
    return {rel for rel, source in sources.items()
            if rel not in seen and (_SCRIPT_FILE.search(rel) or re.search(r"^if __name__ == ['\"]__main__['\"]", source, re.M))}


def generate(pack_dir, describe, out_root, *, code_only=False):
    """Convert a captured pack without executing its code.

    ``code_only`` is for static rescan comparisons only. Such a tree omits
    resources and must never be loaded or published as a complete conversion.
    """
    name = os.path.basename(os.path.normpath(pack_dir))
    pack_out = os.path.join(out_root, name)
    v2 = os.path.join(pack_out, "v2")
    shutil.rmtree(pack_out, ignore_errors=True)
    report = {"pack": name, "rewritten": {}, "leftovers": {}, "nodes": {}, "errors": []}
    def ignored(directory, names):
        return [n for n in names if n in SKIP_DIRS or n.endswith(".pyc") or n.startswith(".fetched")
                or (code_only and not n.endswith((".py", ".cjs") + FRONTEND_SUFFIXES)
                    and not os.path.isdir(os.path.join(directory, n)))]

    shutil.copytree(pack_dir, v2, ignore=ignored)
    with open(os.path.join(pack_out, "__init__.py"), "w") as file:
        file.write('"""Converted pack: its V2 package is v2/ (pack isolation loads it from there)."""\n')
    nodes = describe.get("nodes", {})
    typed = {}
    for node_id, entry in nodes.items():
        # V3 nodes (io.ComfyNode) too: their MODEL / CLIP / VAE inputs are Refs in V2.1 just the same (skimmed_cfg's
        # ``model.clone()`` in ``execute`` went through unrewritten and unflagged)
        if entry.get("kind") == "v3" and not entry.get("function"):
            entry = dict(entry, function="execute")
        if entry.get("kind") in ("v1", "v3") and entry.get("function") and (entry.get("info") or {}).get("input"):
            types = {}
            for section in ("required", "optional"):
                for arg, spec in (entry["info"]["input"].get(section) or {}).items():
                    io_type = spec[0] if isinstance(spec, list) and spec else spec
                    if isinstance(io_type, str):
                        types[arg] = io_type
            typed[entry["class"].split(".")[-1]] = (entry["function"], types)
    typed[OWNED_TYPES_KEY] = (None, owned_types(nodes))
    sources = {}
    for root, dirs, files in os.walk(v2):
        for file_name in files:
            if file_name.endswith(".py"):
                relative = os.path.relpath(os.path.join(root, file_name), v2).replace(os.sep, "/")
                try:
                    sources[relative] = open(os.path.join(root, file_name), encoding="utf-8-sig").read()  # Python accepts a BOM
                except (UnicodeDecodeError, OSError) as error:
                    report["errors"].append(f"{relative}: can't read: {error}")
    # scripts and tests the nodes never load stay as they are
    not_loaded = _not_loaded(sources, describe)
    if not_loaded:
        report["not_loaded"] = sorted(not_loaded)
        sources = {rel: src for rel, src in sources.items() if rel not in not_loaded}
    # ``from X import *`` between the pack's own modules hands engine names on: follow it
    exported = {rel: module_engine_aliases(src, rel) for rel, src in sources.items()}
    named_exported = {rel: module_engine_aliases(src, rel, include_private=True) for rel, src in sources.items()}

    def target(importer, level, module):
        base = os.path.dirname(importer)
        if level:
            for _ in range(level - 1):
                base = os.path.dirname(base)
            roots = [base]
        else:
            roots = ["", base]  # top level (the pack on sys.path), or beside the importer
        for root in roots:
            stem = "/".join(x for x in [root] + (module.split(".") if module else []) if x)
            # Python selects the package before a same-named module file.
            for candidate in (stem + "/__init__.py", stem + ".py"):
                if candidate in sources:
                    return candidate
        return None
    # Recognize a narrowly bounded cross-module metadata export. A facade is
    # safe only for known, ordinary IO strings in a class RETURN_TYPES value;
    # enum protocols, dynamic lookup and other uses remain explicit blockers.
    io_strings = frozenset('STRING IMAGE MASK LATENT BOOLEAN INT FLOAT COMBO CONDITIONING SAMPLER SIGMAS GUIDER NOISE CLIP CONTROL_NET VAE MODEL LORA_MODEL LOSS_MAP CLIP_VISION CLIP_VISION_OUTPUT STYLE_MODEL GLIGEN UPSCALE_MODEL AUDIO WEBCAM POINT FACE_ANALYSIS BBOX SEGS VIDEO'.split())
    trees = {}
    for rel, src in sources.items():
        try:
            trees[rel] = ast.parse(src, rel)
        except SyntaxError:
            pass
    io_exports, conflicted_exports, local_io_blockers = {}, {}, {}
    for rel, tree in trees.items():
        parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in tree.body:
            if (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == 'comfy.comfy_types'
                    and len(node.names) == 1 and node.names[0].name == 'IO' and node.lineno == node.end_lineno):
                name = node.names[0].asname or 'IO'
                conflict = any(isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, (ast.Store, ast.Del))
                               or isinstance(n, ast.arg) and n.arg == name
                               or isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name
                               or isinstance(n, (ast.Import, ast.ImportFrom)) and n is not node and any(
                                   (a.asname or a.name.split('.')[0]) == name for a in n.names)
                               for n in ast.walk(tree))
                io_exports.setdefault(rel, set()).add(name)
                # Do not silently turn enum operations within the exporting
                # module into string-facade operations either.
                for read in ast.walk(tree):
                    if not isinstance(read, ast.Name) or read.id != name or not isinstance(read.ctx, ast.Load):
                        continue
                    member = parents.get(id(read))
                    use = parents.get(id(member))
                    metadata = isinstance(use, (ast.Tuple, ast.List))
                    cursor = use
                    while metadata and id(cursor) in parents and not isinstance(cursor, (ast.Assign, ast.Return)):
                        cursor = parents[id(cursor)]
                        if not isinstance(cursor, (ast.Tuple, ast.List, ast.Dict, ast.Assign, ast.Return)):
                            metadata = False
                    if metadata and isinstance(cursor, ast.Assign):
                        metadata = (isinstance(parents.get(id(cursor)), ast.ClassDef) and any(
                            isinstance(t, ast.Name) and t.id == 'RETURN_TYPES' for t in cursor.targets))
                    elif metadata and isinstance(cursor, ast.Return):
                        scope = parents.get(id(cursor))
                        while scope is not None and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            scope = parents.get(id(scope))
                        metadata = isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) and scope.name == 'INPUT_TYPES'
                    else:
                        metadata = False
                    if not (isinstance(member, ast.Attribute) and member.value is read and member.attr in io_strings
                            and isinstance(member.ctx, ast.Load) and metadata):
                        conflict = True
                        local_io_blockers.setdefault(rel, set()).add(f'comfy.comfy_types.IO (unproven local enum use: {name})')
                if conflict:
                    conflicted_exports.setdefault(rel, set()).add(name)
    facade_exports, io_blockers = {}, local_io_blockers
    for rel, tree in trees.items():
        parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for imp in tree.body:
            if not isinstance(imp, ast.ImportFrom) or not imp.level:
                continue
            for alias in imp.names:
                if alias.name == '*':
                    continue
                imported = '.'.join(x for x in (imp.module, alias.name) if x)
                dest = target(rel, imp.level, imported)
                if not dest or dest not in io_exports:
                    continue
                local = alias.asname or alias.name
                ambiguous = any(isinstance(n, ast.Name) and n.id == local and isinstance(n.ctx, (ast.Store, ast.Del))
                                or isinstance(n, ast.arg) and n.arg == local
                                or isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == local
                                or isinstance(n, (ast.Import, ast.ImportFrom)) and n is not imp and any(
                                    (a.asname or a.name.split('.')[0]) == local for a in n.names)
                                for n in ast.walk(tree))
                for name in io_exports[dest]:
                    uses = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == name
                            and isinstance(n.value, ast.Name) and n.value.id == local]
                    dynamic = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'getattr'
                                  and n.args and isinstance(n.args[0], ast.Name) and n.args[0].id == local
                                  and (len(n.args) < 2 or not isinstance(n.args[1], ast.Constant) or n.args[1].value == name)
                                  for n in ast.walk(tree))
                    if dynamic:
                        io_blockers.setdefault(rel, set()).add(f'comfy.comfy_types.IO (dynamic pack module export: {local}.{name})')
                    if not uses:
                        continue
                    safe = not ambiguous and not dynamic and name not in conflicted_exports.get(dest, ())
                    for use in uses:
                        member = parents.get(id(use))
                        if not (isinstance(member, ast.Attribute) and member.value is use and member.attr in io_strings
                                and isinstance(member.ctx, ast.Load)):
                            safe = False
                            continue
                        ancestor = member
                        while id(ancestor) in parents and not isinstance(ancestor, (ast.Assign, ast.AnnAssign, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            ancestor = parents[id(ancestor)]
                            if isinstance(ancestor, ast.Call):
                                if (not isinstance(ancestor.func, ast.Name) or ancestor.func.id != 'tuple'
                                        or _binds_name(tree, 'tuple')):
                                    safe = False
                            elif not isinstance(ancestor, (ast.List, ast.Tuple, ast.BinOp, ast.Assign, ast.AnnAssign)):
                                safe = False
                        targets = ancestor.targets if isinstance(ancestor, ast.Assign) else [ancestor.target] if isinstance(ancestor, ast.AnnAssign) else []
                        if not any(isinstance(t, ast.Name) and t.id == 'RETURN_TYPES' for t in targets) or not isinstance(parents.get(id(ancestor)), ast.ClassDef):
                            safe = False
                    if safe:
                        facade_exports.setdefault(dest, set()).add(name)
                    else:
                        io_blockers.setdefault(rel, set()).add(f'comfy.comfy_types.IO (unproven pack module export: {local}.{name})')
    # Named and star consumers may rely on the original enum protocol. Until
    # those paths are proven safe, do not substitute a facade for that export.
    blocked_exports = {}
    for rel, tree in trees.items():
        for imp in ast.walk(tree):
            if not isinstance(imp, ast.ImportFrom) or not imp.level:
                continue
            dest = target(rel, imp.level, imp.module or '')
            if dest not in facade_exports:
                continue
            requested = {a.name for a in imp.names}
            blocked = facade_exports[dest] if '*' in requested else facade_exports[dest] & requested
            if blocked:
                for name in set(blocked):
                    io_blockers.setdefault(rel, set()).add(f'comfy.comfy_types.IO (unproven named or star import: {name})')
                blocked_exports.setdefault(dest, set()).update(blocked)
    for dest, names in blocked_exports.items():
        facade_exports[dest] -= names
    star_from = {rel: [t for t in (target(rel, level, mod) for level, mod in star_imports(src, rel)) if t]
                 for rel, src in sources.items()}
    star_aliases = {}
    for rel, targets in star_from.items():
        merged = {}
        for t in targets:
            for name, dotted in exported.get(t, {}).items():
                merged.setdefault(name, dotted)
        star_aliases[rel] = merged
    # ``from .common import folder_paths``: an engine name handed on by name, as ``import *`` does
    named_from = {}
    for rel, src in sources.items():
        for level, mod, names in named_imports(src, rel):
            t = target(rel, level, mod)
            if t and t != rel:
                handed = [n for n in names if n in named_exported.get(t, {})]
                if handed:
                    named_from.setdefault(rel, []).append((t, handed))
                    for name in handed:
                        star_aliases.setdefault(rel, {}).setdefault(name, named_exported[t][name])
    # the pack's probes of ComfyUI, and its variables that hold ComfyUI's modules, resolved for this ComfyUI
    # (normalize.py). A module-level variable another module of the pack may read stays as it is.
    read_elsewhere = {a.name for tree in trees.values() for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names} \
        | {n.attr for tree in trees.values() for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    star_targets = {t for targets in star_from.values() for t in targets}
    originals, probes = dict(sources), {}
    for rel in list(sources):
        if rel in trees:
            sources[rel], probes[rel] = normalize.resolve_probes(
                sources[rel], rel, COMFY_INDEX, _Module, star_aliases.get(rel),
                None if rel in star_targets else frozenset(read_elsewhere))
    # engine-object types that reach a module's functions from the pack's other modules
    resolved_trees = {}
    for rel, src in sources.items():
        try:
            resolved_trees[rel] = ast.parse(src, rel)
        except SyntaxError:
            pass
    handed = _pack_types(resolved_trees, typed, target)

    def typed_for(rel):
        return {**typed, HELPER_TYPES_KEY: (None, handed[rel])} if rel in handed else typed

    results, keep = {}, {}
    for rel, src in sources.items():
        try:
            results[rel] = rewrite(src, rel, typed_for(rel), star_aliases.get(rel), facade_exports=frozenset(facade_exports.get(rel, ())))
            if rel in io_blockers:
                new, done, left = results[rel]
                results[rel] = new, done, sorted(set(left) | io_blockers[rel])
        except SyntaxError as error:
            report["errors"].append(f"{rel}: can't parse: {error}")
        except Exception as error:
            report["errors"].append(f"{rel}: rewrite failed: {type(error).__name__}: {error}")
    for rel, pairs in named_from.items():  # the importer drops names it no longer uses; the rest stay where they come from
        if rel not in results:
            continue
        new, done, left = results[rel]
        try:
            loaded = {n.id for n in ast.walk(ast.parse(new)) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        except SyntaxError:
            continue
        unused = {name for _t, names in pairs for name in names if name not in loaded}
        for t, names in pairs:
            keep.setdefault(t, set()).update(n for n in names if n in loaded)
        if unused:
            results[rel] = (_drop_pack_imports(new, rel, unused), done, left)
    for rel, targets in star_from.items():  # names an importer still uses stay imported where it gets them
        if rel not in results or not targets:
            continue
        try:
            loaded = {n.id for n in ast.walk(ast.parse(results[rel][0])) if isinstance(n, ast.Name)}
        except SyntaxError:
            continue
        for t in targets:
            keep.setdefault(t, set()).update(loaded & set(exported.get(t, {})))
    for rel, names in keep.items():
        if names and rel in results:
            results[rel] = rewrite(sources[rel], rel, typed_for(rel), star_aliases.get(rel), frozenset(names), frozenset(facade_exports.get(rel, ())))
            if rel in io_blockers:
                new, done, left = results[rel]
                results[rel] = new, done, sorted(set(left) | io_blockers[rel])
    for rel in sources:
        if rel in results:
            new, done, left = results[rel]
            results[rel] = (_explicit_submodules(new, rel), done, left)
    for rel, (new, done, left) in results.items():
        done = sorted(set(done) | set(probes.get(rel, ())))
        if new != originals[rel]:
            open(os.path.join(v2, rel), "w", encoding="utf-8").write(new)
        if done:
            report["rewritten"][rel] = done
        if left:
            report["leftovers"][rel] = left
    with open(os.path.join(v2, "_v2_engine.py"), "w") as file:
        file.write(shim_text())
    classes, sources = [], []
    for node_id, entry in sorted(nodes.items()):
        if entry.get("kind") != "v1":
            report["nodes"][node_id] = entry.get("kind")
            continue
        try:
            class_name, source = node_source(node_id, entry)
            classes.append(class_name)
            sources.append(source)
            report["nodes"][node_id] = "generated"
        except Exception as error:
            report["nodes"][node_id] = f"not generated: {error}"
    v3_ids = sorted(n for n, e in nodes.items() if e.get("kind") == "v3")
    for node_id in v3_ids:
        report["nodes"][node_id] = "v3: kept, as an ENGINE_API subclass"
    with open(os.path.join(v2, "_v2_nodes.py"), "w") as file:
        file.write('"""Generated by the V2 converter: the pack\'s V1 nodes as V2 nodes."""\n'
                   "from comfy_api.v4 import io, util  # noqa: F401\n"
                   "import sys\n\n"
                   "from . import _v2_engine as _v2e\n\n"
                   "# the pack's V1 classes, as its __init__ defined them (the converter's entry clears its table after)\n"
                   "_V1 = dict(getattr(sys.modules[__package__], 'NODE_CLASS_MAPPINGS', None) or {})\n"
                   + _v1_fallback(pack_dir, nodes) + "\n\n" + "\n\n\n".join(sources) +
                   f"\n\n\nNODES = [{', '.join(classes)}]\n")
    convert_routes(pack_dir, pack_out, v2, describe, report)
    web = describe.get("web_directory")
    init = os.path.join(v2, "__init__.py")
    original_init = open(init, encoding="utf-8").read() if os.path.exists(init) else ""
    with open(init, "w", encoding="utf-8") as file:  # the pack's own __init__ runs first, as the package
        file.write(original_init.rstrip() + "\n\n\n"
                   "# --------------------------------------------------------------------------------------\n"
                   "# Added by the V2 converter: the pack's V2 entry point. Its V1 table is set to None after,\n"
                   "# so ComfyUI lists the V2 nodes only: ComfyUI's loader reads a V1 table that isn't None and\n"
                   "# then never calls comfy_entrypoint (an empty {} registered no node in process, 2026-11-07).\n"
                   "# --------------------------------------------------------------------------------------\n"
                   "from comfy_api.v4 import ComfyExtension as _V2ComfyExtension  # noqa: E402\n"
                   "_V2_ORIGINAL_ENTRY = globals().get('comfy_entrypoint')\n"
                   "_V2_ORIGINAL_MAPPINGS = dict(globals().get('NODE_CLASS_MAPPINGS') or {})\n"
                   "from . import _v2_nodes as _V2_node_module  # noqa: E402\n"
                   "_V2_node_module._V1.update(_V2_ORIGINAL_MAPPINGS)\n"
                   "_V2_NODES = _V2_node_module.NODES\n"
                   "from . import _v2_engine as _v2_shim  # noqa: E402\n\n"
                   f"_V2_V3_IDS = {v3_ids!r}  # nodes the pack already wrote for the V3 schema API\n\n\n"
                   + V3_ENTRY + ENTRY_V1_TABLE
                   + (f"WEB_DIRECTORY = {web!r}\n" if web else ""))
    return report


if __name__ == "__main__":
    pack_dir, describe_path, out_root = sys.argv[1:4]
    print(json.dumps(generate(os.path.realpath(pack_dir), json.load(open(describe_path)), out_root)))  # noqa: T201
