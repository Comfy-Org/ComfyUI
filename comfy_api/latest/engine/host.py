"""ComfyUI's own engine extensions for ``eng.device``, ``eng.memory``, ``eng.settings``, ``eng.progress``, ``eng.ui`` and ``eng.node``: the process a node runs in, and what it shows the user."""
import asyncio
import inspect
from comfy_api.latest import _runtime


class DeviceOps:
    """The devices and dtypes ComfyUI computes with."""

    def info(self):
        import comfy.model_management as mm

        return {
            "torch_device": str(mm.get_torch_device()),
            "intermediate_device": str(mm.intermediate_device()),
            "intermediate_dtype": str(mm.intermediate_dtype()).removeprefix("torch."),
            "unet_dtype": str(mm.unet_dtype()).removeprefix("torch."),
            "offload_device": str(mm.unet_offload_device()),
            "vae_device": str(mm.vae_device()),
            "vae_dtype": str(mm.vae_dtype()).removeprefix("torch."),
            "text_encoder_device": str(mm.text_encoder_device()),
            "vae_offload_device": str(mm.vae_offload_device()),
            "text_encoder_offload_device": str(mm.text_encoder_offload_device()),
        }


class MemoryOps:
    """Model memory."""

    def empty_cache(self):
        import comfy.model_management as mm

        mm.soft_empty_cache()
        return None

    def load_models(self, models):
        """Load models onto the GPU now, as samplers do before they run."""
        import comfy.model_management as mm

        mm.load_models_gpu([getattr(model, "patcher", model) for model in models])
        return None

    def unload_all(self):
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
        return None

    def free_bytes(self, device=None):
        import torch
        import comfy.model_management as mm

        return int(mm.get_free_memory(torch.device(device) if device else mm.get_torch_device()))


#: Readable settings: name -> how to read it. Nothing else is exposed.
_SETTINGS = {
    "disable_metadata": lambda args: bool(args.disable_metadata),
    "preview_method": lambda args: str(getattr(args.preview_method, "value", args.preview_method)),
    "preview_size": lambda args: int(args.preview_size),
    # --fast: the enabled optimizations' names, sorted (read-only: ComfyUI's own set isn't handed out)
    "fast": lambda args: sorted(str(getattr(f, "value", f)) for f in (args.fast or ())),
}


class SettingsOps:
    """ComfyUI settings a node may read."""

    def get(self, name):
        from comfy.cli_args import args

        reader = _SETTINGS.get(name)
        if reader is None:
            raise ValueError(f"unknown setting {name!r}; readable: {', '.join(sorted(_SETTINGS))}")
        return reader(args)


MAX_PREVIEW_RESOLUTION = 512


class ProgressOps:
    """The running node's progress bar."""

    def update(self, value, total, preview=None, format="JPEG"):
        import comfy.utils

        plan = _runtime.current_runtime().plan
        hook = comfy.utils.PROGRESS_BAR_HOOK
        if hook is None:
            return None
        image = None
        if preview is not None:
            import numpy as np
            from PIL import Image

            if format not in ("JPEG", "PNG"):
                raise ValueError("preview format must be JPEG or PNG")
            pixels = preview[0] if preview.ndim == 4 else preview
            array = (pixels.detach().float().clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
            image = (format, Image.fromarray(array), MAX_PREVIEW_RESOLUTION)
        hook(value, total, image, getattr(plan, "prompt_id", None), getattr(plan, "node_id", None))
        return None


class UiOps:
    """What the running node shows in the UI."""

    def progress_text(self, text):
        import server

        instance = getattr(server.PromptServer, "instance", None)
        plan = _runtime.current_runtime().plan
        node_id = getattr(plan, "node_id", None)
        if instance is not None and node_id is not None:
            instance.send_progress_text(str(text), node_id, getattr(instance, "client_id", None))
        return None

    def send(self, event, data):
        """A custom event to the client that queued the running prompt (``PromptServer.send_sync``)."""
        import server

        if not isinstance(event, str) or not event or len(event) > 128:
            raise ValueError("event is a short name")
        instance = getattr(server.PromptServer, "instance", None)
        if instance is not None:
            instance.send_sync(event, data, getattr(instance, "client_id", None))
        return None

    async def interact(self, kind, payload, reuse_last=False, remember=False, timeout=540.0):
        """Ask the user something and wait, through the interaction broker."""
        import server
        from comfy_api.latest import _runtime
        from comfy_api.latest.interactions import BROKER

        if not isinstance(reuse_last, bool) or not isinstance(remember, bool):
            raise TypeError("interaction cache switches must be booleans")
        rt = _runtime.current_runtime()
        instance = getattr(server.PromptServer, "instance", None)
        return await BROKER.request(
            kind=kind, payload=payload, tenant="", pack=(rt.pack or "pack").split(".")[0] or "pack",
            node_id=str(getattr(rt.plan, "node_id", "") or ""), sid=getattr(instance, "client_id", None),
            timeout=float(timeout), reuse_last=reuse_last, remember=remember)


def _hidden_values():
    """What a node's hidden inputs would hold if the prompt ran it directly."""
    rt = _runtime._active_runtime.get()
    plan = rt.plan if rt is not None else None
    dynprompt = getattr(plan, "dynamic_prompt", None)
    prompt = getattr(plan, "prompt", None)
    if prompt is None and dynprompt is not None:
        prompt = dynprompt.get_original_prompt()
    return {"PROMPT": prompt, "DYNPROMPT": dynprompt, "UNIQUE_ID": getattr(plan, "node_id", None),
            "EXTRA_PNGINFO": getattr(plan, "extra_pnginfo", None)}


def _in_same_mode(fn):
    """Inference mode is per thread, and a worker has to match the prompt's."""
    import torch

    inference = torch.is_inference_mode_enabled()

    def call(**inputs):
        with torch.inference_mode(inference):
            return fn(**inputs)
    return call


async def _call_node(node, inputs):
    import nodes

    cls = nodes.NODE_CLASS_MAPPINGS.get(node)
    if cls is None:
        raise _runtime.CallNotSupported(f"no node named {node!r}")
    hidden = _hidden_values()
    if hasattr(cls, "FUNCTION") and not hasattr(cls, "GET_SCHEMA"):
        declared = getattr(cls, "INPUT_TYPES", dict)().get("hidden", {})
        inputs = {**{k: hidden.get(v) for k, v in declared.items() if not isinstance(v, (list, tuple))}, **inputs}
        fn = getattr(cls(), cls.FUNCTION)
    else:
        from comfy_api.latest import _io
        fields = {_io.Hidden.prompt: hidden["PROMPT"], _io.Hidden.dynprompt: hidden["DYNPROMPT"],
                  _io.Hidden.unique_id: hidden["UNIQUE_ID"], _io.Hidden.extra_pnginfo: hidden["EXTRA_PNGINFO"]}
        fn = cls.PREPARE_CLASS_CLONE({"hidden_inputs": fields}).execute
    if inspect.iscoroutinefunction(fn):
        return await fn(**inputs)
    return await asyncio.to_thread(_in_same_mode(fn), **inputs)


def _node_result(output):
    if isinstance(output, dict):
        return tuple(output.get("result") or ())
    return tuple(getattr(output, "result", output) or ())


class NodeOps:
    """The ``eng.node`` rows, running ComfyUI's own nodes from pack code."""

    async def run(self, node, inputs):
        """Run a registered node and return its outputs as a tuple."""
        return _node_result(await _call_node(node, inputs))

    async def run_output(self, node, inputs):
        """Run a registered node and return ``{"result": tuple, "ui": dict or None}``."""
        output = await _call_node(node, inputs)
        ui = output.get("ui") if isinstance(output, dict) else getattr(output, "ui", None)
        return {"result": _node_result(output), "ui": ui.as_dict() if hasattr(ui, "as_dict") else ui}


EXTENSIONS = {"device": DeviceOps, "memory": MemoryOps, "settings": SettingsOps, "progress": ProgressOps, "ui": UiOps, "node": NodeOps}
