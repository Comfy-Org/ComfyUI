"""Core engine extension: the ``eng.node`` rows, running ComfyUI's own nodes from pack code."""
import asyncio
import inspect

from comfy_engine import _runtime


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


class EngineOps:
    async def run(self, node, inputs):
        """Run a registered node and return its outputs as a tuple."""
        return _node_result(await _call_node(node, inputs))

    async def run_output(self, node, inputs):
        """Run a registered node and return ``{"result": tuple, "ui": dict or None}``."""
        output = await _call_node(node, inputs)
        ui = output.get("ui") if isinstance(output, dict) else getattr(output, "ui", None)
        return {"result": _node_result(output), "ui": ui.as_dict() if hasattr(ui, "as_dict") else ui}
