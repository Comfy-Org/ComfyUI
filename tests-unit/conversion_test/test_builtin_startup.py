"""Describe-time startup initializes ComfyUI's builtin nodes before any pack loads, as normal startup does."""
import os as _os
import sys as _sys
_CONVERSION = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "comfy_api", "conversion")
_sys.path.insert(0, _CONVERSION)
_os.environ.setdefault("COMFY_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_os.environ["PYTHONPATH"] = _os.pathsep.join(p for p in (_CONVERSION, _os.environ["COMFY_ROOT"], _os.environ.get("PYTHONPATH", "")) if p)


def test_cpu_startup_initializes_core_nodes_before_packs():
    from describe import cpu_comfy
    cpu_comfy(as_comfyui_loads_packs=True)
    import nodes
    from comfy_extras.nodes_latent import reshape_latent_to
    assert callable(reshape_latent_to)
    assert 'ImageBlend' in nodes.NODE_CLASS_MAPPINGS
    assert 'SamplerCustom' in nodes.NODE_CLASS_MAPPINGS
