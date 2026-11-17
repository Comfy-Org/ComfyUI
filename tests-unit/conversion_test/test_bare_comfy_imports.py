"""``import model_management``: ComfyUI's comfy/ modules by their bare names become their full names."""
import os as _os
import sys as _sys
_CONVERSION = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "comfy_api", "conversion")
_sys.path.insert(0, _CONVERSION)
_os.environ.setdefault("COMFY_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_os.environ["PYTHONPATH"] = _os.pathsep.join(p for p in (_CONVERSION, _os.environ["COMFY_ROOT"], _os.environ.get("PYTHONPATH", "")) if p)
import convert


def test_bare_names_become_full_names():
    source = ("import os, model_management\n"
              "import model_management as mm  # é\n"
              "from comfy_types import IO, ComfyNodeABC\n"
              "def f():\n"
              "    import ldm.modules.attention\n"
              "    from ldm.util import instantiate_from_config as make\n"
              "import utils, numpy\n")
    assert convert.bare_comfy_imports(source, "nodes.py", {"nodes.py": source}) == (
        "import os; import comfy.model_management as model_management\n"
        "import comfy.model_management as mm  # é\n"
        "from comfy.comfy_types import IO, ComfyNodeABC\n"
        "def f():\n"
        "    import comfy.ldm.modules.attention as _v2_comfy_module; from comfy import ldm\n"
        "    from comfy.ldm.util import instantiate_from_config as make\n"
        "import utils, numpy\n")


def test_the_packs_own_module_of_that_name_is_left():
    source = "import model_management\nfrom samplers import x\n"
    sources = {"sub/nodes.py": source, "sub/model_management.py": "", "samplers/__init__.py": ""}
    assert convert.bare_comfy_imports(source, "sub/nodes.py", sources) == source
    assert convert.bare_comfy_imports(source, "other/nodes.py", sources) == "import comfy.model_management as model_management\nfrom samplers import x\n"
