"""Fixture values must respect widget and executor list contracts."""
import os as _os
import sys as _sys
_CONVERSION = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "comfy_api", "conversion")
_sys.path.insert(0, _CONVERSION)
_os.environ.setdefault("COMFY_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_os.environ["PYTHONPATH"] = _os.pathsep.join(p for p in (_CONVERSION, _os.environ["COMFY_ROOT"], _os.environ.get("PYTHONPATH", "")) if p)
from verify import make_inputs


def test_optional_widget_values_and_connection_only_inputs():
    inputs, reason = make_inputs({"input": {
        "required": {"count": ["INT", {"default": 2}]},
        "optional": {
            "text": ["STRING", {"default": ""}],
            "number": ["FLOAT", {"default": 5, "min": 0, "max": 3}],
            "toggle": ["BOOLEAN", {"default": False}],
            "choice": [["a", "b"], {"default": "b"}],
            "v3_choice": ["COMBO", {"options": ["x", "y"], "default": "y"}],
            "empty_choice": [[]],
            "forced": ["STRING", {"forceInput": True}],
            "lazy": ["INT", {"lazy": True}],
            "model": ["MODEL"],
            "dynamic": ["COMBO", {}],
        },
    }})
    assert reason is None
    assert inputs == {"count": 2, "text": "", "number": 3.0,
                      "toggle": False, "choice": "b", "v3_choice": "y"}


def test_optional_widgets_obey_input_list_contract():
    inputs, reason = make_inputs({"is_input_list": True, "input": {
        "required": {"count": ["INT", {"default": 2}]},
        "optional": {"text": ["STRING", {"default": ""}],
                     "choice": [["a", "b"], {"default": "b"}]},
    }})
    assert reason is None
    assert inputs == {"count": [2], "text": [""], "choice": ["b"]}
