import ast
from pathlib import Path

import pytest


MODEL_VSR = Path(__file__).resolve().parents[2] / "comfy/ldm/wan/model_vsr.py"


@pytest.mark.parametrize("class_name, method_name, progress_name", [
    ("RelayVSRWanModel", "forward_orig", "progress"),
    ("RelayFlash", "decode", "pbar"),
])
def test_relay_vsr_reports_progress_without_writing_a_nested_terminal_bar(class_name, method_name, progress_name):
    """Keep UI progress without replacing the sampler's terminal display."""
    source = ast.parse(MODEL_VSR.read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]

    assert not any(isinstance(call.func, ast.Attribute) and call.func.attr == "model_trange" for call in calls)
    assert any(isinstance(call.func, ast.Attribute) and call.func.attr == "ProgressBar" for call in calls)
    assert any(isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
               and call.func.value.id == progress_name and call.func.attr == "update" for call in calls)
