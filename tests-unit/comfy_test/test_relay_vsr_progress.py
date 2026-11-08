import ast
from pathlib import Path
from types import FunctionType, SimpleNamespace

from tqdm import trange


MODEL_VSR = Path(__file__).resolve().parents[2] / "comfy/ldm/wan/model_vsr.py"


class CursorIncompatibleStream:
    def write(self, text):
        if "\x1b[A" in text:
            raise AttributeError("'NoneType' object has no attribute 'cursor_adjust'")

    def flush(self):
        pass


def test_relay_vsr_progress_works_with_cursor_incompatible_console():
    source = ast.parse(MODEL_VSR.read_text())
    bars = {}
    for node in ast.walk(source):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "model_trange":
            continue
        desc = next(keyword.value.value for keyword in node.keywords if keyword.arg == "desc")
        bars[desc] = compile(ast.Expression(node), str(MODEL_VSR), "eval")

    assert set(bars) == {"RelayVSR", "RelayVSR Flash decode"}
    stream = CursorIncompatibleStream()
    comfy = SimpleNamespace(utils=SimpleNamespace(model_trange=lambda *args, **kwargs: trange(*args, file=stream, **kwargs)))
    scope = {"comfy": comfy, "starts": range(1), "self": SimpleNamespace(blocks=[None]), "frames": 1}

    with trange(1, file=stream):
        for call in bars.values():
            with FunctionType(call, scope)() as bar:
                bar.update(1)
