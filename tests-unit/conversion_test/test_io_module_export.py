import os as _os
import sys as _sys

_CONVERSION = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", "comfy_api", "conversion"
)
_sys.path.insert(0, _CONVERSION)
_os.environ.setdefault(
    "COMFY_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)
_os.environ["PYTHONPATH"] = _os.pathsep.join(
    p
    for p in (_CONVERSION, _os.environ["COMFY_ROOT"], _os.environ.get("PYTHONPATH", ""))
    if p
)
import ast
import importlib
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import convert


def generated(tmp_path, body):
    source = tmp_path / "source"
    source.mkdir()
    (source / "__init__.py").write_text("")
    (source / "bridge.py").write_text(
        'from comfy.comfy_types import IO\n__all__ = ["IO"]\n'
    )
    (source / "nodes.py").write_text("from . import bridge as b\n" + body)
    report = convert.generate(
        str(source),
        {"nodes": {}, "routes": [], "web_directory": None},
        str(tmp_path / "out"),
    )
    v2 = tmp_path / "out" / "source" / "v2"
    return report, (v2 / "bridge.py").read_text(), (v2 / "nodes.py").read_text()


def test_facade_reexport_produces_registration_values(tmp_path):
    report, bridge, consumer = generated(
        tmp_path,
        'class N:\n    RETURN_TYPES = tuple([b.IO.VIDEO] * 2 + ["STRING"])\nNODE_CLASS_MAPPINGS = {"N": N}\n',
    )
    tree = ast.parse(bridge)
    assignment = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "IO" for t in n.targets)
    )
    facade = types.SimpleNamespace(VIDEO="VIDEO")
    ns = {"_v2e": types.SimpleNamespace(IO=facade)}
    exec(  # noqa: S102
        compile(
            ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])),
            "bridge.py",
            "exec",
        ),
        ns,
    )
    nodes = ast.parse(consumer)
    body = [n for n in nodes.body if isinstance(n, (ast.ClassDef, ast.Assign))]
    scope = {"b": types.SimpleNamespace(IO=ns["IO"])}
    exec(  # noqa: S102
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            "nodes.py",
            "exec",
        ),
        scope,
    )
    assert scope["NODE_CLASS_MAPPINGS"]["N"].RETURN_TYPES == (
        "VIDEO",
        "VIDEO",
        "STRING",
    )
    assert '__all__ = ["IO"]' in bridge
    assert not report["leftovers"].get("nodes.py")


def test_generated_modules_import_and_register(tmp_path, monkeypatch):
    generated(
        tmp_path,
        'class N:\n    RETURN_TYPES = (b.IO.VIDEO,)\nNODE_CLASS_MAPPINGS = {"N": N}\n',
    )
    v2 = tmp_path / "out" / "source" / "v2"
    package = types.ModuleType("io_export_test_pack")
    package.__path__ = [str(v2)]
    shim = types.ModuleType("io_export_test_pack._v2_engine")
    shim.IO = types.SimpleNamespace(VIDEO="VIDEO")
    api = types.ModuleType("comfy_api")
    api.__path__ = []
    index = types.ModuleType("comfy_api.v4")
    index.eng = index.ui = index.util = index.io = object()
    for name, module in [
        ("io_export_test_pack", package),
        ("io_export_test_pack._v2_engine", shim),
        ("comfy_api", api),
        ("comfy_api.v4", index),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    nodes = importlib.import_module("io_export_test_pack.nodes")
    assert nodes.NODE_CLASS_MAPPINGS["N"].RETURN_TYPES == ("VIDEO",)
    assert importlib.import_module("io_export_test_pack.bridge").IO is shim.IO
    monkeypatch.delitem(sys.modules, "io_export_test_pack.nodes")
    monkeypatch.delitem(sys.modules, "io_export_test_pack.bridge")


def test_unsupported_class_protocol_or_shadow_is_reported(tmp_path):
    for i, use in enumerate(
        (
            'b.IO("VIDEO")',
            'isinstance("VIDEO", b.IO)',
            "b.IO.MISSING",
            "b.IO.ANY",
            'getattr(b, "IO")',
            "b.IO.VIDEO.value",
        )
    ):
        path = tmp_path / str(i)
        path.mkdir()
        report, bridge, _ = generated(
            path, "class N:\n    RETURN_TYPES = (" + use + ",)\n"
        )
        assert any(
            "comfy.comfy_types.IO" in x for x in report["leftovers"].get("nodes.py", [])
        ), use
        assert "IO = _v2e.IO" not in bridge
    path = tmp_path / "shadow"
    path.mkdir()
    report, bridge, _ = generated(
        path, "def f(b):\n    return b\nclass N:\n    RETURN_TYPES = (b.IO.VIDEO,)\n"
    )
    assert report["leftovers"].get("nodes.py")
    assert "IO = _v2e.IO" not in bridge


def test_rebound_export_is_not_replaced(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "__init__.py").write_text("")
    (source / "bridge.py").write_text(
        "from comfy.comfy_types import IO\nIO = object()\n"
    )
    (source / "nodes.py").write_text(
        "from . import bridge as b\nclass N:\n    RETURN_TYPES = (b.IO.VIDEO,)\n"
    )
    report = convert.generate(
        str(source),
        {"nodes": {}, "routes": [], "web_directory": None},
        str(tmp_path / "out"),
    )
    assert report["leftovers"].get("nodes.py")
    bridge = (tmp_path / "out" / "source" / "v2" / "bridge.py").read_text()
    assert "IO = _v2e.IO" not in bridge


def test_named_class_protocol_consumer_blocks_facade(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "__init__.py").write_text("")
    (source / "bridge.py").write_text("from comfy.comfy_types import IO\n")
    (source / "nodes.py").write_text(
        "from . import bridge as b\nclass N:\n    RETURN_TYPES = (b.IO.VIDEO,)\n"
    )
    (source / "other.py").write_text(
        'from .bridge import IO as enum_type\nx = isinstance("VIDEO", enum_type)\n'
    )
    report = convert.generate(
        str(source),
        {"nodes": {}, "routes": [], "web_directory": None},
        str(tmp_path / "out"),
    )
    assert report["leftovers"].get("other.py")
    bridge = (tmp_path / "out" / "source" / "v2" / "bridge.py").read_text()
    assert "IO = _v2e.IO" not in bridge


def test_exporting_module_enum_operation_remains_blocker(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "__init__.py").write_text("")
    (source / "bridge.py").write_text(
        'from comfy.comfy_types import IO\nx = isinstance("VIDEO", IO)\n'
    )
    (source / "nodes.py").write_text(
        "from . import bridge as b\nclass N:\n    RETURN_TYPES = (b.IO.VIDEO,)\n"
    )
    report = convert.generate(
        str(source),
        {"nodes": {}, "routes": [], "web_directory": None},
        str(tmp_path / "out"),
    )
    assert any("unproven local enum use" in x for x in report["leftovers"]["bridge.py"])
    assert report["leftovers"]["nodes.py"]
    bridge = (tmp_path / "out" / "source" / "v2" / "bridge.py").read_text()
    assert "IO = _v2e.IO" not in bridge


def test_exporter_metadata_context_and_tuple_escape(tmp_path):
    for index, (body, accepted) in enumerate(
        [
            ("class N:\n    RETURN_TYPES = (IO.VIDEO,)\n", True),
            (
                'class N:\n    def INPUT_TYPES(self):\n        return {"required": {"v": (IO.VIDEO,)}}\n',
                True,
            ),
            ("x = type((IO.VIDEO,))\n", False),
        ]
    ):
        case = tmp_path / str(index)
        case.mkdir()
        source = case / "source"
        source.mkdir()
        (source / "__init__.py").write_text("")
        (source / "bridge.py").write_text("from comfy.comfy_types import IO\n" + body)
        (source / "nodes.py").write_text(
            "from . import bridge as b\nclass N:\n    RETURN_TYPES = (b.IO.VIDEO,)\n"
        )
        report = convert.generate(
            str(source),
            {"nodes": {}, "routes": [], "web_directory": None},
            str(case / "out"),
        )
        bridge = (case / "out" / "source" / "v2" / "bridge.py").read_text()
        assert ("IO = _v2e.IO" in bridge) is accepted
        if not accepted:
            assert any(
                "unproven local enum use" in x for x in report["leftovers"]["bridge.py"]
            )


def test_shadowed_tuple_is_not_assumed_to_be_builtin(tmp_path):
    report, bridge, _ = generated(
        tmp_path,
        "def tuple(values):\n    return [type(v) for v in values]\nclass N:\n    RETURN_TYPES = tuple([b.IO.VIDEO])\n",
    )
    assert report["leftovers"]["nodes.py"]
    assert "IO = _v2e.IO" not in bridge


def test_safe_and_unsafe_consumers_remain_blocked(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "__init__.py").write_text("")
    (source / "bridge.py").write_text("from comfy.comfy_types import IO\n")
    (source / "safe.py").write_text(
        "from . import bridge as b\nclass N:\n    RETURN_TYPES = (b.IO.VIDEO,)\n"
    )
    (source / "unsafe.py").write_text(
        'from . import bridge as b\nx = isinstance("VIDEO", b.IO)\n'
    )
    report = convert.generate(
        str(source),
        {"nodes": {}, "routes": [], "web_directory": None},
        str(tmp_path / "out"),
    )
    assert any(
        "unproven pack module export" in x for x in report["leftovers"]["unsafe.py"]
    )


def test_every_named_consumer_is_reported(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "__init__.py").write_text("")
    (source / "bridge.py").write_text("from comfy.comfy_types import IO\n")
    (source / "safe.py").write_text(
        "from . import bridge as b\nclass N:\n    RETURN_TYPES = (b.IO.VIDEO,)\n"
    )
    for name in ("first", "second"):
        (source / f"{name}.py").write_text(
            'from .bridge import IO as T\nx = isinstance("VIDEO", T)\n'
        )
    report = convert.generate(
        str(source),
        {"nodes": {}, "routes": [], "web_directory": None},
        str(tmp_path / "out"),
    )
    for name in ("first", "second"):
        assert any(
            "unproven named or star import" in x
            for x in report["leftovers"][f"{name}.py"]
        )
