import ast
import inspect
import re

from comfy_api import v4
from comfy_api.v4 import _generate, util
from comfy_api.v4.util import folders


STALE = "stale: run python -m comfy_api.v4._generate"


def test_committed_sync_stub_is_what_the_generator_writes():
    assert _generate.STUB.read_text(encoding="utf-8") == _generate.sync_stub(), STALE


def test_committed_reference_is_what_the_generator_writes():
    assert _generate.REFERENCE.read_text(encoding="utf-8") == _generate.reference(), (
        STALE
    )


def test_reference_has_an_entry_for_every_public_name():
    reference = _generate.REFERENCE.read_text(encoding="utf-8")
    entries = set(re.findall(r"^(?:#+ |- )`([\w.]+)`", reference, re.M))
    described_in_prose = {"ComfyAPI", "ComfyAPISync"}

    for module, prefix in ((v4, ""), (util, "util."), (folders, "util.folders.")):
        for name in set(module.__all__) - described_in_prose:
            assert {prefix + name, name} & entries, prefix + name


def test_comfyuis_stub_tool_leaves_v4s_stub_to_v4s_generator(monkeypatch, tmp_path):
    from comfy_api import generate_api_stubs

    written_by_the_tool_itself = _generate.STUB.with_name("ComfyAPISync.pyi")
    monkeypatch.setattr(_generate, "STUB", tmp_path / "stub.pyi")
    monkeypatch.setattr(_generate, "REFERENCE", tmp_path / "reference.md")

    generate_api_stubs.generate_stubs_for_module("comfy_api.v4")

    assert (tmp_path / "stub.pyi").read_text(encoding="utf-8") == _generate.sync_stub()
    assert not written_by_the_tool_itself.exists()


def _stub_parameters(node: ast.FunctionDef) -> list[tuple[str, object, object]]:
    args = node.args
    kind = inspect.Parameter
    positional = [(a, kind.POSITIONAL_ONLY) for a in args.posonlyargs]
    positional = (positional + [(a, kind.POSITIONAL_OR_KEYWORD) for a in args.args])[1:]
    defaults = [None] * (len(positional) - len(args.defaults)) + args.defaults
    parameters = [(a.arg, k, d) for (a, k), d in zip(positional, defaults)]
    if args.vararg:
        parameters.append((args.vararg.arg, kind.VAR_POSITIONAL, None))
    parameters += [
        (a.arg, kind.KEYWORD_ONLY, d) for a, d in zip(args.kwonlyargs, args.kw_defaults)
    ]
    if args.kwarg:
        parameters.append((args.kwarg.arg, kind.VAR_KEYWORD, None))
    return [(name, k, _default(d)) for name, k, d in parameters]


def _default(node: ast.expr | None) -> object:
    """The default's value, or `...` for an expression the stub writes as it is in the source."""
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except ValueError:
        return ...


def _runtime_parameters(call) -> list[tuple[str, object, object]]:
    return [
        (p.name, p.kind, None if p.default is inspect.Parameter.empty else p.default)
        for p in inspect.signature(call).parameters.values()
    ]


def test_sync_stub_declares_exactly_the_calls_and_parameters_of_the_sync_form():
    stub = ast.parse(_generate.STUB.read_text(encoding="utf-8"))
    api_stub = next(n for n in stub.body if isinstance(n, ast.ClassDef))
    group_stubs = {n.name: n for n in api_stub.body if isinstance(n, ast.ClassDef)}
    declared = {
        n.target.id: group_stubs[n.annotation.id]
        for n in api_stub.body
        if isinstance(n, ast.AnnAssign)
    }
    api = v4.ComfyAPISync()

    assert set(vars(api)) == set(declared) | {
        n.name for n in api_stub.body if isinstance(n, ast.FunctionDef)
    } - {"__init__"}
    checked = 0
    for name, group_stub in declared.items():
        group = getattr(api, name)
        methods = {n.name: n for n in group_stub.body if isinstance(n, ast.FunctionDef)}
        assert set(vars(group)) == set(methods), name
        for call, node in methods.items():
            stub = _stub_parameters(node)
            runtime = [
                (n, k, ... if (n, k, ...) in stub else d)
                for n, k, d in _runtime_parameters(getattr(group, call))
            ]
            assert stub == runtime, f"{name}.{call}"
            checked += 1
    assert checked == sum(len(_generate.calls(g)) for g in _generate.groups().values())
