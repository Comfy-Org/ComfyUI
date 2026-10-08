"""Generate the public V2 API inventory without requiring a runtime overlay.

Run from the backend checkout: python -m comfy_api.generate_api_spec
Check the published artifact: python -m comfy_api.generate_api_spec --check
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
from pathlib import Path

MODULES = (
    ("io", "comfy_api.latest._io_public"),
    ("sdk", "comfy_api.latest._sdk_public"),
)
ARTIFACT = Path(__file__).parent / "latest" / "api-spec.json"


class _AnnotationSource:
    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


def _annotation(value) -> str:
    return value if isinstance(value, str) else inspect.formatannotation(value)


def _signature(function) -> str:
    try:
        signature = inspect.signature(function)
    except (ValueError, TypeError):
        return "(...)"
    parameters = []
    for parameter in signature.parameters.values():
        annotation = parameter.annotation
        if isinstance(annotation, str):
            annotation = _AnnotationSource(annotation)
        default = parameter.default
        if default is not inspect.Parameter.empty:
            try:
                ast.parse(repr(default), mode="eval")
            except (SyntaxError, ValueError):
                default = _AnnotationSource("...")
        parameters.append(parameter.replace(annotation=annotation, default=default))
    returns = signature.return_annotation
    if isinstance(returns, str):
        returns = _AnnotationSource(returns)
    return str(signature.replace(parameters=parameters, return_annotation=returns))


def _callable(function, kind: str) -> dict:
    declaration = {
        "kind": kind,
        "async": inspect.iscoroutinefunction(function),
        "signature": _signature(function),
    }
    try:
        signature = inspect.signature(function)
    except (ValueError, TypeError):
        return declaration
    declaration["parameters"] = [
        {
            "name": parameter.name,
            "kind": parameter.kind.name,
            "annotation": None
            if parameter.annotation is inspect.Parameter.empty
            else _annotation(parameter.annotation),
            "required": parameter.default is inspect.Parameter.empty
            and parameter.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD),
        }
        for parameter in signature.parameters.values()
    ]
    declaration["returns"] = (
        None
        if signature.return_annotation is inspect.Signature.empty
        else _annotation(signature.return_annotation)
    )
    return declaration


def _class(cls, seen: frozenset[type] = frozenset()) -> dict:
    if cls in seen:
        return {"kind": "class", "reference": cls.__qualname__}
    seen = seen | {cls}
    members = {}
    for name in sorted(dir(cls)):
        if name.startswith("_") and name != "__init__":
            continue
        member = inspect.getattr_static(cls, name)
        kind = "method"
        if isinstance(member, (classmethod, staticmethod)):
            kind = "classmethod" if isinstance(member, classmethod) else "staticmethod"
            member = member.__func__
        if inspect.isfunction(member):
            members[name] = _callable(member, kind)
        elif isinstance(member, property):
            members[name] = {
                "kind": "property",
                "writable": member.fset is not None,
                "getter": _callable(member.fget, "method"),
            }
        elif inspect.isclass(member) and member.__module__.startswith("comfy_api."):
            members[name] = _class(member, seen)
    annotations = {}
    for base in reversed(cls.__mro__):
        annotations.update(
            {
                name: _annotation(value)
                for name, value in vars(base).get("__annotations__", {}).items()
                if not name.startswith("_")
            }
        )
    return {"kind": "class", "members": members, "annotations": annotations}


def generate_manifest() -> str:
    modules = {}
    for title, module_name in MODULES:
        module = importlib.import_module(module_name)
        names = getattr(module, "__all__", None) or [
            name for name in vars(module) if not name.startswith("_")
        ]
        exports = {}
        for name in sorted(names):
            value = getattr(module, name)
            if inspect.isclass(value):
                exports[name] = _class(value)
            elif callable(value):
                exports[name] = _callable(value, "function")
            elif not inspect.ismodule(value):
                exports[name] = {"kind": "constant", "type": type(value).__name__}
        modules[title] = exports
    return (
        json.dumps(
            {
                "format": "comfy-api-v2",
                "import": "comfy_api.latest",
                "modules": modules,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    generated = generate_manifest()
    if arguments.check:
        if not ARTIFACT.is_file() or ARTIFACT.read_text() != generated:
            parser.exit(
                1, "api-spec.json is stale; run python -m comfy_api.generate_api_spec\n"
            )
        return 0
    ARTIFACT.write_text(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
