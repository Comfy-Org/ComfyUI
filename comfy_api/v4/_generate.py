"""Write the sync form's type stub and the API reference from this package's code.

    python -m comfy_api.v4._generate

Signatures, types and docstrings are read from the source as written. Tests fail while the
committed files differ from what this writes.
"""

import ast
import builtins
import importlib.util
import inspect
import sys
import textwrap
from pathlib import Path

from comfy_api import _exceptions, latest, v4
from comfy_api.v4 import _extension, util
from comfy_api.v4.util import folders

STUB = Path(__file__).parent / "generated" / "ComfyAPISyncStub.pyi"
REFERENCE = Path(__file__).parent / "docs" / "reference.md"
WIDTH = 88


def groups() -> dict[str, type]:
    return dict(v4.ComfyAPI_v4.__annotations__)


def calls(group: type) -> list:
    """A group's calls, chosen as the sync form chooses them: its public coroutine functions."""
    return [
        function
        for name, function in inspect.getmembers(group, inspect.iscoroutinefunction)
        if not name.startswith("_")
    ]


def version_calls() -> list:
    return [
        function
        for name, function in vars(v4.ComfyAPI_v4).items()
        if inspect.isfunction(function) and not name.startswith("_")
    ]


def _definition(function) -> ast.FunctionDef | ast.AsyncFunctionDef:
    return ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]


def _module_tree(module) -> ast.Module:
    return ast.parse(inspect.getsource(module))


def _imports(module) -> dict[str, str]:
    """The names a module binds by import at its top level, each with an import that binds it."""
    bound = {}
    for node in _module_tree(module).body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.partition(".")[0]
                bound[name] = f"import {alias.name}" + (
                    f" as {alias.asname}" if alias.asname else ""
                )
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            source = importlib.util.resolve_name(
                "." * node.level + (node.module or ""), module.__package__
            )
            for alias in node.names:
                bound[alias.asname or alias.name] = (
                    f"from {source} import {alias.name}"
                    + (f" as {alias.asname}" if alias.asname else "")
                )
    return bound


class _Qualified(ast.NodeTransformer):
    """Import each name the way the call's module imports it, or write it as an attribute of
    that module if the module defines it: a module's own imports are not re-exported."""

    def __init__(self, module: str, imports: dict[str, str], members: set[str]):
        self.module = module
        self.members = members
        self.imports = _imports(sys.modules[module])
        self.needed = imports
        self.used = False

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.imports:
            line = self.imports[node.id]
            if self.needed.setdefault(node.id, line) != line:
                raise ValueError(f"{node.id} is imported differently in two modules")
            return node
        if node.id not in vars(sys.modules[self.module]):
            if not hasattr(builtins, node.id):
                raise ValueError(f"{self.module} does not define {node.id}")
            if node.id not in self.members:
                return node
            # A stub class's own call, such as `list`, hides the builtin of that name.
            self.needed["builtins"] = "import builtins"
            return ast.Attribute(ast.Name("builtins"), node.id)
        self.used = True
        return ast.Attribute(ast.Name(self.module.rpartition(".")[2]), node.id)


def _parameter(arg: ast.arg, default: ast.expr | None = None) -> str:
    text = arg.arg
    if arg.annotation is not None:
        text += ": " + ast.unparse(arg.annotation)
    if default is not None:
        text += (" = " if arg.annotation is not None else "=") + ast.unparse(default)
    return text


def _parameters(args: ast.arguments) -> list[str]:
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + args.defaults
    texts = [_parameter(arg, default) for arg, default in zip(positional, defaults)]
    if args.posonlyargs:
        texts.insert(len(args.posonlyargs), "/")
    if args.vararg is not None:
        texts.append("*" + _parameter(args.vararg))
    elif args.kwonlyargs:
        texts.append("*")
    texts += [_parameter(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults)]
    if args.kwarg is not None:
        texts.append("**" + _parameter(args.kwarg))
    return texts


def _signature(node, keyword: str, indent: str = "") -> str:
    parameters = _parameters(node.args)
    head = f"{indent}{keyword} {node.name}("
    tail = ")" + (
        f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    )
    line = head + ", ".join(parameters) + tail
    if len(line) <= WIDTH:
        return line
    inner = indent + "    "
    return head + "\n" + "".join(f"{inner}{p},\n" for p in parameters) + indent + tail


def _docstring(text: str | None, indent: str) -> list[str]:
    if not text:
        return []
    text = inspect.cleandoc(text).replace("\\", "\\\\").replace('"', '\\"')
    lines = text.splitlines()
    if len(lines) == 1:
        return [f'{indent}"""{lines[0]}"""']
    return (
        [f'{indent}"""{lines[0]}']
        + [f"{indent}{line}".rstrip() for line in lines[1:]]
        + [f'{indent}"""']
    )


def _written_plainly(function, node) -> bool:
    """One undecorated definition, with no annotation written as a string: what the stub can copy."""
    owner = _module_tree(sys.modules[function.__module__]).body
    owner = next(
        n.body
        for n in owner
        if getattr(n, "name", None) == function.__qualname__.split(".")[0]
    )
    annotations = [a.annotation for a in ast.walk(node.args) if isinstance(a, ast.arg)]
    return (
        not node.decorator_list
        and sum(getattr(n, "name", None) == node.name for n in owner) == 1
        and not any(
            isinstance(a, ast.Constant) and isinstance(a.value, str)
            for a in annotations + [node.returns]
        )
    )


def _stub_function(
    function, indent: str, modules: set[str], imports: dict[str, str], members=()
) -> list[str]:
    node = _definition(function)
    if not _written_plainly(function, node):
        raise ValueError(
            f"{function.__qualname__} is decorated, overloaded or has a string annotation"
        )
    qualified = _Qualified(function.__module__, imports, set(members))
    node.args = qualified.visit(node.args)
    if node.returns is not None:
        node.returns = qualified.visit(node.returns)
    if qualified.used:
        modules.add(function.__module__)
    lines = [_signature(node, "def", indent) + ":"]
    return lines + (
        _docstring(function.__doc__, indent + "    ") or [indent + "    ..."]
    )


def sync_stub() -> str:
    modules: set[str] = set()
    imports: dict[str, str] = {}
    body = ["class ComfyAPISyncStub:"]
    body += _docstring(v4.ComfyAPISync.__doc__, "    ")
    body += ["", "    def __init__(self) -> None: ..."]
    for function in version_calls():
        body += _stub_function(function, "    ", modules, imports)
    for name, group in groups().items():
        body += ["", f"    class {group.__name__}Sync:"]
        body += _docstring(group.__doc__, "        ")
        members = [function.__name__ for function in calls(group)]
        for function in calls(group):
            body += _stub_function(function, "        ", modules, imports, members)
        body += [f"    {name}: {group.__name__}Sync"]
    packages: dict[str, list[str]] = {}
    for module in sorted(modules):
        package, _, name = module.rpartition(".")
        packages.setdefault(package, []).append(name)
    lines = sorted(set(imports.values()))
    lines += [f"from {p} import {', '.join(names)}" for p, names in packages.items()]
    header = [
        # mypy does not read V3's `io.<type>.Type` names as types; --strict reports a bare `dict`.
        '# mypy: disable-error-code="valid-type, name-defined, type-arg"',
        '"""The sync form of the V4 API, for type checkers and editors.',
        "",
        "Generated by `python -m comfy_api.v4._generate` from the calls' definitions; do not edit.",
        '"""',
        "",
    ]
    return "\n".join(header + lines + ["", ""] + body) + "\n"


class _Definitions:
    """The module-level classes and aliases of the package's modules, by name."""

    def __init__(self, modules):
        self.source: dict[str, str] = {}
        self.nodes: dict[str, ast.AST] = {}
        for module in modules:
            text = inspect.getsource(module)
            for node in _module_tree(module).body:
                if isinstance(node, ast.ClassDef):
                    name = node.name
                elif isinstance(node, ast.Assign) and isinstance(
                    node.targets[0], ast.Name
                ):
                    name = node.targets[0].id
                else:
                    continue
                if name.startswith("__"):
                    continue
                segment = ast.get_source_segment(text, node)
                if self.source.setdefault(name, segment) != segment:
                    raise ValueError(f"{name} is defined differently in two modules")
                self.nodes[name] = node

    def closure(self, names: set[str]) -> list[str]:
        """These names and every name their definitions use, sorted."""
        found, pending = set(), [n for n in names if n in self.source]
        while pending:
            name = pending.pop()
            if name in found:
                continue
            found.add(name)
            pending += [
                node.id
                for node in ast.walk(self.nodes[name])
                if isinstance(node, ast.Name)
                and node.id in self.source
                and node.id != name
            ]
        return sorted(found)


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _reference_entry(title: str, node, lines: list[str], used: set[str]) -> None:
    node.args.posonlyargs = [a for a in node.args.posonlyargs if a.arg != "self"]
    node.args.args = [a for a in node.args.args if a.arg != "self"]
    used |= _names(node.args) | (
        _names(node.returns) if node.returns is not None else set()
    )
    keyword = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    lines += [f"### `{title}`", "", "```python", _signature(node, keyword), "```", ""]
    doc = ast.get_docstring(node)
    if doc:
        lines += [doc, ""]


def _module_entries(prefix: str, module, lines: list[str], used: set[str]) -> None:
    tree = _module_tree(module)
    lines += [ast.get_docstring(tree), ""]
    body = tree.body
    for i, node in enumerate(body):
        if isinstance(node, ast.AnnAssign) and node.target.id in module.__all__:
            lines += [
                f"### `{prefix}.{node.target.id}`",
                "",
                "```python",
                ast.unparse(node),
                "```",
                "",
            ]
            following = body[i + 1] if i + 1 < len(body) else None
            if isinstance(following, ast.Expr) and isinstance(
                following.value, ast.Constant
            ):
                lines += [inspect.cleandoc(following.value.value), ""]
    functions = {
        n.name: n
        for n in body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in sorted(set(functions) & set(module.__all__)):
        _reference_entry(f"{prefix}.{name}", functions[name], lines, used)


def _from_v3(name: str) -> bool:
    value = getattr(v4, name)
    origin = (
        value.__name__ if inspect.ismodule(value) else getattr(value, "__module__", "")
    )
    return origin.startswith("comfy_api.latest")


def _v3_path(name: str) -> str:
    value = getattr(v4, name)
    if getattr(latest, name, None) is value:
        return f"comfy_api.latest.{name}"
    return next(
        f"comfy_api.latest.{owner}.{attribute}"
        for owner, holder in vars(latest).items()
        if inspect.isclass(holder)
        for attribute, member in vars(holder).items()
        if member is value
    )


def reference() -> str:
    used: set[str] = set()
    lines = [
        "# V4 API reference",
        "",
        f"API version {v4.ComfyAPI_v4.VERSION}. Generated from the `comfy_api.v4` package by",
        "`python -m comfy_api.v4._generate`; change the docstrings in the package, not this page.",
        "",
        "```python",
        "from comfy_api.v4 import ComfyAPISync, ComfyExtension, eng, io, ui, util",
        "```",
        "",
        "## `eng`",
        "",
        "`eng` is the API. Its calls are in groups, such as `eng.vae` and `eng.model`, and each",
        "call in a group is a coroutine function: in a node whose `execute` is `async def`, write",
        "`await eng.vae.encode(vae, pixels)`.",
        "",
        "In a node whose `execute` is a plain `def`, create `api = ComfyAPISync()`. It has the same",
        "groups and calls with the same parameters, and each call returns its result:",
        "`api.vae.encode(vae, pixels)`. A call runs on the thread that makes it. In the sync form,",
        "a call that would have to wait for the event loop raises `RuntimeError`.",
        "",
    ]
    for function in version_calls():
        _reference_entry(f"eng.{function.__name__}", _definition(function), lines, used)
    for name, group in groups().items():
        lines += [f"## `eng.{name}`", ""]
        if group.__doc__:
            lines += [inspect.cleandoc(group.__doc__), ""]
        for function in calls(group):
            _reference_entry(
                f"eng.{name}.{function.__name__}", _definition(function), lines, used
            )
    lines += ["## `util`", ""]
    _module_entries("util", util, lines, used)
    lines += ["## `util.folders`", ""]
    _module_entries("util.folders", folders, lines, used)
    lines += [
        "## `ComfyExtension`",
        "",
        "A pack's entry point, `comfy_entrypoint()`, returns an instance of a subclass of",
        "`comfy_api.v4.ComfyExtension`. ComfyUI calls these methods when it loads the pack.",
        "",
    ]
    for name, function in inspect.getmembers(
        _extension.ComfyExtension, inspect.iscoroutinefunction
    ):
        if not name.startswith("_"):
            _reference_entry(
                f"ComfyExtension.{name}", _definition(function), lines, used
            )
    group_modules = {
        sys.modules[f.__module__] for g in groups().values() for f in calls(g)
    }
    definitions = _Definitions(
        sorted(
            group_modules | {_extension, util, folders, _exceptions},
            key=lambda m: m.__name__,
        )
    )
    v3 = sorted(name for name in v4.__all__ if _from_v3(name))
    exported = set(v4.__all__) - {"ComfyAPI", "ComfyAPISync", "ComfyExtension"}
    lines += [
        "## Types",
        "",
        "The types the calls above take and give. Import them from `comfy_api.v4`, except"
        " `ProgressCounter`, which is in `comfy_api.v4.util`.",
        "",
    ]
    for name in definitions.closure(used | exported):
        if name not in v3:
            lines += [
                f"### `{name}`",
                "",
                "```python",
                definitions.source[name],
                "```",
                "",
            ]
    lines += ["## From V3", "", "These names are V3's objects, unchanged:", ""]
    lines += [f"- `{name}`: `{_v3_path(name)}`" for name in v3]
    return "\n".join(lines) + "\n"


def main() -> None:
    STUB.parent.mkdir(exist_ok=True)
    STUB.write_text(sync_stub(), encoding="utf-8", newline="\n")
    REFERENCE.parent.mkdir(exist_ok=True)
    REFERENCE.write_text(reference(), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
