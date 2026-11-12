"""The names V4 hands to packs unchanged from the V3 API keep working.

`comfy_api.v4` passes V3's `io` and `ui` modules and a few V3 classes through to packs. A V4
pack must not break when V3 changes, so `v3_surface.txt` records every such name and how it is
called, and this test fails when one is removed or can no longer be called the recorded way.
Names V3 adds since are not recorded and do not fail. To record them, run this file:

    python tests-unit/comfy_api_test/v4/v3_surface_test.py
"""

import enum
import inspect
import sys
from pathlib import Path

RECORDED = Path(__file__).with_name("v3_surface.txt")
NESTING = 3


def roots() -> dict[str, object]:
    from comfy_api import latest, v4

    passed_through = {
        name: getattr(v4, name)
        for name in ("io", "ui", "Input", "InputImpl", "Types")
        + ("CacheProvider", "CacheContext", "CacheValue")
    }
    return {**passed_through, "ComfyExtension": latest.ComfyExtension}


def _from_v3(obj: object) -> bool:
    return str(getattr(obj, "__module__", "")).startswith("comfy_api")


def _call(function) -> str | None:
    """How `function` is called: its parameters in order, `=` after one with a default."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return None
    kinds = inspect.Parameter
    marks = {kinds.VAR_POSITIONAL: "*", kinds.VAR_KEYWORD: "**"}
    words = []
    for parameter in parameters:
        if parameter.kind is kinds.KEYWORD_ONLY and "*" not in "".join(words):
            words.append("*")
        default = "" if parameter.default is parameter.empty else "="
        words.append(marks.get(parameter.kind, "") + parameter.name + default)
    return "(" + ", ".join(words) + ")"


def _describe(owner: object, name: str) -> tuple[str, object | None]:
    """What `owner.name` is, and the class to look inside if it is one of V3's."""
    static = inspect.getattr_static(owner, name)
    if isinstance(static, (classmethod, staticmethod)):
        static = static.__func__
    if isinstance(static, enum.Enum):
        return "member", None
    if inspect.isclass(static):
        return ("class", static) if _from_v3(static) else ("value", None)
    if inspect.isfunction(static):
        return _call(static) or "value", None
    return "value", None


def _walk(path: str, obj: object, depth: int, seen: set[int], found: dict[str, str]):
    if id(obj) in seen or depth > NESTING:
        return
    seen.add(id(obj))
    if inspect.ismodule(obj):
        names = {n for n in vars(obj) if _from_v3(getattr(obj, n))}
    else:
        # A private base class is not recorded on its own, so its members are recorded here.
        private = [b for b in obj.__mro__ if b.__name__.startswith("_") and _from_v3(b)]
        names = {name for base in [obj, *private] for name in vars(base)}
        if _from_v3(obj.__init__) and _call(obj.__init__):
            found[f"{path}.__init__"] = _call(obj.__init__)
    for name in sorted(n for n in names if not n.startswith("_")):
        found[f"{path}.{name}"], inner = _describe(obj, name)
        if inner is not None:
            _walk(f"{path}.{name}", inner, depth + 1, seen, found)


def surface() -> dict[str, str]:
    found: dict[str, str] = {}
    for name, root in roots().items():
        found[name] = "class" if inspect.isclass(root) else "value"
        _walk(name, root, 1, set(), found)
    return found


def _resolve(path: str) -> object:
    name, *rest = path.split(".")
    obj = roots()[name]
    for part in rest[:-1]:
        obj = getattr(obj, part)
    return obj


def _parameters(call: str) -> tuple[list[str], dict[str, bool]]:
    """A recorded call as its positional names in order, and every name with whether it has a default."""
    positional, defaults, keyword_only = [], {}, False
    for word in filter(None, call.strip("()").split(", ")):
        if word == "*":
            keyword_only = True
        elif word.startswith("*"):
            keyword_only = keyword_only or not word.startswith("**")
        else:
            name = word.rstrip("=")
            defaults[name] = word.endswith("=")
            if not keyword_only:
                positional.append(name)
    return positional, defaults


def _still_called_as(recorded: str, current: str) -> bool:
    """Whether every call the recorded form allowed is still allowed."""
    old_positional, old = _parameters(recorded)
    new_positional, new = _parameters(current)
    takes_any_keyword = "**" in current
    return (
        new_positional[: len(old_positional)] == old_positional
        and all(name in new or takes_any_keyword for name in old)
        and all(
            new.get(name, True) or not had_default for name, had_default in old.items()
        )
        and all(has_default or name in old for name, has_default in new.items())
    )


def broken(recorded: dict[str, str], current: dict[str, str]) -> list[str]:
    problems, gone = [], []
    for path, was in recorded.items():
        owner_path, _, name = path.rpartition(".")
        if any(path.startswith(parent + ".") for parent in gone):
            continue
        if path in current:
            now = current[path]
        elif owner_path and owner_path in current and hasattr(_resolve(path), name):
            now, _ = _describe(_resolve(path), name)
        else:
            problems.append(f"{path} is gone")
            gone.append(path)
            continue
        if was.startswith("(") and now.startswith("("):
            if not _still_called_as(was, now):
                problems.append(f"{path}{was} is now {path}{now}")
        elif was != now and "value" not in (was, now):
            problems.append(f"{path} was a {was} and is now a {now}")
    return problems


def _read() -> dict[str, str]:
    lines = RECORDED.read_text(encoding="utf-8").splitlines()
    return dict(line.split(" ", 1) for line in lines)


def test_everything_v4_passes_through_from_v3_is_still_there_and_called_the_same_way():
    assert broken(_read(), surface()) == []


def test_a_removed_name_or_a_changed_call_is_reported():
    recorded = {
        "io.Image": "class",
        "io.Image.Input.__init__": "(self, id, display_name=)",
        "io.NoSuchType": "class",
        "io.NoSuchType.Input": "class",
    }
    current = {
        "io.Image": "class",
        "io.Image.Input.__init__": "(self, display_name=, *, id)",
    }

    assert broken(recorded, current) == [
        "io.Image.Input.__init__(self, id, display_name=) is now"
        " io.Image.Input.__init__(self, display_name=, *, id)",
        "io.NoSuchType is gone",
    ]


def test_a_new_optional_parameter_or_a_new_name_is_not_a_break():
    recorded = {"io.Image.Input.__init__": "(self, id, display_name=)"}
    current = {
        "io.Image.Input.__init__": "(self, id, display_name=, tooltip=, *, lazy=)",
        "io.NewType": "class",
    }

    assert broken(recorded, current) == []


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parents[3]))
    lines = [f"{path} {what}" for path, what in sorted(surface().items())]
    RECORDED.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
