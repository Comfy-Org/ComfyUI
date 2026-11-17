"""Stand-ins for what a pack needs while it loads that the conversion environment doesn't have.

A pack is converted from what it lists once imported. Most packs that could not be imported here failed on
something outside the pack's own node declarations: a Python package that isn't installed, a name a newer
library no longer has, a model download, a ``pip install`` run at import. A *plan* names exactly those
things, one by one, each found from a real failure of this pack (``derive``), and a ``Session`` puts an inert
stand-in in each one's place while the pack is imported.

Stand-ins only get a pack past its import. When the import is done they are *armed*: any use of one raises
``StandInUsed``. So a node whose inputs are computed from a stand-in is listed as a node that could not be
described, and a node that needs one to run is recorded as not run; nothing is compared against a fake.

Nothing here runs unless a plan asks for it: with an empty plan a pack loads exactly as before.
"""
from __future__ import annotations

import ast
import builtins
import dis
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import json
import os
import re
import subprocess
import sys
import sysconfig
import types
from typing import Any, Optional

PACK = "<pack>"  # the pack's own module, in a plan: the same plan serves the original and the converted pack

_STATE: dict = {"armed": False}


class StandInUsed(RuntimeError):
    """Code used a stand-in after the pack was loaded: the real thing isn't in the conversion environment."""


def _use(name: str) -> None:
    if _STATE["armed"]:
        raise StandInUsed(f"{name} is a stand-in: it is not available in the conversion environment")


def _make(name: str) -> "StandIn":
    stand_in = object.__new__(StandIn)
    object.__setattr__(stand_in, "_standin_name", name)
    return stand_in


def _unpack_count(frame: Any) -> int:
    """How many values the calling code unpacks into (``a, b = stand_in``), read from its bytecode."""
    try:
        code, at = frame.f_code.co_code, frame.f_lasti
        name = dis.opname[code[at]]
        if name == "UNPACK_SEQUENCE":
            return code[at + 1]
        if name == "UNPACK_EX":
            return (code[at + 1] & 0xFF)
    except Exception:
        pass
    return 0


_DECORATED = (types.FunctionType, types.MethodType, type, classmethod, staticmethod)


class StandIn:
    """Anything a missing library would have given: callable, subscriptable, a base class, a decorator."""

    _standin_name = "stand-in"

    def __new__(cls, *args: Any, **kwargs: Any) -> "StandIn":  # a pack's class that derives from a stand-in
        _use(f"{cls.__name__} (derived from {cls._standin_name})")
        return object.__new__(cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __init_subclass__(cls, **kwargs: Any) -> None:
        pass

    def __mro_entries__(self, bases: tuple) -> tuple:
        name = self._standin_name
        return (type(name.rsplit(".", 1)[-1].strip("()[]") or "StandIn", (StandIn,), {"_standin_name": name}),)

    def __getattr__(self, name: str) -> Any:
        if name in ("__name__", "__qualname__"):
            return self._standin_name.rsplit(".", 1)[-1]
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        _use(self._standin_name)
        value = _make(f"{self._standin_name}.{name}")
        try:
            object.__setattr__(self, name, value)  # the same object each time it's asked for
        except Exception:
            pass
        return value

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        _use(self._standin_name)
        if len(args) == 1 and not kwargs and isinstance(args[0], _DECORATED):
            return args[0]  # used as a decorator: what it decorates stays as written
        return _make(f"{self._standin_name}()")

    def __repr__(self) -> str:
        return f"<stand-in for {self._standin_name}>"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return repr(self)

    def __bool__(self) -> bool:
        _use(self._standin_name)
        return True

    def __len__(self) -> int:
        _use(self._standin_name)
        return 0

    def __iter__(self):
        _use(self._standin_name)
        count = _unpack_count(sys._getframe(1))
        return iter([_make(f"{self._standin_name}[{i}]") for i in range(count)])

    def __contains__(self, item: Any) -> bool:
        _use(self._standin_name)
        return False

    def __getitem__(self, key: Any) -> Any:
        _use(self._standin_name)
        return _make(f"{self._standin_name}[...]")

    def __setitem__(self, key: Any, value: Any) -> None:
        _use(self._standin_name)

    def __delitem__(self, key: Any) -> None:
        _use(self._standin_name)

    def __enter__(self) -> "StandIn":
        _use(self._standin_name)
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    async def __aenter__(self) -> "StandIn":
        _use(self._standin_name)
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def __instancecheck__(self, instance: Any) -> bool:
        _use(self._standin_name)
        return False

    def __subclasscheck__(self, subclass: Any) -> bool:
        _use(self._standin_name)
        return False

    def __int__(self) -> int:
        _use(self._standin_name)
        return 0

    __index__ = __int__

    def __float__(self) -> float:
        _use(self._standin_name)
        return 0.0

    # a missing library's version is "newer than anything" to a pack's version check
    def __lt__(self, other: Any) -> bool:
        _use(self._standin_name)
        return False

    __le__ = __lt__

    def __gt__(self, other: Any) -> bool:
        _use(self._standin_name)
        return True

    __ge__ = __gt__

    def _operator(self, *args: Any) -> "StandIn":
        _use(self._standin_name)
        return _make(self._standin_name)

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _operator
    __floordiv__ = __rfloordiv__ = __mod__ = __rmod__ = __pow__ = __rpow__ = __matmul__ = __rmatmul__ = _operator
    __and__ = __rand__ = __or__ = __ror__ = __xor__ = __rxor__ = __lshift__ = __rshift__ = _operator
    __neg__ = __pos__ = __invert__ = __abs__ = _operator


class StandInModule(types.ModuleType):
    """A module that isn't installed: every name in it is a stand-in, and every submodule another of these."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        _use(self.__name__)
        value = _make(f"{self.__name__}.{name}")
        self.__dict__[name] = value
        return value


class _Loader(importlib.abc.Loader):
    def create_module(self, spec: Any) -> types.ModuleType:
        module = StandInModule(spec.name, "Stand-in: this module is not in the conversion environment.")
        module.__dict__.update(__path__=[], __all__=[], __version__="999.0.0",
                               __file__=f"/stand-in/{spec.name.replace('.', '/')}/__init__.py")
        return module

    def exec_module(self, module: types.ModuleType) -> None:
        pass


class _Finder(importlib.abc.MetaPathFinder):
    """Answers for the plan's modules and everything under them."""

    def __init__(self, names: list) -> None:
        self.names = set(names)

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        parts = fullname.split(".")
        if any(".".join(parts[:i]) in self.names for i in range(1, len(parts) + 1)):
            return importlib.machinery.ModuleSpec(fullname, _Loader(), is_package=True)
        return None


class _NamesLoader:
    """Wraps a real module's loader: the names the plan says it lacks are there before and after it runs
    (before: a pack module that's only partly initialised when another imports from it)."""

    def __init__(self, loader: Any, session: "Session", keys: list) -> None:
        self._loader, self._session, self._keys = loader, session, keys

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)

    def create_module(self, spec: Any) -> Any:
        create = getattr(self._loader, "create_module", None)
        return create(spec) if create else None

    def exec_module(self, module: types.ModuleType) -> None:
        for key in self._keys:
            self._session.give_names(module, key, before=True)
        self._loader.exec_module(module)
        for key in self._keys:  # a lazy module (transformers, diffusers) puts another object in sys.modules as it runs
            self._session.give_names(sys.modules.get(module.__name__, module), key, before=False)


class _NamesFinder(importlib.abc.MetaPathFinder):
    def __init__(self, session: "Session", modules: dict) -> None:
        self.session, self.modules = session, modules  # real module name -> the plan's key for it

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname not in self.modules:
            return None
        for finder in sys.meta_path:
            if finder is self or isinstance(finder, (_Finder, _NamesFinder)) or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                    spec.loader = _NamesLoader(spec.loader, self.session, self.modules[fullname])
                return spec
        return None


INSTALLER = re.compile(
    r"(?:^|[\s/\\\"'])(?:pip3?|uv|conda|mamba|micromamba|apt|apt-get|yum|dnf|brew|npm|yarn|pnpm|playwright|winget|choco)"
    r"(?:\.exe)?[\"']?\s.*\b(?:install|sync|add|download|update|upgrade)\b"
    r"|-m[\"']?[\s,]+[\"']?(?:pip|ensurepip|spacy|nltk\.downloader|playwright|unidic)\b"
    r"|(?:^|[\s/\\\"'])git(?:\.exe)?[\"']?\s.*\b(?:clone|pull|fetch|submodule|lfs|checkout)\b"
    r"|\b(?:install|setup|prestartup_script)\.py\b|requirements[\w.-]*\.txt", re.I)


def _command_text(args: Any) -> str:
    if isinstance(args, (bytes, os.PathLike)):
        args = os.fsdecode(args)
    if isinstance(args, str):
        return args
    try:
        return " ".join(os.fsdecode(a) if isinstance(a, (bytes, os.PathLike)) else str(a) for a in args)
    except TypeError:
        return str(args)


def _program(args: Any) -> str:
    if isinstance(args, (bytes, os.PathLike)):
        args = os.fsdecode(args)
    if isinstance(args, str):
        first = args.split()[0] if args.split() else ""
    else:
        try:
            first = str(os.fsdecode(next(iter(args))))
        except (StopIteration, TypeError):
            first = ""
    return os.path.basename(first.strip("\"'")).lower()


class Plan:
    """What stands in for what, for one pack. Kept as plain JSON."""

    LISTS = ("modules", "force", "calls", "commands", "syspath", "mkdirs", "aliases")

    def __init__(self, data: Optional[dict] = None) -> None:
        data = data or {}
        self.modules: list = list(data.get("modules") or [])    # not installed: stood in, with everything under them
        self.force: list = list(data.get("force") or [])        # installed, but their import fails here: stood in
        self.names: dict = {k: list(v) for k, v in (data.get("names") or {}).items()}  # "module[:Class]" -> names it lacks
        self.calls: list = list(data.get("calls") or [])        # "module:function" library calls that fail at load
        self.commands: list = list(data.get("commands") or [])  # programs run at load, not run
        self.installers: bool = bool(data.get("installers"))    # pip/git/... run at load, not run
        self.syspath: list = list(data.get("syspath") or [])    # pack folders it imports from as top-level modules
        self.mkdirs: list = list(data.get("mkdirs") or [])      # folders it expects to be there
        self.aliases: list = list(data.get("aliases") or [])    # names it imports itself by (its folder's name)
        self.skips: list = [dict(s) for s in data.get("skips") or []]  # the pack's own statements that fail at load
        self.leave_caught: bool = bool(data.get("leave_caught"))  # what it catches itself is left as it is
        self.evidence: list = list(data.get("evidence") or [])  # the failure each entry answers

    @classmethod
    def load(cls, path: Optional[str]) -> "Plan":
        if not path or not os.path.exists(path):
            return cls()
        with open(path) as file:
            return cls(json.load(file))

    @classmethod
    def from_environment(cls) -> "Plan":
        return cls.load(os.environ.get("CONV_STANDIN_PLAN"))

    def to_json(self) -> dict:
        out = {key: list(getattr(self, key)) for key in self.LISTS}
        out.update(names={k: list(v) for k, v in self.names.items()}, installers=self.installers, leave_caught=self.leave_caught,
                   skips=[dict(s) for s in self.skips], evidence=list(self.evidence))
        return out

    def save(self, path: str) -> None:
        with open(path + ".tmp", "w") as file:
            json.dump(self.to_json(), file, indent=1)
        os.replace(path + ".tmp", path)

    def __bool__(self) -> bool:
        return bool(self.installers or self.names or self.skips or any(getattr(self, key) for key in self.LISTS))

    def has(self, repair: tuple) -> bool:
        kind = repair[0]
        if kind == "installers":
            return self.installers
        if kind == "names":
            return repair[2] in self.names.get(repair[1], ())
        if kind == "skips":
            return any(s["file"] == repair[1]["file"] and s["id"] == repair[1]["id"] for s in self.skips)
        return repair[1] in getattr(self, kind)

    def add(self, repair: tuple, evidence: str = "") -> bool:
        if self.has(repair):
            return False
        kind = repair[0]
        if kind == "installers":
            self.installers = True
        elif kind == "names":
            self.names.setdefault(repair[1], []).append(repair[2])
        else:
            getattr(self, kind).append(repair[1])
        self.evidence.append({"stand_in": list(repair), "for": evidence[:400]})
        return True


class Session:
    """A plan in force while one pack is imported: ``with Session(plan, entry, module_name): import the pack``."""

    def __init__(self, plan: Optional[Plan], entry: str, module_name: str) -> None:
        self.plan = plan if plan is not None else Plan.from_environment()
        self.entry, self.module_name = entry, module_name
        self.blocked: list = []   # commands that were not run
        self.skipped: list = []   # the pack's own statements that failed and were passed over
        self.caught: list = []    # failures the pack's own code caught while it loaded
        self._undo: list = []

    # -- names in a plan <-> names in this process ------------------------------------------------------
    def real(self, name: str) -> str:
        return self.module_name + name[len(PACK):] if name == PACK or name.startswith(PACK + ".") else name

    def __enter__(self) -> "Session":
        _STATE["armed"] = False
        main = sys.modules.get("__main__")
        if main is not None and os.environ.get("COMFY_ROOT"):  # a pack that asks where ComfyUI's main.py is
            had, old = hasattr(main, "__file__"), getattr(main, "__file__", None)
            main.__file__ = os.path.join(os.environ["COMFY_ROOT"], "main.py")
            self._undo.append(lambda: setattr(main, "__file__", old) if had else delattr(main, "__file__"))
        self._watch()
        if self.plan:
            self._install()
        return self

    def _watch(self) -> None:
        """Keep the failures the pack catches itself as it loads (``try: from .nodes import * / except: print``):
        a pack that goes on without one of its own files loads with nodes missing, and nothing says so."""
        monitoring = getattr(sys, "monitoring", None)
        if monitoring is None:
            return
        caught, mine, tool = self.caught, {}, 4

        def handled(code: Any, offset: int, error: BaseException) -> None:
            name = code.co_filename
            if name not in mine:
                mine[name] = self.rel(name) is not None
            if mine[name] and len(caught) < 80 and isinstance(error, Exception) and not any(error is e for e in caught):
                tb = error.__traceback__ and error.__traceback__.tb_next
                while tb is not None and tb.tb_frame.f_code.co_name != "<module>":
                    tb = tb.tb_next
                if tb is not None or isinstance(error, ImportError):  # it came out of an import
                    caught.append(error)

        def undo() -> None:
            monitoring.set_events(tool, 0)
            monitoring.register_callback(tool, monitoring.events.EXCEPTION_HANDLED, None)
            monitoring.free_tool_id(tool)

        try:
            monitoring.use_tool_id(tool, "pack conversion")
            monitoring.register_callback(tool, monitoring.events.EXCEPTION_HANDLED, handled)
            monitoring.set_events(tool, monitoring.events.EXCEPTION_HANDLED)
            self._undo.append(undo)
        except Exception:
            pass

    def __exit__(self, *exc: Any) -> bool:
        for undo in reversed(self._undo):
            try:
                undo()
            except Exception:
                pass
        self._undo = []
        _STATE["armed"] = True
        return False

    def _install(self) -> None:
        plan = self.plan
        sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, (_Finder, _NamesFinder))]
        for name, module in list(sys.modules.items()):
            if isinstance(module, StandInModule):
                del sys.modules[name]
        for rel in plan.mkdirs:
            try:
                os.makedirs(self._path(rel), exist_ok=True)
            except OSError:
                pass
        for rel in plan.syspath:
            folder = os.path.normpath(os.path.join(self.entry, rel))
            if folder not in sys.path:
                sys.path.append(folder)
        if plan.force:
            for name in list(sys.modules):
                if any(name == forced or name.startswith(forced + ".") for forced in plan.force):
                    del sys.modules[name]
            sys.meta_path.insert(0, _Finder(plan.force))
        if plan.modules:
            sys.meta_path.append(_Finder([self.real(name) for name in plan.modules]))  # last: a real module wins
        waiting = {}
        for key in plan.names:
            if key == PACK or key.startswith((PACK + ".", PACK + ":")):
                continue  # the pack's own files: by file, whatever name they're imported under (_hook_pack_files)
            module_name = self.real(key.split(":", 1)[0])
            module = sys.modules.get(module_name)
            if module is not None and module_name != self.module_name:
                self.give_names(module, key, before=False)
            waiting.setdefault(module_name, []).append(key)  # and whenever it's imported (again: libraries reload)
        if waiting:
            sys.meta_path.insert(0, _NamesFinder(self, waiting))
            self._keep_names(waiting)
        for call in plan.calls:
            self._stand_in_call(call)
        if plan.installers or plan.commands:
            self._block_commands()
        if plan.skips or any(key == PACK or key.startswith((PACK + ".", PACK + ":")) for key in plan.names):
            self._hook_pack_files()

    def rel(self, path: Any) -> Optional[str]:
        """A pack file's path inside the pack (None for any other file)."""
        if not isinstance(path, str):
            return None
        real = os.path.realpath(path)
        return os.path.relpath(real, self.entry) if real.startswith(self.entry + os.sep) else None

    def _hook_pack_files(self) -> None:
        """The pack's own source files, as they're loaded: the names the plan gives them, and its statements that
        fail here passed over (each in a try of its own, so everything up to the failure still happens)."""
        session, loader = self, importlib.machinery.SourceFileLoader
        had = {name: loader.__dict__.get(name) for name in ("get_code", "exec_module")}
        real_get_code, real_exec_module = loader.get_code, loader.exec_module
        keys: dict = {}
        for key in self.plan.names:
            keys.setdefault(key.split(":", 1)[0], []).append(key)

        def get_code(self: Any, fullname: str) -> Any:
            rel = session.rel(getattr(self, "path", None))
            wanted = [s for s in session.plan.skips if s["file"] == rel] if rel else []
            if not wanted:
                return real_get_code(self, fullname)
            tree = ast.parse(self.get_data(self.path), self.path)
            if not _pass_over(tree, rel, {s["id"] for s in wanted}):
                return real_get_code(self, fullname)
            return compile(tree, self.path, "exec", dont_inherit=True)

        def exec_module(self: Any, module: types.ModuleType) -> None:
            rel = session.rel(getattr(self, "path", None))
            mine = keys.get(module_key(rel), ()) if rel else ()
            for key in mine:
                session.give_names(module, key, before=True)
            real_exec_module(self, module)
            for key in mine:
                session.give_names(module, key, before=False)

        def undo() -> None:
            for name, value in had.items():
                if value is None:
                    delattr(loader, name)
                else:
                    setattr(loader, name, value)

        loader.get_code, loader.exec_module = get_code, exec_module  # type: ignore[method-assign]
        self._undo.append(undo)
        builtins.__conv_skip__ = self._pass  # type: ignore[attr-defined]

    def _pass(self, rel: str, ident: str, error: BaseException, namespace: dict, names: list) -> None:
        """A statement of the pack's own failed, as the plan says it does here: the names it would have given
        are stand-ins, and the pack goes on loading."""
        entry = next((s for s in self.plan.skips if s["file"] == rel and s["id"] == ident), {})
        for name in names:
            if name not in namespace:
                namespace[name] = _make(f"{name} ({rel}, line {entry.get('line')})")
        self.skipped.append({"file": rel, "line": entry.get("line"), "code": entry.get("code"),
                             "error": f"{type(error).__name__}: {error}"[:300]})

    def _keep_names(self, wanted: dict) -> None:
        """A library can put a new module object in its place at any time (transformers runs its __init__ again):
        after each import statement, the names the plan gives a module are on whichever object is there now."""
        session, real_import, seen = self, builtins.__import__, {}

        def import_(name: str, *args: Any, **kwargs: Any) -> Any:
            found = real_import(name, *args, **kwargs)
            for module_name, keys in wanted.items():
                module = sys.modules.get(module_name)
                if module is not None and seen.get(module_name) != id(module):
                    seen[module_name] = id(module)
                    for key in keys:
                        session.give_names(module, key, before=False)
            return found

        builtins.__import__ = import_
        self._undo.append(lambda: setattr(builtins, "__import__", real_import))

    def _path(self, rel: str) -> str:
        return os.path.join(self.entry, rel[len(PACK) + 1:]) if rel.startswith(PACK + "/") else (self.entry if rel == PACK else rel)

    def prepare(self, module: types.ModuleType) -> None:
        """The pack's own top module, before it runs (its names are given as its file is loaded). A pack that
        imports itself by its folder's name (``from My_Pack.x import y``: it works where custom_nodes is on the
        import path) finds itself under that name; the name stays, for imports its nodes make when they run."""
        for alias in self.plan.aliases:
            for name in [n for n in sys.modules if n == alias or n.startswith(alias + ".")]:
                del sys.modules[name]  # an earlier load of this pack (its original, when this is its conversion)
            sys.modules[alias] = module
            parent = alias.rpartition(".")[0]
            if parent:  # custom_nodes.<its folder>
                try:
                    importlib.import_module(parent)
                except Exception:
                    sys.modules[parent] = types.ModuleType(parent)
                    sys.modules[parent].__path__ = []  # type: ignore[attr-defined]

    def give_names(self, module: types.ModuleType, key: str, before: bool) -> None:
        owner: Any = module
        path = key.split(":", 1)[1] if ":" in key else ""
        if path:
            if before:
                return
            try:
                for part in path.split("."):
                    owner = getattr(owner, part)
            except Exception:
                return
        for name in self.plan.names.get(key, ()):
            try:
                present = name in vars(owner) or (not before and hasattr(owner, name))
            except Exception:
                present = False
            if not present:
                try:
                    setattr(owner, name, _make(f"{self.real(key)}.{name}"))
                except Exception:
                    pass

    def _stand_in_call(self, call: str) -> None:
        module_name, _, path = call.partition(":")
        try:
            owner: Any = importlib.import_module(self.real(module_name))
            parts = path.split(".")
            for part in parts[:-1]:
                owner = getattr(owner, part)
            original = vars(owner).get(parts[-1], None) if isinstance(owner, type) else getattr(owner, parts[-1])
            if original is None:
                original = getattr(owner, parts[-1])
        except Exception:
            return
        stand_in = SUBSTITUTES.get(call) or _make(f"{module_name}.{path}")
        places = [(owner, parts[-1], original)]
        if not isinstance(owner, type):  # ``from library import function`` elsewhere: the same function there
            for other in list(sys.modules.values()):
                try:
                    names = [n for n, v in vars(other).items() if v is original] if isinstance(other, types.ModuleType) else []
                except Exception:
                    continue
                places += [(other, n, original) for n in names if not (other is owner and n == parts[-1])]
        for place, name, value in places:
            try:
                setattr(place, name, stand_in)
                self._undo.append(lambda place=place, name=name, value=value: setattr(place, name, value))
            except Exception:
                pass

    def _blocks(self, args: Any) -> bool:
        text = _command_text(args)
        return bool((self.plan.installers and INSTALLER.search(text)) or _program(args) in self.plan.commands
                    or (isinstance(args, str) and any(re.search(rf"(?:^|[\s;&|]){re.escape(c)}(?:\s|$)", args) for c in self.plan.commands)))

    def _block_commands(self) -> None:
        session, real_popen, real_system = self, subprocess.Popen, os.system
        nothing = ["/bin/true"] if os.path.exists("/bin/true") else [sys.executable, "-c", "pass"]

        class Popen(real_popen):  # type: ignore[misc, valid-type]
            def __init__(self, args: Any, *rest: Any, **kwargs: Any) -> None:
                if session._blocks(args):
                    session.blocked.append(_command_text(args)[:300])
                    args = nothing
                    kwargs.pop("executable", None)
                    kwargs["shell"] = False
                    rest = rest[:1] if rest else rest  # bufsize; the rest are keywords in practice
                super().__init__(args, *rest, **kwargs)

        def system(command: Any) -> int:
            if session._blocks(command):
                session.blocked.append(_command_text(command)[:300])
                return 0
            return real_system(command)

        subprocess.Popen, os.system = Popen, system  # type: ignore[misc]
        self._undo.append(lambda: (setattr(subprocess, "Popen", real_popen), setattr(os, "system", real_system)))


def _register_nodes_here(nodes_package: str = "nodes") -> tuple:
    """comfy-env's ``register_nodes`` as it is inside one of its own workers: the pack's node folders imported in
    this process (``nodes/`` itself, or each package under it when that gives no nodes), their mappings merged.
    No environment of comfy-env's is built or bound here."""
    caller = sys._getframe(1).f_globals
    package, folder = caller.get("__name__") or "", os.path.join(os.path.dirname(caller.get("__file__") or ""), nodes_package)
    mappings: dict = {}
    display: dict = {}
    if not os.path.isdir(folder):
        return mappings, display
    sources = [nodes_package] if os.path.isfile(os.path.join(folder, "__init__.py")) else []
    for attempt in (sources, [f"{nodes_package}.{name}" for name in sorted(os.listdir(folder))
                              if not name.startswith(("_", ".")) and os.path.isfile(os.path.join(folder, name, "__init__.py"))]):
        for source in attempt:
            module = importlib.import_module(f".{source}", package=package)
            mappings.update(getattr(module, "NODE_CLASS_MAPPINGS", None) or {})
            display.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", None) or {})
        if mappings:
            break
    return mappings, display


# Library calls with a known way to do, in this process, what they'd do with machinery that isn't here.
SUBSTITUTES = {"comfy_env.isolation.wrap:register_nodes": _register_nodes_here}


def module_key(rel: str) -> str:
    """A pack file, as plans name it: ``a/b.py`` is ``<pack>.a.b`` under whatever module name it's imported."""
    parts = rel[:-3].split(os.sep) if rel.endswith(".py") else rel.split(os.sep)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([PACK] + parts)


_BLOCKS = ("body", "orelse", "finalbody")


def _load_time(body: list):
    """(the list a statement is in, its index, the statement) for every statement that runs as a module is
    imported: the module's own, and those inside its ifs, loops, trys, withs and class bodies."""
    for index, node in enumerate(body):
        yield body, index, node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for field in _BLOCKS:
            inner = getattr(node, field, None)
            if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt):
                yield from _load_time(inner)
        for handler in getattr(node, "handlers", None) or []:
            yield from _load_time(handler.body)
        for case in getattr(node, "cases", None) or []:
            yield from _load_time(case.body)


def _span(node: ast.stmt) -> tuple:
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", None) or []])
    return first, getattr(node, "end_lineno", None) or node.lineno


def _statement_id(node: ast.stmt) -> str:
    return hashlib.sha1(ast.dump(node).encode("utf-8", "replace")).hexdigest()[:12]


def statement_at(tree: ast.Module, line: int) -> Optional[ast.stmt]:
    """The innermost load-time statement on that line."""
    best = None
    for _, _, node in _load_time(tree.body):
        first, last = _span(node)
        if first <= line <= last and (best is None or last - first <= _span(best)[1] - _span(best)[0]):
            best = node
    return best


def _binds(node: ast.stmt) -> list:
    """The names a statement gives its module (or class) when it works."""
    names: list = []

    def target(item: Any) -> None:
        if isinstance(item, ast.Name):
            names.append(item.id)
        elif isinstance(item, (ast.Tuple, ast.List)):
            for element in item.elts:
                target(element)
        elif isinstance(item, ast.Starred):
            target(item.value)

    for _, _, inner in _load_time([node]):
        if isinstance(inner, ast.Import):
            names.extend(a.asname or a.name.split(".")[0] for a in inner.names)
        elif isinstance(inner, ast.ImportFrom):
            names.extend(a.asname or a.name for a in inner.names if a.name != "*")
        elif isinstance(inner, ast.Assign):
            for item in inner.targets:
                target(item)
        elif isinstance(inner, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor)):
            target(inner.target)
        elif isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(inner.name)
        elif isinstance(inner, (ast.With, ast.AsyncWith)):
            for item in inner.items:
                target(item.optional_vars)
    return list(dict.fromkeys(names))


def _test_id(node: ast.If) -> str:
    return hashlib.sha1(("test:" + ast.dump(node.test)).encode("utf-8", "replace")).hexdigest()[:12]


def _handed_over(body: list, rel: str, ident: str, names: list, like: ast.AST) -> ast.Try:
    """Those statements in a try that hands a failure to ``__conv_skip__``."""
    call = ast.Expr(ast.Call(ast.Name("__conv_skip__", ast.Load()), [
        ast.Constant(rel), ast.Constant(ident), ast.Name("__conv_error__", ast.Load()),
        ast.Call(ast.Name("locals", ast.Load()), [], []),
        ast.List([ast.Constant(name) for name in names], ast.Load())], []))
    caught = ast.Tuple([ast.Name("Exception", ast.Load()), ast.Name("SystemExit", ast.Load())], ast.Load())
    wrapped = ast.Try(body=body, handlers=[ast.ExceptHandler(type=caught, name="__conv_error__", body=[call])],
                      orelse=[], finalbody=[])
    ast.copy_location(wrapped, like)
    for part in ast.walk(wrapped.handlers[0]):
        ast.copy_location(part, like)
    return wrapped


def _pass_over(tree: ast.Module, rel: str, idents: set) -> int:
    """Put each of those statements in a try of its own that hands the failure to ``__conv_skip__``. An ``if``
    whose condition is what fails keeps its body: the condition is worked out first, in a try of its own, and
    what failed in it is a stand-in (so ``if available():`` goes ahead, and ``if not available():`` doesn't)."""
    found = []
    for body, _, node in _load_time(tree.body):
        if isinstance(node, ast.If) and _test_id(node) in idents:
            found.append((body, node, _test_id(node), True))
        if _statement_id(node) in idents and not (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
            found.append((body, node, _statement_id(node), False))
    for body, node, ident, test in found:
        where = next(i for i, item in enumerate(body) if item is node)
        if not test:
            body[where] = _handed_over([node], rel, ident, _binds(node), node)
            continue
        owner: Any = node
        field = "test"
        while isinstance(getattr(owner, field), ast.UnaryOp) and isinstance(getattr(owner, field).op, ast.Not):
            owner, field = getattr(owner, field), "operand"
        name, value = f"__conv_test_{ident}__", getattr(owner, field)
        setattr(owner, field, ast.copy_location(ast.Name(name, ast.Load()), value))
        first = ast.copy_location(ast.Assign([ast.Name(name, ast.Store())], value), value)
        body[where:where + 1] = [_handed_over([first], rel, ident, [name], value), node]
    ast.fix_missing_locations(tree)
    return len(found)


def forget(entry: str, module_name: str) -> None:
    """Drop a pack's modules (and stand-in modules) from this process, to import it again."""
    root = os.path.realpath(entry) + os.sep
    for name, module in list(sys.modules.items()):
        try:
            file = module.__dict__.get("__file__") if isinstance(module, types.ModuleType) else None
        except Exception:
            file = None
        if name == module_name or name.startswith(module_name + ".") or isinstance(module, StandInModule) \
                or (isinstance(file, str) and os.path.realpath(file).startswith(root)):
            del sys.modules[name]
    importlib.invalidate_caches()


# --------------------------------------------------------------------------------------------------- #
# From a failure to the one stand-in that answers it
# --------------------------------------------------------------------------------------------------- #
PROTECTED = {  # never stood in whole: ComfyUI itself and what it is built on
    "torch", "numpy", "comfy", "comfy_extras", "comfy_api", "comfy_api_nodes", "comfy_execution", "comfy_config",
    "app", "api_server", "folder_paths", "nodes", "server", "execution", "latent_preview", "node_helpers",
    "protocol", "middleware", "utils", "aiohttp", "PIL", "safetensors", "packaging", "typing_extensions",
    "setuptools", "pkg_resources", "pip", "standin", "packload", "describe", "verify",
}
_STDLIB = os.path.realpath(sysconfig.get_paths()["stdlib"]) + os.sep
_NETWORK_STDLIB = ("urllib.", "http.", "socket", "ssl", "ftplib", "smtplib")
_HERE = os.path.dirname(os.path.realpath(__file__)) + os.sep


def _kind(filename: str, entry: str) -> str:
    if filename.startswith("<frozen importlib"):
        return "import"
    if filename.startswith("<"):
        return "other"
    real = os.path.realpath(filename)
    if real.startswith(entry + os.sep):
        return "pack"
    if real.startswith(_HERE):
        return "harness"
    if real.startswith(_STDLIB) and "site-packages" not in real and "dist-packages" not in real:
        return "stdlib"
    return "library"


def _frames(error: BaseException, entry: str) -> list:
    out, tb = [], error.__traceback__
    while tb is not None:
        code = tb.tb_frame.f_code
        out.append({"kind": _kind(code.co_filename, entry), "name": code.co_name,
                    "qualname": getattr(code, "co_qualname", code.co_name),
                    "module": tb.tb_frame.f_globals.get("__name__") or ""})
        tb = tb.tb_next
    return out


def _chain(error: BaseException) -> list:
    out, queue, seen = [], [error], set()
    while queue:
        item = queue.pop(0)
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        out.append(item)
        queue += [item.__cause__, item.__context__]
    return out


def _resolvable(module_name: str, qualname: str) -> bool:
    if "<" in qualname:
        return False
    last = qualname.rsplit(".", 1)[-1]
    if last.startswith("__") and last.endswith("__"):
        return False
    owner: Any = sys.modules.get(module_name)
    if owner is None:
        return False
    try:
        for part in qualname.split("."):
            owner = getattr(owner, part)
    except Exception:
        return False
    return callable(owner)


class Deriver:
    def __init__(self, entry: str, module_name: str, plan: Plan) -> None:
        self.entry, self.module_name, self.plan = os.path.realpath(entry), module_name, plan

    def plan_name(self, module: str) -> str:
        loaded = sys.modules.get(module)
        file = getattr(loaded, "__file__", None) if isinstance(loaded, types.ModuleType) else None
        if isinstance(file, str) and os.path.realpath(file).startswith(self.entry + os.sep):
            return module_key(os.path.relpath(os.path.realpath(file), self.entry))
        if module == self.module_name or module.startswith(self.module_name + "."):
            return PACK + module[len(self.module_name):]
        return module

    def derive(self, error: BaseException) -> Optional[tuple]:
        chain = _chain(error)
        for item in reversed(chain):  # what first went wrong, when it says what's missing
            repair = self._by_type(item)
            if repair and repair[0] == "names" and item is not error and not isinstance(item, ModuleNotFoundError) \
                    and not str(repair[1]).startswith(PACK) and any(e.__context__ is item and e.__cause__ is None for e in chain):
                # a name the pack tried in a ``try`` and then went on without (``from app import x`` / except
                # ImportError: ``from .app import x``): ComfyUI's ``app`` lacking it is the path the pack expects
                continue
            if repair and not self.plan.has(repair):
                return repair
        for item in chain:  # otherwise: where the pack's own code was when it failed
            repair = self._by_place(item)
            if repair and not self.plan.has(repair):
                return repair
        for item in chain:  # otherwise: the statement of its own that can't work here
            repair = self._statement(item)
            if repair and not self.plan.has(repair):
                return repair
        return None

    def _statement(self, error: BaseException) -> Optional[tuple]:
        """The innermost statement of the pack's own that was running, as its files were imported, when it failed."""
        place, tb = None, error.__traceback__
        while tb is not None:
            code = tb.tb_frame.f_code
            if not code.co_flags & 0x1 and _kind(code.co_filename, self.entry) == "pack":  # a module or class body
                place = (os.path.realpath(code.co_filename), tb.tb_lineno)
            tb = tb.tb_next
        if place is None:
            return None
        try:
            with open(place[0], "rb") as file:
                source = file.read()
            tree = ast.parse(source, place[0])
        except (OSError, SyntaxError, ValueError):
            return None
        node = statement_at(tree, place[1])
        if node is None or (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
            return None
        lines = source.decode("utf-8", "replace").splitlines()
        code_line = lines[place[1] - 1].strip()[:160] if 0 < place[1] <= len(lines) else ""
        skip = {"file": os.path.relpath(place[0], self.entry), "id": _statement_id(node), "line": place[1], "code": code_line}
        if isinstance(node, ast.If) and node.test.lineno <= place[1] <= (getattr(node.test, "end_lineno", None) or node.test.lineno):
            skip.update(id=_test_id(node), part="condition")  # its body can still run
        return ("skips", skip)

    # -- what the exception itself names ----------------------------------------------------------------
    def _by_type(self, error: BaseException) -> Optional[tuple]:
        text = str(error)
        if isinstance(error, ModuleNotFoundError):
            name = error.name or (re.search(r"No module named '([^']+)'", text) or [None, None])[1]
            if name:
                return self._missing_module(name)
        if isinstance(error, ImportError):
            found = re.search(r"cannot import name '([^']+)' from (?:partially initialized module )?'([^']+)'", text)
            if found and not isinstance(sys.modules.get(found.group(2)), StandInModule):
                return ("names", self.plan_name(found.group(2)), found.group(1))
        if isinstance(error, AttributeError):
            owner, name = getattr(error, "obj", None), getattr(error, "name", None)
            if isinstance(owner, StandInModule) or isinstance(owner, StandIn):
                return None
            if isinstance(owner, types.ModuleType) and name:
                return ("names", self.plan_name(owner.__name__), name)
            if isinstance(owner, type) and name and "<" not in owner.__qualname__ and owner.__module__ != "builtins" \
                    and getattr(sys.modules.get(owner.__module__), owner.__qualname__.split(".")[0], None) is not None:
                return ("names", f"{self.plan_name(owner.__module__)}:{owner.__qualname__}", name)
        if isinstance(error, NameError) and not isinstance(error, UnboundLocalError) and getattr(error, "name", None):
            return self._starred(error)
        if isinstance(error, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
            return self._command(error.cmd)
        if isinstance(error, (FileNotFoundError, PermissionError)) and error.filename and \
                any(f["module"] == "subprocess" for f in _frames(error, self.entry)):
            return self._command([error.filename])
        return None

    def _starred(self, error: NameError) -> Optional[tuple]:
        """A name a pack module expects from ``from library import *``, when the library is stood in (a stand-in
        module has nothing to give a star import) or no longer exports it."""
        tb = error.__traceback__
        while tb is not None and tb.tb_next is not None:
            tb = tb.tb_next
        if tb is None or _kind(tb.tb_frame.f_code.co_filename, self.entry) != "pack":
            return None
        module = tb.tb_frame.f_globals.get("__name__") or ""
        try:
            with open(tb.tb_frame.f_code.co_filename, encoding="utf-8", errors="replace") as file:
                source = file.read()
        except OSError:
            return None
        if not re.search(r"^\s*from\s+[\w.]+\s+import\s+\*", source, re.M):
            return None
        return ("names", module_key(os.path.relpath(os.path.realpath(tb.tb_frame.f_code.co_filename), self.entry)), error.name)

    def _command(self, command: Any) -> Optional[tuple]:
        if INSTALLER.search(_command_text(command)):
            return ("installers",)
        program = _program(command)
        return ("commands", program) if program else None

    def _missing_module(self, name: str) -> Optional[tuple]:
        if name == self.module_name or name.startswith(self.module_name + "."):
            rel = name[len(self.module_name) + 1:].replace(".", os.sep)
            path = os.path.join(self.entry, rel)
            if os.path.isfile(path + ".py") or (os.path.isdir(path) and any(n.endswith(".py") for n in os.listdir(path))):
                return None  # it's there: something else is wrong
            return ("modules", self.plan_name(name))  # the source lacks it (a git submodule, a file left out)
        top = name.split(".")[0]
        if _folder_name(top) == _folder_name(os.path.basename(self.entry)):
            return ("aliases", top)  # the pack, by its folder's name
        if top == "custom_nodes":  # the pack, by the name its author's folder has
            for alias in ([name] if name.count(".") == 1 else self._custom_nodes_names() if name == top else []):
                if self._is_this_pack(alias):
                    return ("aliases", alias)
        if "." not in name and top not in self.plan.modules:
            folder = self._own_folder_with(top)
            if folder is not None:
                return ("syspath", folder)
        return ("modules", name)

    def _sources(self):
        for root, dirs, files in os.walk(self.entry):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for file_name in files:
                if file_name.endswith(".py"):
                    try:
                        with open(os.path.join(root, file_name), encoding="utf-8", errors="replace") as file:
                            yield file.read()
                    except OSError:
                        continue

    def _custom_nodes_names(self) -> list:
        return sorted({"custom_nodes." + n for source in self._sources() for n in re.findall(r"\bcustom_nodes\.(\w+)", source)})

    def _is_this_pack(self, name: str) -> bool:
        """Whether ``custom_nodes.<folder>`` is this pack under another folder name: the folder's name is its own,
        or everything its code imports from there is a file of its own."""
        if _folder_name(name.split(".")[1]) == _folder_name(os.path.basename(self.entry)):
            return True
        wanted = set()
        for source in self._sources():
            wanted.update(re.findall(re.escape(name) + r"\.(\w+)", source))
            for group in re.findall(r"from\s+" + re.escape(name) + r"\s+import\s+(?:\(([^)]*)\)|([^\n#]+))", source):
                wanted.update(part.split(" as ")[0].strip() for part in "".join(group).split(","))
        wanted.discard("")
        return bool(wanted) and all(os.path.isfile(os.path.join(self.entry, part + ".py"))
                                    or os.path.isdir(os.path.join(self.entry, part)) for part in wanted)

    def _own_folder_with(self, top: str) -> Optional[str]:
        """The pack folder holding ``top`` as a module, when the pack imports its own files as top-level modules."""
        found = []
        for root, dirs, files in os.walk(self.entry):
            depth = os.path.relpath(root, self.entry).count(os.sep) if root != self.entry else -1
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"] if depth < 3 else []
            if top + ".py" in files or (top in dirs and os.path.isfile(os.path.join(root, top, "__init__.py"))):
                found.append(os.path.relpath(root, self.entry))
        found = [f for f in found if f not in self.plan.syspath]
        return sorted(found, key=lambda f: (f.count(os.sep), f))[0] if found else None

    # -- where the pack was when it failed ----------------------------------------------------------------
    def _by_place(self, error: BaseException) -> Optional[tuple]:
        frames = _frames(error, self.entry)
        own = [i for i, f in enumerate(frames) if f["kind"] == "pack"]
        if not own:
            return None
        after = [f for f in frames[own[-1] + 1:] if f["kind"] != "harness"]
        if not after:
            return self._in_pack_code(error)
        if after[0]["kind"] == "import" or (after[0]["name"] == "<module>" and after[0]["kind"] == "library"):
            # the pack imports a library, and that import fails
            for frame in reversed(after):
                top = frame["module"].split(".")[0]
                if frame["kind"] == "library" and frame["name"] == "<module>" and top and top not in PROTECTED:
                    return ("force", top)
            return None
        for frame in after:  # the pack calls a library, and the call fails (a download, a model, a device)
            if frame["kind"] == "library" or (frame["kind"] == "stdlib" and (frame["module"] + ".").startswith(_NETWORK_STDLIB)):
                if frame["module"].split(".")[0] in ("standin", "packload", "describe"):
                    continue
                if _resolvable(frame["module"], frame["qualname"]):
                    return ("calls", f"{frame['module']}:{frame['qualname']}")
        return None

    def _in_pack_code(self, error: BaseException) -> Optional[tuple]:
        if isinstance(error, (FileNotFoundError, NotADirectoryError)) and isinstance(error.filename, str):
            path = os.path.normpath(error.filename if os.path.isabs(error.filename) else os.path.join(os.getcwd(), error.filename))
            if not os.path.splitext(os.path.basename(path))[1] and not os.path.exists(path):
                for root in (self.entry, os.environ.get("COMFY_ROOT") or "/nonexistent", "/tmp"):
                    if path.startswith(root + os.sep):  # a folder the pack lists: an empty one
                        return ("mkdirs", PACK + path[len(self.entry):] if root == self.entry else path)
        return None


def failed_imports(caught: list, entry: str) -> list:
    """Of the failures a pack caught as it loaded, those of an import statement in its own code:
    ``(error, own)``, own when a file of the pack's own was being imported (its nodes may be in it)."""
    entry, out = os.path.realpath(entry), []
    for error in caught:  # each from the frame that caught it down to where it was raised
        own = any(f["kind"] == "pack" and f["name"] == "<module>" for f in _frames(error, entry)[1:])
        if own or isinstance(error, ImportError):
            out.append((error, own))
    return out


def _folder_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def derive(error: BaseException, entry: str, module_name: str, plan: Plan) -> Optional[tuple]:
    try:
        return Deriver(entry, module_name, plan).derive(error)
    except Exception:
        return None


def installed(plan: Plan) -> list:
    """The packages a plan stands in a module of that are installed here: the pack wants a module this version
    of the package doesn't have (the rest of what's stood in isn't installed at all)."""
    tops = sorted({name.split(".")[0] for name in list(plan.modules) + list(plan.force) if not name.startswith(PACK)})
    found = []
    for top in tops:
        try:
            if importlib.machinery.PathFinder.find_spec(top) is not None:
                found.append(top)
        except Exception:
            pass
    return found


def contains_stand_in(value: Any) -> bool:
    return isinstance(value, (StandIn, StandInModule)) or (isinstance(value, type) and issubclass(value, StandIn))
