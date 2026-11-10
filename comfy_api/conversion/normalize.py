"""Resolve a pack's probes of ComfyUI before its code is rewritten.

Packs written for several ComfyUI versions ask ComfyUI what it has, and keep its modules in variables:

    getattr(mm, "soft_empty_cache", None)        hasattr(PromptServer, "instance")
    if PromptServer is not None: ...             mm = comfy.model_management

For the ComfyUI a pack is converted for, each of these has one answer, and ``ComfyIndex`` reads it from
ComfyUI's source without running it. ``resolve_probes`` rewrites them to that answer (``mm.soft_empty_cache``,
``True``, the variable's module written out), so the converter then sees plain dotted engine names: the ones
it knows are rewritten, the rest are reported by their real name instead of as a bare module.

Nothing here guesses. A name the index can't decide (a module with ``import *`` or ``__getattr__``, a class
attribute set at run time) stays as the pack wrote it.
"""
import ast
import os

MODULE, CLASS, FUNCTION, VALUE, OBJECT, ABSENT, UNKNOWN = "module", "class", "function", "value", "object", "absent", "unknown"
#: never None while a pack's code runs
NOT_NONE = {MODULE, CLASS, FUNCTION, OBJECT}
#: set by ComfyUI at run time, not in a class body: the server exists before any pack loads
KNOWN = {"server.PromptServer.instance": OBJECT}
#: objects whose class is known: their attributes are the class's, and what its methods set on ``self``
INSTANCES = {"server.PromptServer.instance": "server.PromptServer"}
#: module values that are one object for ComfyUI's whole life, so a variable holding one is just its name
STABLE_VALUES = {"nodes.NODE_CLASS_MAPPINGS", "nodes.NODE_DISPLAY_NAME_MAPPINGS", "server.PromptServer.instance"}
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
MAX_ROUNDS = 6


class ComfyIndex:
    """What ComfyUI's modules define, read from its source tree."""

    def __init__(self, root):
        self.root = root
        self._modules = {}

    def _path(self, parts):
        base = os.path.join(self.root, *parts)
        if os.path.isfile(base + ".py"):
            return base + ".py"
        if os.path.isfile(os.path.join(base, "__init__.py")):
            return os.path.join(base, "__init__.py")
        if os.path.isdir(base) and any(n.endswith(".py") for n in os.listdir(base)):
            return base  # a namespace package (comfy_extras)
        return None

    def _module(self, parts):
        key = ".".join(parts)
        if key not in self._modules:
            path = self._path(parts)
            if path is None:
                self._modules[key] = None
            elif os.path.isdir(path):
                self._modules[key] = {"names": {}, "open": False, "classes": {}}
            else:
                try:
                    with open(path, encoding="utf-8") as file:
                        self._modules[key] = self._read(ast.parse(file.read(), path))
                except (OSError, SyntaxError, ValueError):
                    self._modules[key] = {"names": {}, "open": True, "classes": {}}
        return self._modules[key]

    @staticmethod
    def _read(tree):
        names, classes, opened = {}, {}, False
        declared_global = {name for node in ast.walk(tree) if isinstance(node, ast.Global) for name in node.names}
        todo = list(tree.body)
        while todo:
            node = todo.pop()
            if isinstance(node, _FUNCTIONS):
                names[node.name] = FUNCTION
                opened = opened or node.name == "__getattr__"
            elif isinstance(node, ast.ClassDef):
                names[node.name] = CLASS
                classes[node.name] = node
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names[alias.asname or alias.name.split(".")[0]] = MODULE
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        opened = True
                    else:
                        names[alias.asname or alias.name] = ("from", node.level, node.module, alias.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
                targets = node.targets if isinstance(node, ast.Assign) else \
                    [i.optional_vars for i in node.items if i.optional_vars is not None] if isinstance(node, (ast.With, ast.AsyncWith)) \
                    else [node.target]
                for target in targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            names.setdefault(name.id, VALUE)
            for field in ("body", "orelse", "finalbody"):
                if not isinstance(node, _SCOPES):
                    todo += [n for n in getattr(node, field, []) if isinstance(n, ast.stmt)]
            for handler in getattr(node, "handlers", []):
                todo += handler.body
        for name in declared_global:  # a function sets it (``global x``): it exists once that ran
            names.setdefault(name, VALUE)
        # the module's namespace changed by hand: nothing absent can be trusted
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("globals", "vars") and not node.args:
                opened = True
        return {"names": names, "open": opened, "classes": classes}

    def kind(self, dotted):
        """What ``dotted`` is in this ComfyUI: a kind, ABSENT (certainly not there) or UNKNOWN."""
        if dotted in KNOWN:
            return KNOWN[dotted]
        for instance, cls in INSTANCES.items():
            if dotted.startswith(instance + "."):
                return self._instance_attribute(cls, dotted[len(instance) + 1:])
        parts = dotted.split(".")
        for split in range(len(parts), 0, -1):
            module = self._module(parts[:split])
            if module is not None:
                break
        else:
            return UNKNOWN
        rest = parts[split:]
        if not rest:
            return MODULE
        found = module["names"].get(rest[0])
        if found is None:
            return UNKNOWN if module["open"] else ABSENT
        if isinstance(found, tuple):  # ``from x import y``: what y is there
            _, level, source, name = found
            if level or not source:
                return UNKNOWN if len(rest) > 1 else VALUE
            inner = self.kind(".".join([source, name] + rest[1:]))
            if len(rest) == 1:  # the name is there, whatever it is (torch's aren't indexed)
                return inner if inner not in (ABSENT, UNKNOWN) else VALUE
            return inner if inner != ABSENT else UNKNOWN
        if len(rest) == 1:
            return found
        if found == CLASS and len(rest) == 2:
            for item in module["classes"][rest[0]].body:
                if isinstance(item, _FUNCTIONS) and item.name == rest[1]:
                    return FUNCTION
                if isinstance(item, (ast.Assign, ast.AnnAssign)):
                    targets = item.targets if isinstance(item, ast.Assign) else [item.target]
                    if any(isinstance(t, ast.Name) and t.id == rest[1] for t in targets):
                        return VALUE
        return UNKNOWN

    def class_values(self, dotted):
        """The names a class body assigns (an Enum's members), in order."""
        module_name, _, class_name = dotted.rpartition(".")
        module = self._module(module_name.split("."))
        node = module and module["classes"].get(class_name)
        return [t.id for item in (node.body if node else []) if isinstance(item, ast.Assign)
                for t in item.targets if isinstance(t, ast.Name)]

    def _instance_attribute(self, cls, attribute):
        if "." in attribute:
            return UNKNOWN
        module_name, _, class_name = cls.rpartition(".")
        module = self._module(module_name.split("."))
        node = module and module["classes"].get(class_name)
        if node is None:
            return UNKNOWN
        for item in ast.walk(node):
            if isinstance(item, _FUNCTIONS) and item.name == attribute and item in node.body:
                return FUNCTION
            if isinstance(item, ast.Attribute) and item.attr == attribute and isinstance(item.ctx, ast.Store) \
                    and isinstance(item.value, ast.Name) and item.value.id == "self":
                return VALUE
        return UNKNOWN  # set from elsewhere, perhaps: not decided here


def _directly_in(scope):
    """The nodes of ``scope``'s own code: not those in a function, lambda or class nested in it."""
    body = scope.body if isinstance(scope.body, list) else [scope.body]
    todo = list(body)
    while todo:
        node = todo.pop()
        yield node
        if isinstance(node, _SCOPES):
            # decorators, defaults and bases run in the enclosing scope
            outside = list(getattr(node, "decorator_list", [])) + list(getattr(node, "bases", []))
            if not isinstance(node, ast.ClassDef):
                outside += node.args.defaults + [d for d in node.args.kw_defaults if d is not None]
            todo += outside
        else:
            todo += list(ast.iter_child_nodes(node))


def _binds(scope, name):
    """Whether ``scope`` (a function, lambda or class) binds ``name`` itself."""
    if not isinstance(scope, ast.ClassDef):
        a = scope.args
        if any(x is not None and x.arg == name for x in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]):
            return True
    for node in _directly_in(scope):
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, (ast.Store, ast.Del)):
            return True
        if isinstance(node, _SCOPES[:2] + (ast.ClassDef,)) and node.name == name:
            return True
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any((x.asname or x.name.split(".")[0]) == name for x in node.names):
            return True
        if isinstance(node, ast.ExceptHandler) and node.name == name:
            return True
        if isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            return True
    return False


def _alias_edits(module, index, parents, exported):
    """A variable that only ever holds one ComfyUI module, class or stable object: its uses become that name.
    ``exported`` None: every module-level name is read elsewhere (``from this import *``)."""
    edits = []
    scopes = ([] if exported is None else [module.tree]) + [n for n in ast.walk(module.tree) if isinstance(n, _FUNCTIONS)]
    globals_declared = {name for node in ast.walk(module.tree) if isinstance(node, ast.Global) for name in node.names}
    for scope in scopes:
        own = list(_directly_in(scope))
        stores = {}
        for node in own:
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                stores.setdefault(node.id, []).append(node)
        for name, targets in stores.items():
            if scope is module.tree and (name in exported or name in globals_declared or name.startswith("__")):
                continue
            if scope is not module.tree and _binds_other_than_assign(scope, name):
                continue
            assigned, dotted, ok = [], None, True
            for target in targets:
                assign = parents.get(id(target))
                if not (isinstance(assign, ast.Assign) and len(assign.targets) == 1 and assign.targets[0] is target):
                    ok = False
                    break
                value = assign.value
                if isinstance(value, ast.Constant) and value.value is None:
                    continue
                full = module.resolve(value) if isinstance(value, (ast.Name, ast.Attribute)) else None
                if not full or (index.kind(full) not in (MODULE, CLASS) and full not in STABLE_VALUES) \
                        or (dotted not in (None, full)):
                    ok = False
                    break
                # run for certain once its statement is reached: only ``try`` bodies may hold it
                child, holder = assign, parents.get(id(assign))
                while holder is not scope and isinstance(holder, ast.Try) and child in holder.body:
                    child, holder = holder, parents.get(id(holder))
                if holder is not scope:
                    ok = False
                    break
                dotted = full
                assigned.append(assign)
            if not ok or not assigned:
                continue
            if scope is module.tree and any(isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef) + _FUNCTIONS)
                                            and _statement_binds(n, name) for n in ast.walk(module.tree)):
                continue
            text = ast.get_source_segment(module.source, assigned[0].value)
            first = min(a.end_lineno for a in assigned)
            loads, bad = [], False
            todo = [(scope, True)]
            while todo and not bad:
                current, direct = todo.pop()
                for node in _directly_in(current):
                    if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load):
                        # in the scope's own code a use before the assignment sees None (or nothing)
                        if direct and node.lineno <= first:
                            bad = True
                        loads.append(node)
                    elif isinstance(node, _SCOPES):
                        if _binds(node, name):
                            if isinstance(node, ast.ClassDef):
                                bad = True  # a class attribute of the same name: left alone
                        else:
                            todo.append((node, False))
            if bad or not loads:
                continue
            for node in loads:
                edits.append((*module.span(node), text))
            for assign in assigned:
                edits.append((*module.span(assign.value), "None"))
    return edits


def _statement_binds(node, name):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return any((x.asname or x.name.split(".")[0]) == name for x in node.names)
    return node.name == name


def _binds_other_than_assign(scope, name):
    a = scope.args
    if any(x is not None and x.arg == name for x in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]):
        return True
    for node in _directly_in(scope):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef) + _FUNCTIONS) and _statement_binds(node, name):
            return True
        if isinstance(node, ast.ExceptHandler) and node.name == name:
            return True
        if isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            return True
    return False


def _module_getter(function, module, index):
    """The ComfyUI module a helper hands back, or None: a function that only imports one and returns it
    (``def _mm(): try: import comfy.model_management as mm; return mm / except Exception: return None``)."""
    a = function.args
    if a.posonlyargs or a.args or a.kwonlyargs or a.vararg or a.kwarg or function.decorator_list:
        return None
    found, todo = None, list(function.body)
    while todo:
        node = todo.pop()
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # a docstring
        if isinstance(node, ast.Try):
            todo += node.body + node.orelse + [s for h in node.handlers for s in h.body]
            if node.finalbody:
                return None
        elif isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass)):
            continue
        elif isinstance(node, ast.Return):
            if node.value is None or (isinstance(node.value, ast.Constant) and node.value.value is None):
                continue
            full = module.resolve(node.value) if isinstance(node.value, (ast.Name, ast.Attribute)) else None
            if not full or index.kind(full) != MODULE or found not in (None, full):
                return None
            found = full
        else:
            return None
    return found


def _getter_edits(module, index, parents):
    """Calls of a module getter become the module, imported under a name of its own; a getter nothing
    calls any more goes."""
    getters = {}
    for node in module.tree.body:
        if isinstance(node, _FUNCTIONS) and not isinstance(node, ast.AsyncFunctionDef):
            full = _module_getter(node, module, index)
            if full:
                getters[node.name] = (node, full)
    edits, imports = [], {}
    for name, (function, full) in getters.items():
        loads = [n for n in ast.walk(module.tree) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)]
        stores = [n for n in ast.walk(module.tree) if isinstance(n, ast.Name) and n.id == name and not isinstance(n.ctx, ast.Load)]
        calls = [parents.get(id(n)) for n in loads]
        if stores or not loads or id(loads[0]) in module.shadowed or any(id(n) in module.shadowed for n in loads) \
                or not all(isinstance(c, ast.Call) and c.func is n and not c.args and not c.keywords for c, n in zip(calls, loads)):
            continue
        local = "_v2m_" + full.replace(".", "_")
        imports[local] = full
        edits += [(*module.span(call), local) for call in calls]
        start = module.pos(function.lineno, 0)
        edits.append((start, module.pos(function.end_lineno, function.end_col_offset), "pass  # V2: a ComfyUI module getter, inlined"))
    if not edits:
        return []
    # the imports go after the docstring and ``from __future__`` lines
    at = 0
    for node in module.tree.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)) \
                or (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
            at = module.pos(node.end_lineno, node.end_col_offset)
        else:
            break
    lines = "".join(f"\nimport {full} as {local}" for local, full in sorted(imports.items())) + "\n"
    return edits + [(at, at, lines)]


def _in_test(node, parents):
    """Whether only the truth of ``node`` is used: an ``if``/``while``/conditional test, ``not``, or an
    operand of ``and``/``or`` that is itself only tested."""
    parent = parents.get(id(node))
    if isinstance(parent, (ast.If, ast.While, ast.IfExp, ast.Assert)) and parent.test is node:
        return True
    if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
        return True
    if isinstance(parent, ast.BoolOp):
        if isinstance(parent.op, ast.And) and parent.values[-1] is not node:
            return True  # truthy: ``x and y`` is ``y`` either way
        return _in_test(parent, parents)
    return False


def _probe_edits(module, index, parents, builtins_free):
    edits = []
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in builtins_free \
                and not node.keywords and len(node.args) in ((2, 3) if node.func.id == "getattr" else (2,)) \
                and isinstance(node.args[0], (ast.Name, ast.Attribute)) \
                and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str) \
                and node.args[1].value.isidentifier() and id(node.func) not in module.shadowed:
            full = module.resolve(node.args[0])
            if not full or index.kind(full) not in (MODULE, CLASS, OBJECT):
                continue
            kind = index.kind(f"{full}.{node.args[1].value}")
            if kind == UNKNOWN:
                continue
            receiver = ast.get_source_segment(module.source, node.args[0])
            if node.func.id == "hasattr":
                edits.append((*module.span(node), "False" if kind == ABSENT else "True"))
            elif kind != ABSENT:
                # getattr evaluates its default even when the attribute exists.
                # Preserve computed defaults, including their side effects and errors.
                if len(node.args) == 3 and not isinstance(node.args[2], ast.Constant):
                    continue
                edits.append((*module.span(node), f"{receiver}.{node.args[1].value}"))
            elif len(node.args) == 3:
                edits.append((*module.span(node), "(" + ast.get_source_segment(module.source, node.args[2]) + ")"))
        elif isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.Is, ast.IsNot)):
            sides = [node.left, node.comparators[0]]
            nones = [s for s in sides if isinstance(s, ast.Constant) and s.value is None]
            others = [s for s in sides if isinstance(s, (ast.Name, ast.Attribute))]
            if len(nones) == 1 and len(others) == 1:
                full = module.resolve(others[0])
                if full and index.kind(full) in NOT_NONE:
                    edits.append((*module.span(node), "True" if isinstance(node.ops[0], ast.IsNot) else "False"))
        elif isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load) and _in_test(node, parents):
            full = module.resolve(node)
            if full and index.kind(full) in NOT_NONE:
                edits.append((*module.span(node), "True"))
    return edits


def resolve_probes(source, path, index, module_class, aliases=None, exported=frozenset()):
    """``source`` with its probes of ComfyUI resolved, and what was resolved (for the report).

    ``module_class`` is the converter's parsed-module class (its ``resolve`` names engine expressions);
    ``aliases``: engine names this module gets from the pack's other modules; ``exported``: module-level
    names other modules of the pack import from this one (their variables are left as they are)."""
    done = []
    for _ in range(MAX_ROUNDS):
        try:
            module = module_class(source, path)
        except SyntaxError:
            break
        for name, dotted in (aliases or {}).items():
            module.aliases.setdefault(name, dotted)
        if not module.aliases:
            break
        parents = {id(child): node for node in ast.walk(module.tree) for child in ast.iter_child_nodes(node)}
        bound = {n.id for n in ast.walk(module.tree) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))} \
            | {n.name for n in ast.walk(module.tree) if isinstance(n, _SCOPES[:2] + (ast.ClassDef,))} \
            | {n.arg for n in ast.walk(module.tree) if isinstance(n, ast.arg)}
        probes = _probe_edits(module, index, parents, {"getattr", "hasattr"} - bound)
        edits = probes or _alias_edits(module, index, parents, exported)
        getters = [] if edits else _getter_edits(module, index, parents)
        edits = edits or getters
        if not edits:
            break
        edits.sort(key=lambda e: (e[0], -e[1]))
        out, last = [], 0
        for start, end, text in edits:
            if start < last:
                continue  # inside an edit already made: the next round sees it
            out += [source[last:start], text]
            last = end
        candidate = "".join(out) + source[last:]
        try:
            compile(candidate, path, "exec")
        except SyntaxError:
            break
        source = candidate
        done.append("ComfyUI probe" if probes else "ComfyUI module getter" if getters else "ComfyUI alias")
    return source, sorted(set(done))
