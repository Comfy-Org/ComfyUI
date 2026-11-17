"""Stand-ins, plans, and the pack's own statements passed over (standin.py)."""
import os as _os
import sys as _sys
_CONVERSION = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "comfy_api", "conversion")
_sys.path.insert(0, _CONVERSION)
_os.environ.setdefault("COMFY_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_os.environ["PYTHONPATH"] = _os.pathsep.join(p for p in (_CONVERSION, _os.environ["COMFY_ROOT"], _os.environ.get("PYTHONPATH", "")) if p)
import ast
import importlib.util
import os
import sys
import tempfile
import textwrap
import unittest

import standin


def write(root, files):
    for rel, text in files.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as file:
            file.write(textwrap.dedent(text))


class Pack:
    """A pack in a folder of its own, imported the way packload imports one."""
    count = 0

    def __init__(self, files):
        self.folder = tempfile.TemporaryDirectory()
        self.entry = os.path.realpath(os.path.join(self.folder.name, "pack"))
        write(self.entry, files)
        Pack.count += 1
        self.name = f"pack_under_test_{Pack.count}"

    def load(self, plan):
        standin.forget(self.entry, self.name)
        session = standin.Session(plan, self.entry, self.name)
        with session:
            spec = importlib.util.spec_from_file_location(
                self.name, os.path.join(self.entry, "__init__.py"), submodule_search_locations=[self.entry])
            module = importlib.util.module_from_spec(spec)
            sys.modules[self.name] = module
            session.prepare(module)
            spec.loader.exec_module(module)
        return module, session

    def tolerant(self, rounds=8):
        """What tolerant.py does over fresh processes, in one: derive a repair from each failure until it loads."""
        plan = standin.Plan()
        for _ in range(rounds):
            try:
                module, session = self.load(plan)
                return module, session, plan
            except BaseException as error:  # noqa: B902 - SystemExit too
                repair = standin.derive(error, self.entry, self.name, plan)
                if repair is None or not plan.add(repair, str(error)):
                    raise
        raise AssertionError("no plan in %d rounds" % rounds)


class StatementsPassedOver(unittest.TestCase):
    def test_statement_at_is_the_innermost_load_time_statement(self):
        tree = ast.parse("if True:\n    x = 1\n    y = (\n        2)\ndef f():\n    z = 3\n")
        self.assertIsInstance(standin.statement_at(tree, 2), ast.Assign)
        self.assertEqual(standin.statement_at(tree, 4).targets[0].id, "y")
        self.assertIsInstance(standin.statement_at(tree, 6), ast.FunctionDef)  # a function's body isn't load time

    def test_binds(self):
        node = ast.parse("try:\n    import a.b, c as d\n    from e import f, g as h\nexcept E:\n    i, (j, *k) = 1\n").body[0]
        self.assertEqual(standin._binds(node), ["a", "d", "f", "h", "i", "j", "k"])

    def test_a_failing_statement_is_passed_over_and_the_rest_loads(self):
        pack = Pack({"__init__.py": """
            import os
            assert os.path.isdir("/no/such/models"), "ERROR: no checkpoints found"
            CONFIG = open("/no/such/file.json").read()
            class Node:
                FOLDER = os.listdir("/no/such/folder")
                RETURN_TYPES = ("IMAGE",)
            NODE_CLASS_MAPPINGS = {"Node": Node}
        """})
        with self.assertRaises(AssertionError):
            pack.load(standin.Plan())
        module, session, plan = pack.tolerant()
        self.assertEqual(sorted(s["line"] for s in plan.skips), [3, 4, 6])
        self.assertEqual(module.NODE_CLASS_MAPPINGS["Node"].RETURN_TYPES, ("IMAGE",))
        self.assertIsInstance(module.CONFIG, standin.StandIn)
        self.assertIsInstance(module.Node.FOLDER, standin.StandIn)
        self.assertEqual(len(session.skipped), 3)
        with self.assertRaises(standin.StandInUsed):  # armed once the pack is loaded
            module.CONFIG.strip()

    def test_a_statement_is_found_again_by_what_it_says_not_where_it_is(self):
        pack = Pack({"__init__.py": "X = 1\nY = int('no')\nNODE_CLASS_MAPPINGS = {}\n"})
        _, _, plan = pack.tolerant()
        write(pack.entry, {"__init__.py": "# a header the converter added\nimport os\nX = 1\nY = int('no')\nNODE_CLASS_MAPPINGS = {}\n"})
        module, session = pack.load(plan)
        self.assertIsInstance(module.Y, standin.StandIn)
        self.assertEqual(len(session.skipped), 1)

    def test_a_failure_inside_a_function_called_at_import_passes_over_the_calling_statement(self):
        pack = Pack({"__init__.py": "from .setup import ready\nNODE_CLASS_MAPPINGS = {}\n",
                     "setup.py": "def check():\n    raise RuntimeError('no GPU')\nready = check()\nother = 2\n"})
        module, _, plan = pack.tolerant()
        self.assertEqual([(s["file"], s["line"]) for s in plan.skips], [("setup.py", 3)])
        self.assertIsInstance(module.ready, standin.StandIn)

    def test_sys_exit_at_import_is_passed_over(self):
        pack = Pack({"__init__.py": "import sys\nif True:\n    sys.exit('missing requirements')\nNODE_CLASS_MAPPINGS = {'a': 1}\n"})
        module, _, plan = pack.tolerant()
        self.assertEqual(plan.skips[0]["line"], 3)
        self.assertEqual(module.NODE_CLASS_MAPPINGS, {"a": 1})

    def test_nothing_is_changed_without_a_plan(self):
        pack = Pack({"__init__.py": "NODE_CLASS_MAPPINGS = {'a': 1}\n"})
        loader = importlib.machinery.SourceFileLoader
        before = (loader.get_code, loader.exec_module)
        pack.load(standin.Plan())
        self.assertEqual((loader.get_code, loader.exec_module), before)
        _, _, plan = Pack({"__init__.py": "Y = int('no')\n"}).tolerant()
        self.assertTrue(plan.skips)
        self.assertEqual((loader.get_code, loader.exec_module), before)  # undone when the session ends


class MissingThings(unittest.TestCase):
    def test_a_missing_package_is_stood_in_for(self):
        pack = Pack({"__init__.py": """
            import no_such_package_xyz
            from no_such_package_xyz.models import Model
            class Node(Model):
                RETURN_TYPES = ("IMAGE",)
            NODE_CLASS_MAPPINGS = {"Node": Node}
        """})
        module, _, plan = pack.tolerant()
        self.assertEqual(plan.modules, ["no_such_package_xyz"])
        self.assertFalse(plan.skips)
        self.assertTrue(standin.contains_stand_in(module.Node))

    def test_a_missing_file_of_its_own_is_keyed_by_the_pack(self):
        pack = Pack({"__init__.py": "from .vendored import thing\nNODE_CLASS_MAPPINGS = {}\n"})
        _, _, plan = pack.tolerant()
        self.assertEqual(plan.modules, ["<pack>.vendored"])

    def test_a_name_a_pack_file_lacks_is_given_by_file(self):
        pack = Pack({"__init__.py": "from .nodes import NODE_CLASS_MAPPINGS, EXTRA\n", "nodes.py": "NODE_CLASS_MAPPINGS = {'a': 1}\n"})
        module, _, plan = pack.tolerant()
        self.assertEqual(plan.names, {"<pack>.nodes": ["EXTRA"]})
        self.assertEqual(module.NODE_CLASS_MAPPINGS, {"a": 1})

    def test_a_pack_that_imports_itself_by_its_folders_name(self):
        files = {"__init__.py": "from pack.standard import Tile\nNODE_CLASS_MAPPINGS = {'Tile': Tile}\n",
                 "standard.py": "class Tile:\n    WHERE = __file__\n"}
        first = Pack(files)
        module, _, plan = first.tolerant()
        self.assertEqual((plan.aliases, plan.modules), (["pack"], []))
        self.assertFalse(standin.contains_stand_in(module.Tile))
        second = Pack(files)  # its conversion, loaded after it in the same process, finds its own files
        again, _ = second.load(plan)
        self.assertTrue(again.Tile.WHERE.startswith(second.entry))

    def test_a_name_the_pack_tries_and_goes_on_without_is_not_stood_in(self):
        # ``from json import X`` fails (a library without it, as ComfyUI's ``app`` is for a pack's own app.py);
        # the pack expects that and imports its own file, which needs a missing package: that package is the repair
        pack = Pack({"__init__.py": """
            try:
                from json import process_image
            except ImportError:
                from .app import process_image
                from .app import table
            NODE_CLASS_MAPPINGS = {k: object for k in table}
        """, "app.py": "import no_such_package_q\ndef process_image():\n    pass\ntable = {'a': 1}\n"})
        module, _, plan = pack.tolerant()
        self.assertEqual((plan.modules, plan.names), (["no_such_package_q"], {}))
        self.assertEqual(list(module.NODE_CLASS_MAPPINGS), ["a"])

    def test_a_plan_round_trips(self):
        pack = Pack({"__init__.py": "import no_such_package_xyz\nY = int('no')\n"})
        _, _, plan = pack.tolerant()
        path = os.path.join(pack.folder.name, "plan.json")
        plan.save(path)
        again = standin.Plan.load(path)
        self.assertEqual(again.to_json(), plan.to_json())
        self.assertFalse(again.add(("skips", dict(plan.skips[0], line=99))))  # the same statement, wherever it is


class ComfyEnv(unittest.TestCase):
    def test_node_folders_are_imported_in_process(self):
        pack = Pack({"__init__.py": "from fake_comfy_env import register_nodes\nNODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes()\n",
                     "nodes/__init__.py": "",
                     "nodes/a/__init__.py": "NODE_CLASS_MAPPINGS = {'A': 1}\nNODE_DISPLAY_NAME_MAPPINGS = {'A': 'a'}\n",
                     "nodes/b/__init__.py": "NODE_CLASS_MAPPINGS = {'B': 2}\n",
                     "nodes/_private/__init__.py": "raise RuntimeError('not a node folder')\n"})
        library = tempfile.TemporaryDirectory()
        write(library.name, {"fake_comfy_env.py": "def register_nodes(nodes_package='nodes'):\n    raise RuntimeError('config rejected')\n"})
        sys.path.insert(0, library.name)
        standin.SUBSTITUTES["fake_comfy_env:register_nodes"] = standin._register_nodes_here
        try:
            module, _, plan = pack.tolerant()
        finally:
            sys.path.remove(library.name)
            del standin.SUBSTITUTES["fake_comfy_env:register_nodes"]
        self.assertEqual(plan.calls, ["fake_comfy_env:register_nodes"])
        self.assertEqual(module.NODE_CLASS_MAPPINGS, {"A": 1, "B": 2})
        self.assertEqual(module.NODE_DISPLAY_NAME_MAPPINGS, {"A": "a"})


class Conditions(unittest.TestCase):
    def test_an_if_whose_condition_fails_keeps_its_body(self):
        pack = Pack({"__init__.py": """\
            def available():
                raise RuntimeError('nothing to ask')
            if available():
                NODES = {'A': 1}
            if not available():
                raise SystemExit('not here')
            elif not not available():
                NODES['B'] = 2
            NODE_CLASS_MAPPINGS = NODES
            """})
        module, session, plan = pack.tolerant()
        self.assertEqual(module.NODE_CLASS_MAPPINGS, {"A": 1, "B": 2})
        self.assertEqual([(s["line"], s.get("part")) for s in plan.skips], [(3, "condition"), (5, "condition"), (7, "condition")])
        self.assertEqual(len(session.skipped), 3)


class CaughtByThePack(unittest.TestCase):
    def test_a_failed_import_of_its_own_that_the_pack_catches_is_seen(self):
        pack = Pack({"__init__.py": """\
            try:
                from .nodes import NODE_CLASS_MAPPINGS
            except Exception as error:
                NODE_CLASS_MAPPINGS = {}
            try:
                import an_optional_library_not_here
            except ImportError:
                pass
            try:
                {}['x']
            except KeyError:
                pass
            """, "nodes.py": "import a_library_not_here\nNODE_CLASS_MAPPINGS = {'A': 1}\n"})
        plan = standin.Plan()
        module, session = pack.load(plan)
        self.assertEqual(module.NODE_CLASS_MAPPINGS, {})
        found = standin.failed_imports(session.caught, pack.entry)
        self.assertEqual([(error.name, own) for error, own in found], [("a_library_not_here", True), ("an_optional_library_not_here", False)])
        self.assertEqual(standin.derive(found[0][0], pack.entry, pack.name, plan), ("modules", "a_library_not_here"))
        plan.add(("modules", "a_library_not_here"))
        module, session = pack.load(plan)
        self.assertEqual(module.NODE_CLASS_MAPPINGS, {"A": 1})
        self.assertEqual([own for _, own in standin.failed_imports(session.caught, pack.entry)], [False])

    def test_a_pack_that_imports_itself_from_custom_nodes_by_another_folders_name(self):
        pack = Pack({"__init__.py": """\
            import importlib
            NODE_CLASS_MAPPINGS = {}
            for name in ['a']:
                try:
                    NODE_CLASS_MAPPINGS.update(importlib.import_module('custom_nodes.Authors_Folder.nodes.' + name).NODES)
                except Exception as error:
                    print(error)
            """, "nodes/__init__.py": "", "nodes/a.py": "NODES = {'A': 1}\n"})
        self.addCleanup(lambda: [sys.modules.pop(n) for n in list(sys.modules) if n.split(".")[0] == "custom_nodes"])
        plan = standin.Plan()
        module, session = pack.load(plan)
        (error, own), = standin.failed_imports(session.caught, pack.entry)
        repair = standin.derive(error, pack.entry, pack.name, plan)
        self.assertEqual(repair, ("aliases", "custom_nodes.Authors_Folder"))
        plan.add(repair)
        module, session = pack.load(plan)
        self.assertEqual(module.NODE_CLASS_MAPPINGS, {"A": 1})
        self.assertEqual(standin.failed_imports(session.caught, pack.entry), [])


if __name__ == "__main__":
    unittest.main(verbosity=1)
