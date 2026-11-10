"""comfy_engine: the public API. Changing this list is an API change."""
import ast
import os

import comfy_engine as api

PUBLIC = ["ComfyExtension", "Input", "InputImpl", "Types", "eng", "io", "ui", "util"]


def test_the_public_surface_is_exactly_this():
    assert sorted(api.__all__) == sorted(PUBLIC)
    for name in PUBLIC:
        assert getattr(api, name) is not None


def test_the_stub_declares_the_same_names():
    stub = os.path.join(os.path.dirname(api.__file__), "__init__.pyi")
    tree = ast.parse(open(stub).read())
    declared = next(ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "__all__")
    assert sorted(declared) == sorted(api.__all__)
