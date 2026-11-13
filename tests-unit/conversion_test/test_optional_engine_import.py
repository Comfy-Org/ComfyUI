"""Optional engine import bindings survive conversion without hiding uses."""
import os as _os
import sys as _sys
_CONVERSION = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "comfy_api", "conversion")
_sys.path.insert(0, _CONVERSION)
_os.environ.setdefault("COMFY_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_os.environ["PYTHONPATH"] = _os.pathsep.join(p for p in (_CONVERSION, _os.environ["COMFY_ROOT"], _os.environ.get("PYTHONPATH", "")) if p)
import ast
import os
import sys
import types
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import convert


def rewrite(src):
    return convert.rewrite(src, 'node.py', {})


def test_optional_nodes_probe_uses_real_facade_and_preserves_fallback():
    source = '''def run():
    try:
        import nodes as available
    except ImportError:
        available = None
    return available is not None
'''
    new, done, left = rewrite(source)
    assert 'available = _v2e' in new
    assert 'available = None' in new
    assert 'nodes availability probe' in done
    assert not left
    ast.parse(new)


def test_optional_alias_fallback_executes_on_import_error():
    src = '''def run():
    try:
        from comfy.utils import ProgressBar as progress
    except ImportError:
        progress = None
    return progress
'''
    new, _, _ = rewrite(src)
    function = next(n for n in ast.parse(new).body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    code = compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), 'node.py', 'exec')
    class Unavailable:
        @property
        def ProgressBar(self):
            raise ImportError('not available')
    marker = object()
    namespace = {'_v2e': Unavailable()}
    exec(code, namespace)  # noqa: S102
    assert namespace['run']() is None
    namespace['_v2e'] = types.SimpleNamespace(ProgressBar=marker)
    assert namespace['run']() is marker


def test_owned_util_uses_dedicated_facade():
    src = '''util = "pack value"
def run():
    try:
        from comfy.utils import common_upscale as upscale
    except ImportError:
        upscale = None
    return util, upscale
'''
    new, done, left = rewrite(src)
    assert 'upscale = _v2util.common_upscale' in new
    assert 'from comfy_api.v4 import util as _v2util' in new
    assert 'comfy.utils.common_upscale' in done and not left
    body = [n for n in ast.parse(new).body if isinstance(n, (ast.Assign, ast.FunctionDef))
            and (not isinstance(n, ast.FunctionDef) or n.name == 'run')]
    marker = object()
    namespace = {'_v2util': types.SimpleNamespace(common_upscale=marker)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), 'node.py', 'exec'), namespace)  # noqa: S102
    assert namespace['run']() == ('pack value', marker)


def test_used_module_alias_keeps_import_and_reports_blocker():
    src = '''def run():
    try:
        import nodes as available
    except ImportError:
        available = None
    return available.unknown if available is not None else None
'''
    new, done, left = rewrite(src)
    assert 'import nodes as available' in new
    assert any('potential local import use' in s for s in left)


def test_colliding_argument_does_not_rewrite_probe():
    src = '''def run(available):
    try:
        import nodes as available
    except ImportError:
        available = None
    return available is not None
'''
    new, done, left = rewrite(src)
    assert 'import nodes as available' in new
    assert 'nodes availability probe' not in done
    assert left


def test_nested_scope_and_chained_fallback_are_not_rewritten():
    for src in ('''def run():
    try:
        import nodes as available
    except ImportError:
        available = None
    def inner():
        return available is not None
    return inner()
''', '''def run():
    try:
        import nodes as available
    except ImportError:
        available = other = None
    return available is not None
''', '''def run():
    try:
        import nodes as available
    except ImportError:
        del available
    return available is not None
'''):
        new, done, left = rewrite(src)
        assert 'import nodes as available' in new
        assert 'nodes availability probe' not in done
        assert left


def test_normal_imports_and_shadowed_argument_unchanged():
    src = '''import json
def run(nodes):
    return nodes is not None and json.dumps({})
'''
    new, done, left = rewrite(src)
    assert new == src and not done and not left


def test_except_fallback_with_known_symbol_uses_facade():
    src = '''def run():
    try:
        from comfy.utils import ProgressBar as progress
    except ImportError:
        progress = None
    return progress
'''
    new, done, left = rewrite(src)
    assert 'progress = _v2e.ProgressBar' in new
    assert 'progress = None' in new
    assert 'comfy.utils.ProgressBar' in done


def test_private_explicit_exports_are_retained_but_not_star_exported():
    src = 'from comfy_extras.nodes_depth_anything_3 import _da3_build_mask\n'
    assert not convert.module_engine_aliases(src, 'adapter.py')
    assert convert.module_engine_aliases(src, 'adapter.py', include_private=True) == {
        '_da3_build_mask': 'comfy_extras.nodes_depth_anything_3._da3_build_mask'}
    new, _, left = convert.rewrite(src, 'adapter.py', {}, keep=frozenset({'_da3_build_mask'}))
    assert src in new
    assert any('retained for pack import' in item for item in left)
