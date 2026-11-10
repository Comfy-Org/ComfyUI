"""Cached positions must preserve source extraction and nested edit ordering."""
import os as _os
import sys as _sys
_CONVERSION = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "comfy_api", "conversion")
_sys.path.insert(0, _CONVERSION)
_os.environ.setdefault("COMFY_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_os.environ["PYTHONPATH"] = _os.pathsep.join(p for p in (_CONVERSION, _os.environ["COMFY_ROOT"], _os.environ.get("PYTHONPATH", "")) if p)
import ast
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import convert


@pytest.mark.parametrize('newline', ['\n', '\r\n', '\r'])
def test_cached_lengths_match_python_source_segments(newline):
    source = newline.join([
        'text = "é中文\\t\u2028\u2029\x85\v\f"',
        'def run(x):',
        '    label = "中文"; return outer(inner("é"),',
        '        other(x, "α"))',
        '',
    ])
    module = convert._Module(source, 'fixture.py')
    nodes = [n for n in ast.walk(module.tree) if isinstance(n, (ast.Call, ast.Expr))]
    assert len(nodes) == 3
    for node in nodes:
        expected = ast.get_source_segment(source, node)
        assert module.length(node) == len(expected)
    assert sorted(nodes, key=module.length) == sorted(nodes, key=lambda n: len(ast.get_source_segment(source, n)))


def test_rewrite_keeps_unicode_columns():
    prefix = 'text = "é中文"\n'
    source = prefix + 'from comfy.utils import common_upscale\nresult = common_upscale(image, 8, 8, "nearest-exact", "disabled")\n'
    converted, done, leftovers = convert.rewrite(source, 'fixture.py', {})
    assert prefix in converted
    assert 'result = util.common_upscale(image, 8, 8, "nearest-exact", "disabled")' in converted
    assert done == ['comfy.utils.common_upscale']
    assert not leftovers
    ast.parse(converted)
