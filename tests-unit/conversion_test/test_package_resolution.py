"""Internal import analysis follows Python's package-before-module lookup."""
import os as _os
import sys as _sys
_CONVERSION = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "comfy_api", "conversion")
_sys.path.insert(0, _CONVERSION)
_os.environ.setdefault("COMFY_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_os.environ["PYTHONPATH"] = _os.pathsep.join(p for p in (_CONVERSION, _os.environ["COMFY_ROOT"], _os.environ.get("PYTHONPATH", "")) if p)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import convert


def test_same_name_package_wins_and_registration_is_preserved(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / '__init__.py').write_text('from .nodes import NODE_CLASS_MAPPINGS\n')
    (source / 'nodes.py').write_text('from nodes import NODE_CLASS_MAPPINGS\n')
    (source / 'nodes').mkdir()
    (source / 'nodes' / '__init__.py').write_text('NODE_CLASS_MAPPINGS = {"PackNode": object}\n')
    convert.generate(str(source), {'nodes': {}, 'routes': [], 'web_directory': None}, str(tmp_path / 'out'))
    new = (tmp_path / 'out' / 'source' / 'v2' / '__init__.py').read_text()
    assert 'from .nodes import NODE_CLASS_MAPPINGS' in new


def test_actual_module_engine_reexport_still_detected(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / '__init__.py').write_text('from .bridge import thing\ndef use():\n    return thing\n')
    (source / 'bridge.py').write_text('from comfy.utils import ProgressBar as thing\n')
    report = convert.generate(str(source), {'nodes': {}, 'routes': [], 'web_directory': None}, str(tmp_path / 'out'))
    new = (tmp_path / 'out' / 'source' / 'v2' / '__init__.py').read_text()
    assert '_v2e.ProgressBar' in new
    assert 'comfy.utils.ProgressBar' in report['rewritten']['__init__.py']
