# ruff: noqa: F821
import os as _os
import sys as _sys
_CONVERSION = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "comfy_api", "conversion")
_sys.path.insert(0, _CONVERSION)
_os.environ.setdefault("COMFY_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_os.environ["PYTHONPATH"] = _os.pathsep.join(p for p in (_CONVERSION, _os.environ["COMFY_ROOT"], _os.environ.get("PYTHONPATH", "")) if p)
import ast
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import convert
import describe


class LocalSamplers:
    SAMPLER_NAMES = ['euler', 'custom']

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'sampler': (cls.SAMPLER_NAMES,)}}


class CoreSamplers:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'sampler': (comfy.samplers.KSampler.SAMPLERS,),
            'scheduler_class': (comfy.samplers.KSampler.SCHEDULERS,),
            'sampler_module': (comfy.samplers.SAMPLER_NAMES,),
            'schedule': (comfy.samplers.SCHEDULER_NAMES,),
            'direct': (SAMPLER_NAMES,),
            'alias': (samplers_alias.SAMPLER_NAMES,),
            'class_alias': (SamplerAlias.SCHEDULERS,),
            'custom': (custom.SAMPLER_NAMES,),
        }}


def test_multiselect_uses_multi_combo_and_preserves_options():
    source = convert._input('actions', [['age', 'gender'], {'multiselect': True, 'default': ['age']}], 'required', {})
    call = ast.parse(source, mode='eval').body
    assert ast.unparse(call.func) == 'io.MultiCombo.Input'
    args = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    assert args['options'] == ['age', 'gender']
    assert args['extra_dict'] == {'multiselect': True, 'default': ['age']}


def test_single_select_stays_combo():
    source = convert._input('mode', [['a', 'b'], {'default': 'a'}], 'optional', {})
    call = ast.parse(source, mode='eval').body
    assert ast.unparse(call.func) == 'io.Combo.Input'
    assert {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}['optional'] is True


def test_only_core_sampler_list_gets_dynamic_hint(monkeypatch):
    module = types.ModuleType('comfy.samplers')
    module.SAMPLER_NAMES = ['euler']
    module.SCHEDULER_NAMES = ['normal']
    module.KSampler = type('KSampler', (), {'SAMPLERS': module.SAMPLER_NAMES, 'SCHEDULERS': module.SCHEDULER_NAMES})
    comfy_module = types.ModuleType('comfy')
    comfy_module.samplers = module
    monkeypatch.setitem(sys.modules, 'comfy.samplers', module)
    monkeypatch.setitem(globals(), 'comfy', comfy_module)
    monkeypatch.setitem(globals(), 'SAMPLER_NAMES', module.SAMPLER_NAMES)
    monkeypatch.setitem(globals(), 'samplers_alias', module)
    monkeypatch.setitem(globals(), 'SamplerAlias', module.KSampler)
    monkeypatch.setitem(globals(), 'custom', types.SimpleNamespace(SAMPLER_NAMES=['custom']))
    assert describe.input_hints(LocalSamplers) == {}
    assert describe.input_hints(CoreSamplers) == {
        'sampler': {'util_constant': 'SAMPLER_NAMES'},
        'scheduler_class': {'util_constant': 'SCHEDULER_NAMES'},
        'sampler_module': {'util_constant': 'SAMPLER_NAMES'},
        'schedule': {'util_constant': 'SCHEDULER_NAMES'},
        'direct': {'util_constant': 'SAMPLER_NAMES'},
        'alias': {'util_constant': 'SAMPLER_NAMES'},
        'class_alias': {'util_constant': 'SCHEDULER_NAMES'},
    }
