"""Legacy setter argument contracts against the real CPU core and canonical engine rows."""
import os as _os
import sys as _sys
_CONVERSION = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "comfy_api", "conversion")
_sys.path.insert(0, _CONVERSION)
_os.environ.setdefault("COMFY_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_os.environ["PYTHONPATH"] = _os.pathsep.join(p for p in (_CONVERSION, _os.environ["COMFY_ROOT"], _os.environ.get("PYTHONPATH", "")) if p)
import asyncio
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import convert


def rewrite(statement, asynchronous=False):
    source = f"class N:\n    {'async ' if asynchronous else ''}def run(self, model, hook, flag, record):\n        {statement}\n        return model\n"
    new, done, left = convert.rewrite(source, "n.py", {"N": ("run", {"model": "MODEL"})})
    return source, new[new.index("class N:"):] if done else new, done, left


@pytest.mark.parametrize("statement", [
    "model.set_model_sampler_cfg_function(hook, bogus=flag)",
    "model.set_model_sampler_post_cfg_function(hook, **flag)",
    "model.set_model_sampler_pre_cfg_function(*hook)",
    "model.set_model_attn1_patch(hook, flag)",
    "model.set_model_denoise_mask_function(hook, unknown=flag)",
    "model.set_model_sampler_cfg_function(hook, True, False)",
    "model.set_model_sampler_cfg_function(hook, sampler_cfg_function=hook)",
])
def test_unrepresentable_arguments_remain_blocked(statement):
    _, new, done, left = rewrite(statement)
    assert not done
    assert statement in new
    assert any("unsupported arguments" in item for item in left)


@pytest.fixture(scope="module")
def core():
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import torch
    from comfy.model_patcher import ModelPatcher
    from comfy_api.latest import _engine

    def model(initial):
        result = ModelPatcher(torch.nn.Linear(1, 1), torch.device("cpu"), torch.device("cpu"))
        if initial:
            result.model_options["disable_cfg1_optimization"] = True
        return result

    ops = _engine.core_extension("model")

    class AsyncOps:
        def __getattr__(self, name):
            async def run(*args):
                return getattr(ops, name)(*args)
            return run

    return model, types.SimpleNamespace(model=AsyncOps())


@pytest.mark.parametrize("setter,parameter,key", [
    ("cfg", "sampler_cfg_function", "sampler_cfg_function"),
    ("post_cfg", "post_cfg_function", "sampler_post_cfg_function"),
    ("pre_cfg", "pre_cfg_function", "sampler_pre_cfg_function"),
])
@pytest.mark.parametrize("form", ["positional", "keyword", "reverse_keywords"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("initial,enabled", [(False, False), (False, True), (True, False), (True, True)])
def test_state_hook_execution_and_evaluation_order(core, setter, parameter, key, form, asynchronous, initial, enabled):
    args = {"positional": "record('hook', hook), record('flag', flag)",
            "keyword": f"{parameter}=record('hook', hook), disable_cfg1_optimization=record('flag', flag)",
            "reverse_keywords": f"disable_cfg1_optimization=record('flag', flag), {parameter}=record('hook', hook)"}[form]
    source, new, done, left = rewrite(f"model.set_model_sampler_{setter}_function({args})", asynchronous)
    assert done and not left
    make_model, eng = core
    observations = []
    for code in (source, new):
        events = []
        class Flag:
            def __bool__(self):
                events.append("bool")
                return enabled
        def record(name, value):
            events.append(name)
            return value
        def hook(args):
            events.append("hook invoked")
            return args["value"] + 7
        namespace = {"eng": eng, "_v2e": types.SimpleNamespace(sync=asyncio.run)}
        exec(code, namespace)  # noqa: S102
        original = make_model(initial)
        result = namespace["N"]().run(original, hook, Flag(), record)
        if asynchronous:
            result = asyncio.run(result)
        installed = result.model_options[key]
        if setter != "cfg":
            assert len(installed) == 1
            installed = installed[0]
        assert installed({"value": 5}) == 12
        observations.append((dict(result.model_options), events))
        if code == new:
            assert result is not original
            assert key not in original.model_options
            assert bool(original.model_options.get("disable_cfg1_optimization")) == initial
    old, converted = observations
    assert old[1] == converted[1]
    assert bool(old[0].get("disable_cfg1_optimization")) == bool(converted[0].get("disable_cfg1_optimization")) == (initial or enabled)


@pytest.mark.parametrize("statement", [
    "model.set_model_attn1_patch(patch=hook)",
    "model.set_model_unet_function_wrapper(unet_wrapper_function=hook)",
    "model.set_model_denoise_mask_function(denoise_mask_function=hook)",
])
def test_single_argument_keywords_compile(statement):
    _, new, done, left = rewrite(statement)
    assert done and not left
    compile(new, "converted", "exec")


def test_keyword_hook_body_keeps_boundary_audit():
    source = '''class N:
    def run(self, model):
        def hook(args):
            return args['model']
        model.set_model_sampler_post_cfg_function(post_cfg_function=hook, disable_cfg1_optimization=True)
        return model
'''
    _, done, left = convert.rewrite(source, "n.py", {"N": ("run", {"model": "MODEL"})})
    assert done
    assert "sampling function args['model'] (not passed to pack code)" in left


@pytest.mark.parametrize("statement", [
    "value = model.set_model_sampler_cfg_function(hook, flag)",
    "record(model.set_model_sampler_post_cfg_function(hook, flag))",
])
def test_setter_result_context_is_explicit_blocker(statement):
    _, _, done, left = rewrite(statement)
    assert not done
    assert any("unsupported arguments or call context" in item for item in left)
