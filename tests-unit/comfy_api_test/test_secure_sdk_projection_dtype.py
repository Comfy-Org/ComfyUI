import asyncio
from types import SimpleNamespace

import pytest
import torch

from comfy_api.latest import _sdk
import comfy.model_management as mm


@pytest.mark.parametrize("dtype,fp16,expected", [(torch.float32, False, "float32"), (torch.float16, False, "float16"), (torch.bfloat16, True, "bfloat16"), (torch.float64, False, "float32"), (torch.float64, True, "float16"), (torch.float8_e4m3fn, True, "float16"), (torch.float8_e4m3fn, False, "float32")])
def test_projection_policy_preserves_selected_precision_and_legacy_fallback(monkeypatch, dtype, fp16, expected):
    monkeypatch.setattr(mm, "unet_dtype", lambda: dtype)
    monkeypatch.setattr(mm, "should_use_fp16", lambda: fp16)
    async def run():
        refs = _sdk.InProcessRefResolver()
        model = _sdk.ModelRef._wrap(await refs.create("MODEL", SimpleNamespace()))
        runtime = _sdk.Runtime(refs=refs, ops=_sdk.InProcessOps(), ctx=SimpleNamespace())
        with _sdk.bind_runtime(runtime.refs, runtime.ctx, runtime.ops):
            assert await model.projection_dtype() == expected
    asyncio.run(run())
