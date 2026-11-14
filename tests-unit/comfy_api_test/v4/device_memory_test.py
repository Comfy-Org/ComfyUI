import pytest
import torch

from comfy import model_management
from comfy.model_patcher import CoreModelPatcher
from comfy.sd import VAE
from comfy_api import v4


@pytest.fixture
def loaded_models(monkeypatch):
    models = []
    monkeypatch.setattr(model_management, "current_loaded_models", models)
    yield models
    for model in models:
        model.model_unload()


@pytest.fixture
def patcher():
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    return CoreModelPatcher(model, torch.device("cpu"), torch.device("cpu"))


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_device_info_gives_comfyuis_devices_and_dtype_names(synchronous):
    info = (
        v4.ComfyAPISync().device.info() if synchronous else await v4.eng.device.info()
    )
    device_fields = {
        "torch_device",
        "intermediate_device",
        "offload_device",
        "vae_device",
        "text_encoder_device",
        "vae_offload_device",
        "text_encoder_offload_device",
    }
    dtype_fields = {"intermediate_dtype", "unet_dtype", "vae_dtype"}
    assert info.keys() == device_fields | dtype_fields
    for key in device_fields:
        assert isinstance(torch.device(info[key]), torch.device)
    for key in dtype_fields:
        assert isinstance(getattr(torch, info[key]), torch.dtype)


@pytest.mark.asyncio
async def test_device_info_reflects_host_policy_and_returns_an_independent_snapshot(
    monkeypatch,
):
    monkeypatch.setattr(model_management.args, "fp16_intermediates", False)
    before = await v4.eng.device.info()
    monkeypatch.setattr(model_management.args, "fp16_intermediates", True)
    monkeypatch.setattr(model_management.args, "bf16_unet", True)
    monkeypatch.setattr(model_management.args, "fp16_vae", True)
    after = await v4.eng.device.info()
    assert before["intermediate_dtype"] == "float32"
    assert (after["intermediate_dtype"], after["unet_dtype"], after["vae_dtype"]) == (
        "float16",
        "bfloat16",
        "float16",
    )
    after["torch_device"] = "invalid"
    assert (await v4.eng.device.info())["torch_device"] != "invalid"


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_free_bytes_can_size_work_on_cpu_and_rejects_invalid_devices(synchronous):
    api = v4.ComfyAPISync().memory if synchronous else v4.eng.memory
    available = api.free_bytes("cpu") if synchronous else await api.free_bytes("cpu")
    chunk_bytes = 4 * 32 * 32
    assert type(available) is int
    assert available // chunk_bytes > 0
    with pytest.raises(RuntimeError, match="device"):
        if synchronous:
            api.free_bytes("invalid")
        else:
            await api.free_bytes("invalid")


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_load_models_handles_native_patchers_and_vae_wrappers(
    loaded_models, patcher, synchronous
):
    vae = VAE(
        sd={"pixel_space_vae": torch.tensor(1.0)},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    supplied = [patcher, vae]
    if synchronous:
        v4.ComfyAPISync().memory.load_models(supplied)
    else:
        await v4.eng.memory.load_models(supplied)
    assert {loaded.model for loaded in loaded_models} == {patcher, vae.patcher}
    value = torch.tensor([[1.0, 2.0]])
    torch.testing.assert_close(patcher.model(value), value)
    pixels = torch.rand(1, 8, 8, 3)
    torch.testing.assert_close(vae.decode(vae.encode(pixels)), pixels)


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_cache_requests_preserve_supplied_and_other_loaded_models(
    loaded_models, patcher, synchronous
):
    other_model = torch.nn.Linear(2, 1)
    other_patcher = CoreModelPatcher(
        other_model, torch.device("cpu"), torch.device("cpu")
    )
    await v4.eng.memory.load_models([patcher])
    model_management.load_models_gpu([other_patcher])
    before = tuple(loaded_models)
    value = torch.tensor([[1.0, 2.0]])
    other_result = other_model(value).detach().clone()
    if synchronous:
        v4.ComfyAPISync().memory.unload_all()
        v4.ComfyAPISync().memory.empty_cache()
    else:
        await v4.eng.memory.unload_all()
        await v4.eng.memory.empty_cache()
    assert tuple(loaded_models) == before
    torch.testing.assert_close(patcher.model(value), value)
    torch.testing.assert_close(other_model(value), other_result)


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.parametrize(
    "options", [None, {"working_memory_bytes": 1048576.5, "full_load": True}]
)
async def test_model_load_options_reach_native_policy_without_changing_output(
    monkeypatch, loaded_models, patcher, synchronous, options
):
    native_load = model_management.load_models_gpu
    observed = []

    def observe_load(models, *, memory_required=0, force_full_load=False):
        observed.append((models, memory_required, force_full_load))
        return native_load(
            models, memory_required=memory_required, force_full_load=force_full_load
        )

    monkeypatch.setattr(model_management, "load_models_gpu", observe_load)
    before = dict(options) if options is not None else None
    if synchronous:
        v4.ComfyAPISync().memory.load_models([patcher], options=options)
    else:
        await v4.eng.memory.load_models([patcher], options=options)
    expected_bytes, expected_full = (1048576.5, True) if options else (0, False)
    assert observed == [([patcher], expected_bytes, expected_full)]
    assert options == before
    assert [loaded.model for loaded in loaded_models] == [patcher]
    value = torch.tensor([[1.0, 2.0]])
    torch.testing.assert_close(patcher.model(value), value)


@pytest.mark.asyncio
async def test_free_bytes_refuses_an_empty_device_name():
    with pytest.raises(RuntimeError):
        await v4.eng.memory.free_bytes("")
