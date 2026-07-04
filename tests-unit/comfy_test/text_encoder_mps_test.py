"""Text encoder device placement on Apple Silicon (MPS).

MPS machines run in VRAMState.SHARED. Text encoders the device supports
(fp16/bf16/fp32) should load on the GPU, while fp8/quantized ones must
stay on the CPU because MPS cannot cast float8 dtypes.
"""

import pytest
import torch

import comfy.model_management as mm
import comfy.ops
import comfy.sd

mps_only = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and mm.is_device_mps(mm.get_torch_device())),
    reason="requires an Apple Silicon MPS device",
)

FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]

# Big enough that text_encoder_initial_device() picks the load device.
LARGE_PARAM_COUNT = 2 * 1024 * 1024 * 1024


class DummyTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        pass


class DummyTEModel(torch.nn.Module):
    def __init__(self, device=None, dtype=None, model_options={}):
        super().__init__()
        operations = model_options.get("custom_operations", comfy.ops.manual_cast)
        self.linear = operations.Linear(8, 8, dtype=dtype, device=device)
        self.dtypes = set([dtype])
        self.construct_device = torch.device(device) if device is not None else None

    def load_sd(self, sd):
        return ([], [])

    def forward(self, x):
        return self.linear(x)


def make_clip(dtype, state_dict=[], clip_class=DummyTEModel):
    class Target:
        params = {}
        clip = clip_class
        tokenizer = DummyTokenizer

    return comfy.sd.CLIP(
        target=Target(),
        parameters=LARGE_PARAM_COUNT,
        model_options={"dtype": dtype},
        state_dict=state_dict,
        disable_dynamic=True,
    )


@pytest.fixture(autouse=True)
def unload_models():
    yield
    mm.unload_all_models()


@mps_only
class TestTextEncoderDeviceMPS:
    def test_vram_state_is_shared(self):
        assert mm.vram_state == mm.VRAMState.SHARED

    def test_text_encoder_device_is_mps(self):
        assert mm.is_device_mps(mm.text_encoder_device())

    def test_fp16_clip_loads_on_mps(self):
        clip = make_clip(torch.float16)
        assert mm.is_device_mps(clip.patcher.load_device)
        weight = clip.cond_stage_model.linear.weight
        assert weight.device.type == "mps"
        x = torch.ones((1, 8), dtype=torch.float16, device=weight.device)
        assert clip.cond_stage_model(x).device.type == "mps"

    @pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
    def test_fp8_clip_falls_back_to_cpu(self, fp8_dtype):
        clip = make_clip(fp8_dtype)
        assert clip.patcher.load_device == clip.patcher.offload_device
        assert clip.cond_stage_model.construct_device.type == "cpu"
        weight = clip.cond_stage_model.linear.weight
        assert weight.device.type == "cpu"
        x = torch.ones((1, 8), dtype=torch.float16, device=weight.device)
        assert clip.cond_stage_model(x).device.type == "cpu"

    @pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
    def test_fp8_state_dict_falls_back_to_cpu(self, fp8_dtype):
        # fp8 weights in the state dict aren't reflected in the declared
        # model dtypes, e.g. quantized checkpoints with fp16 norm layers.
        sd = {"linear.weight": torch.zeros((8, 8), dtype=fp8_dtype)}
        clip = make_clip(torch.float16, state_dict=[sd])
        assert not mm.is_device_mps(clip.patcher.load_device)
        assert clip.cond_stage_model.construct_device.type == "cpu"

    def test_comfy_quant_state_dict_falls_back_to_cpu(self):
        sd = {
            "linear.weight": torch.zeros((8, 8), dtype=torch.uint8),
            "linear.comfy_quant": torch.zeros(16, dtype=torch.uint8),
        }
        clip = make_clip(torch.float16, state_dict=[sd])
        assert not mm.is_device_mps(clip.patcher.load_device)
        assert clip.cond_stage_model.construct_device.type == "cpu"

    @pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
    def test_mixed_declared_dtypes_fall_back_to_cpu(self, fp8_dtype):
        # A secondary declared dtype (e.g. dtype_llama) can be fp8 while the
        # primary dtype is fp16.
        class MixedDtypeTEModel(DummyTEModel):
            def __init__(self, device=None, dtype=None, model_options={}):
                super().__init__(device=device, dtype=dtype, model_options=model_options)
                self.dtypes = set([dtype, fp8_dtype])

        clip = make_clip(torch.float16, clip_class=MixedDtypeTEModel)
        assert not mm.is_device_mps(clip.patcher.load_device)
        assert clip.cond_stage_model.linear.weight.device.type == "cpu"

    @pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
    def test_fp8_cast_still_unsupported_on_mps(self, fp8_dtype):
        # If a torch release adds fp8 casts on MPS, supports_cast() can be
        # updated to let fp8 text encoders onto the GPU (pytorch#132624).
        assert not mm.supports_cast(mm.get_torch_device(), fp8_dtype)
        t = torch.zeros(4, dtype=fp8_dtype, device="mps")
        with pytest.raises((RuntimeError, TypeError)):
            t.to(torch.float16)
