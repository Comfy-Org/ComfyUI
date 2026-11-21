import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.model_management  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.pinned_memory  # noqa: E402


class FakeVRAMBuffer:
    def __init__(self, max_size, device):
        self.allocated = 0

    def size(self):
        return self.allocated

    def get(self, size, offset=0):
        self.allocated = max(self.allocated, size + offset)
        return size


@pytest.mark.parametrize("num_streams", [2, 3, 4])
def test_largest_weight_grows_one_stream_cast_buffer(monkeypatch, num_streams):
    streams = [object() for _ in range(num_streams)]
    counter = [0]

    def get_offload_stream(device):
        counter[0] = (counter[0] + 1) % len(streams)
        return streams[counter[0]]

    buffers = {}
    monkeypatch.setattr(comfy.model_management, "NUM_STREAMS", num_streams)
    monkeypatch.setattr(comfy.model_management, "STREAM_AIMDO_CAST_BUFFERS", buffers)
    monkeypatch.setattr(comfy.model_management, "LARGEST_AIMDO_CASTED_WEIGHT", (None, 0))
    monkeypatch.setattr(comfy.model_management, "get_offload_stream", get_offload_stream)
    monkeypatch.setattr(comfy.model_management, "cast_to_gathered", lambda *args, **kwargs: None)
    monkeypatch.setattr(comfy.model_management.comfy_aimdo.vram_buffer, "VRAMBuffer", FakeVRAMBuffer)
    monkeypatch.setattr(comfy.ops.comfy_aimdo.torch, "aimdo_to_tensor", lambda v, device: torch.empty(v, dtype=torch.uint8))
    monkeypatch.setattr(comfy.ops.comfy_aimdo.model_vbar, "vbar_fault", lambda v: None)
    monkeypatch.setattr(comfy.ops.comfy_aimdo.model_vbar, "vbar_signature_compare", lambda *args: False)
    monkeypatch.setattr(comfy.pinned_memory, "get_pin", lambda *args, **kwargs: None)
    monkeypatch.setattr(comfy.pinned_memory, "pin_memory", lambda *args, **kwargs: None)

    largest = comfy.ops.manual_cast.Linear(64, 64, bias=False)
    largest._v = (None, 0, 0)
    largest._v_signature = None
    largest._pin_state = {"fast_disk": False}
    size = largest.weight.numel() * largest.weight.element_size()

    # Other casts land between uses of the largest weight, so it is offered every stream.
    for other_casts in (0, 1, 0, 1, 1, 2, 0, 3, 2, 1):
        for _ in range(other_casts):
            get_offload_stream(None)
        comfy.ops.cast_modules_with_vbar([largest], None, torch.device("cpu"), None, False)

    assert [b.size() >= size for b in buffers.values()] == [True]
