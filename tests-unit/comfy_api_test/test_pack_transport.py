"""Pack isolation's channel moves large tensors and numpy arrays through shared memory (the secure
overlay's carrier, shm.py), and everything else inline. Nothing is left behind."""
import os
import socket
import threading

import numpy as np
import pytest
import torch

from comfy_api._impl.isolation import channel as channel_module
from comfy_api._impl.isolation.channel import SHM_MIN_BYTES, Channel

pytestmark = pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="POSIX shared memory")


def _segments():
    return {n for n in os.listdir("/dev/shm") if n.startswith(channel_module.segment_prefix())}


@pytest.fixture
def pair():
    """Two ends of a channel in this process; the far end echoes what it gets."""
    a, b = socket.socketpair()
    near = Channel(a, lambda m: None)
    far = Channel(b, lambda m: {"got": m["value"], "type": type(m["value"]).__name__})
    near.shared_memory = far.shared_memory = True
    sent = []

    class Counting:  # what goes through the socket
        def sendall(self, data):
            sent.append(len(data))
            return a.sendall(data)

        def recv(self, size):
            return a.recv(size)
    near.sock = Counting()
    threading.Thread(target=far.serve_forever, daemon=True).start()
    yield near, sent
    near._send({"op": "shutdown", "id": 0})
    a.close()  # noqa


VALUES = {
    "float32": lambda: torch.randn(4, 64, 64, 16),
    "bfloat16": lambda: torch.randn(4, 64, 64, 16).to(torch.bfloat16),
    "strided": lambda: torch.randn(64, 64, 64)[:, ::2],
    "int64": lambda: torch.arange(100_000),
    "numpy": lambda: np.random.default_rng(0).random((128, 256)),
}


@pytest.mark.parametrize("name", sorted(VALUES))
def test_large_values_cross_through_shared_memory(pair, name):
    near, sent = pair
    value = VALUES[name]()
    nbytes = value.nbytes if isinstance(value, np.ndarray) else value.numel() * value.element_size()
    assert nbytes >= SHM_MIN_BYTES
    before = _segments()
    reply = near.request({"op": "echo", "value": value})
    got = reply["got"]
    assert reply["type"] == type(value).__name__  # numpy stays numpy, tensors stay tensors
    if isinstance(value, np.ndarray):
        assert np.array_equal(got, value) and got.dtype == value.dtype
    else:
        assert got.dtype == value.dtype and torch.equal(got, value)
    assert sum(sent) < 4096  # the data didn't go through the socket
    assert _segments() == before  # every segment was read and unlinked


def test_small_values_go_inline_and_subclasses_keep_their_type(pair):
    near, sent = pair
    small = torch.ones(10)
    got = near.request({"op": "echo", "value": small})["got"]
    assert torch.equal(got, small) and sum(sent) > small.numel() * 4  # inline
    sent.clear()
    parameter = torch.nn.Parameter(torch.randn(256, 256))
    got = near.request({"op": "echo", "value": parameter})["got"]
    # A Parameter pickles as "a Parameter around this tensor": the tensor goes through shared
    # memory and the Parameter is rebuilt around it on the other side.
    assert type(got) is torch.nn.Parameter and torch.equal(got, parameter)
    assert sum(sent) < 4096


def test_without_shared_memory_everything_goes_inline():
    a, b = socket.socketpair()
    near, far = Channel(a, lambda m: None), Channel(b, lambda m: m["value"])
    threading.Thread(target=far.serve_forever, daemon=True).start()
    value = torch.randn(256, 256)
    assert torch.equal(near.request({"op": "echo", "value": value}), value)
    near._send({"op": "shutdown", "id": 0})


def test_segments_the_other_side_never_read_are_released():
    a, b = socket.socketpair()
    near = Channel(a, lambda m: None)
    near.shared_memory = True
    before = _segments()
    b.close()  # the other side is gone before it reads anything
    with pytest.raises(Exception):
        near.request({"op": "echo", "value": torch.randn(512, 512)})
    assert _segments() == before


def test_a_pack_process_negotiates_shared_memory_and_moves_tensors_through_it(tmp_path):
    from comfy_api._impl.isolation.host import Worker

    pack = tmp_path / "transport_pack"
    pack.mkdir()
    (pack / "__init__.py").write_text("")
    worker = Worker(str(pack))
    try:
        assert worker.channel.shared_memory  # same machine: it saw ComfyUI's probe
    finally:
        worker.stop()
    assert not {n for n in os.listdir("/dev/shm") if n.startswith(f"cpack_{worker.pid}_")}


def test_a_peer_that_cant_see_the_probe_stays_inline(monkeypatch, tmp_path):
    from comfy_api._impl.isolation import channel
    from comfy_api._impl.isolation.host import Worker

    monkeypatch.setattr(channel, "probe_shared_memory", lambda: ("not-a-segment", lambda: None))
    pack = tmp_path / "transport_pack"
    pack.mkdir()
    (pack / "__init__.py").write_text("")
    worker = Worker(str(pack))
    try:
        assert not worker.channel.shared_memory
    finally:
        worker.stop()
