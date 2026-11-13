import asyncio
import functools
import os
import random
import selectors
import signal
import socket
import threading
import time
import types

import numpy
import psutil
import pytest
import torch

from comfy_api.v4 import _channel, _codec, _pack_processes
from conftest import evaluate, needs_pack_process, start_probe, wait_until


class Side:
    """What `_codec` needs of a side, for values that are data."""

    peer = "the other side"

    def __init__(self):
        self.kept, self.io_types = [], {}

    def keep(self, obj, chain):
        self.kept.append(obj)
        return 0, len(self.kept) - 1

    def forget(self, keys):
        pass

    def hold(self, obj, chain):
        pass

    def resolve(self, key, chain=None):
        return self.kept[key[1]]

    def offers(self, obj):
        return (), {}

    def remote(self, key, type_name, methods, values):
        return _codec.Remote(self, key, type_name, methods, values)


def frame_across(head: bytes, body: bytes, blocks):
    ours, theirs = socket.socketpair()
    writer = threading.Thread(
        target=_channel.write_frame, args=(ours, head, body, blocks)
    )
    writer.start()
    try:
        return _channel.read_frame(theirs, bool)
    finally:
        writer.join()
        ours.close()
        theirs.close()


def across(value):
    side = Side()
    body, blocks = _codec.dumps(value, side, 1)
    _, body, blocks = frame_across(b"h", body, blocks)
    return _codec.loads(body, side, 1, blocks)


def blocks_of(*sizes):
    generator = random.Random(sum(sizes))
    return [memoryview(generator.randbytes(size)) for size in sizes]


@pytest.mark.parametrize(
    "sizes",
    [
        (),
        (0,),
        (0, 0, 0),
        (0, _channel.JOINED, 0),
        (_channel.JOINED - 1,),
        (_channel.JOINED,),
        (_channel.JOINED + 1, 3, _channel.JOINED - 1, 1),
        tuple(random.Random(5).randrange(0, 300) for _ in range(5000)),
    ],
    ids=[
        "none",
        "one empty",
        "three empty",
        "empty around a joined",
        "just under joined",
        "exactly joined",
        "mixed",
        "5000 small",
    ],
)
def test_a_frame_of_any_shape_of_blocks_crosses_whole(sizes):
    sent = blocks_of(*sizes)

    head, body, received = frame_across(b"head", b"body", sent)

    assert (bytes(head), bytes(body)) == (b"head", b"body")
    assert [bytes(block) for block in received] == [bytes(block) for block in sent]


@pytest.mark.parametrize("pieces", [1, 1.5, 3])
def test_a_block_of_a_piece_or_a_part_of_one_more_crosses_whole(monkeypatch, pieces):
    monkeypatch.setattr(_channel, "PIECE", _channel.JOINED * 2)
    sent = blocks_of(int(_channel.PIECE * pieces), 7)

    _, _, received = frame_across(b"", b"", sent)

    assert [bytes(block) for block in received] == [bytes(block) for block in sent]


def test_memory_for_a_frame_is_not_touched_before_its_bytes_arrive():
    ours, theirs = socket.socketpair()
    claimed = 256 << 20
    ours.sendall(_channel._SIZES.pack(0, claimed, 0))
    ours.close()
    process = psutil.Process()
    resident = [process.memory_info().rss]

    def wait():
        resident.append(process.memory_info().rss)

    try:
        with pytest.raises(EOFError):
            _channel.read_frame(theirs, wait)
    finally:
        theirs.close()

    grown = max(resident) - resident[0]
    assert grown < claimed // 2, f"touched {grown >> 20} MiB for 24 bytes received"


class FailsPartWay:
    """A socket whose sends stop with an error that is not an OSError once `limit` bytes
    have gone, as a signal handler's exception or KeyboardInterrupt does in `sendall`."""

    def __init__(self, sock, limit):
        self.sock, self.limit, self.failed = sock, limit, False

    def sendall(self, data):
        data = memoryview(data).cast("B")
        if not self.failed and data.nbytes > self.limit:
            self.sock.sendall(data[: self.limit])
            self.failed = True
            raise KeyboardInterrupt
        self.sock.sendall(data)

    def __getattr__(self, name):
        return getattr(self.sock, name)


def test_a_send_that_stops_part_way_leaves_no_half_frame_for_the_next_message():
    ours, theirs = socket.socketpair()
    channel = _channel.Channel(ours, types.SimpleNamespace(peer="a pack"))
    channel.sock = FailsPartWay(ours, 4096)
    tensor = torch.arange(1 << 16, dtype=torch.float32)
    body, blocks = _codec.dumps(tensor, Side(), 1)

    with pytest.raises(KeyboardInterrupt):
        channel._send({"to": 1, "chain": 1}, body, blocks)
    sent_after = None
    try:
        channel.notify({"output": "stdout", "text": "x" * 300_000})
        sent_after = True
    except _channel.PackStopped:
        sent_after = False
    ours.close()

    theirs.settimeout(5)
    outcome = []
    try:
        head, body, received = _channel.read_frame(theirs, bool)
        value = _codec.loads(body, Side(), 1, received)
        outcome = ["read", torch.equal(value, tensor)]
    except (EOFError, OSError, Exception) as error:
        outcome = ["failed", type(error).__name__]
    theirs.close()

    # Either the channel ended (nothing more is sent, the peer sees the stream end) or the
    # peer gets the tensor whole; never a tensor made of the next message's bytes.
    assert sent_after is False or outcome == ["read", True], (sent_after, outcome)


@pytest.mark.parametrize(
    "tensor",
    [
        torch.rand(3, 4).to(torch.float8_e4m3fn),
        torch.rand(3, 4).to(torch.float8_e5m2),
        torch.rand(3, 4, dtype=torch.complex64).to(torch.complex32),
        torch.rand(3, 4, dtype=torch.complex128),
        (torch.rand(5) * 1000).to(torch.uint16),
        (torch.rand(5) * 1000).to(torch.uint32),
        (torch.rand(5) * 1000).to(torch.uint64),
        torch.rand(4, dtype=torch.complex64).conj(),
        torch._neg_view(torch.rand(4)),
        torch.rand(1, 3).expand(4, 3),
        torch.rand(6, 4)[::2, 1:].t(),
        torch.zeros(0, 3, 2),
        torch.tensor(2.5),
    ],
    ids=[
        "float8_e4m3fn",
        "float8_e5m2",
        "complex32",
        "complex128",
        "uint16",
        "uint32",
        "uint64",
        "conj",
        "neg",
        "expanded",
        "strided transposed",
        "zero size",
        "0-dim",
    ],
)
def test_every_strided_tensor_crosses_with_its_bytes(tensor):
    received = across(tensor)

    expected = tensor.resolve_conj().resolve_neg()
    assert (received.dtype, received.shape) == (expected.dtype, expected.shape)
    assert bytes(received.reshape(-1).view(torch.uint8).numpy()) == bytes(
        expected.reshape(-1).view(torch.uint8).numpy()
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda: torch.quantize_per_tensor(torch.rand(4), 0.1, 0, torch.quint8),
        lambda: torch.empty(3, device="meta"),
        lambda: torch.nested.nested_tensor([torch.rand(2), torch.rand(3)]),
    ],
    ids=["quantized", "meta", "nested"],
)
def test_a_tensor_whose_bytes_cannot_be_sent_stays_or_is_refused_naming_it(make):
    tensor = make()

    try:
        received = across(tensor)
    except Exception as error:
        assert "tensor" in str(error).lower(), str(error)
    else:
        assert type(received) is _codec.Remote


def test_a_small_view_of_a_large_tensor_sends_only_the_view():
    large = torch.zeros(1 << 22)

    _, blocks = _codec.dumps(large[100:110], Side(), 1)

    assert [block.nbytes for block in blocks] == [40]


def test_small_tensors_in_one_message_arrive_aligned_to_their_element_size():
    sent = [torch.ones(3, dtype=torch.bool), torch.arange(4, dtype=torch.float64)]

    received = across(sent)

    assert received[1].data_ptr() % 8 == 0
    assert across([numpy.ones(3, numpy.int8), numpy.arange(4.0)])[1].flags.aligned


@pytest.mark.parametrize(
    "array",
    [
        numpy.arange(10)[::-2],
        numpy.arange(4, dtype=">f4"),
        numpy.array(3.5),
        numpy.zeros((0, 3), numpy.float32),
        numpy.array([True, False]),
    ],
    ids=["negative stride", "big endian", "0-dim", "zero size", "bool"],
)
def test_a_numpy_array_crosses_with_its_dtype_shape_and_values(array):
    received = across(array)

    assert type(received) is numpy.ndarray
    assert (received.dtype, received.shape) == (array.dtype, array.shape)
    assert (received == array).all()


@pytest.mark.parametrize(
    "scalar",
    [numpy.complex64(1 + 2j), numpy.array([1.5], ">f4")[0], numpy.bool_(True)],
    ids=["complex", "big endian", "bool"],
)
def test_a_numpy_scalar_crosses_as_the_same_type_and_value(scalar):
    received = across(scalar)

    assert (type(received), received.dtype, received) == (
        type(scalar),
        scalar.dtype,
        scalar,
    )


@pytest.mark.parametrize(
    "array",
    [
        numpy.array([1, "a"], dtype=object),
        numpy.zeros(2, dtype=[("a", "i4"), ("b", "f4")]),
        numpy.array(["2020-01-01"], dtype="datetime64[D]"),
    ],
    ids=["object", "structured", "datetime"],
)
def test_a_numpy_array_that_is_not_of_numbers_stays_as_a_handle(array):
    assert type(across(array)) is _codec.Remote


@needs_pack_process
def test_large_tensors_sent_both_ways_on_both_lanes_at_once_arrive_whole(probe):
    generator = torch.Generator().manual_seed(3)
    tensors = [
        torch.randint(0, 1 << 30, (8 << 20,), generator=generator, dtype=torch.int32)
        for _ in range(2)
    ]
    expression = "(print('x' * 100_000), value.clone())[1]"

    async def both():
        inputs = [{"expression": expression, "value": t} for t in tensors]
        return await asyncio.gather(
            probe.run(("node", "Eval", "execute"), (inputs[0], {}), "exec"),
            probe.run(("node", "Eval", "execute"), (inputs[1], {}), "server"),
        )

    results = []

    def in_thread():
        results.append(asyncio.run(both()))

    threads = [threading.Thread(target=in_thread) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120)

    assert len(results) == 2
    for answers in results:
        assert [torch.equal(a.args[0], t) for a, t in zip(answers, tensors)] == [
            True,
            True,
        ]


@needs_pack_process
def test_ending_a_pack_whose_process_has_been_reaped_signals_no_process_group(
    tmp_path, monkeypatch
):
    process = start_probe(tmp_path)
    os.kill(process.process.pid, signal.SIGKILL)
    wait_until(lambda: process.process.poll() is not None, "the pack reaped")
    signalled = []
    real_killpg = os.killpg
    monkeypatch.setattr(
        os,
        "killpg",
        lambda pgid, sig: (signalled.append(pgid), real_killpg(pgid, sig)),
    )

    _pack_processes.unload(process.module_path)

    # The pid is free once reaped: another program's process group may have it now.
    assert signalled == []


@needs_pack_process
def test_a_tensor_ends_with_no_shared_memory_and_no_helper_process(probe):
    sent = torch.rand(1 << 20)

    back, shared_in_pack = asyncio.run(
        evaluate(probe, "value * 1, value.is_shared()", sent)
    )
    children = os.popen(f"pgrep -P {os.getpid()} -l").read()

    assert (back.is_shared(), shared_in_pack) == (False, False)
    assert "torch_shm_manager" not in children


@needs_pack_process
def test_the_reader_keeps_ticking_while_a_large_frame_arrives_slowly(monkeypatch):
    """A frame that stops arriving half way stops the reader's ticks: the user's cancel
    then never reaches the pack and no grace period ends it."""
    ours, theirs = socket.socketpair()
    ticks = []
    side = types.SimpleNamespace(
        peer="a pack",
        poll=0.05,
        tick=lambda: ticks.append(time.monotonic()),
        ended=lambda: False,
        stopped=lambda: None,
    )
    channel = _channel.Channel(ours, side)
    channel.start("reader")
    theirs.sendall(_channel._SIZES.pack(0, 1 << 20, 0) + b"x" * 1000)
    time.sleep(0.2)
    before = len(ticks)
    time.sleep(1.0)
    during = len(ticks) - before
    theirs.close()
    channel.reader.join(5)

    assert during >= 5, f"{during} ticks in 1 s while a frame was half sent"


def test_a_frame_cut_anywhere_ends_in_end_of_file():
    blocks = blocks_of(3, _channel.JOINED + 5, 0, 7)
    ours, theirs = socket.socketpair()
    writer = threading.Thread(
        target=_channel.write_frame, args=(ours, b"head", b"body", blocks)
    )
    writer.start()
    whole = b""
    theirs.settimeout(5)
    expected = 24 + 8 * len(blocks) + 8 + sum(b.nbytes for b in blocks)
    while len(whole) < expected:
        whole += theirs.recv(1 << 20)
    writer.join()
    ours.close()
    theirs.close()
    cuts = sorted({0, 1, 23, 24, 30, 56, 60, 64, 65, 67, 68, 1000, len(whole) - 1})

    ended = []
    for cut in cuts:
        sender, reader = socket.socketpair()
        feeder = threading.Thread(
            target=lambda: (sender.sendall(whole[:cut]), sender.close())
        )
        feeder.start()
        try:
            _channel.read_frame(reader, bool)
            ended.append((cut, "read"))
        except EOFError:
            ended.append((cut, "end of file"))
        feeder.join()
        reader.close()

    assert ended == [(cut, "end of file") for cut in cuts]


@needs_pack_process
def test_a_pack_cannot_leave_its_process_group(probe):
    outcome = asyncio.run(evaluate(probe, "caught(os.setsid)"))

    assert outcome[1] == "PermissionError"


FOLDER_PACK_INIT = """\
from comfy_api.v4 import ComfyExtension
from .nodes import Where


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [Where]


async def comfy_entrypoint():
    return Extension()
"""
FOLDER_PACK_NODES = """\
import os

from comfy_api.v4 import io


class Where(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="FolderWhere", outputs=[io.Int.Output()])

    @classmethod
    def execute(cls):
        return io.NodeOutput(os.getpid())
"""


@needs_pack_process
def test_a_pack_that_is_a_folder_with_relative_imports_runs_in_its_own_process(
    tmp_path,
):
    from comfy_api.v4 import _pack_host

    folder = tmp_path / "v4_folder_pack"
    folder.mkdir()
    (folder / "__init__.py").write_text(FOLDER_PACK_INIT, encoding="utf-8")
    (folder / "nodes.py").write_text(FOLDER_PACK_NODES, encoding="utf-8")

    module = asyncio.run(_pack_host.load(str(folder), "v4_folder_pack"))
    try:
        process = _pack_processes.processes[str(folder)]
        answer = asyncio.run(process.run(("node", "FolderWhere", "execute"), ({}, {})))
    finally:
        _pack_processes.unload(str(folder))

    assert answer.args[0] == process.process.pid != os.getpid()
    assert module.WEB_DIRECTORY is None


class RefusesPartWay(FailsPartWay):
    """As `FailsPartWay`, with the error macOS gives a send of 2 GiB or more."""

    def sendall(self, data):
        try:
            super().sendall(data)
        except KeyboardInterrupt:
            raise OSError(22, "Invalid argument") from None


def test_a_send_the_system_refuses_part_way_ends_the_connection():
    ours, theirs = socket.socketpair()
    channel = _channel.Channel(ours, types.SimpleNamespace(peer="a pack"))
    channel.sock = RefusesPartWay(ours, 4096)
    tensor = torch.arange(1 << 16, dtype=torch.float32)
    body, blocks = _codec.dumps(tensor, Side(), 1)

    with pytest.raises(_channel.PackStopped):
        channel._send({"to": 1, "chain": 1}, body, blocks)
    with pytest.raises(_channel.PackStopped):
        channel.notify({"output": "stdout", "text": "x" * 300_000})
    ours.close()
    theirs.settimeout(5)

    with pytest.raises(EOFError):
        _channel.read_frame(theirs, bool)
    theirs.close()


@needs_pack_process
def test_in_a_pack_process_the_folder_table_is_the_one_that_asks_comfyui(probe):
    kind = asyncio.run(evaluate(probe, "type(folders.folder_names_and_paths).__name__"))

    assert kind == "folder_names_and_paths"


def test_a_frame_cut_off_by_a_process_that_ended_with_its_socket_held_ends_the_read():
    class Ended:
        poll = 0.05

        def ended(self):
            return True

        def tick(self):
            pass

    ours, theirs = socket.socketpair()
    # The other process ended part way through a frame; a program it started holds its socket.
    theirs.sendall(_channel._SIZES.pack(0, 100, 0) + b"part")
    channel = _channel.Channel(ours, Ended())
    outcome = []

    def read():
        with selectors.DefaultSelector() as selector:
            selector.register(ours, selectors.EVENT_READ)
            try:
                _channel.read_frame(
                    ours, functools.partial(channel._readable, selector)
                )
            except EOFError:
                outcome.append("end of file")

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    reader.join(5)
    in_time = list(outcome)
    theirs.close()
    reader.join(5)

    assert in_time == ["end of file"]
