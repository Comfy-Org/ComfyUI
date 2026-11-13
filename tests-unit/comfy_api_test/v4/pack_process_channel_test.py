import asyncio
import collections
import contextlib
import hashlib
import io as stdio
import os
import pathlib
import pickle
import socket
import sys
import threading
import time
import types
from fractions import Fraction

import folder_paths
import numpy
import pytest
import torch
from PIL import Image

from comfy_api.latest import InputImpl, Types, _io
from comfy_api.v4 import (
    InterruptProcessingException,
    _channel,
    _codec,
    _pack_host,
    _pack_processes,
)
from comfy_api.v4._channel import PackError, PackStopped, RemoteTraceback
from comfy_execution.graph_utils import ExecutionBlocker
from conftest import MODULE, PROBE, evaluate, in_a_node, needs_pack_process, start_probe

pytestmark = needs_pack_process


@pytest.mark.asyncio
async def test_a_chain_three_deep_runs_each_step_on_the_lane_thread_of_its_process(
    probe,
):
    outer = await evaluate(probe, "outer")

    def host_step(inner):
        return [(os.getpid(), threading.current_thread().name)] + inner()

    pack_pid = (await evaluate(probe, "where()"))[0]
    assert in_a_node(outer, host_step) == [
        (pack_pid, "MainThread"),
        (os.getpid(), threading.current_thread().name),
        (pack_pid, "MainThread"),
    ]
    assert pack_pid != os.getpid()


@pytest.mark.asyncio
async def test_an_object_freed_on_comfyuis_reader_thread_cannot_call_the_pack_and_hang_it(
    probe, monkeypatch
):
    make = await evaluate(probe, "make")
    refused = []

    class CallsInDel:
        def __del__(self):
            try:
                make()
            except RuntimeError as error:
                refused.append((threading.current_thread().name, str(error)))

    def tick():
        monkeypatch.undo()
        CallsInDel()

    async def refusal():
        while not refused:
            await asyncio.sleep(0.01)

    monkeypatch.setattr(probe, "tick", tick)
    await asyncio.wait_for(refusal(), 10)

    assert refused == [
        (
            f"{probe.peer} reader",
            f"{probe.peer} cannot be called from its reader thread",
        )
    ]
    assert await asyncio.wait_for(evaluate(probe, "1 + 1"), 10) == 2


def host_object():
    return object()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value, seen_in_pack",
    [
        (None, "builtins.NoneType"),
        (True, "builtins.bool"),
        (7, "builtins.int"),
        (2.5, "builtins.float"),
        ("text", "builtins.str"),
        (b"\x00bytes", "builtins.bytes"),
        ([1, [2, "three"]], "builtins.list"),
        ((1, (2,)), "builtins.tuple"),
        ({1: "an int key", "nested": {"t": (1, 2)}}, "builtins.dict"),
        ({1, 2, 3}, "builtins.set"),
        (Fraction(24000, 1001), "fractions.Fraction"),
        (_io.FolderType.output, "comfy_api.latest._io.FolderType"),
        (_io.NodeReplace("New", "Old"), "comfy_api.latest._io.NodeReplace"),
    ],
)
async def test_plain_data_and_value_classes_cross_both_ways_as_the_same_type_and_value(
    probe, value, seen_in_pack
):
    back, seen = await evaluate(probe, "value, seen(value)", value)

    assert seen == seen_in_pack
    assert type(back) is type(value)
    assert (
        vars(back) == vars(value)
        if isinstance(value, _io.NodeReplace)
        else back == value
    )


@pytest.mark.asyncio
async def test_comfyui_value_objects_cross_both_ways_as_the_same_class(probe):
    image = Image.new("RGB", (3, 2), (10, 20, 30))
    video = InputImpl.VideoFromComponents(
        Types.VideoComponents(images=torch.rand(2, 4, 4, 3), frame_rate=Fraction(25))
    )
    values = [ExecutionBlocker("why"), image, video, stdio.BytesIO(b"file")]

    back, seen = await evaluate(probe, "value, [seen(v) for v in value]", values)

    assert [type(v) for v in back] == [type(v) for v in values]
    assert seen == [f"{type(v).__module__}.{type(v).__qualname__}" for v in values]
    assert back[0].message == "why"
    assert back[1].tobytes() == image.tobytes()
    assert torch.equal(back[2].get_components().images, video.get_components().images)
    assert back[3].getvalue() == b"file"


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [host_object(), host_object, {}.keys()])
async def test_a_comfyui_object_that_is_not_data_reaches_the_pack_as_a_handle_and_comes_back_itself(
    probe, value
):
    back, seen = await evaluate(probe, "value, seen(value)", value)

    assert seen == "comfy_api.v4._codec.Remote"
    assert back is value


@pytest.mark.asyncio
async def test_an_object_of_a_packs_own_class_arrives_as_a_handle_and_its_module_is_not_imported(
    probe,
):
    pair = await evaluate(probe, "Pair(1, 2)")

    assert type(pair) is _codec.Remote and "Pair" in repr(pair)
    with pytest.raises(AttributeError, match="Pair lives in pack v4_probe_pack.py"):
        pair.left
    assert await evaluate(probe, "type(value).__name__, value.left", pair) == (
        "Pair",
        1,
    )
    assert MODULE not in sys.modules


@pytest.mark.asyncio
async def test_a_handle_put_into_text_raises_naming_its_type(probe):
    name = await evaluate(probe, "Name('abc')")

    assert type(name) is _codec.Remote
    with pytest.raises(TypeError, match="Name lives in pack v4_probe_pack.py"):
        f"{name}_00001_.png"
    with pytest.raises(TypeError, match="Name lives in pack v4_probe_pack.py"):
        "{}_00001_.png".format(name)


@pytest.mark.asyncio
async def test_a_member_of_a_packs_own_str_or_int_enum_arrives_as_its_value_and_another_as_a_handle(
    probe,
):
    shade = await evaluate(probe, "Shade.DARK")
    level = await evaluate(probe, "Level.HIGH")
    color = await evaluate(probe, "Color.RED")

    assert type(shade) is str and shade == "dark"
    assert type(level) is int and level == 2
    assert type(color) is _codec.Remote
    with pytest.raises(AttributeError, match="Color lives in pack"):
        color.value
    assert await evaluate(probe, "value is Color.RED", color) is True


@pytest.mark.parametrize(
    "crafted",
    [
        b"\x80\x05c" + MODULE.encode() + b"\nPair\n.",
        b"\x80\x05c" + MODULE.encode() + b"\nPair\n(K\x01K\x02tR.",
        pickle.dumps(os.system),
        pickle.dumps(numpy.dtype),
    ],
    ids=[
        "a pack class",
        "an instance of a pack class",
        "os.system",
        "a class outside the list",
    ],
)
def test_a_message_naming_a_class_outside_the_allowlist_is_refused_without_importing_it(
    tmp_path, monkeypatch, crafted
):
    (tmp_path / f"{MODULE}.py").write_text(
        "raise SystemExit('imported')\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(pickle.UnpicklingError, match="refused"):
        _codec.loads(crafted, None)
    assert MODULE not in sys.modules


def file_values(path):
    return {
        "a video": InputImpl.VideoFromFile(str(path), start_time=1.0, duration=2.0),
        "a 3D file": Types.File3D(str(path), "glb"),
    }


def held(value):
    """What a value that holds a file holds, and the rest of what it says of itself."""
    if isinstance(value, Types.File3D):
        return value.get_source(), value.format
    return value.get_stream_source(), value.get_active_trim_window()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["a video", "a 3D file"])
async def test_a_value_that_holds_a_file_by_path_crosses_both_ways_holding_its_bytes(
    probe, tmp_path, kind
):
    path = tmp_path / "clip.bin"
    path.write_bytes(b"the file's bytes")
    sent = file_values(path)[kind]

    in_pack = await evaluate(
        probe,
        "type(value).__name__, type(source := (value.get_source if hasattr(value, 'get_source')"
        " else value.get_stream_source)()).__name__, source.getvalue()",
        sent,
    )
    back = await evaluate(probe, "value", sent)
    made_there = await evaluate(
        probe,
        f"__import__('comfy_api.latest').latest.Types.File3D({str(path)!r}, 'glb')",
    )

    assert in_pack == (type(sent).__name__, "BytesIO", b"the file's bytes")
    source, rest = held(back)
    assert type(back) is type(sent) and type(source) is stdio.BytesIO
    assert (source.getvalue(), rest) == (b"the file's bytes", held(sent)[1])
    assert held(sent)[0] == str(path)
    assert held(made_there)[0].getvalue() == b"the file's bytes"


class Crafting(pickle.Pickler):
    """Pickles `Crafting.MARK` as the persistent reference given."""

    MARK = object()

    def __init__(self, file, reference):
        super().__init__(file, protocol=5)
        self.reference = reference

    def persistent_id(self, obj):
        return self.reference if obj is self.MARK else None


@pytest.mark.parametrize("kind", ["a video", "a 3D file"])
def test_a_message_that_gives_a_value_a_path_for_its_file_is_refused(tmp_path, kind):
    path = tmp_path / "private.bin"
    path.write_bytes(b"not the sender's")
    value = file_values(path)[kind]
    crafted = stdio.BytesIO()
    Crafting(crafted, ("file", type(value).__name__, dict(vars(value)))).dump(
        Crafting.MARK
    )

    with pytest.raises(pickle.UnpicklingError, match="bytes, not a path"):
        _codec.loads(crafted.getvalue(), None)
    with pytest.raises(pickle.UnpicklingError, match="refused"):
        _codec.loads(pickle.dumps(value), None)


def test_a_message_cannot_build_a_class_v3_does_not_publish_as_a_value():
    from comfy_api.latest._input_impl import video_types

    for cls in (video_types.FrameChunks, video_types.HostFrames):
        with pytest.raises(pickle.UnpicklingError, match="refused"):
            _codec.loads(pickle.dumps(cls), None)


@pytest.mark.asyncio
async def test_a_python_error_crosses_both_ways_as_the_same_class(probe):
    def host_fails():
        raise KeyError("missing")

    with pytest.raises(ValueError, match="invalid literal") as error:
        await evaluate(probe, "int('x')")
    assert type(error.value) is ValueError
    assert "_pack_main.py" in str(error.value.__cause__)
    assert await evaluate(probe, "caught(value)", host_fails) == (
        "builtins",
        "KeyError",
        "'missing'",
    )


@pytest.mark.asyncio
async def test_the_cancel_crosses_both_ways_as_comfyuis_cancel_exception(probe):
    def host_cancels():
        raise InterruptProcessingException()

    with pytest.raises(InterruptProcessingException):
        await evaluate(probe, "fail(InterruptProcessingException())")
    module, qualname, _ = await evaluate(probe, "caught(value)", host_cancels)
    caught = await evaluate(probe, "caught")
    # Here the cancel is raised on the thread that waits for the pack's answer.
    caught_while_waiting = in_a_node(caught, host_cancels)

    assert (module, qualname) == (
        "comfy_api._exceptions",
        "InterruptProcessingException",
    )
    assert caught_while_waiting[:2] == (module, qualname)


@pytest.mark.asyncio
async def test_comfyuis_exception_that_is_not_an_exception_reaches_the_pack_as_one_and_goes_on_here(
    probe,
):
    keep_error = await evaluate(probe, "keep_error")

    def host_exits():
        raise SystemExit(5)

    with pytest.raises(SystemExit):
        # In a node, so that the pack's exec lane has recorded it before the next request.
        in_a_node(keep_error, host_exits)

    assert await evaluate(probe, "errors[-1]") == ("SystemExit", True)


@pytest.mark.asyncio
async def test_a_packs_own_error_keeps_its_module_and_name_and_is_a_pack_error(probe):
    with pytest.raises(PackError, match="seed out of range") as error:
        await evaluate(probe, "fail(SeedError('seed out of range'))")

    assert (type(error.value).__module__, type(error.value).__qualname__) == (
        MODULE,
        "SeedError",
    )
    assert type(error.value.__cause__) is RemoteTraceback
    assert "SeedError: seed out of range" in str(error.value.__cause__)


DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16, torch.int64, torch.bool]
)
@pytest.mark.parametrize("size", [(3, 5), (256, 512), (0, 4), ()])
async def test_a_tensor_crosses_both_ways_with_its_values_dtype_shape_and_device(
    probe, device, dtype, size
):
    tensor = (torch.rand(size) * 100).to(dtype).to(device)

    back, seen = await evaluate(probe, "value, seen(value)", tensor)

    assert seen == ("torch.Tensor", str(dtype), size, device)
    assert (back.dtype, back.shape, back.device.type) == (dtype, tensor.shape, device)
    assert torch.equal(back, tensor)


def digest(tensor):
    return hashlib.sha256(memoryview(tensor.view(torch.uint8).numpy())).hexdigest()


@pytest.mark.asyncio
async def test_a_64_mib_tensor_crosses_both_ways_bit_for_bit(probe):
    generator = torch.Generator().manual_seed(64)
    sent = torch.randint(0, 256, (64 << 20,), dtype=torch.uint8, generator=generator)
    sent = sent.view(torch.float32)

    digest_in_pack, back = await evaluate(probe, "digest(value), value", sent)

    assert digest_in_pack == digest(sent) == digest(back)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(4,), (1024, 1024)], ids=["small", "large"])
async def test_a_tensor_that_crosses_is_the_receivers_own_memory_on_both_sides(
    probe, size
):
    sent = torch.zeros(size)

    shared_in_pack = await evaluate(
        probe, "keep('sent', value).add_(1).is_shared()", sent
    )
    received = await evaluate(probe, "keep('made', torch.zeros(value))", size)
    sent.add_(5)
    received.add_(2)
    in_pack = await evaluate(
        probe, "kept['sent'].sum().item(), kept['made'].add_(3).sum().item()"
    )

    count = sent.numel()
    assert (shared_in_pack, sent.is_shared(), received.is_shared()) == (False,) * 3
    assert (sent.sum().item(), received.sum().item()) == (5 * count, 2 * count)
    assert in_pack == (count, 3 * count)


class Recording:
    """A channel's socket that records how much each send and receive is given."""

    def __init__(self, sock):
        self.sock, self.sent, self.received = sock, [], []

    def sendall(self, data):
        self.sent.append(memoryview(data).nbytes)
        self.sock.sendall(data)

    def recv_into(self, view):
        self.received.append(len(view))
        return self.sock.recv_into(view)

    def __getattr__(self, name):
        return getattr(self.sock, name)


@pytest.mark.parametrize("pieces", [4, 4.5], ids=["a multiple", "not a multiple"])
def test_a_block_longer_than_a_piece_crosses_whole_in_pieces_both_ways(
    probe, monkeypatch, pieces
):
    piece = 4096
    tensor = torch.arange(int(pieces * piece) // 4, dtype=torch.float32)
    monkeypatch.setattr(_channel, "PIECE", piece)
    recording = Recording(probe.channel.sock)
    monkeypatch.setattr(probe.channel, "sock", recording)
    set_piece = (
        "setattr(__import__('comfy_api.v4._channel').v4._channel, 'PIECE', value)"
    )
    asyncio.run(evaluate(probe, set_piece, piece))
    try:
        back, in_pack = asyncio.run(evaluate(probe, "value, value.clone()", tensor))
    finally:
        asyncio.run(evaluate(probe, set_piece, 256 << 20))

    assert torch.equal(back, tensor) and torch.equal(in_pack, tensor)
    assert max(recording.sent) <= piece and max(recording.received) <= piece
    assert recording.sent.count(piece) >= 4 and recording.received.count(piece) >= 4


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(4, 6), (300, 400)])
async def test_a_non_contiguous_tensor_or_a_view_crosses_with_its_values(probe, size):
    base = torch.rand(size)
    views = [base.t(), base[1:3], base[:, ::2], base[size[0] // 2 :]]

    back = await evaluate(probe, "value", views)

    for view, received in zip(views, back):
        assert received.shape == view.shape and torch.equal(received, view)


@pytest.mark.asyncio
@pytest.mark.parametrize("dtype", [numpy.int32, numpy.complex64])
async def test_a_numpy_array_crosses_both_ways_as_a_numpy_array(probe, dtype):
    array = numpy.arange(12, dtype=dtype).reshape(3, 4).T

    back, seen = await evaluate(probe, "value, seen(value)", array)

    assert seen == "numpy.ndarray"
    assert type(back) is numpy.ndarray and back.dtype == array.dtype
    assert (back == array).all()


def test_a_tensor_comfyui_received_stays_valid_after_the_pack_process_is_killed(
    tmp_path,
):
    process = start_probe(tmp_path)

    async def receive():
        return await evaluate(process, "torch.full((64, 1024), 7.0)")

    tensor = asyncio.run(receive())
    _pack_processes.unload(process.module_path)
    process.process.wait()

    assert tensor.sum().item() == 7.0 * 64 * 1024
    tensor.add_(1)
    assert tensor.sum().item() == 8.0 * 64 * 1024


@pytest.mark.asyncio
async def test_the_pack_runs_each_call_in_the_grad_and_inference_modes_of_comfyuis_thread(
    probe,
):
    modes = await evaluate(probe, "modes")

    with torch.inference_mode():
        in_inference = await evaluate(probe, "modes()"), modes()
    with torch.enable_grad():
        with_grad = await evaluate(probe, "modes()"), modes()

    assert in_inference == ((True, False), (True, False))
    assert with_grad == ((False, True), (False, True))


@pytest.mark.asyncio
async def test_util_folders_answers_with_comfyuis_folders_while_the_pack_is_imported(
    probe,
):
    assert (
        await evaluate(probe, "OUTPUT_AT_IMPORT") == folder_paths.get_output_directory()
    )


@pytest.mark.asyncio
async def test_a_call_to_comfyui_from_a_thread_the_pack_started_is_refused(probe):
    module, qualname, message = await evaluate(
        probe, "in_a_thread(lambda: ComfyAPISync().device.info())"
    )

    assert (module, qualname) == ("builtins", "RuntimeError")
    assert "from the thread that runs it" in message


def test_a_reply_that_cannot_be_sent_leaves_the_waiting_call_to_end_as_stopped(
    tmp_path,
):
    process = start_probe(tmp_path)
    outer = asyncio.run(evaluate(process, "outer"))

    def host_step(inner):
        process.channel.sock.shutdown(socket.SHUT_WR)
        return []

    try:
        with pytest.raises(PackStopped, match="stopped"):
            outer(host_step)
        with pytest.raises(PackStopped):
            outer(host_step)
    finally:
        _pack_processes.unload(process.module_path)


@pytest.mark.asyncio
async def test_an_answer_comfyui_stopped_waiting_for_is_decoded_so_what_it_holds_is_released(
    probe, monkeypatch
):
    make, make_slowly, alive = [
        await evaluate(probe, name) for name in ("make", "make_slowly", "alive")
    ]
    held = make()
    held_alive = alive()
    del held
    dropped = threading.Event()
    drop = probe.channel.drop
    monkeypatch.setattr(
        probe.channel, "drop", lambda reply: (drop(reply), dropped.set())
    )

    with pytest.raises(TimeoutError):
        probe.request(("call", make_slowly._key), timeout=0.05)
    assert dropped.wait(30)

    # The release goes with the next message.
    assert (held_alive, alive()) == (1, 0)


def test_an_object_of_a_stopped_pack_process_fails_clearly_once_the_pack_runs_again(
    tmp_path,
):
    process = start_probe(tmp_path)
    try:
        make = asyncio.run(evaluate(process, "make"))
        process.kill()
        with pytest.raises(PackStopped, match="exit code"):
            make()
        asyncio.run(evaluate(process, "where()"))

        with pytest.raises(
            LookupError, match="made by a pack process that has stopped"
        ):
            make()
    finally:
        _pack_processes.unload(process.module_path)


# What numpy arithmetic, `tensor.shape`, `pathlib` and `torch.load` give a pack.
SAME_TYPE = {
    "numpy.float64(2.5)": numpy.float64(2.5),
    "numpy.float32(0.5)": numpy.float32(0.5),
    "numpy.int64(7)": numpy.int64(7),
    "numpy.bool_(True)": numpy.bool_(True),
    "torch.zeros(2, 3).shape": torch.Size([2, 3]),
    "pathlib.Path('/tmp/v4')": pathlib.Path("/tmp/v4"),
    "pathlib.PurePosixPath('a/b')": pathlib.PurePosixPath("a/b"),
    "collections.OrderedDict(w=1, b=[2])": collections.OrderedDict(w=1, b=[2]),
    "bytearray(b'ab')": bytearray(b"ab"),
    "frozenset({1, 2})": frozenset({1, 2}),
    "pathlib.PureWindowsPath('C:/v4')": pathlib.PureWindowsPath("C:/v4"),
    "numpy.complex64(1 + 2j)": numpy.complex64(1 + 2j),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("expression", SAME_TYPE)
async def test_a_value_crosses_both_ways_as_the_same_type_and_value(probe, expression):
    expected = SAME_TYPE[expression]

    value = await evaluate(probe, expression)
    seen_in_pack = await evaluate(probe, "seen(value), value == " + expression, value)

    assert type(value) is type(expected) and value == expected
    assert seen_in_pack == (
        f"{type(expected).__module__}.{type(expected).__qualname__}",
        True,
    )


# Kinds that stay in the pack, and say so.
HANDLES = {
    "Box(3)": "Box",
    "Pair(1, 2)": "Pair",
    "Name('abc')": "Name",
    "numpy.str_('text')": "str_",
    "datetime.datetime(2026, 1, 1)": "datetime",
    "decimal.Decimal('1.5')": "Decimal",
    "collections.defaultdict(int, a=1)": "defaultdict",
    "complex(1, 2)": "complex",
    "range(3)": "range",
    "torch.eye(3).to_sparse()": "Tensor",
    "PosixPath('a')": "PosixPath",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("expression", HANDLES)
async def test_a_value_of_another_kind_stays_in_the_pack_and_names_its_type_on_use(
    probe, expression
):
    value = await evaluate(probe, expression)

    assert type(value) is _codec.Remote
    assert f"<{HANDLES[expression]} in pack" in str(value)
    with pytest.raises(AttributeError, match=f"{HANDLES[expression]} lives in pack"):
        value.anything
    with pytest.raises(TypeError, match=f"{HANDLES[expression]} lives in pack"):
        f"{value}_00001_.png"
    assert await evaluate(probe, "type(value).__name__", value) == HANDLES[expression]


@pytest.mark.asyncio
async def test_a_tensors_requires_grad_and_parameter_class_cross_both_ways(probe):
    parameter = torch.nn.Parameter(torch.ones(2))

    back, seen = await evaluate(
        probe,
        "value, [(type(v).__name__, v.requires_grad) for v in value]",
        [parameter, torch.ones(2, requires_grad=True), torch.ones(2)],
    )

    assert seen == [("Parameter", True), ("Tensor", True), ("Tensor", False)]
    assert [(type(v), v.requires_grad) for v in back] == [
        (torch.nn.Parameter, True),
        (torch.Tensor, True),
        (torch.Tensor, False),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(4, 4), (256, 256)], ids=["small", "large"])
async def test_a_conjugated_or_negated_tensor_crosses_with_its_values(probe, size):
    tensor = torch.randn(size, dtype=torch.complex64)

    back, imaginary = await evaluate(probe, "value.conj(), value.conj().imag", tensor)
    negated = await evaluate(probe, "value", tensor[:1, :1].conj().imag)

    assert torch.equal(back.resolve_conj(), tensor.conj().resolve_conj())
    assert torch.equal(imaginary, -tensor.imag)
    assert torch.equal(negated, -tensor[:1, :1].imag)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(2, 3), (256, 512)], ids=["small", "large"])
async def test_a_tensor_is_an_inference_tensor_on_both_sides_when_the_call_is_in_inference_mode(
    probe, size
):
    async def crossing(tensor):
        in_pack, made = await evaluate(
            probe, "value.is_inference(), torch.ones(value.shape)", tensor
        )
        return in_pack, made.is_inference()

    with torch.inference_mode():
        inference = await crossing(torch.ones(size))
    normal = await crossing(torch.ones(size))

    assert (inference, normal) == ((True, True), (False, False))


@pytest.mark.asyncio
async def test_a_copy_of_a_packs_object_is_the_object_itself(probe):
    import copy
    import gc

    make = await evaluate(probe, "make")
    copies = copy.copy(make), copy.deepcopy({"ui": [make]})["ui"][0]
    del make
    gc.collect()

    assert copies[0] is copies[1]
    assert type(copies[0]()) is _codec.Remote


@pytest.mark.asyncio
async def test_a_handle_given_in_one_request_is_refused_in_another_while_both_run(
    probe,
):
    held = asyncio.create_task(evaluate(probe, "hold_until_asked(value)", host_object))
    asked = await probe.run(
        ("node", "Eval", "execute"), ({"expression": "ask_held()"}, {}), "server"
    )
    module, qualname, message = asked.args[0]
    await held

    assert (module, qualname) == ("builtins", "ValueError")
    assert "valid only in the request it was given in" in message


NO_MESSAGE = (
    "type('Unprintable', (Exception,), {'__str__': lambda self: 1 / 0})(object())"
)


class Unprintable(Exception):
    def __str__(self):
        raise ZeroDivisionError


@pytest.mark.asyncio
async def test_an_error_whose_message_cannot_be_made_still_fails_the_call(probe):
    with pytest.raises(PackError, match="Unprintable"):
        await asyncio.wait_for(evaluate(probe, f"fail({NO_MESSAGE})"), 30)


@pytest.mark.asyncio
async def test_comfyui_raising_an_error_whose_message_cannot_be_made_still_answers_the_pack(
    probe,
):
    def host_fails():
        raise Unprintable(object())

    _, qualname, _ = await asyncio.wait_for(
        evaluate(probe, "caught(value)", host_fails), 30
    )

    assert qualname == "Unprintable"


@pytest.mark.asyncio
async def test_a_node_result_that_cannot_cross_fails_the_call(probe):
    with pytest.raises(NotImplementedError):
        await asyncio.wait_for(evaluate(probe, "torch.empty(2, device='meta')"), 30)


@pytest.mark.asyncio
async def test_a_packs_object_in_an_answer_that_cannot_cross_is_let_go_in_the_pack(
    probe,
):
    unsent = "[watch('unsent', Thing()), torch.empty(1, device='meta')]"
    sent = "watch('sent', Thing())"

    with pytest.raises(NotImplementedError):
        await evaluate(probe, unsent)
    handle = await evaluate(probe, sent)

    assert await evaluate(probe, "(gc.collect(), sorted(watched))[1]") == ["sent"]
    assert type(handle) is _codec.Remote


@pytest.mark.asyncio
async def test_an_io_type_a_pack_makes_crosses_as_an_io_type_with_its_parts(probe):
    made, made_input = await evaluate(
        probe, "(lambda made: (made, made.Input))(io.Custom('V4_THING'))"
    )

    assert issubclass(made, _io._ComfyType) and made.io_type == "V4_THING"
    assert made_input is made.Input


def test_comfyuis_table_of_waiting_threads_is_empty_after_nested_calls(probe):
    outer = asyncio.run(evaluate(probe, "outer"))

    for _ in range(3):
        outer(lambda inner: inner())

    assert probe.channel.waiting == {}


def test_crafted_references_name_only_what_they_may(tmp_path):
    def crafted(pid):
        class Crafting(pickle.Pickler):
            def persistent_id(self, obj):
                return pid if obj == "here" else None

        buffer = stdio.BytesIO()
        Crafting(buffer, protocol=5).dump(["here"])
        return buffer.getvalue()

    for message in (
        b"\x80\x05ccomfy_api.latest._io\nCustom\n.",  # a function of an allowed module
        crafted(("dtype", "zeros")),
        crafted(("io", "Custom", "X", None)),
        crafted(("io", "Int", "INT", "__class__")),
        crafted(("numpy scalar", "|O", b"")),
        crafted(("path", "Path", "/")),
    ):
        with pytest.raises(pickle.UnpicklingError):
            _codec.loads(message, None)


def test_only_the_apis_public_calls_are_served():
    with pytest.raises(LookupError, match="not a call of the API"):
        _pack_host._call_api("execution", "__init__", (), {})


@pytest.fixture(scope="module")
def other(tmp_path_factory):
    """A second pack process, of the same source, shared by the tests of this module."""
    path = tmp_path_factory.mktemp("other") / "v4_other_probe.py"
    path.write_text(PROBE, encoding="utf-8")
    asyncio.run(_pack_host.load(str(path), "v4_other_probe"))
    yield _pack_processes.processes[str(path)]
    _pack_processes.unload(str(path))


def test_a_chain_through_two_pack_processes_comes_back_to_the_first_packs_waiting_thread(
    probe, other
):
    p_outer = asyncio.run(evaluate(probe, "outer"))
    q_outer = asyncio.run(evaluate(other, "outer"))
    p_where = asyncio.run(evaluate(probe, "where"))

    steps = in_a_node(p_outer, lambda inner: q_outer(lambda inner: [p_where()]))

    assert [name for _, name in steps] == ["MainThread"] * 3


def test_while_comfyui_waits_on_a_pack_it_runs_that_packs_waiting_calls_and_no_other_packs(
    probe, other
):
    """The pack's own call must run inside the wait, or the two would wait for each other
    (see `_channel`); another pack's call waits for the loop, as it would in one process."""

    async def main():
        hook = await evaluate(probe, "lambda: time.sleep(1.0) or 1")
        in_hook, seen = [False], {}

        def comfyui_work(who):
            return lambda: seen.setdefault(who, in_hook[0])

        calls = [
            asyncio.create_task(evaluate(pack, "value()", comfyui_work(who)))
            for pack, who in ((probe, "same pack"), (other, "another pack"))
        ]
        await asyncio.sleep(0)  # both nodes are sent
        # ComfyUI's thread runs something else meanwhile; both packs' calls arrive.
        time.sleep(0.5)
        in_hook[0] = True
        in_a_node(hook)
        in_hook[0] = False
        await asyncio.gather(*calls)
        return seen

    assert asyncio.run(main()) == {"same pack": True, "another pack": False}


def test_a_packs_object_freed_on_another_thread_does_not_wait_for_the_packs_node(probe):
    """A patched model freed on ComfyUI's server thread moves its patch (`to`)."""
    import gc

    class Patcher:
        def __init__(self, patch):
            self.patch, self.cycle = patch, self

        def __del__(self):
            self.patch.to("cpu")

    async def main():
        patch = await evaluate(probe, "torch.nn.Linear(1, 1)")
        Patcher(patch)  # a clone freed; another clone still holds the patch
        busy = asyncio.create_task(evaluate(probe, "time.sleep(3)"))
        await asyncio.sleep(0.3)
        started = time.monotonic()
        await asyncio.to_thread(gc.collect)
        took = time.monotonic() - started
        await busy
        assert patch._release.alive
        return took

    gc.disable()  # the patcher is freed by the collection on the other thread, not earlier
    try:
        assert asyncio.run(main()) < 2.0
    finally:
        gc.enable()


@pytest.mark.asyncio
async def test_an_error_that_cannot_be_rebuilt_from_its_message_keeps_its_type_name_and_message(
    probe,
):
    with pytest.raises(PackError) as error:
        await evaluate(probe, "b'\\xff'.decode()")

    assert str(error.value).startswith("UnicodeDecodeError: ")
    assert "can't decode byte 0xff" in str(error.value)


def test_calls_of_a_pack_that_wait_together_for_comfyuis_loop_are_all_answered(probe):
    async def main():
        node = {"expression": "value()", "value": lambda: "node"}
        route = {"expression": "value()", "value": lambda: "route"}
        calls = [
            asyncio.create_task(probe.run(("node", "Eval", "execute"), (node, {}))),
            asyncio.create_task(
                probe.run(("node", "Eval", "execute"), (route, {}), "server")
            ),
        ]
        await asyncio.sleep(0)  # both are sent
        time.sleep(0.5)  # this loop is busy while both of the pack's calls arrive
        answers = await asyncio.wait_for(asyncio.gather(*calls), 30)
        return [answer.args[0] for answer in answers]

    assert asyncio.run(main()) == ["node", "route"]


def test_a_cancel_never_waits_for_room_in_the_socket():
    ours, theirs = socket.socketpair()
    channel = _channel.Channel(
        ours, types.SimpleNamespace(peer="a pack that does not read")
    )
    ours.setblocking(False)
    for size in (1 << 16, 1 << 12, 1 << 8, 1):
        with contextlib.suppress(BlockingIOError):
            while True:
                ours.send(b"x" * size)
    ours.setblocking(True)
    sent = []
    cancel = threading.Thread(
        target=lambda: sent.append(channel.cancel(1, "exec")), daemon=True
    )

    cancel.start()
    cancel.join(5)
    theirs.close()  # a send still waiting fails now
    cancel.join(5)
    ours.close()

    assert sent == [False]
