"""What a message from a pack process can make ComfyUI do: messages written by hand, as a pack
process that does not use `_codec` writes them."""

import asyncio
import fractions
import io
import os
import pickle
import struct
import time
import tracemalloc

import pytest
import torch

import folder_paths
from comfy_api.latest import Types, _io
from comfy_api.v4 import _channel, _codec, _pack_host, _pack_processes
from comfy_execution.graph_utils import ExecutionBlocker

from conftest import evaluate, needs_pack_process, start_probe


def text(value: str) -> bytes:
    data = value.encode()
    return pickle.BINUNICODE + struct.pack("<I", len(data)) + data


def whole(value: int) -> bytes:
    return pickle.BININT + struct.pack("<i", value)


def found(module: str, name: str) -> bytes:
    return text(module) + text(name) + pickle.STACK_GLOBAL


def items(*parts: bytes) -> bytes:
    return pickle.MARK + b"".join(parts) + pickle.TUPLE


def attributes(**values: bytes) -> bytes:
    pairs = b"".join(text(name) + value for name, value in values.items())
    return pickle.EMPTY_DICT + pickle.MARK + pairs + pickle.SETITEMS


def reference(*parts: bytes) -> bytes:
    return items(*parts) + pickle.BINPERSID


def message(body: bytes) -> bytes:
    return pickle.PROTO + b"\x05" + body + pickle.STOP


PACK_OBJECT = reference(
    text("object"),
    items(whole(1), whole(7)),
    text("Thing"),
    pickle.EMPTY_TUPLE,
    pickle.EMPTY_DICT,
)
EMPTY_BYTES_IO = found("_io", "BytesIO") + pickle.EMPTY_TUPLE + pickle.NEWOBJ


def class_attribute(module: str, name: str, attribute: str) -> bytes:
    """A message that gives a class an attribute: BUILD with slot state on the class."""
    state = items(pickle.NONE, attributes(**{attribute: PACK_OBJECT}))
    return message(found(module, name) + state + pickle.BUILD)


class Released:
    def release(self, key):
        pass


@pytest.fixture
def comfyui():
    """ComfyUI's side of a pack process, with one request whose objects the pack can name."""
    side = _pack_host.PackProcess("/nowhere/pack", "pack")
    side.channel = Released()
    side.chains[1] = _pack_host._Chain(1, "exec", None, None)
    return side


@pytest.mark.parametrize(
    "module, name, attribute",
    [
        ("comfy_api.latest._io", "ComfyNode", "PREPARE_CLASS_CLONE"),
        ("comfy_api.latest._io", "NodeOutput", "__init__"),
        ("fractions", "Fraction", "limit_denominator"),
        ("comfy_execution.graph_utils", "ExecutionBlocker", "__init__"),
    ],
)
def test_a_message_cannot_change_a_class_comfyui_has(comfyui, module, name, attribute):
    cls = pickle.Unpickler.find_class(pickle.Unpickler(io.BytesIO()), module, name)
    before = vars(cls).get(attribute)
    try:
        with pytest.raises(pickle.UnpicklingError, match="did not make it"):
            _codec.loads(class_attribute(module, name, attribute), comfyui)
        after = vars(cls).get(attribute)
    finally:
        if vars(cls).get(attribute) is not before:
            if before is None:
                delattr(cls, attribute)
            else:
                setattr(cls, attribute, before)

    assert after is before


def test_a_message_cannot_change_an_object_comfyui_gave_the_pack(comfyui):
    class Given:
        pass

    given = Given()
    comfyui.chains[1].objects.append(given)
    named = reference(text("yours"), items(whole(1), whole(0)))

    with pytest.raises(pickle.UnpicklingError, match="did not make it"):
        _codec.loads(
            message(named + attributes(patched=text("x")) + pickle.BUILD), comfyui, 1
        )

    assert vars(given) == {}


@pytest.mark.parametrize(
    "opcode", ["REDUCE", "INST", "OBJ", "NEWOBJ_EX", "NEWOBJ with arguments"]
)
def test_a_message_cannot_call_a_constructor(comfyui, tmp_path, monkeypatch, opcode):
    monkeypatch.setattr(folder_paths, "temp_directory", str(tmp_path))
    preview = found("comfy_api.latest._ui", "PreviewText")
    body = {
        "REDUCE": preview + items(text("x")) + pickle.REDUCE,
        "INST": b"(S'x'\nicomfy_api.latest._ui\nPreviewText\n",
        "OBJ": pickle.MARK + preview + text("x") + pickle.OBJ,
        "NEWOBJ_EX": preview + items(text("x")) + pickle.EMPTY_DICT + pickle.NEWOBJ_EX,
        "NEWOBJ with arguments": preview + items(text("x")) + pickle.NEWOBJ,
    }[opcode]

    with pytest.raises(pickle.UnpicklingError, match="refused"):
        _codec.loads(message(body) if opcode != "INST" else body + b".", comfyui)


def test_a_preview_cannot_be_written_by_a_message(comfyui, tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "temp_directory", str(tmp_path))
    tensor = reference(
        text("tensor"),
        text("float32"),
        items(whole(1), whole(8), whole(8)),
        text("cpu"),
        pickle.NEWFALSE,
        pickle.NEWFALSE,
    )
    body = found("comfy_api.latest._ui", "PreviewMask") + items(tensor) + pickle.REDUCE

    with pytest.raises(pickle.UnpicklingError, match="refused"):
        _codec.loads(message(body), comfyui, None, [bytearray(4 * 64)])

    assert os.listdir(tmp_path) == []


def test_a_memo_index_in_a_message_allocates_nothing(comfyui):
    put = pickle.NONE + pickle.LONG_BINPUT + struct.pack("<I", (1 << 31) - 1)

    tracemalloc.start()
    try:
        _codec.loads(message(put), comfyui)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert peak < 1 << 20


def test_a_fraction_with_an_exponent_is_refused_at_once(comfyui):
    body = found("fractions", "Fraction") + items(text("1e10000000")) + pickle.REDUCE

    started = time.monotonic()
    with pytest.raises(pickle.UnpicklingError, match="refused"):
        _codec.loads(message(body), comfyui)

    assert time.monotonic() - started < 1


@pytest.mark.parametrize(
    "value",
    [
        fractions.Fraction(-3, 4),
        __import__("collections").OrderedDict(a=1),
        _io.Hidden.unique_id,
        ExecutionBlocker("stop"),
        io.BytesIO(b"bytes"),
    ],
    ids=["Fraction", "OrderedDict", "enum", "value object", "BytesIO"],
)
def test_what_pickle_makes_of_a_value_still_crosses(comfyui, value):
    body, blocks = _codec.dumps(value, comfyui, 1)

    back = _codec.loads(body, comfyui, 1, blocks)

    assert type(back) is type(value)
    if isinstance(value, io.BytesIO):
        assert back.getvalue() == value.getvalue()
    elif isinstance(value, ExecutionBlocker):
        assert back.message == value.message
    else:
        assert back == value


def test_a_file_value_cannot_be_given_a_path_after_its_check(comfyui, tmp_path):
    decoy = tmp_path / "decoy.txt"
    decoy.write_text("decoy")
    checked = reference(
        text("file"),
        text("VideoFromFile"),
        attributes(_VideoFromFile__file=EMPTY_BYTES_IO),
    )
    then = attributes(_VideoFromFile__file=text(str(decoy)))

    with pytest.raises(pickle.UnpicklingError, match="did not make it"):
        _codec.loads(message(checked + then + pickle.BUILD), comfyui, 1)


def test_a_file_value_with_attributes_its_class_does_not_set_is_refused(comfyui):
    state = attributes(
        _VideoFromFile__file=EMPTY_BYTES_IO, get_stream_source=PACK_OBJECT
    )

    with pytest.raises(pickle.UnpicklingError, match="not a value that holds a file"):
        _codec.loads(
            message(reference(text("file"), text("VideoFromFile"), state)), comfyui, 1
        )


@pytest.mark.parametrize(
    "file_format", ["glb/../../../outside", "glb\\..\\x", ".glb", 5]
)
def test_a_3d_file_whose_format_is_not_an_extension_is_refused(comfyui, file_format):
    model = Types.File3D(io.BytesIO(b"glTF"))
    model._format = file_format
    body, blocks = _codec.dumps(model, comfyui, 1)

    with pytest.raises(pickle.UnpicklingError, match="not the format of a file"):
        _codec.loads(body, comfyui, 1, blocks)


def test_a_3d_file_with_an_extension_crosses(comfyui):
    body, blocks = _codec.dumps(Types.File3D(io.BytesIO(b"glTF"), "glb"), comfyui, 1)

    back = _codec.loads(body, comfyui, 1, blocks)

    assert (back.format, back.get_data().getvalue()) == ("glb", b"glTF")


SEND_RAW_HEAD = """
import gc
from comfy_api.v4 import _channel
channel = next(o for o in gc.get_objects() if type(o) is _channel.Channel)
with channel.send_lock:
    _channel.write_frame(channel.sock, value, b"")
"""


@needs_pack_process
def test_a_pack_process_whose_message_would_change_a_class_is_ended_and_the_class_is_not(
    tmp_path,
):
    process = start_probe(tmp_path)
    attribute = "PREPARE_CLASS_CLONE"
    before = vars(_io.ComfyNode).get(attribute)
    head = class_attribute("comfy_api.latest._io", "ComfyNode", attribute)
    try:
        with pytest.raises(_channel.PackStopped):
            asyncio.run(
                evaluate(
                    process,
                    "exec(value, {})",
                    SEND_RAW_HEAD.replace("value", repr(head)),
                )
            )
        after = vars(_io.ComfyNode).get(attribute)
    finally:
        if vars(_io.ComfyNode).get(attribute) is not before:
            if before is None:
                delattr(_io.ComfyNode, attribute)
            else:
                setattr(_io.ComfyNode, attribute, before)
        _pack_processes.unload(process.module_path)

    assert after is before


def test_decoding_a_message_costs_little_more(comfyui):
    schema = _io.Schema(
        node_id="Node",
        inputs=[_io.Image.Input("image"), _io.Int.Input("n", default=3)],
        outputs=[_io.Image.Output()],
    )
    body, blocks = _codec.dumps(([schema] * 20, torch.zeros(4)), comfyui, 1)

    started = time.perf_counter()
    for _ in range(100):
        _codec.loads(body, comfyui, 1, [bytearray(16)])

    assert (time.perf_counter() - started) / 100 < 0.05


def test_a_log_record_from_a_pack_process_is_not_formatted_again(comfyui, caplog):
    record = {"name": "v4.pack", "levelno": 30, "levelname": "WARNING"}
    record |= {"msg": "%0300000000d", "args": (1,)}

    with caplog.at_level("WARNING", logger="v4.pack"):
        comfyui.output({"output": "log", "record": record})

    assert caplog.messages == ["%0300000000d"]
