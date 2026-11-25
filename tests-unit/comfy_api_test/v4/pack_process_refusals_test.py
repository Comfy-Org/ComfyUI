import fractions
import io as stdio
import pickle
import pickletools

import pytest

from comfy_api.latest import _io
from comfy_api.v4 import _codec, _pack_host

from conftest import needs_pack_process

pytestmark = needs_pack_process


class _Side:
    peer = "pack test"
    io_types = {}

    def __init__(self):
        self.objects = {}

    def remote(self, key, type_name, methods, values):
        return _codec.Remote(self, key, type_name, methods, values)

    def resolve(self, key, chain):
        return self.objects[key]


class _Sent:
    """Stands for one of ComfyUI's objects that the pack sends back."""


SENT = _Sent()


def _ops(value):
    buffer = stdio.BytesIO()
    pickler = pickle.Pickler(buffer, protocol=2)
    pickler.fast = True
    pickler.persistent_id = lambda obj: ("yours", "held") if obj is SENT else None
    pickler.dump(value)
    return buffer.getvalue()[2:-1]


class _Held:
    pass


@pytest.mark.parametrize(
    "target, opcodes",
    [
        (
            "a class",
            b"ccomfy_api.latest._io\nComfyNode\n"
            + _ops((None, {"changed": 1}))
            + b"b.",
        ),
        ("a class", b"cfractions\nFraction\n" + _ops((None, {"changed": 1})) + b"b."),
        (
            "an enum member",
            b"ccomfy_api.latest._io\nFolderType\n"
            + _ops(("output",))
            + b"R"
            + _ops(({"_value_": "x"}, None))
            + b"b.",
        ),
        (
            "an object it did not make",
            _ops(SENT) + _ops((None, {"changed": 1})) + b"b.",
        ),
        ("an object it did not make", _ops(SENT) + _ops("key") + _ops("value") + b"s."),
    ],
)
def test_a_message_cannot_change_a_class_an_enum_member_or_an_object_it_did_not_make(
    target, opcodes
):
    side = _Side()
    held = side.objects["held"] = _Held()

    with pytest.raises(pickle.UnpicklingError, match="refused to change"):
        _codec.loads(b"\x80\x05" + opcodes, side, 1)

    assert "changed" not in vars(_io.ComfyNode) and "changed" not in vars(
        fractions.Fraction
    )
    assert _io.FolderType.output.value == "output"
    assert vars(held) == {}


CHECKED = "ADDITEMS APPEND APPENDS BUILD NEWOBJ REDUCE SETITEM SETITEMS"
REFUSED = "EXT1 EXT2 EXT4 INST NEWOBJ_EX OBJ"
AS_PYTHON = """
    BINBYTES BINBYTES8 BINFLOAT BINGET BININT BININT1 BININT2 BINPERSID BINPUT BINSTRING
    BINUNICODE BINUNICODE8 BYTEARRAY8 DICT DUP EMPTY_DICT EMPTY_LIST EMPTY_SET EMPTY_TUPLE FLOAT
    FRAME FROZENSET GET GLOBAL INT LIST LONG LONG1 LONG4 LONG_BINGET LONG_BINPUT MARK MEMOIZE
    NEWFALSE NEWTRUE NEXT_BUFFER NONE PERSID POP POP_MARK PROTO PUT READONLY_BUFFER
    SHORT_BINBYTES SHORT_BINSTRING SHORT_BINUNICODE STACK_GLOBAL STOP STRING TUPLE TUPLE1 TUPLE2
    TUPLE3 UNICODE
"""


BARE_IMAGE = b"cPIL.Image\nImage\n)\x81"
BARE_COMPONENTS = b"ccomfy_api.latest._util.video_types\nVideoComponents\n)\x81"


@pytest.mark.parametrize(
    "opcodes",
    [
        BARE_IMAGE + b".",
        BARE_COMPONENTS + b".",
        b"(" + BARE_IMAGE + b"l.",
        BARE_COMPONENTS + _ops({"images": 1}) + b"b.",
    ],
    ids=["an image", "a V3 value", "an image in a list", "a V3 value short of a field"],
)
def test_a_message_cannot_make_an_object_without_the_state_its_class_needs(opcodes):
    with pytest.raises(pickle.UnpicklingError, match="without"):
        _codec.loads(b"\x80\x05" + opcodes, _Side(), 1)


def test_a_reply_whose_error_holds_an_object_made_without_state_is_refused():
    reply = (
        b"\x80\x05\x95w\x00\x00\x00\x00\x00\x00\x00}\x94(\x8c\x02to\x94K\x07\x8c\x05chain"
        b"\x94K\x01\x8c\x05error\x94}\x94(\x8c\x06module\x94\x8c\x04evil\x94\x8c\x08qualname"
        b"\x94\x8c\x04Boom\x94\x8c\x04args\x94\x8c\tPIL.Image\x94\x8c\x05Image\x94\x93\x94)"
        b"\x81\x94\x85\x94\x8c\ttraceback\x94\x8c\x02tb\x94uu."
    )

    with pytest.raises(pickle.UnpicklingError, match="without the state"):
        _codec.loads(reply, _Side(), 1)


def test_a_file_value_short_of_an_attribute_is_refused():
    without_its_format = (
        b"\x80\x05\x95C\x00\x00\x00\x00\x00\x00\x00\x8c\x04file\x94\x8c\x06File3D\x94}\x94"
        b"\x8c\x07_source\x94\x8c\x03_io\x94\x8c\x07BytesIO\x94\x93\x94)\x81\x94C\x04glTF"
        b"\x94K\x00N\x87\x94bs\x87\x94Q."
    )

    with pytest.raises(pickle.UnpicklingError, match="holds a file"):
        _codec.loads(without_its_format, _Side(), 1)


def nested(levels):
    """A list that names the list below it twice, `levels` deep: 2**levels places to walk."""
    value = [1]
    for _ in range(levels):
        value = [value, value]
    return value


@pytest.mark.parametrize("levels", [21, 60])
def test_a_value_too_large_to_walk_through_its_shared_parts_is_refused(levels):
    message = pickle.dumps(nested(levels), protocol=5)

    with pytest.raises(pickle.UnpicklingError, match="too large to walk"):
        _codec.loads(message, _Side(), 1)
    assert len(message) < 400


@pytest.mark.parametrize(
    "value",
    [
        nested(10),
        [[0.0] * 3] * 1000,
        ({"samples": [1, 2]},) * 2,
        [list(range(1000)) for _ in range(2000)],
    ],
    ids=[
        "a little sharing",
        "one row a thousand times",
        "one latent twice",
        "two million places, none shared",
    ],
)
def test_a_value_with_shared_parts_that_is_small_to_walk_crosses(value):
    assert _codec.loads(pickle.dumps(value, protocol=5), _Side(), 1) == value


def test_a_value_that_holds_itself_crosses():
    value = [1]
    value.append(value)

    decoded = _codec.loads(pickle.dumps(value, protocol=5), _Side(), 1)

    assert decoded[0] == 1 and decoded[1] is decoded


def test_a_message_cannot_name_one_of_its_objects_by_anything_but_text():
    named_by_a_tuple = _ops(("object", 1, ("a", "tuple"), (), {}))

    with pytest.raises(pickle.UnpicklingError, match="described"):
        _codec.loads(b"\x80\x05" + named_by_a_tuple + b"Q.", _Side(), 1)


@pytest.mark.parametrize(
    "key",
    ["k" * 64, (1,), (1, 2, 3), (1, "2"), (1, 1 << 64), (-1, 2), (True, 2)],
    ids=["text", "one", "three", "text in it", "too large", "negative", "not a number"],
)
def test_a_message_cannot_name_one_of_its_objects_by_a_key_that_is_not_two_numbers(key):
    keyed = _ops(("object", key, "Thing", (), {}))

    with pytest.raises(pickle.UnpicklingError, match="key"):
        _codec.loads(b"\x80\x05" + keyed + b"Q.", _Side(), 1)


@pytest.mark.parametrize(
    "description",
    [
        {"args": (_Held(),)},
        {"args": [1]},
        {"module": 1},
        {"qualname": None},
        {"traceback": _Held()},
    ],
    ids=["an object", "a list of args", "module", "qualname", "traceback"],
)
def test_an_error_described_by_anything_but_text_and_plain_values_is_one_comfyui_can_print(
    description,
):
    from comfy_api.v4 import _channel

    plain = {"module": "pack", "qualname": "Boom", "args": ("x",), "traceback": "tb"}

    error = _channel.rebuild(plain | description)

    assert isinstance(error, _channel.PackError)
    assert "could not be read" in str(error) and str(error.__cause__)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["StopIteration", "StopAsyncIteration"])
async def test_an_error_a_waiting_coroutine_cannot_be_given_arrives_as_a_pack_error(name):
    import asyncio

    from comfy_api.v4 import _channel

    error = _channel.rebuild(
        {"module": "builtins", "qualname": name, "args": ("x",), "traceback": "tb"}
    )
    waiting = asyncio.get_running_loop().create_future()
    waiting.set_exception(error)

    with pytest.raises(_channel.PackError, match="x"):
        await waiting
    assert type(error).__name__ == name


def test_the_decoder_has_decided_every_instruction_of_pythons_unpickler():
    """The decoder is built on `pickle._Unpickler`, which Python may change: an instruction it
    gains, or one the decoder no longer checks, fails here before a message can use it."""
    python, ours = pickle._Unpickler.dispatch, _codec._Unpickler.dispatch
    names = {ord(opcode.code): opcode.name for opcode in pickletools.opcodes}

    def decided(code):
        if ours[code] is _codec._refuse:
            return "refused"
        return "as Python" if ours[code] is python[code] else "checked"

    assert {names[code]: decided(code) for code in python} == {
        **dict.fromkeys(CHECKED.split(), "checked"),
        **dict.fromkeys(REFUSED.split(), "refused"),
        **dict.fromkeys(AS_PYTHON.split(), "as Python"),
    }


@pytest.mark.parametrize("name", REFUSED.split())
def test_a_message_with_an_instruction_no_value_needs_is_refused(name):
    code = next(o.code for o in pickletools.opcodes if o.name == name).encode("latin-1")

    with pytest.raises(pickle.UnpicklingError, match="a value does not need"):
        _codec.loads(b"\x80\x05" + _ops(()) + code + b".", _Side(), 1)


def test_comfyui_keeps_only_what_it_uses_of_what_a_pack_offers():
    process = _pack_host.PackProcess("/nowhere/pack", "pack")
    process.channel = type("Channel", (), {"release": staticmethod(lambda key: None)})()

    remote = process.remote(
        (0, 1),
        "Video",
        ("get_stream_source", "generate_noise"),
        {"seed": 3, "format": "gltf"},
    )

    assert remote.seed == 3 and callable(remote.generate_noise)
    for name in ("get_stream_source", "format"):
        with pytest.raises(AttributeError):
            getattr(remote, name)


@pytest.mark.parametrize(
    "web, refused",
    [
        (None, False),
        ("web", False),
        ("web/../js", False),
        ("..", True),
        ("/etc", True),
        ("link", True),
        (3, True),
    ],
)
def test_a_web_folder_of_a_pack_process_is_served_only_from_inside_the_pack(
    tmp_path, web, refused
):
    pack = tmp_path / "pack"
    (pack / "web").mkdir(parents=True)
    (pack / "link").symlink_to(tmp_path)

    assert _pack_host._web_directory(str(pack), web) == (None if refused else web)
