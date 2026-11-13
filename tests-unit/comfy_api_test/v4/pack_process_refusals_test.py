import fractions
import io as stdio
import pickle

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
