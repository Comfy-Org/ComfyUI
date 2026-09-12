from comfy_api.latest import io
from execution import get_input_data

MISSING = object()


class OptionalWidgetNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                "seed": ("INT", {"default": 7}),
                "mode": (["fast", "slow"], {"default": "slow"}),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    def run(self, image, seed, mode, mask=None):
        return (image,)


class SentinelDefaultNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {"seed": ("INT", {"default": 7})}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "run"

    def run(self, seed=MISSING):
        return (seed,)


class KwargsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {"seed": ("INT", {"default": 7})}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "run"

    def run(self, **kwargs):
        return (kwargs.get("seed"),)


class V3OptionalNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="V3OptionalNode",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("seed", optional=True, default=7),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, seed) -> io.NodeOutput:
        return io.NodeOutput(image)


def input_data(class_def, inputs):
    input_data_all, _, _ = get_input_data(inputs, class_def, "1")
    return input_data_all


def test_omitted_optional_widget_gets_its_schema_default():
    """A prompt saved before the input existed does not carry it."""
    data = input_data(OptionalWidgetNode, {"image": ["2", 0]})
    assert data["seed"] == [7]
    assert data["mode"] == ["slow"]


def test_provided_value_wins():
    data = input_data(OptionalWidgetNode, {"image": ["2", 0], "seed": 42})
    assert data["seed"] == [42]


def test_optional_socket_is_left_out():
    """Sockets have no schema default and must stay absent so the node can tell."""
    assert "mask" not in input_data(OptionalWidgetNode, {"image": ["2", 0]})


def test_sentinel_default_is_left_alone():
    """SoftSwitchNode and friends use their own default to detect a missing input."""
    assert "seed" not in input_data(SentinelDefaultNode, {})


def test_kwargs_only_node_is_left_alone():
    assert "seed" not in input_data(KwargsNode, {})


def test_v3_node_uses_its_execute_signature():
    """V3 FUNCTION is a (*args, **kwargs) trampoline, execute is the real signature."""
    assert input_data(V3OptionalNode, {"image": ["2", 0]})["seed"] == [7]
