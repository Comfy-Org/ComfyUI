from typing import TypedDict
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_api.latest import _io

# sentinel for missing inputs
MISSING = object()


class NotNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyNotNode",
            display_name="Not",
            category="utilities/logic",
            description="Logical NOT operation. Returns true if the value is falsy. Uses Python's rules for truthiness.",
            search_aliases=["invert", "toggle", "negate", "flip boolean"],
            inputs=[
                io.AnyType.Input("value"),
            ],
            outputs=[
                io.Boolean.Output(),
            ],
        )

    @classmethod
    def execute(cls, value) -> io.NodeOutput:
        return io.NodeOutput(not value)


class AndNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplatePrefix(
            input=io.AnyType.Input("value"),
            prefix="value",
            min=1,
        )
        return io.Schema(
            node_id="ComfyAndNode",
            display_name="And",
            category="utilities/logic",
            description="Logical AND operation. Returns true if all of the values are truthy. Uses Python's rules for truthiness.",
            search_aliases=["all", "every"],
            inputs=[
                io.Autogrow.Input("values", template=template),
            ],
            outputs=[
                io.Boolean.Output(),
            ],
        )

    @classmethod
    def execute(cls, values: io.Autogrow.Type) -> io.NodeOutput:
        return io.NodeOutput(all(values.values()))


class OrNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplatePrefix(
            input=io.AnyType.Input("value"),
            prefix="value",
            min=1,
        )
        return io.Schema(
            node_id="ComfyOrNode",
            display_name="Or",
            category="utilities/logic",
            description="Logical OR operation. Returns true if any of the values are truthy. Uses Python's rules for truthiness.",
            search_aliases=["any", "some"],
            inputs=[
                io.Autogrow.Input("values", template=template),
            ],
            outputs=[
                io.Boolean.Output(),
            ],
        )

    @classmethod
    def execute(cls, values: io.Autogrow.Type) -> io.NodeOutput:
        return io.NodeOutput(any(values.values()))


class SwitchNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("switch")
        return io.Schema(
            node_id="ComfySwitchNode",
            search_aliases=["if", "then", "switch", "conditional", "branch"],
            display_name="If/Else Switch",
            category="utilities/logic",
            is_experimental=True,
            inputs=[
                io.Boolean.Input("switch"),
                io.MatchType.Input("on_false", template=template, lazy=True),
                io.MatchType.Input("on_true", template=template, lazy=True),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="output"),
            ],
        )

    @classmethod
    def check_lazy_status(cls, switch, on_false=None, on_true=None):
        if switch and on_true is None:
            return ["on_true"]
        if not switch and on_false is None:
            return ["on_false"]

    @classmethod
    def execute(cls, switch, on_true, on_false) -> io.NodeOutput:
        return io.NodeOutput(on_true if switch else on_false)


class SoftSwitchNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("switch")
        return io.Schema(
            node_id="ComfySoftSwitchNode",
            display_name="Soft Switch",
            category="utilities/logic",
            is_experimental=True,
            inputs=[
                io.Boolean.Input("switch"),
                io.MatchType.Input("on_false", template=template, lazy=True, optional=True),
                io.MatchType.Input("on_true", template=template, lazy=True, optional=True),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="output"),
            ],
        )

    @classmethod
    def check_lazy_status(cls, switch, on_false=MISSING, on_true=MISSING):
        # We use MISSING instead of None, as None is passed for connected-but-unevaluated inputs.
        # This trick allows us to ignore the value of the switch and still be able to run execute().

        # One of the inputs may be missing, in which case we need to evaluate the other input
        if on_false is MISSING:
            return ["on_true"]
        if on_true is MISSING:
            return ["on_false"]
        # Normal lazy switch operation
        if switch and on_true is None:
            return ["on_true"]
        if not switch and on_false is None:
            return ["on_false"]

    @classmethod
    def validate_inputs(cls, switch, on_false=MISSING, on_true=MISSING):
        # This check happens before check_lazy_status(), so we can eliminate the case where
        # both inputs are missing.
        if on_false is MISSING and on_true is MISSING:
            return "At least one of on_false or on_true must be connected to Switch node"
        return True

    @classmethod
    def execute(cls, switch, on_true=MISSING, on_false=MISSING) -> io.NodeOutput:
        if on_true is MISSING:
            return io.NodeOutput(on_false)
        if on_false is MISSING:
            return io.NodeOutput(on_true)
        return io.NodeOutput(on_true if switch else on_false)


class CustomComboNode(io.ComfyNode):
    """
    Frontend node that allows user to write their own options for a combo.
    This is here to make sure the node has a backend-representation to avoid some annoyances.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CustomCombo",
            display_name="Custom Combo",
            category="utilities",
            is_experimental=True,
            inputs=[io.Combo.Input("choice", options=[])],
            outputs=[
                io.String.Output(display_name="STRING"),
                io.Int.Output(display_name="INDEX"),
            ],
            accept_all_inputs=True,
        )

    @classmethod
    def validate_inputs(cls, choice: io.Combo.Type, index: int = 0, **kwargs) -> bool:
        # NOTE: DO NOT DO THIS unless you want to skip validation entirely on the node's inputs.
        # I am doing that here because the widgets (besides the combo dropdown) on this node are fully frontend defined.
        # I need to skip checking that the chosen combo option is in the options list, since those are defined by the user.
        return True

    @classmethod
    def execute(cls, choice: io.Combo.Type, index: int = 0, **kwargs) -> io.NodeOutput:
        return io.NodeOutput(choice, index)


class DCTestNode(io.ComfyNode):
    class DCValues(TypedDict):
        combo: str
        string: str
        integer: int
        image: io.Image.Type
        subcombo: dict[str]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DCTestNode",
            display_name="DCTest",
            category="utilities/logic",
            is_output_node=True,
            inputs=[io.DynamicCombo.Input("combo", options=[
                io.DynamicCombo.Option("option1", [io.String.Input("string")]),
                io.DynamicCombo.Option("option2", [io.Int.Input("integer")]),
                io.DynamicCombo.Option("option3", [io.Image.Input("image")]),
                io.DynamicCombo.Option("option4", [
                    io.DynamicCombo.Input("subcombo", options=[
                        io.DynamicCombo.Option("opt1", [io.Float.Input("float_x"), io.Float.Input("float_y")]),
                        io.DynamicCombo.Option("opt2", [io.Mask.Input("mask1", optional=True)]),
                    ])
                ])]
            )],
            outputs=[io.AnyType.Output()],
        )

    @classmethod
    def execute(cls, combo: DCValues) -> io.NodeOutput:
        combo_val = combo["combo"]
        if combo_val == "option1":
            return io.NodeOutput(combo["string"])
        elif combo_val == "option2":
            return io.NodeOutput(combo["integer"])
        elif combo_val == "option3":
            return io.NodeOutput(combo["image"])
        elif combo_val == "option4":
            return io.NodeOutput(f"{combo['subcombo']}")
        else:
            raise ValueError(f"Invalid combo: {combo_val}")


class AutogrowNamesTestNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = _io.Autogrow.TemplateNames(input=io.Float.Input("float"), names=["a", "b", "c"])
        return io.Schema(
            node_id="AutogrowNamesTestNode",
            display_name="AutogrowNamesTest",
            category="utilities/logic",
            inputs=[
                _io.Autogrow.Input("autogrow", template=template)
            ],
            outputs=[io.String.Output()],
        )

    @classmethod
    def execute(cls, autogrow: _io.Autogrow.Type) -> io.NodeOutput:
        vals = list(autogrow.values())
        combined = ",".join([str(x) for x in vals])
        return io.NodeOutput(combined)

class AutogrowPrefixTestNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = _io.Autogrow.TemplatePrefix(input=io.Float.Input("float"), prefix="float", min=1, max=10)
        return io.Schema(
            node_id="AutogrowPrefixTestNode",
            display_name="AutogrowPrefixTest",
            category="utilities/logic",
            inputs=[
                _io.Autogrow.Input("autogrow", template=template)
            ],
            outputs=[io.String.Output()],
        )

    @classmethod
    def execute(cls, autogrow: _io.Autogrow.Type) -> io.NodeOutput:
        vals = list(autogrow.values())
        combined = ",".join([str(x) for x in vals])
        return io.NodeOutput(combined)

class ComboOutputTestNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComboOptionTestNode",
            display_name="ComboOptionTest",
            category="utilities/logic",
            inputs=[io.Combo.Input("combo", options=["option1", "option2", "option3"]),
                    io.Combo.Input("combo2", options=["option4", "option5", "option6"])],
            outputs=[io.Combo.Output(), io.Combo.Output()],
        )

    @classmethod
    def execute(cls, combo: io.Combo.Type, combo2: io.Combo.Type) -> io.NodeOutput:
        return io.NodeOutput(combo, combo2)

class ConvertStringToComboNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ConvertStringToComboNode",
            search_aliases=["string to dropdown", "text to combo"],
            display_name="Convert String to Combo",
            category="utilities/logic",
            inputs=[io.String.Input("string")],
            outputs=[io.Combo.Output()],
        )

    @classmethod
    def execute(cls, string: str) -> io.NodeOutput:
        return io.NodeOutput(string)

class InvertBooleanNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="InvertBooleanNode",
            search_aliases=["not", "toggle", "negate", "flip boolean"],
            display_name="Invert Boolean",
            category="utilities/logic",
            inputs=[io.Boolean.Input("boolean")],
            outputs=[io.Boolean.Output()],
        )

    @classmethod
    def execute(cls, boolean: bool) -> io.NodeOutput:
        return io.NodeOutput(not boolean)

class LogicExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SwitchNode,
            CustomComboNode,
            NotNode,
            AndNode,
            OrNode,
            # SoftSwitchNode,
            # ConvertStringToComboNode,
            # DCTestNode,
            # AutogrowNamesTestNode,
            # AutogrowPrefixTestNode,
            # ComboOutputTestNode,
            # InvertBooleanNode,
        ]

async def comfy_entrypoint() -> LogicExtension:
    return LogicExtension()
