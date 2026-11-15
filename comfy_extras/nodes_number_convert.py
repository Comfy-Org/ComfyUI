"""Number Convert node for unified numeric type conversion.

Provides a single node that converts INT, FLOAT, STRING, and BOOL
inputs into FLOAT and INT outputs.
"""


import math

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


class NumberConvertNode(io.ComfyNode):
    """Converts various types to numeric FLOAT and INT outputs."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ComfyNumberConvert",
            display_name="Convert Number",
            category="utilities",
            search_aliases=[
                "int to float", "float to int", "number convert",
                "int2float", "float2int", "cast", "parse number",
                "string to number", "bool to int",
            ],
            inputs=[
                io.MultiType.Input(
                    "value",
                    [io.Int, io.Float, io.String, io.Boolean],
                    display_name="value",
                ),
            ],
            outputs=[
                io.Float.Output(display_name="FLOAT"),
                io.Int.Output(display_name="INT"),
            ],
        )

    @classmethod
    def execute(cls, value) -> io.NodeOutput:
        if isinstance(value, bool):
            float_val = 1.0 if value else 0.0
            int_val = 1 if value else 0
        elif isinstance(value, int):
            float_val = float(value)
            int_val = value
        elif isinstance(value, float):
            float_val = value
            int_val = int(value)
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError("Cannot convert empty string to number.")
            try:
                float_val = float(text)
            except ValueError:
                raise ValueError(
                    f"Cannot convert string to number: {value!r}"
                ) from None
            if not math.isfinite(float_val):
                raise ValueError(
                    f"Cannot convert non-finite value to number: {float_val}"
                )
            try:
                int_val = int(text)
            except ValueError:
                int_val = int(float_val)
        else:
            raise TypeError(
                f"Unsupported input type: {type(value).__name__}"
            )

        if not math.isfinite(float_val):
            raise ValueError(
                f"Cannot convert non-finite value to number: {float_val}"
            )

        return io.NodeOutput(float_val, int_val)


class NumberConvertExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [NumberConvertNode]


async def comfy_entrypoint() -> NumberConvertExtension:
    return NumberConvertExtension()
