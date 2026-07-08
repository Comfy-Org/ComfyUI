"""Math expression node using simpleeval for safe evaluation.

Provides a ComfyMathExpression node that evaluates math expressions
against dynamically-grown numeric inputs.
"""


import math
import string

from simpleeval import simple_eval
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


MAX_EXPONENT = 4000


def _variadic_sum(*args):
    """Support both sum(values) and sum(a, b, c)."""
    if len(args) == 1 and hasattr(args[0], "__iter__"):
        return sum(args[0])
    return sum(args)


def _safe_pow(base, exp):
    """Wrap pow() with an exponent cap to prevent DoS via huge exponents.

    The ** operator is already guarded by simpleeval's safe_power, but
    pow() as a callable bypasses that guard.
    """
    if abs(exp) > MAX_EXPONENT:
        raise ValueError(f"Exponent {exp} exceeds maximum allowed ({MAX_EXPONENT})")
    return pow(base, exp)


MATH_FUNCTIONS = {
    "sum": _variadic_sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "pow": _safe_pow,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "int": int,
    "float": float,
}


class MathExpressionNode(io.ComfyNode):
    """Evaluates a math expression against dynamically-grown inputs."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        autogrow = io.Autogrow.TemplateNames(
            input=io.MultiType.Input("value", [io.Float, io.Int, io.Boolean]),
            names=list(string.ascii_lowercase),
            min=1,
        )
        return io.Schema(
            node_id="ComfyMathExpression",
            display_name="Math Expression",
            category="utilities",
            search_aliases=[
                "expression", "formula", "calculate", "calculator",
                "eval", "math",
            ],
            inputs=[
                io.String.Input("expression", default="a + b", multiline=True),
                io.Autogrow.Input("values", template=autogrow),
            ],
            outputs=[
                io.Float.Output(display_name="FLOAT"),
                io.Int.Output(display_name="INT"),
                io.Boolean.Output(display_name="BOOL"),
            ],
        )

    @classmethod
    def execute(
        cls, expression: str, values: io.Autogrow.Type
    ) -> io.NodeOutput:
        if not expression.strip():
            raise ValueError("Expression cannot be empty.")

        context: dict = dict(values)
        context["values"] = list(values.values())

        result = simple_eval(expression, names=context, functions=MATH_FUNCTIONS)
        # bool check must come first because bool is a subclass of int in Python
        if not isinstance(result, (int, float)):
            raise ValueError(
                f"Math Expression '{expression}' must evaluate to a numeric result, "
                f"got {type(result).__name__}: {result!r}"
            )
        try:
            float_result = float(result)
        except OverflowError:
            raise ValueError(
                f"Math Expression '{expression}' produced a result too large to "
                f"represent as a float: {result}"
            ) from None
        if not math.isfinite(float_result):
            raise ValueError(
                f"Math Expression '{expression}' produced a non-finite result: {result}"
            )
        return io.NodeOutput(float_result, int(result), bool(result))


class MathExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MathExpressionNode]


async def comfy_entrypoint() -> MathExtension:
    return MathExtension()
