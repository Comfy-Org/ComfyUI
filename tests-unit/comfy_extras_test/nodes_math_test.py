import math

import pytest
from collections import OrderedDict
from unittest.mock import patch, MagicMock

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_server = MagicMock()

with patch.dict("sys.modules", {"nodes": mock_nodes, "server": mock_server}):
    from comfy_extras.nodes_math import MathExpressionNode


class TestMathExpressionExecute:
    @staticmethod
    def _exec(expression: str, **kwargs) -> object:
        values = OrderedDict(kwargs)
        return MathExpressionNode.execute(expression, values)

    def test_addition(self):
        result = self._exec("a + b", a=3, b=4)
        assert result[0] == 7.0
        assert result[1] == 7

    def test_subtraction(self):
        result = self._exec("a - b", a=10, b=3)
        assert result[0] == 7.0
        assert result[1] == 7

    def test_multiplication(self):
        result = self._exec("a * b", a=3, b=5)
        assert result[0] == 15.0
        assert result[1] == 15

    def test_division(self):
        result = self._exec("a / b", a=10, b=4)
        assert result[0] == 2.5
        assert result[1] == 2

    def test_single_input(self):
        result = self._exec("a * 2", a=5)
        assert result[0] == 10.0
        assert result[1] == 10

    def test_three_inputs(self):
        result = self._exec("a + b + c", a=1, b=2, c=3)
        assert result[0] == 6.0
        assert result[1] == 6

    def test_float_inputs(self):
        result = self._exec("a + b", a=1.5, b=2.5)
        assert result[0] == 4.0
        assert result[1] == 4

    def test_mixed_int_float_inputs(self):
        result = self._exec("a * b", a=1024, b=1.5)
        assert result[0] == 1536.0
        assert result[1] == 1536

    def test_mixed_resolution_scale(self):
        result = self._exec("a * b", a=512, b=0.75)
        assert result[0] == 384.0
        assert result[1] == 384

    def test_sum_values_array(self):
        result = self._exec("sum(values)", a=1, b=2, c=3)
        assert result[0] == 6.0

    def test_sum_variadic(self):
        result = self._exec("sum(a, b, c)", a=1, b=2, c=3)
        assert result[0] == 6.0

    def test_min_values(self):
        result = self._exec("min(values)", a=5, b=2, c=8)
        assert result[0] == 2.0

    def test_max_values(self):
        result = self._exec("max(values)", a=5, b=2, c=8)
        assert result[0] == 8.0

    def test_abs_function(self):
        result = self._exec("abs(a)", a=-7)
        assert result[0] == 7.0
        assert result[1] == 7

    def test_sqrt(self):
        result = self._exec("sqrt(a)", a=16)
        assert result[0] == 4.0
        assert result[1] == 4

    def test_ceil(self):
        result = self._exec("ceil(a)", a=2.3)
        assert result[0] == 3.0
        assert result[1] == 3

    def test_floor(self):
        result = self._exec("floor(a)", a=2.7)
        assert result[0] == 2.0
        assert result[1] == 2

    def test_sin(self):
        result = self._exec("sin(a)", a=0)
        assert result[0] == 0.0

    def test_log10(self):
        result = self._exec("log10(a)", a=100)
        assert result[0] == 2.0
        assert result[1] == 2

    def test_float_output_type(self):
        result = self._exec("a + b", a=1, b=2)
        assert isinstance(result[0], float)

    def test_int_output_type(self):
        result = self._exec("a + b", a=1, b=2)
        assert isinstance(result[1], int)

    def test_non_numeric_result_raises(self):
        with pytest.raises(ValueError, match="must evaluate to a numeric result"):
            self._exec("'hello'", a=42)

    def test_undefined_function_raises(self):
        with pytest.raises(Exception, match="not defined"):
            self._exec("str(a)", a=42)

    def test_boolean_result(self):
        result = self._exec("a > b", a=5, b=3)
        assert result[2] is True
        result = self._exec("a > b", a=3, b=5)
        assert result[2] is False

    def test_empty_expression_raises(self):
        with pytest.raises(ValueError, match="Expression cannot be empty"):
            self._exec("", a=1)

    def test_whitespace_only_expression_raises(self):
        with pytest.raises(ValueError, match="Expression cannot be empty"):
            self._exec("   ", a=1)

    # --- Missing function coverage (round, pow, log, log2, cos, tan) ---

    def test_round(self):
        result = self._exec("round(a)", a=2.7)
        assert result[0] == 3.0
        assert result[1] == 3

    def test_round_with_ndigits(self):
        result = self._exec("round(a, 2)", a=3.14159)
        assert result[0] == pytest.approx(3.14)

    def test_pow(self):
        result = self._exec("pow(a, b)", a=2, b=10)
        assert result[0] == 1024.0
        assert result[1] == 1024

    def test_log(self):
        result = self._exec("log(a)", a=math.e)
        assert result[0] == pytest.approx(1.0)

    def test_log2(self):
        result = self._exec("log2(a)", a=8)
        assert result[0] == pytest.approx(3.0)

    def test_cos(self):
        result = self._exec("cos(a)", a=0)
        assert result[0] == 1.0

    def test_tan(self):
        result = self._exec("tan(a)", a=0)
        assert result[0] == 0.0

    # --- int/float converter functions ---

    def test_int_converter(self):
        result = self._exec("int(a / b)", a=7, b=2)
        assert result[1] == 3

    def test_float_converter(self):
        result = self._exec("float(a)", a=5)
        assert result[0] == 5.0

    # --- Error path tests ---

    def test_division_by_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            self._exec("a / b", a=1, b=0)

    def test_sqrt_negative_raises(self):
        with pytest.raises(ValueError, match="math domain error"):
            self._exec("sqrt(a)", a=-1)

    def test_overflow_inf_raises(self):
        with pytest.raises(ValueError, match="non-finite result"):
            self._exec("a * b", a=1e308, b=10)

    def test_pow_huge_exponent_raises(self):
        with pytest.raises(ValueError, match="Exponent .* exceeds maximum"):
            self._exec("pow(a, b)", a=10, b=10000000)

    def test_huge_int_result_raises_value_error(self):
        # Exponent is within the allowed MAX_EXPONENT range, so the result is a
        # finite Python int that is nonetheless too large to convert to float.
        # This must raise a clean ValueError, not an uncaught OverflowError.
        with pytest.raises(ValueError, match="too large to represent as a float"):
            self._exec("2 ** 3999")
