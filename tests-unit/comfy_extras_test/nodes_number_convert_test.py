import pytest
from unittest.mock import patch, MagicMock

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_server = MagicMock()

with patch.dict("sys.modules", {"nodes": mock_nodes, "server": mock_server}):
    from comfy_extras.nodes_number_convert import NumberConvertNode


class TestNumberConvertExecute:
    @staticmethod
    def _exec(value) -> object:
        return NumberConvertNode.execute(value)

    # --- INT input ---

    def test_int_input(self):
        result = self._exec(42)
        assert result[0] == 42.0
        assert result[1] == 42

    def test_int_zero(self):
        result = self._exec(0)
        assert result[0] == 0.0
        assert result[1] == 0

    def test_int_negative(self):
        result = self._exec(-7)
        assert result[0] == -7.0
        assert result[1] == -7

    # --- FLOAT input ---

    def test_float_input(self):
        result = self._exec(3.14)
        assert result[0] == 3.14
        assert result[1] == 3

    def test_float_truncation_toward_zero(self):
        result = self._exec(-2.9)
        assert result[0] == -2.9
        assert result[1] == -2  # int() truncates toward zero, not floor

    def test_float_output_type(self):
        result = self._exec(5)
        assert isinstance(result[0], float)

    def test_int_output_type(self):
        result = self._exec(5.7)
        assert isinstance(result[1], int)

    # --- BOOL input ---

    def test_bool_true(self):
        result = self._exec(True)
        assert result[0] == 1.0
        assert result[1] == 1

    def test_bool_false(self):
        result = self._exec(False)
        assert result[0] == 0.0
        assert result[1] == 0

    # --- STRING input ---

    def test_string_integer(self):
        result = self._exec("42")
        assert result[0] == 42.0
        assert result[1] == 42

    def test_string_float(self):
        result = self._exec("3.14")
        assert result[0] == 3.14
        assert result[1] == 3

    def test_string_negative(self):
        result = self._exec("-5.5")
        assert result[0] == -5.5
        assert result[1] == -5

    def test_string_with_whitespace(self):
        result = self._exec("  7.0  ")
        assert result[0] == 7.0
        assert result[1] == 7

    def test_string_scientific_notation(self):
        result = self._exec("1e3")
        assert result[0] == 1000.0
        assert result[1] == 1000

    # --- Large number precision (string input) ---

    def test_string_large_int_above_2_53(self):
        """Text-to-int must not lose precision for integers beyond 2^53."""
        big = 2**53 + 1  # 9007199254740993
        result = self._exec(str(big))
        assert result[1] == big

    def test_string_large_negative_int_above_2_53(self):
        big = -(2**53 + 1)
        result = self._exec(str(big))
        assert result[1] == big

    def test_string_very_large_int(self):
        big = 2**63 + 42
        result = self._exec(str(big))
        assert result[1] == big

    def test_string_large_int_float_output_is_float(self):
        """FLOAT output is still a float (may lose precision, but must be float type)."""
        result = self._exec(str(2**53 + 1))
        assert isinstance(result[0], float)

    # --- Large number precision (int input) ---

    def test_int_large_above_2_53(self):
        """Native int input must preserve its value in the INT output."""
        big = 2**53 + 1
        result = self._exec(big)
        assert result[1] == big

    def test_int_large_negative_above_2_53(self):
        big = -(2**53 + 1)
        result = self._exec(big)
        assert result[1] == big

    def test_int_very_large(self):
        big = 2**100
        result = self._exec(big)
        assert result[1] == big

    # --- String decimal / scientific notation fallback ---

    def test_string_decimal_still_truncates(self):
        """Strings with decimal points fall back to int(float(...)) truncation."""
        result = self._exec("3.7")
        assert result[1] == 3

    def test_string_negative_decimal_truncates(self):
        result = self._exec("-2.9")
        assert result[1] == -2

    def test_string_scientific_large(self):
        result = self._exec("1e18")
        assert result[0] == 1e18
        assert result[1] == 10**18

    # --- STRING error paths ---

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Cannot convert empty string"):
            self._exec("")

    def test_whitespace_only_string_raises(self):
        with pytest.raises(ValueError, match="Cannot convert empty string"):
            self._exec("   ")

    def test_non_numeric_string_raises(self):
        with pytest.raises(ValueError, match="Cannot convert string to number"):
            self._exec("abc")

    def test_string_inf_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            self._exec("inf")

    def test_string_nan_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            self._exec("nan")

    def test_string_negative_inf_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            self._exec("-inf")

    # --- Unsupported type ---

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported input type"):
            self._exec([1, 2, 3])
