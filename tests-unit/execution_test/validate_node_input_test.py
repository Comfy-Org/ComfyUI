import pytest
from comfy_execution.validation import validate_node_input


def test_exact_match():
    """Test cases where types match exactly"""
    assert validate_node_input("STRING", "STRING")
    assert validate_node_input("STRING,INT", "STRING,INT")
    assert validate_node_input("INT,STRING", "STRING,INT")  # Order shouldn't matter


def test_strict_mode():
    """Test strict mode validation"""
    # Should pass - received type is subset of input type
    assert validate_node_input("STRING", "STRING,INT", strict=True)
    assert validate_node_input("INT", "STRING,INT", strict=True)
    assert validate_node_input("STRING,INT", "STRING,INT,BOOLEAN", strict=True)

    # Should fail - received type is not subset of input type
    assert not validate_node_input("STRING,INT", "STRING", strict=True)
    assert not validate_node_input("STRING,BOOLEAN", "STRING", strict=True)
    assert not validate_node_input("INT,BOOLEAN", "STRING,INT", strict=True)


def test_non_strict_mode():
    """Test non-strict mode validation (default behavior)"""
    # Should pass - types have overlap
    assert validate_node_input("STRING,BOOLEAN", "STRING,INT")
    assert validate_node_input("STRING,INT", "INT,BOOLEAN")
    assert validate_node_input("STRING", "STRING,INT")

    # Should fail - no overlap in types
    assert not validate_node_input("BOOLEAN", "STRING,INT")
    assert not validate_node_input("FLOAT", "STRING,INT")
    assert not validate_node_input("FLOAT,BOOLEAN", "STRING,INT")


def test_whitespace_handling():
    """Test that whitespace is handled correctly"""
    assert validate_node_input("STRING, INT", "STRING,INT")
    assert validate_node_input("STRING,INT", "STRING, INT")
    assert validate_node_input(" STRING , INT ", "STRING,INT")
    assert validate_node_input("STRING,INT", " STRING , INT ")


def test_empty_strings():
    """Test behavior with empty strings"""
    assert validate_node_input("", "")
    assert not validate_node_input("STRING", "")
    assert not validate_node_input("", "STRING")


def test_single_vs_multiple():
    """Test single type against multiple types"""
    assert validate_node_input("STRING", "STRING,INT,BOOLEAN")
    assert validate_node_input("STRING,INT,BOOLEAN", "STRING", strict=False)
    assert not validate_node_input("STRING,INT,BOOLEAN", "STRING", strict=True)


def test_non_string():
    """Test non-string types"""
    obj1 = object()
    obj2 = object()
    assert validate_node_input(obj1, obj1)
    assert not validate_node_input(obj1, obj2)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "received_type,input_type,expected",
    [
        (["simple", "karras"], ["simple", "karras", "AYS SD1"], True),
        (["karras", "simple"], ["simple", "karras"], True),
        (["simple", "karras"], ["simple"], False),
        (["simple", "karras"], ["karras", "exponential"], False),
        (["simple"], ["exponential"], False),
        (["a,b"], ["a", "b"], False),
        ([1, 2], [1, 2, 3], True),
        ([[1]], [[1], [2]], True),
        ([[3]], [[1], [2]], False),
        ([1], [1, [2]], True),
        ([[1]], [1, 2], False),
    ],
)
def test_combo_options(received_type, input_type, strict, expected):
    assert validate_node_input(received_type, input_type, strict) == expected


def test_large_combo_options_do_not_require_quadratic_comparisons():
    class Option(str):
        comparisons = 0
        __hash__ = str.__hash__

        def __eq__(self, other):
            Option.comparisons += 1
            return super().__eq__(other)

    count = 1000
    supported = [Option(str(index)) for index in range(count)]
    received = [Option(str(index)) for index in reversed(range(count))]

    assert validate_node_input(received, supported)
    assert Option.comparisons < count * 4


class NotEqualsOverrideTest(str):
    """Test class for ``__ne__`` override."""

    def __ne__(self, value: object) -> bool:
        if self == "*" or value == "*":
            return False
        if self == "LONGER_THAN_2":
            return not len(value) > 2
        raise TypeError("This is a class for unit tests only.")


def test_ne_override():
    """Test ``__ne__`` any override"""
    any = NotEqualsOverrideTest("*")
    invalid_type = "INVALID_TYPE"
    obj = object()
    assert validate_node_input(any, any)
    assert validate_node_input(any, invalid_type)
    assert validate_node_input(any, obj)
    assert validate_node_input(any, {})
    assert validate_node_input(any, [])
    assert validate_node_input(any, [1, 2, 3])


def test_ne_custom_override():
    """Test ``__ne__`` custom override"""
    special = NotEqualsOverrideTest("LONGER_THAN_2")

    assert validate_node_input(special, special)
    assert validate_node_input(special, "*")
    assert validate_node_input(special, "INVALID_TYPE")
    assert validate_node_input(special, [1, 2, 3])

    # Should fail
    assert not validate_node_input(special, [1, 2])
    assert not validate_node_input(special, "TY")


@pytest.mark.parametrize(
    "received,input_type,strict,expected",
    [
        ("STRING", "STRING", False, True),
        ("STRING,INT", "STRING,INT", False, True),
        ("STRING", "STRING,INT", True, True),
        ("STRING,INT", "STRING", True, False),
        ("BOOLEAN", "STRING,INT", False, False),
        ("STRING,BOOLEAN", "STRING,INT", False, True),
    ],
)
def test_parametrized_cases(received, input_type, strict, expected):
    """Parametrized test cases for various scenarios"""
    assert validate_node_input(received, input_type, strict) == expected
