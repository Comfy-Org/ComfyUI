import pytest

from api_server.utils.query_params import parse_optional_int_query_param


def test_parse_optional_int_query_param_returns_none_when_missing():
    assert parse_optional_int_query_param({}, "offset") is None


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("0", 0),
        ("5", 5),
        ("-1", -1),
    ],
)
def test_parse_optional_int_query_param_parses_integers(raw_value, expected):
    query = {"offset": raw_value}

    assert parse_optional_int_query_param(query, "offset") == expected


@pytest.mark.parametrize(
    ("name", "raw_value"),
    [
        ("offset", "not-an-integer"),
        ("offset", "1.5"),
        ("offset", ""),
        ("max_items", "not-an-integer"),
    ],
)
def test_parse_optional_int_query_param_rejects_invalid_integers(name, raw_value):
    query = {name: raw_value}

    with pytest.raises(ValueError) as exc_info:
        parse_optional_int_query_param(query, name)

    assert str(exc_info.value) == f"{name} must be an integer"
