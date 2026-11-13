import pytest

from comfy_api.latest._io import build_nested_inputs, create_input_dict_v1, get_finalized_class_inputs
from comfy_extras.nodes_string import StringConcatenateMulti


def _rows(values):
    schema = create_input_dict_v1(StringConcatenateMulti.define_schema().inputs)
    _, _, v3_data = get_finalized_class_inputs(schema, values)
    return build_nested_inputs(values, v3_data)['texts']


@pytest.mark.parametrize('values,delimiter,expected', [
    ({'texts.0.text': 'a', 'texts.1.text': 'b'}, ', ', 'a, b'),
    ({'texts.0.text': 'a', 'texts.1.text': 'b', 'texts.2.text': 'c', 'texts.3.text': 'd'}, '-', 'a-b-c-d'),
    ({'texts.0.text': 'a', 'texts.2.text': 'c'}, '+', 'a+c'),
    ({'texts.0.text': '', 'texts.1.text': 'b'}, '|', '|b'),
])
def test_joins_rows_in_order(values, delimiter, expected):
    assert StringConcatenateMulti.execute(_rows(values), delimiter).result == (expected,)


@pytest.mark.parametrize('values', [{}, {'texts.0.text': 'a'}])
def test_rejects_fewer_than_two_rows(values):
    schema = create_input_dict_v1(StringConcatenateMulti.define_schema().inputs)
    with pytest.raises(ValueError, match='expected between 2 and 20'):
        get_finalized_class_inputs(schema, values)


def test_rejects_rows_past_the_limit():
    schema = create_input_dict_v1(StringConcatenateMulti.define_schema().inputs)
    with pytest.raises(ValueError, match='exceeds the index limit of 19'):
        get_finalized_class_inputs(schema, {'texts.20.text': 'a'})
