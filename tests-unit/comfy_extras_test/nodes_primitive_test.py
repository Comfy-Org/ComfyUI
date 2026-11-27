from comfy_extras.nodes_primitive import Float


def test_float_step_allows_hundredths():
    schema = Float.define_schema()
    value_input = next(i for i in schema.inputs if i.id == "value")
    assert value_input.step == 0.01
