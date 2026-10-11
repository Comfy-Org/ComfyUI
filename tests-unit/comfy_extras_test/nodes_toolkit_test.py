from comfy_extras.nodes_toolkit import CreateList, GetItemFromList


def test_create_list_accepts_heterogeneous_inputs():
    inputs = CreateList.INPUT_TYPES()

    assert inputs["required"]["inputs"][1]["template"]["input"] == {
        "required": {"input": ("*", {})}
    }
    assert CreateList.RETURN_TYPES == ["*"]
    assert CreateList.OUTPUT_IS_LIST == [True]
    assert CreateList.execute({"input0": ["text"], "input1": [42]}).result == (["text", 42],)


def test_get_item_from_list_uses_integer_index():
    assert GetItemFromList.OUTPUT_IS_LIST == [False]
    assert GetItemFromList.execute(["text", 42], [1]).result == (42,)
    assert GetItemFromList.execute(
        [["image 1", "image 2"], ["text 1", "text 2"]], [0]
    ).result == (["image 1", "image 2"],)
