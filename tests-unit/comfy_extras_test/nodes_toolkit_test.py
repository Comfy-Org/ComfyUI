from comfy_extras.nodes_toolkit import CreateList, GetItemFromList


def test_create_list_accepts_heterogeneous_inputs():
    inputs = CreateList.INPUT_TYPES()

    assert inputs["required"]["inputs"][1]["template"]["input"] == {
        "required": {"input": ("*", {})}
    }
    assert CreateList.RETURN_TYPES == ["*"]
    assert CreateList.OUTPUT_IS_LIST == [True]
    assert CreateList.execute({"input0": ["text"], "input1": [42]}).result == (["text", 42],)


def test_get_item_from_list_output_matches_list_type():
    info = GetItemFromList.GET_NODE_INFO_V1()
    list_type, list_options = info["input"]["required"]["list"]

    assert list_type == "COMFY_MATCHTYPE_V3"
    assert list_options["template"]["allowed_types"] == "*"
    assert info["output"] == ["COMFY_MATCHTYPE_V3"]
    assert info["output_matchtypes"] == [list_options["template"]["template_id"]]


def test_get_item_from_list_uses_integer_index():
    assert GetItemFromList.OUTPUT_IS_LIST == [False]
    assert GetItemFromList.execute(["a", "b", "c"], [2]).result == ("c",)
    assert GetItemFromList.execute(["text", 42], [1]).result == (42,)
    assert GetItemFromList.execute(
        [["image 1", "image 2"], ["text 1", "text 2"]], [0]
    ).result == (["image 1", "image 2"],)
