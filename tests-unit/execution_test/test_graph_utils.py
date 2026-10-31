from comfy_execution.graph_utils import is_link


def test_is_link_accepts_plain_list_with_integer_output_index():
    assert is_link(["node", 0])


def test_is_link_rejects_float_output_index():
    assert not is_link(["node", 0.0])


def test_is_link_rejects_negative_output_index():
    assert not is_link(["node", -1])


def test_is_link_rejects_list_subclasses():
    class LinkList(list):
        pass

    assert not is_link(LinkList(["node", 0]))


def test_is_link_rejects_non_plain_node_id_and_bool_index():
    class NodeId(str):
        pass

    assert not is_link([NodeId("node"), 0])
    assert not is_link(["node", True])
