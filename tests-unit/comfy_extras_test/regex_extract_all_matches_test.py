from unittest.mock import patch, MagicMock

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

with patch.dict("sys.modules", {"nodes": mock_nodes, "server": MagicMock()}):
    from comfy_extras.nodes_string import RegexExtract


def _all_matches(string, pattern):
    return RegexExtract.execute(
        string=string,
        regex_pattern=pattern,
        mode="All Matches",
        case_insensitive=True,
        multiline=False,
        dotall=False,
        group_index=1,
    )[0]


def test_all_matches_returns_whole_matches_with_one_group():
    assert _all_matches("10px 20px", r"(\d+)px") == "10px\n20px"


def test_all_matches_returns_whole_matches_with_several_groups():
    assert _all_matches("w=640 h=480", r"(\w)=(\d+)") == "w=640\nh=480"


def test_all_matches_without_groups_is_unchanged():
    assert _all_matches("a1 b2 c3", r"[a-z]\d") == "a1\nb2\nc3"
    assert _all_matches("abc", r"\d") == ""
