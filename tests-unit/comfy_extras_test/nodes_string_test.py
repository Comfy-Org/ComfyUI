from unittest.mock import patch, MagicMock

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_server = MagicMock()

with patch.dict("sys.modules", {"nodes": mock_nodes, "server": mock_server}):
    from comfy_extras.nodes_string import RegexExtract


def _extract(string, pattern, mode, group_index=1):
    return RegexExtract.execute(
        string=string,
        regex_pattern=pattern,
        mode=mode,
        case_insensitive=True,
        multiline=False,
        dotall=False,
        group_index=group_index,
    )[0]


class TestRegexExtractNonParticipatingGroup:
    def test_first_group_non_participating_returns_empty_string(self):
        # An alternation where the requested group is on the branch that did
        # not match: match.group(1) is None. The node output is a String, so
        # it must return "" rather than leaking None downstream.
        result = _extract("hello", r"(\d+)-(\d+)|(\w+)", "First Group", 1)
        assert result == ""
        assert isinstance(result, str)

    def test_all_groups_non_participating_does_not_crash(self):
        # Previously appended None and crashed in join() with a TypeError.
        assert _extract("hello world", r"(\d+)|(\w+)", "All Groups", 1) == ""

    def test_all_groups_mixes_participating_and_non(self):
        assert _extract("x 5 y", r"([a-z])|(\d)", "All Groups", 1) == "x\ny"

    def test_first_group_normal_still_works(self):
        assert _extract("2026-07", r"(\d+)-(\d+)", "First Group", 2) == "07"
