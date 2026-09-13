from __future__ import annotations

import os


def stored_path(path: str) -> str:
    return os.path.abspath(path)


def prefix_case_paths(root: str) -> list[tuple[str, bool]]:
    parent = os.path.dirname(root)
    name = os.path.basename(root)
    upper = name.upper()
    assert upper != name, f"root basename {name!r} has no distinct uppercase form"

    return [
        (root, True),
        (os.path.join(root, "child.png"), True),
        (os.path.join(root, "sub", "deep.png"), True),
        (os.path.join(root, "meta%_*?[].png"), True),
        (os.path.join(parent, upper), False),
        (os.path.join(parent, upper, "case.png"), False),
        (root + "-other" + os.sep + "lexical.png", False),
        (root + "extra", False),
        (os.path.join(parent, "unrelated.png"), False),
        (os.path.join(root, os.pardir, "escaped.png"), False),
        (os.path.join(root, "sub", os.pardir, "in.png"), True),
        (os.path.join(name, "relative.png"), False),
        (root + os.sep + os.sep + "doubled.png", True),
        (os.path.join(root, "trailing.png") + os.sep, True),
    ]


def expected_prefix_case_paths(root: str) -> set[str]:
    return {os.path.abspath(path) for path, expected in prefix_case_paths(root) if expected}
