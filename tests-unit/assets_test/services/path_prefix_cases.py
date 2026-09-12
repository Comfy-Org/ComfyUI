from __future__ import annotations

import os


def stored_path(path: str) -> str:
    return os.path.abspath(path)


def prefix_case_paths(root: str) -> list[str]:
    parent = os.path.dirname(root)
    name = os.path.basename(root)
    upper = name.upper()
    assert upper != name, f"root basename {name!r} has no distinct uppercase form"

    return [
        root,
        os.path.join(root, "child.png"),
        os.path.join(root, "sub", "deep.png"),
        os.path.join(root, "meta%_*?[].png"),
        os.path.join(parent, upper),
        os.path.join(parent, upper, "case.png"),
        root + "-other" + os.sep + "lexical.png",
        root + "extra",
        os.path.join(parent, "unrelated.png"),
        os.path.join(root, os.pardir, "escaped.png"),
        os.path.join(root, "sub", os.pardir, "in.png"),
        os.path.join(name, "relative.png"),
        root + os.sep + os.sep + "doubled.png",
        os.path.join(root, "trailing.png") + os.sep,
    ]
