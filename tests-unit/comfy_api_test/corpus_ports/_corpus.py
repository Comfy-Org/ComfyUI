"""Nodes ported from the pack corpus to the V2 API, each next to the original it's checked against.

``original(path, names, namespace)`` takes definitions straight out of a corpus pack's source file
(unmodified), so a test can run the V1 original in ComfyUI's process and compare it with the port.
"""
import ast
import os

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4, "overlay-r2", "pack-db", "packs")


def corpus_file(pack_glob: str, relative: str) -> str:
    import glob
    found = glob.glob(os.path.join(CORPUS, pack_glob, "*", "*-HEAD", relative))
    if not found:
        raise FileNotFoundError(f"{pack_glob}/{relative} isn't in the pack corpus")
    return found[0]


def original(path: str, names: list, namespace: dict) -> dict:
    """The named top-level definitions of a corpus source file, run in ``namespace``."""
    with open(path, encoding="utf-8") as file:
        tree = ast.parse(file.read(), path)
    wanted = [node for node in tree.body
              if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign)) and
              (getattr(node, "name", None) in names or
               (isinstance(node, ast.Assign) and any(getattr(t, "id", None) in names for t in node.targets)))]
    exec(compile(ast.Module(body=wanted, type_ignores=[]), path, "exec"), namespace)  # noqa: S102
    return namespace
