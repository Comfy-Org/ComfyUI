"""Nodes ported from the pack corpus to the V2 API, each next to the original it's checked against.

``original(path, names, namespace)`` takes definitions straight out of a corpus pack's source file
(unmodified), so a test can run the V1 original in ComfyUI's process and compare it with the port.
"""
from ._corpus import CORPUS, corpus_file, original  # noqa: F401
