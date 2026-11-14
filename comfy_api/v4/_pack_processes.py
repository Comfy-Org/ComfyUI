"""Which V4 packs run in their own process, and the processes ComfyUI runs for them.

`--pack-process NAME ...` names the packs, by the name ComfyUI's loader gives each. The option is
not supported on Windows in this release: there the packs it names run inside ComfyUI. When
ComfyUI exits, `_packs` takes every pack away, which ends its process with the programs it
started, and then ends the processes still loading.
"""

import logging
import sys
from typing import TYPE_CHECKING

from comfy.cli_args import args

if TYPE_CHECKING:
    from ._pack_host import PackProcess

# The packs running in their own process, by the path they were loaded from.
processes: dict[str, "PackProcess"] = {}
# The `--pack-process` names the loader has met.
_met: set[str] = set()


def isolated(module_path: str) -> bool:
    """Whether the pack loaded from `module_path` runs in its own process."""
    import nodes  # nodes imports this module

    name = nodes.get_module_name(module_path)
    if name not in args.pack_process:
        return False
    first = not _met
    _met.add(name)
    if sys.platform != "win32":
        return True
    if first:
        logging.warning(
            "--pack-process is not supported on Windows in this release; "
            "the named packs run inside ComfyUI."
        )
    return False


def warn_unmatched() -> None:
    """Warn of each `--pack-process` name that no custom node pack the loader met has."""
    for name in sorted(set(args.pack_process) - _met):
        logging.warning(
            f"--pack-process {name}: no custom node pack of that name was loaded."
        )


def unload(module_path: str) -> None:
    """End the process of the pack loaded from `module_path`, if it runs in one: ComfyUI has
    refused or flushed the pack, or it failed to load."""
    process = processes.pop(module_path, None)
    if process is not None:
        process.stop()


def end_all() -> None:
    """End every pack process there still is."""
    for module_path in list(processes):
        unload(module_path)
