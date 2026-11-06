"""Does this build still load the custom-node packs a release is judged on?

The stable-release bar names three by hand: "A staff member reproduces a break
in KJNodes, VideoHelperSuite or ComfyUI-Manager" stops a release. Until now
that reproduction was a person's job. These checks do it against a running
server, so a core change that stops a pack importing, or quietly drops a block
of its nodes, fails here instead.

They are deliberately shallow -- loaded, registered, well-formed, answering --
because the depth already exists on the frontend side
(ComfyUI_frontend's custom-node suite exercises widgets, connectivity and
execution). What that suite cannot see is this repo's Python node API: a pack
that fails to import never reaches a browser at all. Two of these three cannot
even be imported outside a running server (both touch
`PromptServer.instance` at import time), which is why this talks to a live
server rather than importing nodes in-process.

Run against a server that already has the pinned packs installed:

    COMFYUI_URL=http://127.0.0.1:8188 \
    COMFYUI_STARTUP_LOG=console_output.log \
        pytest tests/custom_nodes

`.github/workflows/test-custom-nodes.yml` installs the pins, starts the server
and captures the log.
"""
import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((Path(__file__).parent / "manifest.json").read_text(encoding="utf-8"))
PACKS = MANIFEST["packs"]
PACK_IDS = [pack["pack"] for pack in PACKS]

SERVER_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
# Set by CI to the server's captured stdout. Unset locally means the log checks
# have nothing to read and say so, rather than passing on an empty string.
STARTUP_LOG = os.environ.get("COMFYUI_STARTUP_LOG")


def fetch(path: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(f"{SERVER_URL}{path}", timeout=120) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")


@pytest.fixture(scope="session")
def object_info() -> dict:
    status, body = fetch("/object_info")
    assert status == 200, f"/object_info returned {status}: the server did not come up"
    return json.loads(body)


@pytest.fixture(scope="session")
def startup_log() -> str:
    if STARTUP_LOG is None:
        pytest.skip("COMFYUI_STARTUP_LOG is not set, so there is no startup log to read")
    path = Path(STARTUP_LOG)
    assert path.is_file(), f"COMFYUI_STARTUP_LOG points at {path}, which does not exist"
    return path.read_text(encoding="utf-8", errors="replace")


def nodes_of(object_info: dict, pack: str) -> dict[str, dict]:
    """The nodes a pack registered, by the module path object_info reports."""
    module = f"custom_nodes.{pack}"
    return {
        name: entry
        for name, entry in object_info.items()
        if entry.get("python_module") == module
    }


@pytest.mark.parametrize("pack", PACKS, ids=PACK_IDS)
def test_the_pack_under_test_is_the_pinned_commit(pack):
    """A drifted checkout would otherwise report on code nobody reviewed."""
    path = REPO_ROOT / "custom_nodes" / pack["pack"]
    assert path.is_dir(), f"{pack['pack']} is not installed in custom_nodes/"
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert head == pack["pin"], (
        f"{pack['pack']} is checked out at {head}, not the pinned {pack['pin']}"
    )


@pytest.mark.parametrize("pack", PACKS, ids=PACK_IDS)
def test_the_pack_imported_without_failing(pack, startup_log):
    """The server starts anyway when a pack fails to import, so the log is the
    only place that says so."""
    failures = [
        line
        for line in startup_log.splitlines()
        if pack["pack"] in line
        and ("Cannot import" in line or "IMPORT FAILED" in line)
    ]
    assert not failures, f"{pack['pack']} failed to import:\n" + "\n".join(failures)


@pytest.mark.parametrize("pack", PACKS, ids=PACK_IDS)
def test_the_pack_registers_the_nodes_it_is_known_for(pack, object_info):
    registered = nodes_of(object_info, pack["pack"])
    missing = [name for name in pack["expectedNodes"] if name not in registered]
    assert not missing, (
        f"{pack['pack']} no longer registers {missing} "
        f"(it registered {len(registered)} nodes)"
    )


@pytest.mark.parametrize("pack", PACKS, ids=PACK_IDS)
def test_the_pack_does_not_silently_register_fewer_nodes(pack, object_info):
    """A failed sub-import drops a block of nodes while the server still
    starts, the pack still appears, and nothing else complains."""
    count = len(nodes_of(object_info, pack["pack"]))
    assert count >= pack["minNodeCount"], (
        f"{pack['pack']} registered {count} nodes, below its floor of "
        f"{pack['minNodeCount']} (it registered {pack['nodeCount']} when pinned)"
    )


@pytest.mark.parametrize("pack", PACKS, ids=PACK_IDS)
def test_every_node_the_pack_registers_is_well_formed(pack, object_info):
    """A node whose schema does not survive this build's node API would reach
    the frontend as an unusable node rather than as an error."""
    malformed = []
    for name, entry in nodes_of(object_info, pack["pack"]).items():
        if not isinstance(entry.get("input"), dict):
            malformed.append(f"{name}: input is not an object")
            continue
        if not isinstance(entry.get("input_order"), dict):
            malformed.append(f"{name}: input_order is not an object")
            continue
        if not isinstance(entry.get("output"), list):
            malformed.append(f"{name}: output is not a list")
            continue
        # ComfyUI builds one order list per INPUT_TYPES() section, and the
        # frontend reads them per section too: an optional input listed under
        # input_order.required is ignored, not honoured. Every section has to
        # be present on both sides, as a list, or the frontend is reading an
        # order the node did not declare.
        order = entry["input_order"]
        for section in sorted(set(entry["input"]) | set(order)):
            declared = entry["input"].get(section)
            ordered = order.get(section)
            if not isinstance(declared, dict):
                malformed.append(
                    f"{name}: input_order.{section} exists but input.{section} "
                    "does not"
                )
                continue
            if not isinstance(ordered, list):
                malformed.append(
                    f"{name}: input.{section} exists but input_order.{section} "
                    f"is {type(ordered).__name__}, not a list"
                )
                continue
            for input_name in ordered:
                if input_name not in declared:
                    malformed.append(
                        f"{name}: input_order.{section} lists {input_name}, "
                        f"which is not one of the node's {section} inputs"
                    )
    assert not malformed, f"{pack['pack']} registered malformed nodes:\n" + "\n".join(
        malformed
    )


@pytest.mark.parametrize("pack", PACKS, ids=PACK_IDS)
def test_every_route_the_pack_adds_answers(pack, object_info):
    """ComfyUI-Manager registers no nodes at all, so its own routes are the
    only evidence it loaded and is serving."""
    if not pack["routes"]:
        pytest.skip(f"{pack['pack']} adds no routes of its own")
    for route in pack["routes"]:
        status, body = fetch(route)
        assert status == 200, f"{pack['pack']} route {route} returned {status}"
        assert body.strip(), f"{pack['pack']} route {route} returned an empty body"
