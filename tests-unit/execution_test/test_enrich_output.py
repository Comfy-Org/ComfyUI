import copy
import os
import tempfile
from collections import namedtuple
from unittest.mock import patch

import folder_paths
import pytest

from app.assets.manager import NoAssets
from comfy_execution.asset_enrichment import (
    emit_cached_output,
    register_cached_outputs,
    register_executed_outputs,
)
from test_inmemory_assets import AssetCall, InMemoryAssets

_CacheEntry = namedtuple("_CacheEntry", ["ui", "outputs"])

_BASE = os.path.join(tempfile.gettempdir(), "asset-enrichment-test-base")


class _ArgsStub:
    enable_assets = False
    enable_asset_hashing = False


class _Server:
    last_node_id: str | None = None
    sockets_metadata: dict[str, dict[str, object]] = {}

    def __init__(self, client_id: str | None = None) -> None:
        self.client_id = client_id
        self.sent: list[tuple] = []

    def send_sync(self, event, data, sid=None):
        self.sent.append((event, data, sid))

    def queue_updated(self) -> None:
        pass


@pytest.fixture
def output_path_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(folder_paths, "get_directory_by_type", lambda _type: _BASE)
    monkeypatch.setattr(os.path, "isfile", lambda _path: True)


def _output(filename: str, *, subfolder: str = "", type_: str = "output") -> dict:
    return {"images": [{"filename": filename, "subfolder": subfolder, "type": type_}]}


def _wrapper(filename: str, node_id: str = "1") -> dict:
    return {
        "meta": {
            "node_id": node_id,
            "display_node": node_id,
            "parent_node": None,
            "real_node_id": node_id,
        },
        "output": _output(filename),
    }


def _find_ids(value) -> list:
    found: list = []
    if isinstance(value, dict):
        for key, sub in value.items():
            if key == "id":
                found.append(sub)
            found.extend(_find_ids(sub))
    elif isinstance(value, list):
        for item in value:
            found.extend(_find_ids(item))
    return found


# REQUIRED test names (invoked verbatim downstream). Do not rename.
def test_executed_new_path_gets_fresh_id(output_path_environment) -> None:
    manager = InMemoryAssets()
    output_ui = _output("new.png")

    enriched = register_executed_outputs(output_ui, "job-1", manager)

    assert enriched["images"][0]["id"] == "asset-1"
    assert "id" not in output_ui["images"][0]
    assert manager.calls == [
        AssetCall(
            "register_executed_output", (os.path.join(_BASE, "new.png"), "job-1")
        )
    ]
    deliveries = manager.deliveries_by_path[os.path.join(_BASE, "new.png")]
    assert [(delivery.asset.id, delivery.asset.job_id) for delivery in deliveries] == [
        ("asset-1", "job-1")
    ]


def test_executed_multiple_entries_register_each_output(output_path_environment) -> None:
    manager = InMemoryAssets()
    filenames = ["first.png", "second.png", "third.png"]
    output_ui = {
        "images": [
            {"filename": filename, "subfolder": "batch", "type": "output"}
            for filename in filenames
        ]
    }

    register_executed_outputs(output_ui, "job-list", manager)

    expected_paths = [os.path.join(_BASE, "batch", filename) for filename in filenames]
    assert manager.calls == [
        AssetCall("register_executed_output", (path, "job-list"))
        for path in expected_paths
    ]
    assert [
        (delivery.abs_path, delivery.asset.job_id, delivery.superseded)
        for path in expected_paths
        for delivery in manager.deliveries_by_path[path]
    ] == [(path, "job-list", False) for path in expected_paths]


def test_executed_mixed_types_use_their_own_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = InMemoryAssets()
    output_root = f"{_BASE}-output"
    temp_root = f"{_BASE}-temp"
    roots = {"output": output_root, "temp": temp_root}
    monkeypatch.setattr(
        folder_paths,
        "get_directory_by_type",
        lambda output_type: roots[output_type],
    )
    monkeypatch.setattr(os.path, "isfile", lambda _path: True)
    output_ui = {
        "images": [
            {"filename": "kept.png", "subfolder": "saved", "type": "output"},
            {"filename": "preview.png", "subfolder": "staged", "type": "temp"},
        ]
    }

    register_executed_outputs(output_ui, "job-mixed", manager)

    assert manager.calls == [
        AssetCall(
            "register_executed_output",
            (os.path.join(output_root, "saved", "kept.png"), "job-mixed"),
        ),
        AssetCall(
            "register_executed_output",
            (os.path.join(temp_root, "staged", "preview.png"), "job-mixed"),
        ),
    ]


def test_executed_repeated_calls_keep_distinct_path_deliveries(
    output_path_environment,
) -> None:
    manager = InMemoryAssets()
    filenames = ["iteration-1.png", "iteration-2.png", "iteration-3.png"]

    for filename in filenames:
        register_executed_outputs(_output(filename), "loop-job", manager)

    expected_paths = [os.path.join(_BASE, filename) for filename in filenames]
    assert manager.calls == [
        AssetCall("register_executed_output", (path, "loop-job"))
        for path in expected_paths
    ]
    assert [
        (delivery.abs_path, delivery.asset.job_id, delivery.superseded)
        for path in expected_paths
        for delivery in manager.deliveries_by_path[path]
    ] == [(path, "loop-job", False) for path in expected_paths]


def test_executed_empty_entry_list_is_noop(output_path_environment) -> None:
    manager = InMemoryAssets()

    enriched = register_executed_outputs({"images": []}, "job", manager)

    assert enriched == {"images": []}
    assert manager.calls == []
    assert manager.deliveries_by_path == {}


def test_executed_over_existing_path_gets_new_id_and_marks_old_missing(output_path_environment) -> None:
    manager = InMemoryAssets()

    first = register_executed_outputs(_output("same.png"), "job-1", manager)
    old_id = first["images"][0]["id"]

    second = register_executed_outputs(_output("same.png"), "job-2", manager)
    new_id = second["images"][0]["id"]

    deliveries = manager.deliveries_by_path[os.path.join(_BASE, "same.png")]
    assert new_id != old_id
    assert deliveries[0].superseded is True
    assert deliveries[-1].asset.id == new_id


def test_cached_replay_creates_delivery_with_current_job_id(output_path_environment) -> None:
    manager = InMemoryAssets()

    register_executed_outputs(_output("replay.png"), "seed-job", manager)
    enriched = register_cached_outputs(_wrapper("replay.png"), "replay-job", manager)

    replay_id = enriched["output"]["images"][0]["id"]
    deliveries = manager.deliveries_by_path[os.path.join(_BASE, "replay.png")]
    assert (replay_id, "replay-job") == (deliveries[-1].asset.id, deliveries[-1].asset.job_id)
    assert replay_id != "asset-1"


def test_cached_registration_happens_without_client(output_path_environment) -> None:
    manager = InMemoryAssets()
    server = _Server(client_id=None)
    ui_outputs: dict = {}

    register_executed_outputs(_output("noclient.png"), "seed-job", manager)
    emit_cached_output(
        server,
        "node-1",
        "node-1",
        _CacheEntry(ui=_wrapper("noclient.png"), outputs=[]),
        "job-x",
        ui_outputs,
        manager,
    )

    deliveries = manager.deliveries_by_path[os.path.join(_BASE, "noclient.png")]
    assert any(delivery.asset.job_id == "job-x" for delivery in deliveries)
    assert "node-1" in ui_outputs
    assert ui_outputs["node-1"]["output"]["images"][0]["id"] is not None
    assert server.sent == []


def test_cache_entry_contains_no_asset_ids(output_path_environment) -> None:
    manager = InMemoryAssets()
    output_ui = _output("keep.png")

    enriched = register_executed_outputs(output_ui, "job", manager)

    cache_entry = _CacheEntry(
        ui={"meta": {"node_id": "1"}, "output": output_ui}, outputs=[]
    )
    assert _find_ids(cache_entry.ui) == []
    assert _find_ids(enriched) == ["asset-1"]


def test_cached_ui_object_unmodified_after_emission(output_path_environment) -> None:
    manager = InMemoryAssets()
    server = _Server(client_id="client-1")
    cached = _CacheEntry(ui=_wrapper("immut.png"), outputs=[])
    snapshot = copy.deepcopy(cached.ui)

    register_executed_outputs(_output("immut.png"), "seed-job", manager)
    emit_cached_output(server, "1", "1", cached, "prompt-1", {}, manager)

    assert cached.ui == snapshot


def test_double_emission_yields_single_delivery(output_path_environment) -> None:
    manager = InMemoryAssets()
    server = _Server(client_id="client-1")
    ui_outputs: dict = {}
    cached = _CacheEntry(ui=_wrapper("dbl.png"), outputs=[])

    register_executed_outputs(_output("dbl.png"), "seed-job", manager)
    emit_cached_output(server, "1", "1", cached, "prompt-1", ui_outputs, manager)
    emit_cached_output(server, "1", "1", cached, "prompt-1", ui_outputs, manager)

    deliveries = manager.deliveries_by_path[os.path.join(_BASE, "dbl.png")]
    assert len([delivery for delivery in deliveries if delivery.asset.job_id == "prompt-1"]) == 1


def test_executed_disabled_returns_unenriched_copy(output_path_environment) -> None:
    manager = NoAssets(_ArgsStub())
    output_ui = _output("a.png")

    with patch.object(
        manager,
        "register_executed_output",
        wraps=manager.register_executed_output,
    ) as register_executed_output:
        enriched = register_executed_outputs(output_ui, "job", manager)

    assert enriched is not output_ui
    assert "id" not in enriched["images"][0]
    register_executed_output.assert_not_called()


def test_executed_missing_file_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = InMemoryAssets()
    monkeypatch.setattr(folder_paths, "get_directory_by_type", lambda _type: _BASE)
    monkeypatch.setattr(os.path, "isfile", lambda _path: False)

    enriched = register_executed_outputs(_output("gone.png"), "job", manager)

    assert "id" not in enriched["images"][0]
    assert manager.calls == []


def test_executed_path_escape_is_skipped(output_path_environment) -> None:
    manager = InMemoryAssets()
    output_ui = {"images": [{"filename": "passwd", "subfolder": "../../etc", "type": "output"}]}

    enriched = register_executed_outputs(output_ui, "job", manager)

    assert "id" not in enriched["images"][0]
    assert manager.calls == []


def test_executed_non_list_value_passes_through(output_path_environment) -> None:
    manager = InMemoryAssets()

    enriched = register_executed_outputs({"text": "hello"}, "job", manager)

    assert enriched["text"] == "hello"


def test_executed_registration_failure_never_raises(
    output_path_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = InMemoryAssets()

    def boom(_abs_path: str, job_id: str | None) -> None:
        raise RuntimeError("registration blew up")

    monkeypatch.setattr(manager, "register_executed_output", boom)

    enriched = register_executed_outputs(_output("boom.png"), "job", manager)

    assert "id" not in enriched["images"][0]


def test_cached_strips_legacy_ids_before_replay(output_path_environment) -> None:
    manager = InMemoryAssets()
    wrapper = _wrapper("legacy.png")
    wrapper["output"]["images"][0]["id"] = "stale-id"

    register_executed_outputs(_output("legacy.png"), "seed-job", manager)
    enriched = register_cached_outputs(wrapper, "replay-job", manager)

    assert enriched["output"]["images"][0]["id"] != "stale-id"
    assert wrapper["output"]["images"][0]["id"] == "stale-id"


def test_cached_none_wrapper_returns_none(output_path_environment) -> None:
    manager = InMemoryAssets()

    result = register_cached_outputs(None, "job", manager)

    assert result is None
    assert manager.calls == []


def test_cached_missing_live_content_is_nonevent(output_path_environment) -> None:
    manager = InMemoryAssets()

    enriched = register_cached_outputs(_wrapper("orphan.png"), "job", manager)

    assert "id" not in enriched["output"]["images"][0]
    assert manager.deliveries_by_path == {}


def test_emit_cached_sends_enriched_output_to_client(output_path_environment) -> None:
    manager = InMemoryAssets()
    server = _Server(client_id="client-1")
    ui_outputs: dict = {}

    register_executed_outputs(_output("send.png"), "seed-job", manager)
    emit_cached_output(
        server,
        "1",
        "1",
        _CacheEntry(ui=_wrapper("send.png"), outputs=[]),
        "prompt-1",
        ui_outputs,
        manager,
    )

    assert len(server.sent) == 1
    event, payload, client_id = server.sent[0]
    assert event == "executed"
    assert client_id == "client-1"
    assert payload["output"]["images"][0]["id"] is not None
