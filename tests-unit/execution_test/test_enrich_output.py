import copy
import os
import sys
import tempfile
import types
from collections import namedtuple
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

_CacheEntry = namedtuple("_CacheEntry", ["ui", "outputs"])

_BASE = os.path.join(tempfile.gettempdir(), "asset-enrichment-test-base")


class _FakeAssetDB:

    def __init__(self) -> None:
        self.live_id_by_path: dict[str, str] = {}
        self.missing: list[str] = []
        self.deliveries: list[tuple[str, str, str | None]] = []
        self._counter = 0

    def _new_id(self) -> str:
        self._counter += 1
        return f"asset-{self._counter}"

    def register_executed(self, abs_path: str, job_id: str | None = None):
        old = self.live_id_by_path.get(abs_path)
        if old is not None:
            self.missing.append(old)
        new_id = self._new_id()
        self.live_id_by_path[abs_path] = new_id
        self.deliveries.append((new_id, abs_path, job_id))
        return types.SimpleNamespace(
            id=new_id,
            content_id=f"content-{new_id}",
            job_id=job_id,
            name=os.path.basename(abs_path),
        )

    def register_cached(self, abs_path: str, job_id: str | None = None):
        if abs_path not in self.live_id_by_path:
            return None
        new_id = self._new_id()
        self.deliveries.append((new_id, abs_path, job_id))
        return types.SimpleNamespace(
            id=new_id,
            content_id=f"content-{self.live_id_by_path[abs_path]}",
            job_id=job_id,
            name=os.path.basename(abs_path),
        )


class _Server:
    def __init__(self, client_id: str | None = None) -> None:
        self.client_id = client_id
        self.sent: list[tuple] = []

    def send_sync(self, event, payload, client_id):
        self.sent.append((event, payload, client_id))


@contextmanager
def _patched(
    fake: _FakeAssetDB,
    *,
    enable_assets: bool = True,
    directory: str | None = _BASE,
    file_exists: bool = True,
    executed_side_effect=None,
    cached_side_effect=None,
):
    reg_exec = MagicMock(side_effect=executed_side_effect or fake.register_executed)
    reg_cached = MagicMock(side_effect=cached_side_effect or fake.register_cached)
    modules = {
        "comfy.cli_args": MagicMock(
            args=types.SimpleNamespace(enable_assets=enable_assets)
        ),
        "folder_paths": MagicMock(
            get_directory_by_type=MagicMock(return_value=directory)
        ),
        "app.assets.services.ingest": MagicMock(
            register_executed_output=reg_exec,
            register_cached_output=reg_cached,
        ),
    }
    with patch.dict(sys.modules, modules), patch(
        "os.path.isfile", return_value=file_exists
    ):
        import comfy_execution.asset_enrichment as module

        yield module, reg_exec, reg_cached


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
def test_executed_new_path_gets_fresh_id() -> None:
    fake = _FakeAssetDB()
    output_ui = _output("new.png")

    with _patched(fake) as (module, reg_exec, _):
        enriched = module.register_executed_outputs(output_ui, "job-1")

    assert enriched["images"][0]["id"] == "asset-1"
    assert "id" not in output_ui["images"][0]
    reg_exec.assert_called_once()
    _, kwargs = reg_exec.call_args
    assert kwargs.get("job_id") == "job-1"
    assert fake.deliveries == [("asset-1", os.path.join(_BASE, "new.png"), "job-1")]


def test_executed_over_existing_path_gets_new_id_and_marks_old_missing() -> None:
    fake = _FakeAssetDB()

    first_output = _output("same.png")
    with _patched(fake) as (module, _, _c):
        first = module.register_executed_outputs(first_output, "job-1")
    old_id = first["images"][0]["id"]

    second_output = _output("same.png")
    with _patched(fake) as (module, _, _c):
        second = module.register_executed_outputs(second_output, "job-2")
    new_id = second["images"][0]["id"]

    assert new_id != old_id
    assert old_id in fake.missing
    assert fake.live_id_by_path[os.path.join(_BASE, "same.png")] == new_id


def test_cached_replay_creates_delivery_with_current_job_id() -> None:
    fake = _FakeAssetDB()

    with _patched(fake) as (module, _, _c):
        module.register_executed_outputs(_output("replay.png"), "seed-job")
        enriched = module.register_cached_outputs(_wrapper("replay.png"), "replay-job")

    replay_id = enriched["output"]["images"][0]["id"]
    assert (replay_id, os.path.join(_BASE, "replay.png"), "replay-job") in fake.deliveries
    assert replay_id != "asset-1"


def test_cached_registration_happens_without_client() -> None:
    fake = _FakeAssetDB()
    server = _Server(client_id=None)
    ui_outputs: dict = {}

    with _patched(fake) as (module, _, _c):
        module.register_executed_outputs(_output("noclient.png"), "seed-job")
        module.emit_cached_output(
            server, "node-1", "node-1", _CacheEntry(ui=_wrapper("noclient.png"), outputs=[]),
            "job-x", ui_outputs,
        )

    assert any(job == "job-x" for (_id, _path, job) in fake.deliveries)
    assert "node-1" in ui_outputs
    assert ui_outputs["node-1"]["output"]["images"][0]["id"] is not None
    assert server.sent == []


def test_cache_entry_contains_no_asset_ids() -> None:
    fake = _FakeAssetDB()
    output_ui = _output("keep.png")

    with _patched(fake) as (module, _, _c):
        enriched = module.register_executed_outputs(output_ui, "job")

    cache_entry = _CacheEntry(
        ui={"meta": {"node_id": "1"}, "output": output_ui}, outputs=[]
    )
    assert _find_ids(cache_entry.ui) == []
    assert _find_ids(enriched) == ["asset-1"]


def test_cached_ui_object_unmodified_after_emission() -> None:
    fake = _FakeAssetDB()
    server = _Server(client_id="client-1")
    cached = _CacheEntry(ui=_wrapper("immut.png"), outputs=[])
    snapshot = copy.deepcopy(cached.ui)

    with _patched(fake) as (module, _, _c):
        module.register_executed_outputs(_output("immut.png"), "seed-job")
        module.emit_cached_output(
            server, "1", "1", cached, "prompt-1", {}
        )

    assert cached.ui == snapshot


def test_double_emission_yields_single_delivery() -> None:
    fake = _FakeAssetDB()
    server = _Server(client_id="client-1")
    ui_outputs: dict = {}
    cached = _CacheEntry(ui=_wrapper("dbl.png"), outputs=[])

    with _patched(fake) as (module, _, _c):
        module.register_executed_outputs(_output("dbl.png"), "seed-job")
        module.emit_cached_output(server, "1", "1", cached, "prompt-1", ui_outputs)
        module.emit_cached_output(server, "1", "1", cached, "prompt-1", ui_outputs)

    cached_deliveries = [d for d in fake.deliveries if d[2] == "prompt-1"]
    assert len(cached_deliveries) == 1


def test_executed_disabled_returns_unenriched_copy() -> None:
    fake = _FakeAssetDB()
    output_ui = _output("a.png")

    with _patched(fake, enable_assets=False) as (module, reg_exec, _):
        enriched = module.register_executed_outputs(output_ui, "job")

    assert enriched is not output_ui
    assert "id" not in enriched["images"][0]
    reg_exec.assert_not_called()


def test_executed_missing_file_is_skipped() -> None:
    fake = _FakeAssetDB()

    with _patched(fake, file_exists=False) as (module, reg_exec, _):
        enriched = module.register_executed_outputs(_output("gone.png"), "job")

    assert "id" not in enriched["images"][0]
    reg_exec.assert_not_called()


def test_executed_path_escape_is_skipped() -> None:
    fake = _FakeAssetDB()
    output_ui = {"images": [{"filename": "passwd", "subfolder": "../../etc", "type": "output"}]}

    with _patched(fake) as (module, reg_exec, _):
        enriched = module.register_executed_outputs(output_ui, "job")

    assert "id" not in enriched["images"][0]
    reg_exec.assert_not_called()


def test_executed_non_list_value_passes_through() -> None:
    fake = _FakeAssetDB()

    with _patched(fake) as (module, _, _c):
        enriched = module.register_executed_outputs({"text": "hello"}, "job")

    assert enriched["text"] == "hello"


def test_executed_registration_failure_never_raises() -> None:
    fake = _FakeAssetDB()

    def boom(_abs_path, job_id=None):
        raise RuntimeError("registration blew up")

    with _patched(fake, executed_side_effect=boom) as (module, _, _c):
        enriched = module.register_executed_outputs(_output("boom.png"), "job")

    assert "id" not in enriched["images"][0]


def test_cached_strips_legacy_ids_before_replay() -> None:
    fake = _FakeAssetDB()
    wrapper = _wrapper("legacy.png")
    wrapper["output"]["images"][0]["id"] = "stale-id"

    with _patched(fake) as (module, _, _c):
        module.register_executed_outputs(_output("legacy.png"), "seed-job")
        enriched = module.register_cached_outputs(wrapper, "replay-job")

    assert enriched["output"]["images"][0]["id"] != "stale-id"
    assert wrapper["output"]["images"][0]["id"] == "stale-id"


def test_cached_none_wrapper_returns_none() -> None:
    fake = _FakeAssetDB()

    with _patched(fake) as (module, _, reg_cached):
        result = module.register_cached_outputs(None, "job")

    assert result is None
    reg_cached.assert_not_called()


def test_cached_missing_live_content_is_nonevent() -> None:
    fake = _FakeAssetDB()

    with _patched(fake) as (module, _, _c):
        enriched = module.register_cached_outputs(_wrapper("orphan.png"), "job")

    assert "id" not in enriched["output"]["images"][0]
    assert fake.deliveries == []


def test_emit_cached_sends_enriched_output_to_client() -> None:
    fake = _FakeAssetDB()
    server = _Server(client_id="client-1")
    ui_outputs: dict = {}

    with _patched(fake) as (module, _, _c):
        module.register_executed_outputs(_output("send.png"), "seed-job")
        module.emit_cached_output(
            server, "1", "1", _CacheEntry(ui=_wrapper("send.png"), outputs=[]),
            "prompt-1", ui_outputs,
        )

    assert len(server.sent) == 1
    event, payload, client_id = server.sent[0]
    assert event == "executed"
    assert client_id == "client-1"
    assert payload["output"]["images"][0]["id"] is not None
