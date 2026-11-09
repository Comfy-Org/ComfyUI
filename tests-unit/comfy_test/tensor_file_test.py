import os
import gc
import sys
import weakref

import pytest
import safetensors.torch
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_state  # noqa: E402
import comfy.model_patcher  # noqa: E402
from comfy.memory_management import read_tensor_file_slice_into, write_tensor_file_slice  # noqa: E402
from comfy.utils import create_safetensors  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.name != "nt" and sys.platform != "linux",
    reason="anonymous files require Linux or Windows",
)


def test_sparse_safetensors_and_write_validation():
    geometry = {"key": torch.empty((3, 5), device="meta"), "tail": torch.empty((7,), dtype=torch.uint8, device="meta")}
    state, file = create_safetensors(geometry)

    try:
        expected = torch.zeros(3, 5)
        restored = torch.empty_like(expected)
        assert read_tensor_file_slice_into(state["key"], restored)
        assert torch.equal(restored, expected)
        file.seek(0)
        saved = safetensors.torch.load(file.read())
        assert torch.equal(saved["key"], expected)
        assert torch.equal(saved["tail"], torch.zeros(7, dtype=torch.uint8))

        with pytest.raises(ValueError):
            write_tensor_file_slice(state["key"], expected[1])
        with pytest.raises(ValueError):
            write_tensor_file_slice(expected, expected)
        with pytest.raises(ValueError):
            write_tensor_file_slice(state["key"], torch.empty_like(expected, device="meta"))
    finally:
        file.close()

    large, file = create_safetensors({"kv": torch.empty((1 << 30,), dtype=torch.uint8, device="meta")})
    try:
        stat = os.fstat(file.fileno())
        assert stat.st_size > 1 << 30
        if hasattr(stat, "st_blocks"):
            assert stat.st_blocks * 512 < 65536
        file.seek(-32, os.SEEK_END)
        assert file.read() == bytes(32)
    finally:
        file.close()


def test_anonymous_safetensors():
    state, file = create_safetensors({"kv": torch.empty((3, 5), dtype=torch.int32, device="meta")})
    info = state["kv"].untyped_storage()._comfy_tensor_file_slice
    try:
        result = torch.empty(3, 5, dtype=torch.int32)
        assert read_tensor_file_slice_into(state["kv"], result)
        assert torch.equal(result, torch.zeros_like(result))
        expected = torch.arange(15, dtype=torch.int32).reshape(3, 5)
        file.seek(info.offset)
        file.write(expected.numpy().tobytes())
        assert read_tensor_file_slice_into(state["kv"], result)
        assert torch.equal(result, expected)
        file.seek(0)
        assert torch.equal(safetensors.torch.load(file.read())["kv"], expected)
    finally:
        file.close()


class _KV(torch.nn.Module):
    def __init__(self, rows=4):
        super().__init__()
        self.register_buffer("k", torch.empty(rows, 8))
        self.register_buffer("v", torch.empty(rows, 8, dtype=torch.float16))


def test_model_state_persists_values_and_metadata(tmp_path):
    cpu = torch.device("cpu")
    path = str(tmp_path / "state.safetensors")
    state = comfy.model_state.ModelState.open(path)
    assert state.tensors == {} and state.metadata == {}
    model = state.load(_KV(), cpu)
    assert not model.k.count_nonzero() and not model.v.count_nonzero()
    model.k.fill_(3.0)
    model.v.fill_(5.0)
    state.metadata["position"] = 7
    state.flush()
    assert state.patcher.model is model
    state.save(path)

    saved = safetensors.torch.load_file(path)
    assert torch.equal(saved["k"], torch.full((4, 8), 3.0)) and torch.equal(saved["v"], torch.full((4, 8), 5.0, dtype=torch.float16))

    reopened = comfy.model_state.ModelState.open(path, create="never")
    assert reopened.metadata == {"position": 7}

    # The next turn continues on a copy; the opened state only reaches the file when saved.
    turn = reopened.copy()
    model = turn.load(_KV(), cpu)
    assert torch.equal(model.k, saved["k"]) and torch.equal(model.v, saved["v"])
    model.k[0, 0] = 9.0
    turn.metadata["position"] = 8
    turn.flush()
    reopened.save(path)
    assert safetensors.torch.load_file(path)["k"][0, 0] == 3.0
    turn.save(path)
    assert safetensors.torch.load_file(path)["k"][0, 0] == 9.0
    assert comfy.model_state.ModelState.open(path).metadata == {"position": 8}

    with pytest.raises(FileNotFoundError):
        comfy.model_state.ModelState.open(str(tmp_path / "missing.safetensors"), create="never")
    with pytest.raises(RuntimeError):
        comfy.model_state.ModelState.open(path).load(_KV(rows=1), cpu)

    recreated = comfy.model_state.ModelState.open(path, create="always")
    assert recreated.metadata == {} and recreated.tensors == {}
    assert not recreated.load(_KV(), cpu).k.count_nonzero()
    assert safetensors.torch.load_file(path)["k"][0, 0] == 9.0


def test_model_state_rejects_plain_safetensors(tmp_path):
    path = str(tmp_path / "plain.safetensors")
    safetensors.torch.save_file({"k": torch.zeros(4, 8)}, path)
    with pytest.raises(ValueError):
        comfy.model_state.ModelState.open(path)


def test_model_state_copies_are_independent(tmp_path):
    cpu = torch.device("cpu")
    first = comfy.model_state.ModelState()
    model = first.load(_KV(), cpu)
    model.k.fill_(2.0)
    first.metadata["position"] = 3
    first.flush()

    second = first.copy()
    assert second.metadata == {"position": 3} and second.metadata is not first.metadata
    model = second.load(_KV(), cpu)
    model.k[0, 0] = 1.0
    second.metadata["position"] = 4
    second.flush()

    third = second.copy()
    assert third.metadata == {"position": 4}
    assert torch.equal(third.load(_KV(), cpu).k, model.k) and model.k[0, 0] == 1.0
    assert torch.equal(first.load(_KV(), cpu).k, torch.full((4, 8), 2.0)) and first.metadata == {"position": 3}

    # Releasing a loaded state drops values its metadata may already describe, so it cannot reach another node.
    first.patcher.model.k.fill_(5.0)
    first.release()
    with pytest.raises(RuntimeError):
        first.copy()
    with pytest.raises(RuntimeError):
        first.save(str(tmp_path / "released.safetensors"))
    # A written back state stays usable.
    third.flush()
    expected = third.tensors["k"].clone()
    third.release()
    assert torch.equal(third.copy().load(_KV(), cpu).k, expected)


def test_shared_model_survives_nodes_and_snapshot_switches():
    cpu = torch.device("cpu")
    owner = comfy.model_patcher.ModelPatcher(_KV(), cpu, cpu)
    cloned_owner = owner.clone()
    first = comfy.model_state.ModelState()
    first.owner = owner
    model = first.load(_KV(), cpu)
    patcher = first.patcher
    model.k.fill_(2)
    second = first.copy()
    assert second.load(_KV(), cpu) is model
    assert second.patcher is patcher
    model.k.fill_(3)
    other = comfy.model_state.ModelState()
    other.owner = cloned_owner
    assert other.load(_KV(), cpu) is model
    assert other.patcher is patcher
    assert not model.k.count_nonzero()
    assert first.load(_KV(), cpu) is model
    assert torch.equal(model.k, torch.full((4, 8), 2.0))
    assert second.load(_KV(), cpu) is model
    assert torch.equal(model.k, torch.full((4, 8), 3.0))
    late_clone = owner.clone()
    old_patcher = weakref.ref(patcher)
    for state in (first, second, other):
        state.release()
    replacement = comfy.model_state.ModelState()
    replacement.owner = late_clone
    assert replacement.load(_KV(rows=1), cpu).k.shape == (1, 8)
    assert late_clone.get_attachment("model_state_model") is owner.get_attachment("model_state_model")
    del patcher, model
    gc.collect()
    assert old_patcher() is None


def test_deferred_save_uses_resident_snapshot_and_flushes_on_eviction(tmp_path):
    cpu = torch.device("cpu")
    path = str(tmp_path / "deferred.safetensors")
    state = comfy.model_state.ModelState()
    model = state.load(_KV(), cpu)
    model.k.fill_(7)
    state.metadata["position"] = 9
    state.save(path, writethrough=False)
    assert not os.path.exists(path)
    assert state.tensors is None
    reopened = comfy.model_state.ModelState.open(path, "never")
    assert reopened._values.version == state._values.version and reopened.patcher is state.patcher
    assert reopened.tensors is not state.tensors and state.tensors["k"].eq(7).all()
    state.patcher.unpin_all_weights()
    assert safetensors.torch.load_file(path)["k"].eq(7).all()
    assert not state.patcher.get_attachment("model_state_saves")
    assert comfy.model_state.ModelState.open(path).metadata == {"position": 9}


def test_deferred_save_keeps_latest_snapshot_and_external_edits_win(tmp_path):
    cpu = torch.device("cpu")
    path = str(tmp_path / "latest.safetensors")
    first = comfy.model_state.ModelState()
    model = first.load(_KV(), cpu)
    model.k.fill_(1)
    first.save(path, writethrough=False)
    second = first.copy()
    model = second.load(_KV(), cpu)
    model.k.fill_(2)
    second.save(path, writethrough=False)
    assert not os.path.exists(path)
    second.patcher.unpin_all_weights()
    assert safetensors.torch.load_file(path)["k"].eq(2).all()
    safetensors.torch.save_file({"k": torch.ones(4, 8)}, path, {comfy.model_state.METADATA_KEY: '{"external": true}'})
    assert comfy.model_state.ModelState.open(path).metadata == {"external": True}


def test_deferred_save_survives_state_collection(tmp_path):
    cpu = torch.device("cpu")
    path = str(tmp_path / "collected.safetensors")
    state = comfy.model_state.ModelState()
    model = state.load(_KV(), cpu)
    model.k.fill_(11)
    state.save(path, writethrough=False)
    assert not os.path.exists(path)
    del state, model
    gc.collect()
    assert safetensors.torch.load_file(path)["k"].eq(11).all()


@pytest.mark.parametrize("resident_target", [False, True])
def test_live_state_preserves_values_when_changing_owner(resident_target):
    cpu = torch.device("cpu")
    owner = comfy.model_patcher.ModelPatcher(_KV(), cpu, cpu)
    target_owner = comfy.model_patcher.ModelPatcher(_KV(), cpu, cpu)
    state = comfy.model_state.ModelState()
    state.owner = owner
    source = state.load(_KV(), cpu)
    source.k.fill_(2)
    state.metadata["position"] = 3
    if resident_target:
        target_state = comfy.model_state.ModelState()
        target_state.owner = target_owner
        target = target_state.load(_KV(), cpu)
        target.k.fill_(9)
    state.owner = target_owner
    moved = state.load(_KV(), cpu)
    assert moved is not source and moved.k.eq(2).all()
    assert state.metadata == {"position": 3}
    moved.k.fill_(4)
    assert source.k.eq(2).all()
    if resident_target:
        assert moved is target
        assert target_state.load(_KV(), cpu).k.eq(9).all()


def test_incompatible_live_geometry_preserves_state():
    cpu = torch.device("cpu")
    state = comfy.model_state.ModelState()
    model = state.load(_KV(), cpu)
    model.k.fill_(2)
    model.v.fill_(3)
    state.metadata["position"] = 4
    with pytest.raises(RuntimeError):
        state.load(_KV(rows=1), cpu)
    assert state.tensors["k"].eq(2).all() and state.tensors["v"].eq(3).all()
    restored = state.load(_KV(), cpu)
    assert restored.k.eq(2).all() and restored.v.eq(3).all()
    assert state.metadata == {"position": 4}


def test_independent_model_override_does_not_inherit_state_model():
    cpu = torch.device("cpu")
    owner = comfy.model_patcher.ModelPatcher(_KV(), cpu, cpu)
    state = comfy.model_state.ModelState()
    state.owner = owner
    source = state.load(_KV(), cpu)
    source.k.fill_(2)
    independent = comfy.model_patcher.ModelPatcher(_KV(), cpu, cpu)
    clone = owner.clone(model_override=independent.get_clone_model_override())
    other = comfy.model_state.ModelState()
    other.owner = clone
    model = other.load(_KV(), cpu)
    assert other.patcher is not state.patcher and model is not source
    assert not model.k.count_nonzero()
    model.k.fill_(3)
    assert source.k.eq(2).all()
    assert state.load(_KV(), cpu) is source


@pytest.mark.parametrize("trigger", ["eviction", "collection"])
def test_external_replacement_cancels_pending_save(tmp_path, trigger):
    path = str(tmp_path / "replaced.safetensors")
    state = comfy.model_state.ModelState()
    model = state.load(_KV(), torch.device("cpu"))
    model.k.fill_(7)
    state.metadata["position"] = 1
    state.save(path)
    state.save(path, writethrough=False)
    stat = os.stat(path)
    replacement = str(tmp_path / "external.safetensors")
    safetensors.torch.save_file(
        {"k": torch.full_like(model.k, 99), "v": torch.full_like(model.v, -99)},
        replacement, {comfy.model_state.METADATA_KEY: '{"position": 2}'})
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    os.replace(replacement, path)
    assert os.stat(path).st_mtime_ns == stat.st_mtime_ns
    assert os.stat(path).st_size == stat.st_size
    external = (tmp_path / "replaced.safetensors").read_bytes()
    if trigger == "eviction":
        state.patcher.unpin_all_weights()
        assert not state.patcher.get_attachment("model_state_saves")
    else:
        del state, model
        gc.collect()
    assert (tmp_path / "replaced.safetensors").read_bytes() == external
    opened = comfy.model_state.ModelState.open(path, "never")
    assert opened.metadata == {"position": 2} and opened.tensors["k"].eq(99).all()


@pytest.mark.parametrize("writethrough", [False, True])
def test_loading_saved_state_does_not_mutate_saved_snapshot(tmp_path, writethrough):
    path = str(tmp_path / "snapshot.safetensors")
    state = comfy.model_state.ModelState()
    model = state.load(_KV(), torch.device("cpu"))
    model.k.fill_(7)
    state.metadata["turn"] = {"position": 9}
    state.save(path, writethrough=writethrough)
    state.metadata["turn"]["position"] = 10
    metadata = state.metadata
    model = state.load(_KV(), torch.device("cpu"))
    assert state.metadata is metadata
    model.k.fill_(8)
    state.metadata["turn"]["position"] = 11
    opened = comfy.model_state.ModelState.open(path, "never")
    assert opened.metadata == {"turn": {"position": 9}}
    assert opened.tensors["k"].eq(7).all()
    opened.load(_KV(), torch.device("cpu")).k.fill_(12)
    opened.metadata["turn"]["position"] = 13
    reopened = comfy.model_state.ModelState.open(path, "never")
    assert reopened.metadata == {"turn": {"position": 9}} and reopened.tensors["k"].eq(7).all()


@pytest.mark.parametrize("prior_deferred", [False, True])
def test_failed_writethrough_preserves_saved_catalog(tmp_path, monkeypatch, prior_deferred):
    path = str(tmp_path / "failed.safetensors")
    cpu = torch.device("cpu")
    if prior_deferred:
        previous = comfy.model_state.ModelState()
        previous.load(_KV(), cpu).k.fill_(1)
        previous.save(path, writethrough=False)
    state = comfy.model_state.ModelState()
    state.load(_KV(), cpu).k.fill_(2)

    def fail_save(*args, **kwargs):
        raise OSError("injected save failure")

    with monkeypatch.context() as patch:
        patch.setattr(comfy.utils, "save_torch_file", fail_save)
        with pytest.raises(OSError, match="injected save failure"):
            state.save(path)
    assert not os.path.exists(path) and not state.patcher.get_attachment("model_state_saves")
    if prior_deferred:
        assert previous.patcher.get_attachment("model_state_saves")[path] is previous._values
        assert comfy.model_state.ModelState.open(path, "never").tensors["k"].eq(1).all()
        previous.patcher.unpin_all_weights()
        assert safetensors.torch.load_file(path)["k"].eq(1).all()
    else:
        with pytest.raises(FileNotFoundError):
            comfy.model_state.ModelState.open(path, "never")


@pytest.mark.parametrize("writethrough", [False, True])
@pytest.mark.parametrize("trigger", ["eviction", "release"])
def test_save_release_does_not_retain_readback_or_catalog(tmp_path, writethrough, trigger):
    paths = [str(tmp_path / name) for name in ("readback.safetensors", "alias.safetensors")]
    state = comfy.model_state.ModelState()
    state.load(_KV(), torch.device("cpu")).k.fill_(7)
    for path in paths:
        state.save(path, writethrough=writethrough)
    assert state.tensors is None
    if trigger == "eviction":
        state.patcher.unpin_all_weights()
        assert state.tensors is None and not state.patcher.get_attachment("model_state_saves")
    state.release()
    assert state.tensors is None and state.patcher is None
    for path in paths:
        assert path not in comfy.model_state.SAVED_STATES
        opened = comfy.model_state.ModelState.open(path, "never")
        assert opened.patcher is None and opened.tensors["k"].eq(7).all()


@pytest.mark.parametrize("writethrough", [False, True])
def test_metadata_resave_changes_fingerprint_without_changing_opened_snapshot(tmp_path, writethrough):
    path = str(tmp_path / "metadata.safetensors")
    state = comfy.model_state.ModelState()
    state.load(_KV(), torch.device("cpu")).k.fill_(7)
    state.metadata["turn"] = {"position": 1}
    state.save(path, writethrough=writethrough)
    opened = comfy.model_state.ModelState.open(path, "never")
    before = comfy.model_state.ModelState.fingerprint(path)
    state.metadata["turn"]["position"] = 2
    assert comfy.model_state.ModelState.fingerprint(path) == before
    state.save(path, writethrough=writethrough)
    after = comfy.model_state.ModelState.fingerprint(path)
    assert after != before
    assert opened.metadata == {"turn": {"position": 1}} and opened.tensors["k"].eq(7).all()
    assert comfy.model_state.ModelState.open(path, "never").metadata == {"turn": {"position": 2}}
    state.patcher.unpin_all_weights()
    assert comfy.model_state.ModelState.fingerprint(path) == after


@pytest.mark.parametrize("canceled", [False, True])
def test_partial_write_failure_retries_only_unfinished_destination(tmp_path, monkeypatch, canceled):
    paths = [str(tmp_path / name) for name in ("first.safetensors", "second.safetensors")]
    state = comfy.model_state.ModelState()
    model = state.load(_KV(), torch.device("cpu"))
    model.k.fill_(7)
    for path in paths:
        state.save(path)
        state.save(path, writethrough=False)
    save = comfy.model_state._save_tensors
    file_version = comfy.model_state.ModelState.file_version
    finished, calls = [], []
    failing = True

    def replace_first(path):
        if not finished:
            save({"k": torch.full_like(model.k, 99), "v": model.v}, path,
                 "{}")
            finished.append(path)
        return file_version(path)

    def fail_partial_save(tensors, path, metadata=None):
        with open(path, "wb") as file:
            file.write(b"partial safetensors write")
        raise OSError("injected second destination failure")

    def save_with_failure(tensors, path, metadata=None):
        calls.append(path)
        if failing and finished:
            with monkeypatch.context() as patch:
                patch.setattr(comfy.utils, "save_torch_file", fail_partial_save)
                save(tensors, path, metadata)
        save(tensors, path, metadata)
        finished.append(path)

    if canceled:
        monkeypatch.setattr(comfy.model_state.ModelState, "file_version", staticmethod(replace_first))
    monkeypatch.setattr(comfy.model_state, "_save_tensors", save_with_failure)
    with pytest.raises(OSError, match="injected second destination failure"):
        state.patcher.unpin_all_weights()
    unfinished = calls[-1]
    assert unfinished != finished[0]
    assert state.patcher.get_attachment("model_state_saves") == {unfinished: state._values}
    assert not list(tmp_path.glob(".model-state-*"))
    completed = safetensors.torch.load_file(finished[0])["k"].clone()
    assert completed.eq(99 if canceled else 7).all()
    failing = False
    calls.clear()
    state.patcher.unpin_all_weights()
    assert calls == [unfinished] and not state.patcher.get_attachment("model_state_saves")
    assert torch.equal(safetensors.torch.load_file(finished[0])["k"], completed)
    assert safetensors.torch.load_file(unfinished)["k"].eq(7).all()


def test_cpu_cache_tensor_accounting_before_and_after_flush():
    state = comfy.model_state.ModelState()
    model = state.load(_KV(), torch.device("cpu"))
    resident = state._comfy_cache_tensors()
    expected = model.k.nbytes + model.v.nbytes
    assert state.tensors is None
    assert sum(value.nbytes for value in resident.values()) == expected
    assert resident["k"].data_ptr() == model.k.data_ptr()
    state.flush()
    snapshot = state._comfy_cache_tensors()
    assert snapshot is state.tensors
    assert sum(value.nbytes for value in snapshot.values()) == expected
    assert all(value.data_ptr() != resident[name].data_ptr() for name, value in snapshot.items())


def test_snapshot_reuse_does_not_change_patcher_identity(monkeypatch):
    cpu = torch.device("cpu")
    first = comfy.model_state.ModelState()
    model = first.load(_KV(), cpu)
    model.k.fill_(3)
    patcher = first.patcher
    patch_uuid = patcher.patches_uuid
    version = first._values.version
    second = first.copy()

    def unexpected_restore(*args, **kwargs):
        pytest.fail("The resident snapshot should not be restored")

    with monkeypatch.context() as patch:
        patch.setattr(model, "to_empty", unexpected_restore)
        assert second.load(_KV(), cpu) is model
    assert first._values.version == version and second._values.version != version
    model.k.fill_(5)
    assert first.load(_KV(), cpu) is model and model.k.eq(3).all()
    assert second.load(_KV(), cpu) is model and model.k.eq(5).all()
    assert patcher.patches_uuid == patch_uuid
    assert model.current_weight_patches_uuid == patch_uuid


def test_releasing_unused_copy_does_not_discard_active_writes(monkeypatch):
    state = comfy.model_state.ModelState()
    state.load(_KV(), torch.device("cpu")).k.fill_(7)
    unused = state.copy()
    discarded = []
    monkeypatch.setattr(state.patcher, "is_dynamic", lambda: True)
    monkeypatch.setattr(state.patcher, "discard_writes", lambda: discarded.append(True), raising=False)
    unused.release()
    assert not discarded
    assert state.patcher.get_attachment("model_state") is state._values
    state.release()
    assert discarded == [True]
