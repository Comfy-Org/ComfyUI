import os
import shutil
import time

import pytest
import torch

from comfy.text_encoders import llm_state_cache as cache

MB = 2**20


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache, "_entries", [])
    return tmp_path


def write(name, mb, age=0):
    path = os.path.join(cache.CACHE_DIR, name)
    cache._write(path, {"t": torch.zeros(mb * MB, dtype=torch.uint8)}, {"key": "k", "ids": [1, 2]})
    if age:
        t = time.time() - age
        os.utime(path, (t, t))
    return path


def files(d):
    return sorted(p.name for p in d.iterdir())


def test_disk_budget_evicts_least_recently_used(cache_dir, monkeypatch):
    monkeypatch.setattr(cache, "DISK_BUDGET", 1000 * 2**40)
    old = write("spill-old.safetensors", 4, age=300)
    used = write("prefix-used.safetensors", 4, age=200)
    write("spill-mid.safetensors", 4, age=100)
    assert cache._read(used) is not None  # a read counts as a use
    monkeypatch.setattr(cache, "DISK_BUDGET", 9 * MB)
    write("spill-new.safetensors", 4)
    assert files(cache_dir) == ["prefix-used.safetensors", "spill-new.safetensors"]
    assert not os.path.exists(old)


def test_disk_budget_drops_spilled_entries_and_old_runs_files(cache_dir, monkeypatch):
    monkeypatch.setattr(cache, "DISK_BUDGET", 1000 * 2**40)
    orphan = write("spill-2-140729398259792.safetensors", 4, age=3 * 3600)  # an earlier run's, unknown to this one
    partial = os.path.join(cache.CACHE_DIR, "spill-x.safetensors.tmp2")
    with open(partial, "wb") as f:
        f.write(b"\0" * MB)
    os.utime(partial, (time.time() - 2 * 3600,) * 2)
    entry = cache._Entry("k", None, 0)
    entry.file = write("spill-mine.safetensors", 4, age=60)
    cache._entries.append(entry)
    unrelated = cache_dir / "notes.txt"
    unrelated.write_text("x")
    monkeypatch.setattr(cache, "DISK_BUDGET", 5 * MB)
    write("prefix-new.safetensors", 4)
    assert files(cache_dir) == ["notes.txt", "prefix-new.safetensors"]
    assert not os.path.exists(orphan)
    assert cache._entries == []


def test_recent_partial_write_is_left_alone(cache_dir, monkeypatch):
    partial = os.path.join(cache.CACHE_DIR, "spill-x.safetensors.tmp7")
    with open(partial, "wb") as f:
        f.write(b"\0" * 4 * MB)
    monkeypatch.setattr(cache, "DISK_BUDGET", 1 * MB)
    write("prefix-new.safetensors", 4)
    assert files(cache_dir) == ["prefix-new.safetensors", "spill-x.safetensors.tmp7"]


def test_room_is_made_before_writing(cache_dir, monkeypatch):
    # the folder stays under the budget while a file is written, counting another write still in progress
    monkeypatch.setattr(cache, "DISK_BUDGET", 1000 * 2**40)
    old = write("spill-old.safetensors", 4, age=100)
    partial = os.path.join(cache.CACHE_DIR, "spill-x.safetensors.tmp7")
    with open(partial, "wb") as f:
        f.write(b"\0" * 4 * MB)
    monkeypatch.setattr(cache, "DISK_BUDGET", 9 * MB)
    seen = []
    save = cache.safetensors.torch.save_file
    monkeypatch.setattr(cache.safetensors.torch, "save_file", lambda t, p, m: seen.append(files(cache_dir)) or save(t, p, m))
    write("spill-new.safetensors", 4)
    assert seen == [["spill-x.safetensors.tmp7"]]
    assert not os.path.exists(old)
    assert files(cache_dir) == ["spill-new.safetensors", "spill-x.safetensors.tmp7"]


def test_missing_or_broken_file_is_a_miss(cache_dir):
    assert cache._read(os.path.join(cache.CACHE_DIR, "spill-gone.safetensors")) is None
    broken = cache_dir / "prefix-broken.safetensors"
    broken.write_bytes(b"not a safetensors file")
    assert cache._read(str(broken)) is None


def free_bytes(path):
    return shutil.disk_usage(path).free


def test_restored_states_do_not_hold_deleted_spill_files(cache_dir, monkeypatch):
    monkeypatch.setattr(cache, "DISK_BUDGET", 1000 * 2**40)
    restored = []
    free_before = free_bytes(cache_dir)
    for i in range(20):
        path = os.path.join(cache.CACHE_DIR, f"spill-{i}.safetensors")
        cache._write(path, {"t": torch.ones(16 * MB, dtype=torch.uint8), "ckpt.5.0.conv": torch.ones(4, dtype=torch.uint8),
                            "ckpt.5.0.rec": torch.ones(4, dtype=torch.uint8)}, {"key": "k", "ids": [1, 2]})
        tensors, checkpoints, _ = cache._read(path)
        restored.append((cache._padded(tensors, {"t": torch.empty(16 * MB, dtype=torch.uint8)}), checkpoints))
        del tensors
        cache._remove(path)
    assert all(state["t"].sum() == 16 * MB for state, _ in restored)
    assert free_before - free_bytes(cache_dir) < 160 * MB, "deleted spill files' space is freed while their states live"
    if os.path.exists("/proc/self/maps"):
        with open("/proc/self/maps") as f:
            held = [line for line in f if str(cache_dir) in line]
        assert held == [], "no deleted spill file stays mapped"


def test_file_deleted_after_a_complete_read_is_still_a_hit(cache_dir, monkeypatch):
    path = write("spill-a.safetensors", 1)

    def gone(p, *a, **k):
        raise FileNotFoundError(p)
    monkeypatch.setattr(cache.os, "utime", gone)  # another server deleted it once the read was done
    assert cache._read(path) is not None


def mapped(path):
    with open("/proc/self/maps") as f:
        return [line for line in f if path in line]


@pytest.mark.skipif(not os.path.exists("/proc/self/maps"), reason="needs /proc")
def test_restoring_a_spill_deletes_its_file_after_the_views_are_gone(cache_dir, monkeypatch):
    monkeypatch.setattr(cache, "DISK_BUDGET", 1000 * 2**40)
    path = os.path.join(cache.CACHE_DIR, "spill-a.safetensors")
    cache._write(path, {"t": torch.zeros(MB, dtype=torch.uint8), "ckpt.2.0.conv": torch.zeros(4), "ckpt.2.0.rec": torch.zeros(4)},
                 {"key": "k", "ids": [1, 2]})
    entry = cache._Entry("k", None, 64)
    entry.file, entry.ids, entry.checkpoints = path, [1, 2], {2: {}}
    cache._entries.append(entry)
    monkeypatch.setattr(cache, "_load", lambda model, cap, device, dtype, tensors: (object(), object()))
    monkeypatch.setattr(cache, "_restore_linear", lambda *a: None)
    monkeypatch.setattr(cache, "_set_index", lambda *a: None)
    remove = cache._remove
    at_delete = []

    def checked_remove(p):
        at_delete.append(mapped(p))
        remove(p)
    monkeypatch.setattr(cache, "_remove", checked_remove)
    taken, _, n = cache._take(None, "k", [1, 2] + [3] * (cache.MIN_SUFFIX + 1), 4, "cpu", None)
    assert taken is entry and n == 2
    assert at_delete == [[]], "Windows doesn't free a file deleted while a view of it lives"
    assert not os.path.exists(path)
