import os
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


def test_generate_without_mtp_head_drops_cache_ids(monkeypatch):
    # a model with no MTP head (e.g. Qwen Image 2.1's prompt enhancer) takes the plain generate path, which has no
    # cache_ids argument: mtp auto must not pass it through
    from comfy.cli_args import args
    monkeypatch.setattr(args, "cpu", True)  # importing the model code must not need a GPU
    from comfy.text_encoders import qwen35, llama
    seen = {}

    def plain_generate(self, embeds=None, do_sample=True, max_length=256, temperature=1.0, top_k=50, top_p=0.9,
                       min_p=0.0, repetition_penalty=1.0, seed=42, stop_tokens=None, model_state=None, position_ids=None,
                       presence_penalty=0.0):
        seen.update(model_state=model_state, position_ids=position_ids)
        return "ok"

    monkeypatch.setattr(llama.BaseGenerate, "generate", plain_generate)
    model = qwen35.Qwen35.__new__(qwen35.Qwen35)
    torch.nn.Module.__init__(model)
    model.mtp = None
    out = model.generate(embeds=None, mtp=True, cache_ids=("a", "b"), model_state="state", position_ids="pos")
    assert out == "ok"
    assert seen == {"model_state": "state", "position_ids": "pos"}
