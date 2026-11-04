import types

import pytest
import torch

from qwen35_decode_test import _ExactPacked, _tiny_model

import comfy.text_encoders.llm_state_cache as cache  # noqa: E402

END = list(cache.MSG_END)


class _Pages16(_ExactPacked):
    page = 16  # a 48-row window: room to resume a turn after the previous reply's draft rows


@pytest.fixture(params=["bf16", "int8"])
def model(request, monkeypatch, tmp_path):
    import comfy.text_encoders.llama as llama
    import comfy.text_encoders.qwen35 as qwen35
    model = _tiny_model(monkeypatch)
    if request.param == "int8":
        monkeypatch.setattr(llama, "Int8DecodeCache", _Pages16, raising=False)
        monkeypatch.setattr(qwen35.GatedAttention, "kv_cache_type", lambda self, device, dtype: qwen35.Int8FixedKVCache)
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache, "ROUND", 64)
    monkeypatch.setattr(cache, "HEADROOM", 0)  # a later turn outgrows the first one's capacity
    cache._entries.clear()
    cache._current.clear()
    yield model
    for entry in list(cache._entries):
        cache._drop(entry)


_table = {}


def _embeds(model, ids):
    # generated tokens embed as the model embeds them; other ids (message ends, prompt text) get fixed vectors
    rows = []
    for i in ids:
        if i < 97:
            rows.append(model.model.embed_tokens(torch.tensor([i]))[0])
        else:
            rows.append(_table.setdefault(i, torch.randn(64, generator=torch.Generator().manual_seed(i))))
    return torch.stack(rows).unsqueeze(0)


def _text(seed, n):
    return [1000 + seed * 100 + k for k in range(n)]


def _generate(model, ids, max_length, mode):
    """(generated ids, first logits after the prompt) with the cache on or off."""
    first = []
    logits = model.logits

    def record(x):
        out = logits(x)
        if not first:
            first.append(out.detach().clone())
        return out
    model.logits = record
    try:
        with cache.call_mode(mode):
            kwargs = {"cache_ids": list(ids)} if mode == "on" else {}
            out = model.generate(_embeds(model, ids), do_sample=False, max_length=max_length, mtp=True, **kwargs)
    finally:
        model.logits = logits
    return out, first[0]


def _check(model, ids, max_length=8):
    got, got_logits = _generate(model, ids, max_length, "on")
    stats = dict(cache.stats)
    want, want_logits = _generate(model, ids, max_length, "off")
    assert got == want
    assert torch.allclose(got_logits, want_logits, atol=1e-4), stats
    return stats


def test_next_turn_resumes_the_conversation(model):
    system = _text(0, 12) + END
    turn1 = system + _text(1, 20) + END
    stats = _check(model, turn1)
    assert stats["prefix_len"] == 0 and stats.get("shared_start_saved")
    reply = cache._entries[-1].ids[len(turn1):]  # the reply tokens the state holds
    turn2 = turn1 + reply + END + _text(2, 30) + END
    stats = _check(model, turn2)
    assert stats["prefix_len"] >= len(turn1) and stats["restore"] in ("resident", "grow")


def test_spilled_conversation_reloads_from_disk(model, monkeypatch):
    monkeypatch.setattr(cache, "MAX_ENTRIES", 1)
    # longer than the int8 window, so the pages must survive the spill
    a = _text(0, 12) + END + _text(1, 70) + END
    b = _text(5, 12) + END + _text(6, 20) + END  # another system prompt: no shared start
    _check(model, a)
    reply = cache._entries[-1].ids[len(a):]
    _check(model, b)
    assert sum(e.state is None for e in cache._entries) == 1  # a was written to disk
    stats = _check(model, a + reply + END + _text(2, 15) + END)
    assert stats["restore"] == "from-disk" and stats["prefix_len"] >= len(a)


def test_new_conversation_starts_from_the_shared_start_file(model):
    system = _text(0, 70) + END  # longer than the int8 window, so the file must hold its pages
    _check(model, system + _text(1, 20) + END)
    for entry in list(cache._entries):  # as after a restart: only the files remain
        cache._drop(entry)
    stats = _check(model, system + _text(3, 25) + END)
    assert stats["restore"] == "shared-start-file" and stats["prefix_len"] == len(system)


def test_int8_resumes_only_inside_its_window():
    entry = types.SimpleNamespace(window=(48, 16), written=200)
    # the refresh before pos reads the page holding pos - 1 and the one before it; the window holds rows 152..199
    assert not cache._resumable(entry, 170)  # needs rows from 144
    assert cache._resumable(entry, 180)  # needs rows from 160
    assert cache._resumable(types.SimpleNamespace(window=None, written=200), 10)
