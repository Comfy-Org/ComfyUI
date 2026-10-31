import os

import pytest

from comfy_api.latest._weight_cache import WeightCache, weight_identity


def _write(tmp_path, name, body="w"):
    path = tmp_path / name
    path.write_text(body)
    return str(path)


def test_hit_reuses_entry_without_reloading(tmp_path):
    path = _write(tmp_path, "a.bin")
    cache = WeightCache(load=lambda p: object(), max_entries=2)

    first = cache.get(path)
    second = cache.get(path)

    assert first is second
    assert (cache.loads, cache.hits) == (1, 1)


def test_evicts_least_recently_used_not_most_recent(tmp_path):
    """A cache that evicts the newest entry pins whatever loaded first.

    The copy-pasted caches this replaces used ``dict.popitem()``, which drops
    the most recently inserted entry, so a workflow alternating between two
    models reloaded both every time while the first one it ever saw was never
    reclaimed.
    """
    a, b, c = (_write(tmp_path, n) for n in ("a.bin", "b.bin", "c.bin"))
    cache = WeightCache(load=lambda p: p, max_entries=2)

    cache.get(a)
    cache.get(b)
    cache.get(a)          # a is now the most recently used
    cache.get(c)          # must evict b, the least recently used

    loads_before = cache.loads
    cache.get(a)
    assert cache.loads == loads_before, "a was evicted despite recent use"
    cache.get(b)
    assert cache.loads == loads_before + 1, "b should have been the eviction"


def test_release_runs_on_eviction(tmp_path):
    a, b = (_write(tmp_path, n) for n in ("a.bin", "b.bin"))
    released = []
    cache = WeightCache(
        load=lambda p: p, max_entries=1, release=released.append)

    cache.get(a)
    cache.get(b)

    assert released == [a]


def test_discriminators_separate_entries_from_one_file(tmp_path):
    path = _write(tmp_path, "a.bin")
    cache = WeightCache(load=lambda p, variant: variant, max_entries=4)

    assert cache.get(path, "base") == "base"
    assert cache.get(path, "large") == "large"
    assert cache.loads == 2


def test_rewritten_file_is_not_served_from_cache(tmp_path, monkeypatch):
    path = _write(tmp_path, "a.bin", "one")
    cache = WeightCache(load=lambda p: open(p).read(), max_entries=2)
    assert cache.get(path) == "one"

    # Same path, new contents: identity must change so the stale entry is not
    # returned. mtime_ns can collide on a coarse clock, so force it forward.
    with open(path, "w") as handle:
        handle.write("two")
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    assert cache.get(path) == "two"


def test_identity_changes_when_contents_change(tmp_path):
    path = _write(tmp_path, "a.bin", "one")
    before = weight_identity(path)
    with open(path, "w") as handle:
        handle.write("a much longer body")
    assert weight_identity(path) != before


def test_max_entries_must_be_positive():
    with pytest.raises(ValueError):
        WeightCache(load=lambda p: p, max_entries=0)


def test_clear_releases_all_and_reports_count(tmp_path):
    a, b = (_write(tmp_path, n) for n in ("a.bin", "b.bin"))
    released = []
    cache = WeightCache(
        load=lambda p: p, max_entries=4, release=released.append)
    cache.get(a)
    cache.get(b)

    assert cache.clear() == 2
    assert sorted(released) == sorted([a, b])
    assert cache.clear() == 0
