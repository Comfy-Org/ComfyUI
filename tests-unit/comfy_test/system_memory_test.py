import sys
from types import SimpleNamespace

import pytest

import comfy.system_memory as system_memory

GIB = 1024 ** 3
HOST_TOTAL = 128 * GIB
HOST_AVAILABLE = 100 * GIB


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def v2(directory, limit=None, current=None, inactive_file=None):
    if limit is not None:
        write(directory / "memory.max", f"{limit}\n")
    if current is not None:
        write(directory / "memory.current", f"{current}\n")
    if inactive_file is not None:
        write(directory / "memory.stat", f"anon 1\nfile 2\ninactive_file {inactive_file}\nactive_file 3\n")


def v1(directory, limit=None, usage=None, inactive_file=None):
    if limit is not None:
        write(directory / "memory.limit_in_bytes", f"{limit}\n")
    if usage is not None:
        write(directory / "memory.usage_in_bytes", f"{usage}\n")
    if inactive_file is not None:
        write(directory / "memory.stat", f"cache 1\ninactive_file 9\ntotal_inactive_file {inactive_file}\n")


@pytest.fixture
def cgroup(tmp_path, monkeypatch):
    v2_root = tmp_path / "sys_fs_cgroup"
    v1_root = v2_root / "memory"
    proc = tmp_path / "proc_self_cgroup"
    v2_root.mkdir()
    monkeypatch.setattr(system_memory, "CGROUP_V2_ROOT", str(v2_root))
    monkeypatch.setattr(system_memory, "CGROUP_V1_MEMORY_ROOT", str(v1_root))
    monkeypatch.setattr(system_memory, "PROC_SELF_CGROUP", str(proc))
    monkeypatch.setattr(system_memory, "_cgroup_dirs", None)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(system_memory.psutil, "virtual_memory", lambda: SimpleNamespace(total=HOST_TOTAL, available=HOST_AVAILABLE))
    return SimpleNamespace(v2_root=v2_root, v1_root=v1_root, proc_path=proc, proc=lambda text: write(proc, text))


def assert_host_values():
    assert system_memory.cgroup_memory_limit() is None
    assert system_memory.virtual_memory_total() == HOST_TOTAL
    assert system_memory.virtual_memory_available() == HOST_AVAILABLE


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_non_linux_returns_psutil_values(cgroup, monkeypatch, platform):
    cgroup.proc("0::/\n")
    v2(cgroup.v2_root, limit=32 * GIB, current=8 * GIB, inactive_file=0)
    monkeypatch.setattr(sys, "platform", platform)
    assert_host_values()


def test_no_cgroup_files_returns_psutil_values(cgroup):
    cgroup.proc("0::/\n")
    assert_host_values()


def test_v2_private_namespace_root_limit(cgroup):
    cgroup.proc("0::/\n")
    v2(cgroup.v2_root, limit=32 * GIB, current=10 * GIB, inactive_file=0)
    assert system_memory.cgroup_memory_limit() == 32 * GIB
    assert system_memory.virtual_memory_total() == 32 * GIB
    assert system_memory.virtual_memory_available() == 22 * GIB


@pytest.mark.parametrize("limit", ["max", "0", "-1", ""])
def test_v2_unlimited_and_unusable_limit_values_are_ignored(cgroup, limit):
    cgroup.proc("0::/\n")
    v2(cgroup.v2_root, limit=limit, current=8 * GIB, inactive_file=0)
    assert_host_values()


def test_v2_nested_host_namespace_path(cgroup):
    cgroup.proc("0::/kubepods.slice/pod1/ctr1\n")
    v2(cgroup.v2_root, current=100 * GIB, inactive_file=0)
    v2(cgroup.v2_root / "kubepods.slice" / "pod1" / "ctr1", limit=16 * GIB, current=4 * GIB, inactive_file=0)
    assert system_memory.virtual_memory_total() == 16 * GIB
    assert system_memory.virtual_memory_available() == 12 * GIB


def test_v2_parent_limit_applies_when_own_cgroup_is_unlimited(cgroup):
    cgroup.proc("0::/pod/ctr\n")
    v2(cgroup.v2_root / "pod" / "ctr", limit="max", current=4 * GIB, inactive_file=0)
    v2(cgroup.v2_root / "pod", limit=16 * GIB, current=6 * GIB, inactive_file=0)
    assert system_memory.virtual_memory_total() == 16 * GIB
    assert system_memory.virtual_memory_available() == 10 * GIB


def test_v2_leaf_headroom_binds_when_parent_has_more_headroom(cgroup):
    cgroup.proc("0::/pod/ctr\n")
    v2(cgroup.v2_root / "pod" / "ctr", limit=16 * GIB, current=4 * GIB, inactive_file=0)
    v2(cgroup.v2_root / "pod", limit=64 * GIB, current=40 * GIB, inactive_file=0)
    assert system_memory.virtual_memory_total() == 16 * GIB
    assert system_memory.virtual_memory_available() == 12 * GIB


def test_v2_parent_headroom_binds_when_smaller_than_leaf_headroom(cgroup):
    cgroup.proc("0::/pod/ctr\n")
    v2(cgroup.v2_root / "pod" / "ctr", limit=16 * GIB, current=4 * GIB, inactive_file=0)
    v2(cgroup.v2_root / "pod", limit=32 * GIB, current=31 * GIB, inactive_file=0)
    assert system_memory.virtual_memory_total() == 16 * GIB
    assert system_memory.virtual_memory_available() == GIB


def test_v2_equal_limits_count_sibling_usage_at_parent(cgroup):
    cgroup.proc("0::/pod/ctr\n")
    v2(cgroup.v2_root / "pod" / "ctr", limit=32 * GIB, current=4 * GIB, inactive_file=0)
    v2(cgroup.v2_root / "pod", limit=32 * GIB, current=20 * GIB, inactive_file=0)
    assert system_memory.virtual_memory_total() == 32 * GIB
    assert system_memory.virtual_memory_available() == 12 * GIB


def test_v1_limit_and_usage(cgroup):
    cgroup.proc("4:memory:/docker/abc\n3:cpu,cpuacct:/docker/abc\n")
    v1(cgroup.v1_root / "docker" / "abc", limit=32 * GIB, usage=10 * GIB, inactive_file=2 * GIB)
    assert system_memory.cgroup_memory_limit() == 32 * GIB
    assert system_memory.virtual_memory_total() == 32 * GIB
    assert system_memory.virtual_memory_available() == 24 * GIB


def test_v1_docker_bind_mounted_root(cgroup):
    cgroup.proc("4:memory:/docker/abc\n")
    v1(cgroup.v1_root, limit=32 * GIB, usage=10 * GIB, inactive_file=2 * GIB)
    assert system_memory.virtual_memory_total() == 32 * GIB
    assert system_memory.virtual_memory_available() == 24 * GIB


def test_v1_line_without_memory_controller_is_ignored(cgroup):
    cgroup.proc("3:cpu,cpuacct:/foo\n")
    v1(cgroup.v1_root / "foo", limit=8 * GIB, usage=GIB, inactive_file=0)
    assert_host_values()


def test_hybrid_layout_finds_v1_memory_limit(cgroup):
    cgroup.proc("4:memory:/docker/abc\n0::/\n")
    v1(cgroup.v1_root / "docker" / "abc", limit=32 * GIB, usage=10 * GIB, inactive_file=0)
    assert system_memory.virtual_memory_total() == 32 * GIB
    assert system_memory.virtual_memory_available() == 22 * GIB


@pytest.mark.parametrize("limit", [HOST_TOTAL, 2 * HOST_TOTAL])
def test_limit_at_or_above_host_total_is_unlimited(cgroup, limit):
    cgroup.proc("0::/\n")
    v2(cgroup.v2_root, limit=limit, current=8 * GIB, inactive_file=0)
    assert_host_values()


def test_page_cache_is_not_counted_as_used(cgroup):
    cgroup.proc("0::/\n")
    v2(cgroup.v2_root, limit=96_600_000_000, current=96_600_000_000, inactive_file=49_500_000_000)
    assert system_memory.virtual_memory_available() == 49_500_000_000


def test_available_is_capped_by_host_available(cgroup, monkeypatch):
    cgroup.proc("0::/\n")
    v2(cgroup.v2_root, limit=64 * GIB, current=0, inactive_file=0)
    monkeypatch.setattr(system_memory.psutil, "virtual_memory", lambda: SimpleNamespace(total=HOST_TOTAL, available=10 * GIB))
    assert system_memory.virtual_memory_available() == 10 * GIB


def test_available_is_never_negative(cgroup):
    cgroup.proc("0::/\n")
    v2(cgroup.v2_root, limit=16 * GIB, current=20 * GIB, inactive_file=0)
    assert system_memory.virtual_memory_available() == 0


def test_available_falls_back_to_limit_when_usage_is_unreadable(cgroup):
    cgroup.proc("0::/\n")
    v2(cgroup.v2_root, limit=32 * GIB)
    assert system_memory.virtual_memory_total() == 32 * GIB
    assert system_memory.virtual_memory_available() == 32 * GIB


@pytest.mark.parametrize("stat", [None, "inactive_file abc\n", "anon 1\n"])
def test_unusable_memory_stat_counts_all_usage(cgroup, stat):
    cgroup.proc("0::/\n")
    v2(cgroup.v2_root, limit=32 * GIB, current=10 * GIB)
    if stat is not None:
        write(cgroup.v2_root / "memory.stat", stat)
    assert system_memory.virtual_memory_available() == 22 * GIB


def test_runtime_limit_change_is_observed(cgroup):
    cgroup.proc("0::/\n")
    v2(cgroup.v2_root, limit=64 * GIB, current=0, inactive_file=0)
    assert system_memory.virtual_memory_total() == 64 * GIB
    v2(cgroup.v2_root, limit=32 * GIB)
    assert system_memory.virtual_memory_total() == 32 * GIB
    assert system_memory.virtual_memory_available() == 32 * GIB
    v2(cgroup.v2_root, limit="max")
    assert_host_values()


def test_cgroup_path_is_resolved_once(cgroup):
    cgroup.proc("0::/a\n")
    v2(cgroup.v2_root / "a", limit=16 * GIB, current=0, inactive_file=0)
    v2(cgroup.v2_root / "b", limit=8 * GIB, current=0, inactive_file=0)
    assert system_memory.virtual_memory_total() == 16 * GIB
    cgroup.proc("0::/b\n")
    assert system_memory.virtual_memory_total() == 16 * GIB


@pytest.mark.parametrize("proc", [None, b"0::/\xff\n"])
def test_unusable_proc_self_cgroup_falls_back_to_well_known_roots(cgroup, proc):
    if proc is not None:
        cgroup.proc_path.write_bytes(proc)
    v2(cgroup.v2_root, limit=32 * GIB, current=8 * GIB, inactive_file=0)
    assert system_memory.virtual_memory_total() == 32 * GIB
    assert system_memory.virtual_memory_available() == 24 * GIB


def test_well_known_roots_are_not_consulted_when_proc_lists_a_cgroup(cgroup):
    cgroup.proc("0::/a\n")
    (cgroup.v2_root / "a").mkdir()
    v1(cgroup.v1_root, limit=8 * GIB, usage=GIB, inactive_file=0)
    assert_host_values()
