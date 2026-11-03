"""Tests for the local installation snapshot + models collector used by deploy to cloud"""

import json
import platform

import pytest

import folder_paths
from app import deploy_to_cloud
from comfyui_version import __version__

SHA_A = "a" * 40
SHA_B = "b" * 40
SHA_C = "c" * 40
SHA_D = "d" * 40
SHA_E = "e" * 40
NODE_KEYS = {"id", "type", "dirName", "enabled", "version", "commit", "url"}


def write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def git_repo(gitdir, head, origin=None):
    write(gitdir / "HEAD", head + "\n")
    if origin:
        write(gitdir / "config", f'[core]\n\tbare = false\n[remote "origin"]\n\turl = {origin}\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n')


@pytest.fixture
def custom_nodes(tmp_path):
    root = tmp_path / "custom_nodes"
    cnr = root / "cnr_pack"
    write(cnr / ".tracking")
    write(cnr / "pyproject.toml", '[project]\nname = "comfyui-cnr-pack"\nversion = "1.2.3"\n')

    git_repo(root / "loose_ref" / ".git", "ref: refs/heads/main")
    write(root / "loose_ref" / ".git" / "refs" / "heads" / "main", SHA_A + "\n")

    git_repo(root / "packed_ref" / ".git", "ref: refs/heads/main")
    write(root / "packed_ref" / ".git" / "packed-refs", f"# pack-refs with: peeled fully-peeled sorted\n{SHA_C} refs/heads/dev\n{SHA_B} refs/heads/main\n^{SHA_C}\n")

    git_repo(root / "detached" / ".git", SHA_C)

    git_repo(tmp_path / "modules" / "linked", SHA_D)
    write(root / "gitdir_file" / ".git", "gitdir: ../../modules/linked\n")

    git_repo(root / "with_origin" / ".git", SHA_E, origin="https://github.com/a/b.git")

    write(root / "plain_dir" / "nodes.py")
    write(root / "toml_only" / "pyproject.toml", '[project]\nname = "x"\n')
    (root / "stray_dir").mkdir()
    write(root / "stray_dir" / "readme.txt")
    write(root / "bad_gitfile" / ".git", "not a gitdir pointer\n")
    write(root / "foo.py")
    write(root / "websocket_image_save.py")
    write(root / "__init__.py")
    write(root / "__pycache__" / "x.pyc")
    write(root / "notes.txt")
    git_repo(root / "corrupt_head" / ".git", "not a sha")
    write(root / "renamed_off.disabled" / "__init__.py")
    write(root / "bar.py.disabled")
    write(root / ".disabled" / "old_pack" / "__init__.py")
    write(root / ".disabled" / "foo_off.py")
    (root / ".disabled" / "stray_off").mkdir()
    return root


def test_scan_custom_nodes_identifies_every_kind(custom_nodes):
    assert deploy_to_cloud.scan_custom_nodes([str(custom_nodes)]) == [
        {"id": "comfyui-cnr-pack", "type": "cnr", "dirName": "cnr_pack", "enabled": True, "version": "1.2.3"},
        {"id": "corrupt_head", "type": "git", "dirName": "corrupt_head", "enabled": True},
        {"id": "detached", "type": "git", "dirName": "detached", "enabled": True, "commit": SHA_C},
        {"id": "foo", "type": "file", "dirName": "foo.py", "enabled": True},
        {"id": "gitdir_file", "type": "git", "dirName": "gitdir_file", "enabled": True, "commit": SHA_D},
        {"id": "loose_ref", "type": "git", "dirName": "loose_ref", "enabled": True, "commit": SHA_A},
        {"id": "packed_ref", "type": "git", "dirName": "packed_ref", "enabled": True, "commit": SHA_B},
        {"id": "plain_dir", "type": "git", "dirName": "plain_dir", "enabled": True},
        {"id": "renamed_off.disabled", "type": "git", "dirName": "renamed_off.disabled", "enabled": False},
        {"id": "toml_only", "type": "git", "dirName": "toml_only", "enabled": True},
        {"id": "with_origin", "type": "git", "dirName": "with_origin", "enabled": True, "url": "https://github.com/a/b.git", "commit": SHA_E},
        {"id": "old_pack", "type": "git", "dirName": "old_pack", "enabled": False},
    ]


def test_scan_skips_what_desktop_skips(custom_nodes):
    names = {n["dirName"] for n in deploy_to_cloud.scan_custom_nodes([str(custom_nodes)])}
    # core file node, stray dir without pyproject/Python, .git file without gitdir:, .disabled/ files and stray dirs, *.py.disabled
    assert not names & {"websocket_image_save.py", "stray_dir", "bad_gitfile", "foo_off.py", "stray_off", "__init__.py", "notes.txt", "bar.py.disabled"}


def test_scan_skips_missing_dirs(tmp_path):
    assert deploy_to_cloud.scan_custom_nodes([str(tmp_path / "missing")]) == []


def test_scan_skips_unreadable_dir(custom_nodes, monkeypatch):
    real_listdir = deploy_to_cloud.os.listdir

    def listdir(path):
        if str(path).endswith("plain_dir"):
            raise PermissionError(path)
        return real_listdir(path)

    monkeypatch.setattr(deploy_to_cloud.os, "listdir", listdir)
    names = [n["dirName"] for n in deploy_to_cloud.scan_custom_nodes([str(custom_nodes)])]
    assert "plain_dir" not in names
    assert "loose_ref" in names


def test_count_custom_nodes_counts_enabled_only(custom_nodes, monkeypatch):
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "custom_nodes", ([str(custom_nodes)], set()))
    assert deploy_to_cloud.count_custom_nodes() == 10


@pytest.mark.parametrize("pyproject, expected", [
    ('[project]\nname = "dq"\nversion = "1.0"\n', {"id": "dq", "version": "1.0"}),
    ("[project]\nname = 'sq'\nversion = '2.0'\n", {"id": "sq", "version": "2.0"}),
    ('[project]\nversion = "3.0"\n\n[tool.x]\nname = "not-this"\n', {"id": "pack", "version": "3.0"}),
    ('[tool.comfy]\nname = "x"\nversion = "9"\n', {"id": "pack"}),
    (None, {"id": "pack"}),
])
def test_cnr_pyproject_parsing(tmp_path, pyproject, expected):
    write(tmp_path / "pack" / ".tracking")
    if pyproject is not None:
        write(tmp_path / "pack" / "pyproject.toml", pyproject)
    assert deploy_to_cloud.scan_custom_nodes([str(tmp_path)]) == [{"type": "cnr", "dirName": "pack", "enabled": True, **expected}]


@pytest.mark.parametrize("url, expected", [
    ("https://user:tok@github.com/a/b.git", "https://github.com/a/b.git"),
    ("http://tok@example.com:8080/a/b?x=1#f", "http://example.com:8080/a/b"),
    ("https://github.com/a/b.git", "https://github.com/a/b.git"),
    ("git@github.com:a/b.git", "git@github.com:a/b.git"),
    ("ssh://user:secret@host/x.git", "ssh://host/x.git"),
    ("ssh://git@github.com/org/repo.git", "ssh://git@github.com/org/repo.git"),
    ("ssh://git:@github.com:22/org/repo.git", "ssh://github.com:22/org/repo.git"),
    ("ssh://al%40ice@host/x.git", "ssh://host/x.git"),
    ("https://git@github.com/org/repo.git", "https://github.com/org/repo.git"),
    ("git://user@host:9418/x.git", "git://host:9418/x.git"),
    ("HTTPS://github.com/a/b.git", "https://github.com/a/b.git"),
    # only http(s)/ssh/git URLs with a host, and scp-like remotes on a dotted host, are reported
    ("deploy@gitlab.example.org:team/repo.git", "deploy@gitlab.example.org:team/repo.git"),
    ("user@internal-host:repo.git", None),
    ("git@github.com:/abs/repo.git", None),
    ("https::https://a:b@h/x", None),
    ("ext::ssh -i key host %S x", None),
    ("https://[::1]/x.git", None),
    ("https://user:123/pw@host/x.git", None),
    ("https://user@/x.git", None),
    ("file:///srv/x.git", None),
    ("ftp://host/x.git", None),
    ("/srv/x.git", None),
    ("../x.git", None),
    ("C:\\repos\\x.git", None),
    # git's value syntax: quotes, escapes and comments are decoded before the URL is classified
    ('"https://alice:secret@github.com/org/repo.git"', "https://github.com/org/repo.git"),
    ('"https://al\\"ice:se\\\\cret@github.com/org/my\\"repo.git"', 'https://github.com/org/my"repo.git'),
    ('"ssh://alice:secret@host/x.git" # pinned', "ssh://host/x.git"),
    ("https://alice:secret@github.com/a/b.git ; set by script", "https://github.com/a/b.git"),
    ("https://github.com/a/b.git\n\t\tpushurl = https://alice:secret@github.com/a/b.git", "https://github.com/a/b.git"),
    ('"https://alice:secret@github.com/a/b.git', None),
    ('"https://alice:se\\qret@github.com/a/b.git"', None),
    ("https://alice:secret@github.com/a/b.git\\", None),
])
def test_git_origin_url_strips_userinfo(tmp_path, url, expected):
    git_repo(tmp_path, SHA_A, origin=url)
    assert deploy_to_cloud._git_origin_url(str(tmp_path)) == expected


def test_git_origin_url_tolerates_bad_config(tmp_path):
    write(tmp_path / "config", "garbage without section\n")
    assert deploy_to_cloud._git_origin_url(str(tmp_path)) is None
    assert deploy_to_cloud._git_origin_url(str(tmp_path / "missing")) is None


def test_git_head_rejects_ref_escaping_gitdir(tmp_path):
    write(tmp_path / "outside", SHA_A)
    write(tmp_path / "g" / "HEAD", "ref: refs/../../outside\n")
    assert deploy_to_cloud._git_head(str(tmp_path / "g")) is None


def test_pip_packages_are_name_to_version():
    packages = deploy_to_cloud.pip_packages()
    assert "aiohttp" in packages
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in packages.items())


def dist_info(site, name, version):
    write(site / f"{name}-{version}.dist-info" / "METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A fake sys.prefix; each test sets sys.path to the site directories it builds."""
    prefix = tmp_path / "env"
    prefix.mkdir()
    monkeypatch.setattr(deploy_to_cloud.sys, "prefix", str(prefix))
    return prefix


def test_pip_packages_keep_first_distribution_of_a_name(env, monkeypatch):
    dist_info(env / "first", "dup", "1.0")
    dist_info(env / "second", "dup", "2.0")
    write(env / "second" / "nameless-1.0.dist-info" / "METADATA", "Metadata-Version: 2.1\nVersion: 1.0\n")
    monkeypatch.setattr(deploy_to_cloud.sys, "path", [str(env / "first"), str(env / "second")])
    assert deploy_to_cloud.pip_packages() == {"dup": "1.0"}


def test_pip_packages_read_only_sys_path_entries_under_sys_prefix(tmp_path, env, monkeypatch):
    dist_info(env / "lib" / "site-packages", "inside", "1.0")
    dist_info(tmp_path / "user-site", "user-only", "1.0")
    dist_info(tmp_path / "base" / "site-packages", "base-only", "1.0")
    dist_info(tmp_path / "env_sibling" / "site-packages", "sibling-only", "1.0")
    monkeypatch.setattr(deploy_to_cloud.sys, "path", [
        str(tmp_path / "comfyui"), str(tmp_path / "user-site"), str(env / "lib" / "site-packages"),
        str(tmp_path / "base" / "site-packages"), str(tmp_path / "env_sibling" / "site-packages"),
    ])
    assert deploy_to_cloud.pip_packages() == {"inside": "1.0"}


def test_pip_packages_compare_resolved_paths(tmp_path, env, monkeypatch):
    dist_info(env / "site", "inside", "1.0")
    dist_info(tmp_path / "user-site", "user-only", "1.0")
    try:
        (tmp_path / "env-link").symlink_to(env, target_is_directory=True)
        (env / "escape").symlink_to(tmp_path / "user-site", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    monkeypatch.setattr(deploy_to_cloud.sys, "prefix", str(tmp_path / "env-link"))
    monkeypatch.setattr(deploy_to_cloud.sys, "path", [str(env / "site"), str(env / "escape")])
    assert deploy_to_cloud.pip_packages() == {"inside": "1.0"}


@pytest.mark.parametrize("direct_url, expected", [
    (None, "1.0"),
    ({"url": "file:///home/alice/src/pkg", "dir_info": {"editable": True}}, "file://local"),
    ({"url": "file:///home/alice/src/pkg", "dir_info": {}}, "file://local"),
    ({"url": "https://alice:tok@github.com/a/b.git", "vcs_info": {"vcs": "git", "commit_id": SHA_A, "requested_revision": "main"}},
     f"git+https://github.com/a/b.git@{SHA_A}"),
    ({"url": "ssh://git@github.com/a/b.git", "vcs_info": {"vcs": "git", "commit_id": SHA_B}}, f"git+ssh://git@github.com/a/b.git@{SHA_B}"),
    ({"url": "https://tok@files.example.com/x/pkg-1.0.tar.gz?sig=s#sha256=00", "archive_info": {"hash": "sha256=00"}},
     "https://files.example.com/x/pkg-1.0.tar.gz"),
    ({"url": "file:///home/alice/wheels/pkg-1.0-py3-none-any.whl", "archive_info": {}}, "file://local"),
    ({"url": "file:///home/alice/repo", "vcs_info": {"vcs": "git", "commit_id": SHA_A}}, "file://local"),
    ({"url": "https://github.com/a/b.git", "vcs_info": {"vcs": "git"}}, "file://local"),
    ({"url": "https://github.com/a/b.git", "vcs_info": "git"}, "file://local"),
    ({"url": 5, "archive_info": {}}, "file://local"),
    ({"archive_info": {}}, "file://local"),
    (["https://github.com/a/b.git"], "file://local"),
    ("{not json", "file://local"),
])
def test_pip_packages_report_direct_url_installs_without_paths_or_credentials(env, monkeypatch, direct_url, expected):
    dist_info(env / "site", "pkg", "1.0")
    if direct_url is not None:
        write(env / "site" / "pkg-1.0.dist-info" / "direct_url.json", direct_url if isinstance(direct_url, str) else json.dumps(direct_url))
    monkeypatch.setattr(deploy_to_cloud.sys, "path", [str(env / "site")])
    assert deploy_to_cloud.pip_packages() == {"pkg": expected}


def test_build_snapshot_envelope(custom_nodes, monkeypatch):
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "custom_nodes", ([str(custom_nodes)], set()))
    monkeypatch.setattr(folder_paths, "base_path", str(custom_nodes.parent / "MyComfy"))
    s = deploy_to_cloud.build_snapshot()
    assert s["type"] == "comfyui-desktop-2-snapshot"
    assert s["version"] == 2
    assert s["installationName"] == "MyComfy"
    assert s["exportedAt"].endswith("Z")
    [snap] = s["snapshots"]
    tag = "v" + __version__
    assert snap["createdAt"] == s["exportedAt"]
    assert (snap["version"], snap["trigger"], snap["label"]) == (2, "manual", None)
    assert snap["comfyui"] == {"ref": tag, "commit": None, "releaseTag": tag, "variant": "comfyui-core", "baseTag": tag}
    assert snap["pythonVersion"] == platform.python_version()
    assert "aiohttp" in snap["pipPackages"]
    assert snap["customNodes"] == deploy_to_cloud.scan_custom_nodes([str(custom_nodes)])


def test_build_snapshot_golden_json(custom_nodes, monkeypatch):
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "custom_nodes", ([str(custom_nodes)], set()))
    parsed = json.loads(json.dumps(deploy_to_cloud.build_snapshot()))
    nodes = parsed["snapshots"][0]["customNodes"]
    assert len(nodes) == 12
    for node in nodes:
        assert set(node) <= NODE_KEYS


def test_list_models(monkeypatch):
    files = {
        "checkpoints": ["sd.safetensors", "sub/dir/model.GGUF", "readme.txt", "preview.png"],
        "configs": ["v1.yaml", "weird.safetensors"],
        "custom_nodes": ["pack/model.safetensors"],
        "text_encoders": ["t5.onnx", "t5.yaml"],
    }
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", {k: ([], set()) for k in files})
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda folder: files[folder])
    assert deploy_to_cloud.list_models() == [
        {"directory": "checkpoints", "filename": "sd.safetensors"},
        {"directory": "checkpoints", "filename": "sub/dir/model.GGUF"},
        {"directory": "text_encoders", "filename": "t5.onnx"},
    ]


def test_collect_returns_snapshot_and_models(monkeypatch):
    monkeypatch.setattr(deploy_to_cloud, "build_snapshot", lambda: {"type": "s"})
    monkeypatch.setattr(deploy_to_cloud, "list_models", lambda: [{"directory": "d", "filename": "f"}])
    assert deploy_to_cloud.collect() == ({"type": "s"}, [{"directory": "d", "filename": "f"}])
