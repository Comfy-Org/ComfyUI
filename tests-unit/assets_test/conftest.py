import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable, Iterator, Optional

import pytest
import requests


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Allow overriding the database URL used by the spawned ComfyUI process.
    Priority:
      1) --db-url command line option
      2) ASSETS_TEST_DB_URL environment variable (used by CI)
      3) default: None (will use file-backed sqlite in temp dir)
    """
    parser.addoption(
        "--db-url",
        action="store",
        default=os.environ.get("ASSETS_TEST_DB_URL"),
        help="SQLAlchemy DB URL (e.g. sqlite:///path/to/db.sqlite3)",
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_base_dirs(root: Path) -> None:
    for sub in ("models", "custom_nodes", "input", "output", "temp", "user"):
        (root / sub).mkdir(parents=True, exist_ok=True)


def _wait_http_ready(base: str, session: requests.Session, timeout: float = 90.0) -> None:
    start = time.time()
    last_err = None
    while time.time() - start < timeout:
        try:
            r = session.get(base + "/api/assets", timeout=5)
            if r.status_code in (200, 400):
                return
        except Exception as e:
            last_err = e
        time.sleep(0.25)
    raise RuntimeError(f"ComfyUI HTTP did not become ready: {last_err}")


@pytest.fixture(scope="session")
def comfy_tmp_base_dir() -> Path:
    env_base = os.environ.get("ASSETS_TEST_BASE_DIR")
    created_by_fixture = False
    if env_base:
        tmp = Path(env_base)
        tmp.mkdir(parents=True, exist_ok=True)
    else:
        tmp = Path(tempfile.mkdtemp(prefix="comfyui-assets-tests-"))
        created_by_fixture = True
    _make_base_dirs(tmp)
    yield tmp
    if created_by_fixture:
        with contextlib.suppress(Exception):
            for p in sorted(tmp.rglob("*"), reverse=True):
                if p.is_file() or p.is_symlink():
                    p.unlink(missing_ok=True)
            for p in sorted(tmp.glob("**/*"), reverse=True):
                with contextlib.suppress(Exception):
                    p.rmdir()
            tmp.rmdir()


@pytest.fixture(scope="session")
def comfy_url_and_proc(comfy_tmp_base_dir: Path, request: pytest.FixtureRequest):
    """
    Boot ComfyUI subprocess with:
      - sandbox base dir
      - file-backed sqlite DB in temp dir
      - autoscan disabled
    Returns (base_url, process, port)
    """
    port = _free_port()
    db_url = request.config.getoption("--db-url")
    if not db_url:
        # Use a file-backed sqlite database in the temp directory
        db_path = comfy_tmp_base_dir / "assets-test.sqlite3"
        db_url = f"sqlite:///{db_path}"

    logs_dir = comfy_tmp_base_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    out_log = open(logs_dir / "stdout.log", "w", buffering=1)
    err_log = open(logs_dir / "stderr.log", "w", buffering=1)

    comfy_root = Path(__file__).resolve().parent.parent.parent
    if not (comfy_root / "main.py").is_file():
        raise FileNotFoundError(f"main.py not found under {comfy_root}")

    proc = subprocess.Popen(
        args=[
            sys.executable,
            "main.py",
            f"--base-directory={str(comfy_tmp_base_dir)}",
            f"--database-url={db_url}",
            "--enable-assets",
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
            "--cpu",
        ],
        stdout=out_log,
        stderr=err_log,
        cwd=str(comfy_root),
        env={**os.environ},
    )

    for _ in range(50):
        if proc.poll() is not None:
            out_log.flush()
            err_log.flush()
            raise RuntimeError(f"ComfyUI exited early with code {proc.returncode}")
        time.sleep(0.1)

    base_url = f"http://127.0.0.1:{port}"
    try:
        with requests.Session() as s:
            _wait_http_ready(base_url, s, timeout=90.0)
        yield base_url, proc, port
    except Exception as e:
        with contextlib.suppress(Exception):
            proc.terminate()
            proc.wait(timeout=10)
        with contextlib.suppress(Exception):
            out_log.flush()
            err_log.flush()
        raise RuntimeError(f"ComfyUI did not become ready: {e}")

    if proc and proc.poll() is None:
        with contextlib.suppress(Exception):
            proc.terminate()
            proc.wait(timeout=15)
    out_log.close()
    err_log.close()


@pytest.fixture
def http() -> Iterator[requests.Session]:
    with requests.Session() as s:
        s.timeout = 120
        yield s


@pytest.fixture
def api_base(comfy_url_and_proc) -> str:
    base_url, _proc, _port = comfy_url_and_proc
    return base_url


def _post_multipart_asset(
    session: requests.Session,
    base: str,
    *,
    name: str,
    tags: list[str],
    meta: dict,
    data: bytes,
    extra_fields: Optional[dict] = None,
) -> tuple[int, dict]:
    files = {"file": (name, data, "application/octet-stream")}
    form_data = {
        "tags": json.dumps(tags),
        "name": name,
        "user_metadata": json.dumps(meta),
    }
    if extra_fields:
        for k, v in extra_fields.items():
            form_data[k] = v
    r = session.post(base + "/api/assets", files=files, data=form_data, timeout=120)
    return r.status_code, r.json()


@pytest.fixture
def make_asset_bytes() -> Callable[[str, int], bytes]:
    # Salt content per test so it never collides with assets left over from
    # earlier tests. Delete is now always a soft delete (content is preserved),
    # so the suite can no longer rely on hard-deleting content for isolation.
    # Deterministic within a test: the same (name, size) yields the same bytes.
    salt = uuid.uuid4().bytes

    def _make(name: str, size: int = 8192) -> bytes:
        seed = sum(ord(c) for c in name) % 251
        body = bytearray((i * 31 + seed) % 256 for i in range(size))
        body[: len(salt)] = salt[:size]
        return bytes(body)
    return _make


@pytest.fixture
def asset_factory(http: requests.Session, api_base: str):
    """
    Returns create(name, tags, meta, data) -> response dict
    Tracks created ids and deletes them after the test.
    """
    created: list[str] = []

    def create(name: str, tags: list[str], meta: dict, data: bytes) -> dict:
        status, body = _post_multipart_asset(http, api_base, name=name, tags=tags, meta=meta, data=data)
        assert status in (200, 201), body
        created.append(body["id"])
        return body

    yield create

    for aid in created:
        with contextlib.suppress(Exception):
            http.delete(f"{api_base}/api/assets/{aid}", timeout=30)


@pytest.fixture
def seeded_asset(request: pytest.FixtureRequest, http: requests.Session, api_base: str) -> dict:
    """
    Upload one asset with ".safetensors" extension into models/checkpoints/unit-tests/<name>.
    Returns response dict with id, asset_hash, tags, etc.
    """
    name = "unit_1_example.safetensors"
    p = getattr(request, "param", {}) or {}
    tags: Optional[list[str]] = p.get("tags")
    if tags is None:
        tags = ["models", "checkpoints", "unit-tests", "alpha"]
    meta = {"purpose": "test", "epoch": 1, "flags": ["x", "y"], "nullable": None}
    # Unique content per test so the seed always creates a fresh asset (201).
    # Delete is now always a soft delete, so content from a prior test survives
    # and would otherwise dedup this upload into an existing asset (200).
    content = uuid.uuid4().bytes + b"A" * (4096 - 16)
    files = {"file": (name, content, "application/octet-stream")}
    form_data = {
        "tags": json.dumps(tags),
        "name": name,
        "user_metadata": json.dumps(meta),
    }
    r = http.post(api_base + "/api/assets", files=files, data=form_data, timeout=120)
    body = r.json()
    assert r.status_code == 201, body
    from helpers import assert_hash_fields_consistent
    assert_hash_fields_consistent(body)
    return body


@pytest.fixture(autouse=True)
def autoclean_unit_test_assets(http: requests.Session, api_base: str):
    """Ensure isolation by removing all AssetInfo rows tagged with 'unit-tests' after each test."""
    yield

    while True:
        r = http.get(
            api_base + "/api/assets",
            params={"include_tags": "unit-tests", "limit": "500", "sort": "name"},
            timeout=30,
        )
        if r.status_code != 200:
            break
        body = r.json()
        ids = [a["id"] for a in body.get("assets", [])]
        if not ids:
            break
        for aid in ids:
            with contextlib.suppress(Exception):
                http.delete(f"{api_base}/api/assets/{aid}", timeout=30)
