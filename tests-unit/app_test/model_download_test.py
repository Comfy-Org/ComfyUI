import os

import pytest

import folder_paths
from app.model_download import (
    ModelDownloadError,
    ModelDownloadRequest,
    is_allowed_model_download_url,
    normalize_model_relative_path,
    open_model_download_response,
    parse_model_download_request,
    resolve_model_download_destination,
)


class _FakeResponse:
    """Minimal stand-in for ``aiohttp.ClientResponse`` for the redirect tests."""

    def __init__(self, status, headers=None):
        self.status = status
        self.headers = headers or {}
        self.released = False

    def release(self):
        self.released = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.released = True


class _FakeSession:
    """Hands out queued ``_FakeResponse`` objects in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def get(self, url, allow_redirects, timeout):
        self.calls.append((url, allow_redirects))
        if not self._responses:
            raise AssertionError("Unexpected extra session.get call")
        return self._responses.pop(0)


def test_parse_model_download_request_allows_huggingface_model_url():
    request = parse_model_download_request({
        "name": "nested/model.safetensors",
        "url": "https://huggingface.co/org/repo/resolve/main/model.safetensors",
        "directory": "checkpoints",
    })

    assert request == ModelDownloadRequest(
        name="nested/model.safetensors",
        url="https://huggingface.co/org/repo/resolve/main/model.safetensors",
        directory="checkpoints",
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/model.safetensors",
        "http://huggingface.co/org/repo/resolve/main/model.safetensors",
        "https://example.com/model.safetensors",
        "https://huggingface.co.evil.com/model.safetensors",
    ],
)
def test_download_url_allowlist_rejects_untrusted_or_plain_http_urls(url):
    assert is_allowed_model_download_url(url) is False


@pytest.mark.parametrize(
    "url",
    [
        # Direct HF model URLs.
        "https://huggingface.co/org/repo/resolve/main/model.safetensors",
        # HF LFS CDN subdomains: this is where `/resolve/main/...` redirects
        # land, so the allowlist must accept them or downloads break.
        "https://cdn-lfs.huggingface.co/repos/abc/def/model.safetensors",
        "https://cdn-lfs-us-1.huggingface.co/repos/abc/def/model.safetensors",
        # Civitai download endpoints (PR objective: support Civitai too).
        "https://civitai.com/api/download/models/12345",
        "https://civitai.red/api/download/models/12345",
    ],
)
def test_download_url_allowlist_accepts_huggingface_and_civitai_urls(url):
    assert is_allowed_model_download_url(url) is True


@pytest.mark.parametrize(
    "name, expected",
    [
        ("model.safetensors", "model.safetensors"),
        ("sub/model.safetensors", "sub/model.safetensors"),
        ("nested/dir/model.safetensors", "nested/dir/model.safetensors"),
        # Backslashes are normalized to forward slashes so Windows-style
        # paths land in the same place as the POSIX equivalents.
        ("nested\\dir\\model.safetensors", "nested/dir/model.safetensors"),
    ],
)
def test_normalize_model_relative_path_accepts_safe_paths(name, expected):
    assert normalize_model_relative_path(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "../model.safetensors",
        "nested/../../model.safetensors",
        "/absolute/model.safetensors",
        "model.safetensors\x00",
    ],
)
def test_normalize_model_relative_path_rejects_unsafe_paths(name):
    with pytest.raises(ModelDownloadError):
        normalize_model_relative_path(name)


def test_parse_model_download_request_rejects_unsupported_extensions():
    with pytest.raises(ModelDownloadError):
        parse_model_download_request({
            "name": "model.gguf",
            "url": "https://huggingface.co/org/repo/resolve/main/model.gguf",
            "directory": "checkpoints",
        })


def test_resolve_model_download_destination_uses_configured_model_folder(tmp_path, monkeypatch):
    model_root = tmp_path / "models" / "checkpoints"
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", {
        "checkpoints": ([str(model_root)], {".safetensors"}),
    })

    destination = resolve_model_download_destination(ModelDownloadRequest(
        name="sub/model.safetensors",
        url="https://huggingface.co/org/repo/resolve/main/model.safetensors",
        directory="checkpoints",
    ))

    assert destination.directory == "checkpoints"
    assert destination.relative_path == "sub/model.safetensors"
    assert destination.full_path == os.path.join(str(model_root), "sub", "model.safetensors")
    assert destination.already_exists is False


def test_resolve_model_download_destination_reuses_existing_model(tmp_path, monkeypatch):
    model_root = tmp_path / "models" / "checkpoints"
    model_root.mkdir(parents=True)
    existing = model_root / "model.safetensors"
    existing.write_bytes(b"model")
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", {
        "checkpoints": ([str(model_root)], {".safetensors"}),
    })

    destination = resolve_model_download_destination(ModelDownloadRequest(
        name="model.safetensors",
        url="https://huggingface.co/org/repo/resolve/main/model.safetensors",
        directory="checkpoints",
    ))

    assert destination.full_path == str(existing)
    assert destination.already_exists is True


@pytest.mark.parametrize("directory", ["configs", "custom_nodes", "unknown"])
def test_resolve_model_download_destination_rejects_blocked_or_unknown_directories(tmp_path, monkeypatch, directory):
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", {
        "configs": ([str(tmp_path / "configs")], {".yaml"}),
        "custom_nodes": ([str(tmp_path / "custom_nodes")], set()),
    })

    with pytest.raises(ModelDownloadError):
        resolve_model_download_destination(ModelDownloadRequest(
            name="model.safetensors",
            url="https://huggingface.co/org/repo/resolve/main/model.safetensors",
            directory=directory,
        ))


@pytest.mark.asyncio
async def test_open_model_download_response_follows_allowed_subdomain_redirect():
    """HF redirects /resolve/main/... to cdn-lfs.huggingface.co; that must work."""
    session = _FakeSession([
        _FakeResponse(302, {"Location": "https://cdn-lfs.huggingface.co/repos/abc/model.safetensors"}),
        _FakeResponse(200),
    ])

    response = await open_model_download_response(
        session, "https://huggingface.co/org/repo/resolve/main/model.safetensors"
    )

    assert response.status == 200
    assert session.calls == [
        ("https://huggingface.co/org/repo/resolve/main/model.safetensors", False),
        ("https://cdn-lfs.huggingface.co/repos/abc/model.safetensors", False),
    ]


@pytest.mark.asyncio
async def test_open_model_download_response_rejects_offsite_redirect():
    """A redirect leaving the allowlist must surface as a 403 instead of being followed."""
    session = _FakeSession([
        _FakeResponse(302, {"Location": "https://attacker.example.com/payload"}),
    ])

    with pytest.raises(ModelDownloadError) as exc_info:
        await open_model_download_response(
            session, "https://huggingface.co/org/repo/resolve/main/model.safetensors"
        )

    assert exc_info.value.status == 403
    # The initial request was issued with redirects disabled, otherwise
    # the validation above would be a no-op.
    assert session.calls[0][1] is False


@pytest.mark.asyncio
async def test_open_model_download_response_rejects_redirect_without_location():
    session = _FakeSession([_FakeResponse(302)])

    with pytest.raises(ModelDownloadError) as exc_info:
        await open_model_download_response(
            session, "https://huggingface.co/org/repo/resolve/main/model.safetensors"
        )

    assert exc_info.value.status == 502


@pytest.mark.asyncio
async def test_open_model_download_response_stops_after_too_many_redirects():
    session = _FakeSession(
        [_FakeResponse(302, {"Location": "https://cdn-lfs.huggingface.co/loop"})] * 10
    )

    with pytest.raises(ModelDownloadError) as exc_info:
        await open_model_download_response(
            session, "https://huggingface.co/org/repo/resolve/main/model.safetensors"
        )

    assert exc_info.value.status == 502
