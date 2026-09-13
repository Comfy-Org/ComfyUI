import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

from app.assets.api import routes


def make_request(remote="127.0.0.1", headers=None, query=None):
    return SimpleNamespace(
        match_info={"id": str(uuid.uuid4())},
        remote=remote,
        headers=headers or {},
        query=query or {},
    )


def run_route(request):
    return asyncio.run(routes.open_asset_location_route.__wrapped__(request))


def run_delete_route(request):
    return asyncio.run(routes.delete_asset_route.__wrapped__(request))


def test_open_location_rejects_lan_clients(monkeypatch):
    get_detail = Mock()
    monkeypatch.setattr(routes, "get_asset_detail", get_detail)

    response = run_route(make_request(remote="192.168.1.50"))

    assert response.status == 403
    assert json.loads(response.text)["error"]["code"] == "LOCAL_ACCESS_REQUIRED"
    get_detail.assert_not_called()


def test_open_location_rejects_cross_site_requests(monkeypatch):
    get_detail = Mock()
    monkeypatch.setattr(routes, "get_asset_detail", get_detail)

    response = run_route(make_request(headers={"Sec-Fetch-Site": "cross-site"}))

    assert response.status == 403
    assert json.loads(response.text)["error"]["code"] == "CROSS_SITE_REQUEST_FORBIDDEN"
    get_detail.assert_not_called()


def test_open_location_reveals_managed_generated_file(tmp_path, monkeypatch):
    source = tmp_path / "output" / "video" / "render.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    reveal = Mock()

    monkeypatch.setattr(
        routes,
        "USER_MANAGER",
        SimpleNamespace(get_request_user_id=lambda _request: "default"),
    )
    monkeypatch.setattr(
        routes,
        "get_asset_detail",
        lambda **_kwargs: SimpleNamespace(
            tags=["output"], ref=SimpleNamespace(file_path=str(source))
        ),
    )
    monkeypatch.setattr(
        routes.folder_paths,
        "get_output_directory",
        lambda: str(tmp_path / "output"),
    )
    monkeypatch.setattr(routes, "reveal_file_in_file_manager", reveal)

    response = run_route(make_request())

    assert response.status == 204
    reveal.assert_called_once_with(str(source))


def test_open_location_returns_structured_error_for_unknown_user(monkeypatch):
    monkeypatch.setattr(
        routes,
        "USER_MANAGER",
        SimpleNamespace(get_request_user_id=Mock(side_effect=KeyError("unknown"))),
    )

    response = run_route(make_request())

    assert response.status == 500
    assert json.loads(response.text)["error"]["code"] == "INTERNAL"


def test_open_location_rejects_non_output_assets(tmp_path, monkeypatch):
    source = tmp_path / "output" / "image.png"
    source.parent.mkdir()
    source.write_bytes(b"image")

    monkeypatch.setattr(
        routes,
        "USER_MANAGER",
        SimpleNamespace(get_request_user_id=lambda _request: "default"),
    )
    monkeypatch.setattr(
        routes,
        "get_asset_detail",
        lambda **_kwargs: SimpleNamespace(
            tags=["input"], ref=SimpleNamespace(file_path=str(source))
        ),
    )
    monkeypatch.setattr(
        routes.folder_paths,
        "get_output_directory",
        lambda: str(tmp_path / "output"),
    )
    reveal = Mock()
    monkeypatch.setattr(routes, "reveal_file_in_file_manager", reveal)

    response = run_route(make_request())

    assert response.status == 403
    assert json.loads(response.text)["error"]["code"] == "ASSET_LOCATION_FORBIDDEN"
    reveal.assert_not_called()


def test_open_location_rejects_output_asset_outside_output_root(
    tmp_path, monkeypatch
):
    source = tmp_path / "outside" / "image.png"
    source.parent.mkdir()
    source.write_bytes(b"image")

    monkeypatch.setattr(
        routes,
        "USER_MANAGER",
        SimpleNamespace(get_request_user_id=lambda _request: "default"),
    )
    monkeypatch.setattr(
        routes,
        "get_asset_detail",
        lambda **_kwargs: SimpleNamespace(
            tags=["output"], ref=SimpleNamespace(file_path=str(source))
        ),
    )
    monkeypatch.setattr(
        routes.folder_paths,
        "get_output_directory",
        lambda: str(tmp_path / "output"),
    )
    reveal = Mock()
    monkeypatch.setattr(routes, "reveal_file_in_file_manager", reveal)

    response = run_route(make_request())

    assert response.status == 403
    assert json.loads(response.text)["error"]["code"] == "ASSET_LOCATION_FORBIDDEN"
    reveal.assert_not_called()


def test_delete_content_rejects_paths_outside_managed_root(tmp_path, monkeypatch):
    source = tmp_path / "outside" / "render.png"
    source.parent.mkdir()
    source.write_bytes(b"image")
    delete_with_file = Mock()

    monkeypatch.setattr(
        routes,
        "USER_MANAGER",
        SimpleNamespace(get_request_user_id=lambda _request: "default"),
    )
    monkeypatch.setattr(
        routes,
        "get_asset_detail",
        lambda **_kwargs: SimpleNamespace(
            tags=["output"], ref=SimpleNamespace(file_path=str(source))
        ),
    )
    monkeypatch.setattr(
        routes.folder_paths,
        "get_output_directory",
        lambda: str(tmp_path / "output"),
    )
    monkeypatch.setattr(routes, "delete_asset_reference_with_file", delete_with_file)

    response = run_delete_route(make_request(query={"delete_content": "true"}))

    assert response.status == 403
    assert json.loads(response.text)["error"]["code"] == "ASSET_DELETE_FORBIDDEN"
    delete_with_file.assert_not_called()


def test_delete_content_uses_guarded_file_service(tmp_path, monkeypatch):
    source = tmp_path / "output" / "render.png"
    source.parent.mkdir()
    source.write_bytes(b"image")
    delete_with_file = Mock(return_value=True)

    monkeypatch.setattr(
        routes,
        "USER_MANAGER",
        SimpleNamespace(get_request_user_id=lambda _request: "default"),
    )
    monkeypatch.setattr(
        routes,
        "get_asset_detail",
        lambda **_kwargs: SimpleNamespace(
            tags=["output"], ref=SimpleNamespace(file_path=str(source))
        ),
    )
    monkeypatch.setattr(
        routes.folder_paths,
        "get_output_directory",
        lambda: str(tmp_path / "output"),
    )
    monkeypatch.setattr(
        routes.folder_paths,
        "get_temp_directory",
        lambda: str(tmp_path / "temp"),
    )
    monkeypatch.setattr(routes.user_manager.args, "multi_user", False)
    monkeypatch.setattr(routes, "delete_asset_reference_with_file", delete_with_file)

    request = make_request(query={"delete_content": "true"})
    response = run_delete_route(request)

    assert response.status == 204
    delete_with_file.assert_called_once_with(
        reference_id=request.match_info["id"],
        owner_id="default",
        staging_directory=str(tmp_path / "temp"),
        expected_file_path=str(source),
        allowed_directories=[str(tmp_path / "output")],
        allow_ownerless=True,
    )


def test_delete_returns_structured_error_for_unknown_user(monkeypatch):
    delete_reference = Mock()
    monkeypatch.setattr(
        routes,
        "USER_MANAGER",
        SimpleNamespace(get_request_user_id=Mock(side_effect=KeyError("unknown"))),
    )
    monkeypatch.setattr(routes, "delete_asset_reference", delete_reference)

    response = run_delete_route(make_request())

    assert response.status == 500
    assert json.loads(response.text)["error"]["code"] == "INTERNAL"
    delete_reference.assert_not_called()


def test_delete_does_not_expose_os_permission_error_path(tmp_path, monkeypatch):
    source = tmp_path / "output" / "private" / "render.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"image")
    delete_with_file = Mock(
        side_effect=PermissionError(13, "Permission denied", str(source))
    )
    monkeypatch.setattr(
        routes,
        "USER_MANAGER",
        SimpleNamespace(get_request_user_id=lambda _request: "default"),
    )
    monkeypatch.setattr(
        routes,
        "get_asset_detail",
        lambda **_kwargs: SimpleNamespace(
            tags=["output"], ref=SimpleNamespace(file_path=str(source))
        ),
    )
    monkeypatch.setattr(
        routes.folder_paths,
        "get_output_directory",
        lambda: str(tmp_path / "output"),
    )
    monkeypatch.setattr(
        routes.folder_paths,
        "get_temp_directory",
        lambda: str(tmp_path / "temp"),
    )
    monkeypatch.setattr(routes.user_manager.args, "multi_user", False)
    monkeypatch.setattr(routes, "delete_asset_reference_with_file", delete_with_file)

    response = run_delete_route(make_request(query={"delete_content": "true"}))

    assert response.status == 500
    assert json.loads(response.text)["error"]["code"] == "INTERNAL"
    assert str(source) not in response.text
