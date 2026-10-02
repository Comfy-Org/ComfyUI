import pytest
import asyncio
import aiohttp
import base64
import json
import struct
from io import BytesIO
from PIL import Image
from aiohttp import web
from unittest.mock import patch
from app.model_manager import (
    ModelFileManager, ModelDownloadResolver, _download_file,
    _is_model_download_allowed, _resolve_download_destination, _validate_model_url,
)

pytestmark = (
    pytest.mark.asyncio
)  # This applies the asyncio mark to all test functions in the module

class DummyPromptServer:
    def __init__(self):
        self.events = asyncio.Queue()

    def send_sync(self, event, payload, client_id):
        self.events.put_nowait((event, payload, client_id))

@pytest.fixture
def prompt_server():
    return DummyPromptServer()

@pytest.fixture
def model_manager(prompt_server):
    return ModelFileManager(prompt_server.send_sync)

@pytest.fixture
def app(model_manager):
    app = web.Application()
    routes = web.RouteTableDef()
    model_manager.add_routes(routes)
    app.add_routes(routes)
    return app

async def test_get_model_folders_includes_registered_extensions(aiohttp_client, app, tmp_path):
    """Folders expose their registered extension set verbatim; an empty list
    means match-all (filter_files_extensions semantics)."""
    with patch('folder_paths.folder_names_and_paths', {
        'test_checkpoints': ([str(tmp_path)], {'.safetensors', '.ckpt'}),
        'test_configs': ([str(tmp_path)], ['.yaml']),
        'test_match_all': ([str(tmp_path)], set()),
        'configs': ([str(tmp_path)], ['.yaml']),
    }):
        client = await aiohttp_client(app)
        response = await client.get('/experiment/models')

        assert response.status == 200
        folders = {f['name']: f for f in await response.json()}

        assert 'configs' not in folders  # blocklisted
        assert folders['test_checkpoints']['folders'] == [str(tmp_path)]
        assert folders['test_checkpoints']['extensions'] == ['.ckpt', '.safetensors']
        assert folders['test_configs']['extensions'] == ['.yaml']
        # Match-all registrations are exposed honestly, not substituted.
        assert folders['test_match_all']['extensions'] == []

async def test_get_model_preview_safetensors(aiohttp_client, app, tmp_path):
    img = Image.new('RGB', (100, 100), 'white')
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    img_b64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    safetensors_file = tmp_path / "test_model.safetensors"
    header_bytes = json.dumps({
        "__metadata__": {
            "ssmd_cover_images": json.dumps([img_b64])
        }
    }).encode('utf-8')
    length_bytes = struct.pack('<Q', len(header_bytes))
    with open(safetensors_file, 'wb') as f:
        f.write(length_bytes)
        f.write(header_bytes)

    with patch('folder_paths.folder_names_and_paths', {
        'test_folder': ([str(tmp_path)], None)
    }):
        client = await aiohttp_client(app)
        response = await client.get('/experiment/models/preview/test_folder/0/test_model.safetensors')

        # Verify response
        assert response.status == 200
        assert response.content_type == 'image/webp'

        # Verify the response contains valid image data
        img_bytes = BytesIO(await response.read())
        img = Image.open(img_bytes)
        assert img.format
        assert img.format.lower() == 'webp'

        # Clean up
        img.close()


@pytest.mark.parametrize("payload", [None, [], {}, {"models": []},
    {"models": [{}], "client_id": None, "batch_id": "batch"},
    {"models": [{}], "client_id": "client", "batch_id": None},
    {"models": [{"name": None}], "client_id": "client", "batch_id": "batch"}])
async def test_download_rejects_malformed_request(aiohttp_client, app, payload):
    client = await aiohttp_client(app)
    response = await client.post('/experiment/models/download_missing', json=payload)
    assert response.status == 400
    assert 'error' in await response.json()


@pytest.mark.parametrize("url", [
    'http://localhost:8188/model.safetensors',
    'https://127.0.0.1/model.safetensors',
    'https://huggingface.co.evil.example/model.safetensors',
    'https://user:secret@huggingface.co/model.safetensors',
    'https://huggingface.co:8443/model.safetensors',
    'https://evil.example/model.safetensors',
])
async def test_model_url_policy_covers_redirects(url):
    with pytest.raises(ValueError):
        _validate_model_url(url, redirect=True)


async def test_model_source_and_file_policy():
    assert _is_model_download_allowed('model.safetensors', 'https://huggingface.co/repo/model.safetensors')[0]
    assert not _is_model_download_allowed('model.pkl', 'https://huggingface.co/repo/model.pkl')[0]
    assert not _is_model_download_allowed('plugin.py', 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth')[0]
    _validate_model_url('https://cas-bridge.xethub.hf.co/model', redirect=True)


async def test_model_download_resolver_rejects_private_addresses(monkeypatch):
    async def resolve(*args):
        return [{"host": "127.0.0.1"}]
    monkeypatch.setattr(aiohttp.ThreadedResolver, 'resolve', resolve)
    resolver = ModelDownloadResolver()
    try:
        with pytest.raises(OSError, match='non-public'):
            await resolver.resolve('huggingface.co')
    finally:
        await resolver.close()


async def test_download_destination_contains_paths_and_checks_all_roots(tmp_path, monkeypatch):
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    (second / 'model.safetensors').write_bytes(b'existing')
    monkeypatch.setattr('folder_paths.folder_names_and_paths', {
        'text_encoders': ([str(first), str(second)], {'.safetensors'}),
        'custom_nodes': ([str(first)], {'.safetensors'}),
    })
    assert _resolve_download_destination('clip', 'model.safetensors') == str(second / 'model.safetensors')
    assert _resolve_download_destination('clip', 'sub/new.safetensors') == str(first / 'sub/new.safetensors')
    for name in ['../escape.safetensors', r'..\escape.safetensors', '/absolute.safetensors', r'C:\escape.safetensors']:
        with pytest.raises(ValueError):
            _resolve_download_destination('clip', name)
    with pytest.raises(ValueError):
        _resolve_download_destination('custom_nodes', 'model.safetensors')
    (first / 'outside').symlink_to(second, target_is_directory=True)
    with pytest.raises(ValueError):
        _resolve_download_destination('clip', 'outside/new.safetensors')


async def test_download_batch_places_skips_and_reports_files(aiohttp_client, aiohttp_server, app, model_manager, prompt_server, tmp_path, monkeypatch):
    content = b'small model fixture'
    source = web.Application()
    async def serve_model(request):
        return web.Response(body=content)
    source.router.add_get('/model', serve_model)
    source_server = await aiohttp_server(source)
    monkeypatch.setattr('app.model_manager._validate_model_url', lambda *args, **kwargs: None)
    monkeypatch.setattr('folder_paths.folder_names_and_paths', {'checkpoints': ([str(tmp_path)], {'.safetensors'})})
    (tmp_path / 'existing.safetensors').write_bytes(b'keep me')
    model = {'name': 'sub/model.safetensors', 'directory': 'checkpoints', 'url': str(source_server.make_url('/model'))}
    client = await aiohttp_client(app)
    response = await client.post('/experiment/models/download_missing', json={
        'models': [model, model, {**model, 'name': 'existing.safetensors'}, {**model, 'name': '../escape.safetensors'}],
        'client_id': 'client', 'batch_id': 'batch',
    })
    result = await response.json()
    assert response.status == 200
    assert (result['downloaded'], result['skipped'], result['failed']) == (1, 1, 1)
    assert (tmp_path / 'sub/model.safetensors').read_bytes() == content
    assert (tmp_path / 'existing.safetensors').read_bytes() == b'keep me'
    assert not list(tmp_path.rglob('*.temp'))
    events = []
    while not prompt_server.events.empty():
        event, data, sid = prompt_server.events.get_nowait()
        assert event == 'missing_model_download'
        assert sid == 'client'
        assert data['batch_id'] == 'batch'
        events.append(data)
    assert any(e['status'] == 'running' for e in events)
    assert any(e['status'] == 'completed' and e['bytes_downloaded'] == len(content) for e in events)


async def test_cancel_stalled_transfer_is_scoped_and_cleans_up(aiohttp_client, aiohttp_server, app, model_manager, prompt_server, tmp_path, monkeypatch):
    release = asyncio.Event()
    async def stalled(request):
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b'partial')
        await release.wait()
        return response
    source = web.Application()
    source.router.add_get('/model', stalled)
    source_server = await aiohttp_server(source)
    monkeypatch.setattr('app.model_manager._validate_model_url', lambda *args, **kwargs: None)
    monkeypatch.setattr('folder_paths.folder_names_and_paths', {'checkpoints': ([str(tmp_path)], {'.safetensors'})})
    model = {'name': 'model.safetensors', 'directory': 'checkpoints', 'url': str(source_server.make_url('/model'))}
    client = await aiohttp_client(app)
    pending = asyncio.create_task(client.post('/experiment/models/download_missing', json={'models': [model], 'client_id': 'client', 'batch_id': 'batch'}))
    try:
        _, data, _ = await asyncio.wait_for(prompt_server.events.get(), 5)
        payload = {'task_id': data['task_id'], 'client_id': 'other', 'batch_id': 'batch'}
        rejected = await client.post('/experiment/models/download_missing/cancel', json=payload)
        assert rejected.status == 404
        concurrent = await client.post('/experiment/models/download_missing', json={'models': [model], 'client_id': 'client2', 'batch_id': 'batch2'})
        assert (await concurrent.json())['failed'] == 1
        response = await client.post('/experiment/models/download_missing/cancel', json={**payload, 'client_id': 'client'})
        assert response.status == 200
        result = await (await asyncio.wait_for(pending, 5)).json()
        assert result['canceled'] == 1
        assert not (tmp_path / 'model.safetensors').exists()
        assert not list(tmp_path.glob('*.temp'))
        assert not model_manager._model_downloads
        assert not model_manager._download_destinations
        response = await client.post('/experiment/models/download_missing/cancel', json={**payload, 'client_id': 'client'})
        assert response.status == 404
    finally:
        release.set()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.parametrize('scenario', ['redirect', 'truncated', 'http_error', 'existing'])
async def test_transfer_failure_never_installs_partial_or_overwrites(aiohttp_server, tmp_path, monkeypatch, scenario):
    async def source(request):
        if scenario == 'redirect':
            raise web.HTTPFound('https://evil.example/model.safetensors')
        if scenario == 'http_error':
            raise web.HTTPForbidden()
        response = web.StreamResponse(headers={'Content-Length': '100'} if scenario == 'truncated' else {})
        await response.prepare(request)
        await response.write(b'partial')
        if scenario == 'truncated':
            request.transport.close()
        return response
    app = web.Application()
    app.router.add_get('/model', source)
    server = await aiohttp_server(app)
    source_url = str(server.make_url('/model'))
    def validate(url, **kwargs):
        if url != source_url:
            _validate_model_url(url, **kwargs)
    monkeypatch.setattr('app.model_manager._validate_model_url', validate)
    destination = tmp_path / 'model.safetensors'
    if scenario == 'existing':
        destination.write_bytes(b'original')
    async def progress(size):
        pass
    async with aiohttp.ClientSession() as session:
        with pytest.raises((ValueError, aiohttp.ClientError, FileExistsError)):
            await _download_file(session, source_url, str(destination), progress)
    if scenario == 'existing':
        assert destination.read_bytes() == b'original'
    else:
        assert not destination.exists()
    assert not list(tmp_path.glob('*.temp'))
