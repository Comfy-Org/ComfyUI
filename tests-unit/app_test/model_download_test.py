import asyncio

import pytest
from aiohttp import web

import folder_paths
from app import model_download
from app.model_download import ModelDownloadManager, _valid_download_url


@pytest.fixture
def manager_app():
    manager = ModelDownloadManager()
    app = web.Application()
    routes = web.RouteTableDef()
    manager.add_routes(routes)
    app.add_routes(routes)
    return manager, app


@pytest.mark.asyncio
async def test_download_batch_installs_each_model_and_keeps_failures_independent(
    aiohttp_client, manager_app, monkeypatch, tmp_path
):
    manager, app = manager_app
    model_folder = tmp_path / 'models'
    monkeypatch.setattr(folder_paths, 'folder_names_and_paths', {
        'diffusion_models': ([str(model_folder)], {'.safetensors'})
    })
    monkeypatch.setattr(model_download, '_valid_download_url', lambda url, redirect=False: True)

    source = web.Application()

    async def serve_model(request):
        name = request.match_info['name']
        if name == 'failed.safetensors':
            return web.Response(status=403)
        return web.Response(body=name.encode(), content_type='application/octet-stream')

    source.router.add_get('/{name}', serve_model)
    source_client = await aiohttp_client(source)
    client = await aiohttp_client(app)
    names = ['first.safetensors', 'failed.safetensors', 'last.safetensors']
    response = await client.post('/models/download', json={
        'models': [
            {'name': name, 'directory': 'diffusion_models', 'url': str(source_client.make_url(f'/{name}'))}
            for name in names
        ]
    })

    assert response.status == 202
    batch_id = (await response.json())['batch_id']
    await asyncio.gather(*tuple(manager.tasks))
    status = await client.get(f'/models/download/{batch_id}')
    assert status.status == 200
    models = (await status.json())['models']
    assert [model['status'] for model in models] == ['completed', 'failed', 'completed']
    assert models[1]['error'] == 'HTTP 403'
    assert (model_folder / names[0]).read_bytes() == names[0].encode()
    assert not (model_folder / names[1]).exists()
    assert (model_folder / names[2]).read_bytes() == names[2].encode()
    assert list(model_folder.glob('.model-download-*')) == []


@pytest.mark.asyncio
async def test_download_does_not_replace_an_existing_model(
    aiohttp_client, manager_app, monkeypatch, tmp_path
):
    manager, app = manager_app
    model_folder = tmp_path / 'models'
    model_folder.mkdir()
    existing = model_folder / 'existing.safetensors'
    existing.write_bytes(b'original')
    monkeypatch.setattr(folder_paths, 'folder_names_and_paths', {
        'vae': ([str(model_folder)], {'.safetensors'})
    })
    monkeypatch.setattr(model_download, '_valid_download_url', lambda url, redirect=False: True)
    client = await aiohttp_client(app)

    response = await client.post('/models/download', json={
        'models': [{
            'name': existing.name,
            'directory': 'vae',
            'url': 'https://huggingface.co/org/repo/resolve/main/existing.safetensors',
        }]
    })
    batch_id = (await response.json())['batch_id']
    await asyncio.gather(*tuple(manager.tasks))
    status = await client.get(f'/models/download/{batch_id}')

    assert (await status.json())['models'][0]['status'] == 'failed'
    assert existing.read_bytes() == b'original'


@pytest.mark.parametrize('model', [
    {'name': '../escape.safetensors', 'directory': 'vae', 'url': 'https://huggingface.co/a/b'},
    {'name': r'C:\escape.safetensors', 'directory': 'vae', 'url': 'https://huggingface.co/a/b'},
    {'name': 'CON.safetensors', 'directory': 'vae', 'url': 'https://huggingface.co/a/b'},
    {'name': 'model.exe', 'directory': 'vae', 'url': 'https://huggingface.co/a/b'},
    {'name': 'model.safetensors', 'directory': 'custom_nodes', 'url': 'https://huggingface.co/a/b'},
    {'name': 'model.safetensors', 'directory': 'vae', 'url': 'http://127.0.0.1/file'},
])
@pytest.mark.asyncio
async def test_download_batch_rejects_untrusted_inputs(
    aiohttp_client, manager_app, monkeypatch, tmp_path, model
):
    manager, app = manager_app
    monkeypatch.setattr(folder_paths, 'folder_names_and_paths', {
        'vae': ([str(tmp_path)], {'.safetensors'}),
        'custom_nodes': ([str(tmp_path)], set()),
    })
    client = await aiohttp_client(app)

    response = await client.post('/models/download', json={'models': [model]})

    assert response.status == 400
    assert manager.batches == {}
    assert list(tmp_path.iterdir()) == []


def test_model_download_redirects_stay_on_known_hosts():
    assert _valid_download_url('https://huggingface.co/a/b')
    assert _valid_download_url('https://github.com/org/repo/releases/download/v1/model.safetensors')
    assert _valid_download_url('https://cas-bridge.xethub.hf.co/a', redirect=True)
    assert _valid_download_url('https://release-assets.githubusercontent.com/a', redirect=True)
    assert _valid_download_url('https://civitai-delivery-worker-prod.account.r2.cloudflarestorage.com/a', redirect=True)
    assert not _valid_download_url('https://cas-bridge.xethub.hf.co/a')
    assert not _valid_download_url('https://other-worker.account.r2.cloudflarestorage.com/a', redirect=True)
    assert not _valid_download_url('https://127.0.0.1/a', redirect=True)
    assert not _valid_download_url('http://huggingface.co/a')
