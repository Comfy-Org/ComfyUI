import asyncio
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import PureWindowsPath
from urllib.parse import urljoin, urlsplit

import aiohttp
from aiohttp import web

import folder_paths


MODEL_EXTENSIONS = {'.safetensors', '.sft', '.ckpt', '.pth', '.pt'}
SOURCE_HOSTS = {'huggingface.co', 'civitai.com', 'civitai.red', 'github.com'}
REDIRECT_HOSTS = {'release-assets.githubusercontent.com', 'objects.githubusercontent.com'}
REDIRECT_HOST_SUFFIXES = ('.hf.co', '.civitai.com', '.civitai.red')
R2_DELIVERY_SUFFIX = '.r2.cloudflarestorage.com'
MAX_MODELS_PER_BATCH = 1000
MAX_REDIRECTS = 10
MAX_SAVED_BATCHES = 100


def _valid_download_url(url: str, *, redirect: bool = False) -> bool:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ''
        port = parsed.port
    except ValueError:
        return False
    if parsed.scheme != 'https' or port not in (None, 443) or parsed.username or parsed.password:
        return False
    if host in SOURCE_HOSTS:
        return True
    return redirect and (
        host in REDIRECT_HOSTS
        or any(host.endswith(suffix) for suffix in REDIRECT_HOST_SUFFIXES)
        or (host.startswith('civitai-delivery-worker-prod.') and host.endswith(R2_DELIVERY_SUFFIX))
    )


@dataclass
class ModelDownload:
    name: str
    directory: str
    url: str
    destination: str
    status: str = 'queued'
    bytes_downloaded: int = 0
    bytes_total: int | None = None
    error: str | None = None

    def response(self) -> dict:
        return {
            'name': self.name,
            'directory': self.directory,
            'status': self.status,
            'bytes_downloaded': self.bytes_downloaded,
            'bytes_total': self.bytes_total,
            'error': self.error,
        }


class ModelDownloadManager:
    def __init__(self):
        self.batches: dict[str, list[ModelDownload]] = {}
        self.tasks: set[asyncio.Task] = set()
        self.queue_lock = asyncio.Lock()

    def add_routes(self, routes):
        @routes.post('/models/download')
        async def start_downloads(request):
            try:
                payload = await request.json()
                models = payload['models']
                if not isinstance(models, list) or not 1 <= len(models) <= MAX_MODELS_PER_BATCH:
                    raise ValueError(f'models must contain 1 to {MAX_MODELS_PER_BATCH} entries')
                downloads = [self._parse_model(model) for model in models]
                destinations = [download.destination for download in downloads]
                if len(destinations) != len(set(destinations)):
                    raise ValueError('models must have unique destinations')
            except (KeyError, TypeError, ValueError) as error:
                return web.json_response({'error': str(error)}, status=400)

            for old_id, old_downloads in list(self.batches.items()):
                if len(self.batches) < MAX_SAVED_BATCHES:
                    break
                if all(item.status in ('completed', 'failed') for item in old_downloads):
                    del self.batches[old_id]
            if len(self.batches) >= MAX_SAVED_BATCHES:
                return web.json_response({'error': 'too many active model downloads'}, status=429)

            batch_id = uuid.uuid4().hex
            self.batches[batch_id] = downloads
            task = asyncio.create_task(self._run_batch(downloads))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
            return web.json_response({'batch_id': batch_id, 'models': [item.response() for item in downloads]}, status=202)

        @routes.get('/models/download/{batch_id}')
        async def get_downloads(request):
            batch_id = request.match_info['batch_id']
            downloads = self.batches.get(batch_id)
            if downloads is None:
                return web.Response(status=404)
            return web.json_response({'models': [item.response() for item in downloads]})

    def _parse_model(self, model) -> ModelDownload:
        if not isinstance(model, dict):
            raise ValueError('model must be an object')
        name = model.get('name')
        directory = model.get('directory')
        url = model.get('url')
        if not all(isinstance(value, str) for value in (name, directory, url)):
            raise ValueError('name, directory and url must be strings')
        if (name != os.path.basename(name) or name.startswith('.') or '\\' in name
                or any(ord(char) < 32 for char in name)
                or ':' in name or len(name) > 255 or PureWindowsPath(name).is_reserved()):
            raise ValueError('invalid model filename')
        if os.path.splitext(name)[1].lower() not in MODEL_EXTENSIONS:
            raise ValueError('unsupported model extension')
        if directory not in folder_paths.folder_names_and_paths or directory in ('configs', 'custom_nodes', 'datasets'):
            raise ValueError('unknown model directory')
        if len(url) > 4096 or not _valid_download_url(url):
            raise ValueError('unsupported model URL')
        paths = folder_paths.get_folder_paths(directory)
        if not paths:
            raise ValueError('model directory has no configured path')
        destination = os.path.join(paths[0], name)
        return ModelDownload(name, directory, url, destination)

    async def _run_batch(self, downloads: list[ModelDownload]):
        async with self.queue_lock:
            for download in downloads:
                download.status = 'running'
                try:
                    await self._download(download)
                except Exception as error:
                    logging.exception('Model download failed for %s in %s', download.name, download.directory)
                    download.status = 'failed'
                    download.error = f'HTTP {error.status}' if isinstance(error, aiohttp.ClientResponseError) else str(error)
                else:
                    download.status = 'completed'

    async def _download(self, download: ModelDownload):
        directory = os.path.dirname(download.destination)
        os.makedirs(directory, exist_ok=True)
        if os.path.lexists(download.destination):
            raise ValueError('model file already exists')

        temporary_path = None
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=120)
        try:
            async with aiohttp.ClientSession(timeout=timeout, trust_env=True, auto_decompress=False) as session:
                url = download.url
                for _ in range(MAX_REDIRECTS + 1):
                    async with session.get(url, allow_redirects=False) as response:
                        if response.status in (301, 302, 303, 307, 308):
                            location = response.headers.get('Location')
                            if not location:
                                raise ValueError('model URL redirected without a location')
                            url = urljoin(url, location)
                            if not _valid_download_url(url, redirect=True):
                                raise ValueError('model URL redirected to an unsupported host')
                            continue
                        response.raise_for_status()
                        if response.content_type.startswith('text/') or response.content_type.endswith('json'):
                            raise ValueError('model URL did not return a model file')
                        download.bytes_total = response.content_length
                        with tempfile.NamedTemporaryFile(dir=directory, prefix='.model-download-', delete=False) as temporary:
                            temporary_path = temporary.name
                            async for chunk in response.content.iter_chunked(1024 * 1024):
                                await asyncio.to_thread(temporary.write, chunk)
                                download.bytes_downloaded += len(chunk)
                        if download.bytes_total is not None and download.bytes_downloaded != download.bytes_total:
                            raise ValueError('model download ended before the expected size')
                        if download.bytes_downloaded == 0:
                            raise ValueError('model download was empty')
                        os.link(temporary_path, download.destination)
                        os.unlink(temporary_path)
                        temporary_path = None
                        return
                raise ValueError('model URL redirected too many times')
        finally:
            if temporary_path is not None:
                os.unlink(temporary_path)
