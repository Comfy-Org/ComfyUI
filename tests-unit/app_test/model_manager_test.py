import pytest
import base64
import json
import struct
from io import BytesIO
from PIL import Image
from aiohttp import web
from unittest.mock import patch
from app.model_manager import ModelFileManager

pytestmark = (
    pytest.mark.asyncio
)  # This applies the asyncio mark to all test functions in the module

@pytest.fixture
def model_manager():
    return ModelFileManager()

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
