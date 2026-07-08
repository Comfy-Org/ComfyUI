import errno
import os
import sys
import asyncio
import traceback
import time

import nodes
import folder_paths
import execution
from comfy_execution.jobs import (
    JobStatus,
    get_job,
    get_all_jobs,
    validate_job_id,
    cancel_job,
    CANCEL_PENDING,
    CANCEL_RUNNING,
)
import uuid
import urllib
import json
import glob
import struct
import ssl
import socket
import ipaddress
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
from io import BytesIO

import aiohttp
from aiohttp import web
import logging

import mimetypes
from comfy.cli_args import args
from comfy.deploy_environment import get_deploy_environment
import comfy.utils
import comfy.model_management
from comfy_api import feature_flags
import node_helpers
from comfyui_version import __version__
from app.frontend_management import FrontendManager, parse_version
from comfy_api.internal import _ComfyNodeInternal
from app.assets.seeder import asset_seeder
from app.assets.api.routes import register_assets_routes
from app.assets.services.ingest import register_file_in_place
from app.assets.services.asset_management import resolve_hash_to_path

from app.user_manager import UserManager
from app.model_manager import ModelFileManager
from app.custom_node_manager import CustomNodeManager
from app.subgraph_manager import SubgraphManager
from app.node_replace_manager import NodeReplaceManager
from typing import Optional, Union
from api_server.routes.internal.internal_routes import InternalRoutes
from protocol import BinaryEventTypes

# Import cache control middleware
from middleware.cache_middleware import cache_control

if args.enable_manager:
    import comfyui_manager


def _remove_sensitive_from_queue(queue: list) -> list:
    """Remove sensitive data (index 5) from queue item tuples."""
    return [item[:5] for item in queue]


async def send_socket_catch_exception(function, message):
    try:
        await function(message)
    except (aiohttp.ClientError, aiohttp.ClientPayloadError, ConnectionResetError, BrokenPipeError, ConnectionError) as err:
        logging.warning("send error: {}".format(err))

# Track deprecated paths that have been warned about to only warn once per file
_deprecated_paths_warned = set()

@web.middleware
async def deprecation_warning(request: web.Request, handler):
    """Middleware to warn about deprecated frontend API paths"""
    path = request.path

    if path.startswith("/scripts/ui") or path.startswith("/extensions/core/"):
        # Only warn once per unique file path
        if path not in _deprecated_paths_warned:
            _deprecated_paths_warned.add(path)
            logging.warning(
                f"[DEPRECATION WARNING] Detected import of deprecated legacy API: {path}. "
                f"This is likely caused by a custom node extension using outdated APIs. "
                f"Please update your extensions or contact the extension author for an updated version."
            )

    response: web.Response = await handler(request)
    return response


@web.middleware
async def compress_body(request: web.Request, handler):
    accept_encoding = request.headers.get("Accept-Encoding", "")
    response: web.Response = await handler(request)
    if not isinstance(response, web.Response):
        return response
    if response.content_type not in ["application/json", "text/plain"]:
        return response
    if response.body and "gzip" in accept_encoding:
        response.enable_compression()
    return response


def create_cors_middleware(allowed_origin: str):
    @web.middleware
    async def cors_middleware(request: web.Request, handler):
        if request.method == "OPTIONS":
            # Pre-flight request. Reply successfully:
            response = web.Response()
        else:
            response = await handler(request)

        response.headers['Access-Control-Allow-Origin'] = allowed_origin
        response.headers['Access-Control-Allow-Methods'] = 'POST, GET, DELETE, PUT, OPTIONS, PATCH'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response

    return cors_middleware

def is_loopback(host):
    if host is None:
        return False
    try:
        if ipaddress.ip_address(host).is_loopback:
            return True
        else:
            return False
    except:
        pass

    loopback = False
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            r = socket.getaddrinfo(host, None, family, socket.SOCK_STREAM)
            for family, _, _, _, sockaddr in r:
                if not ipaddress.ip_address(sockaddr[0]).is_loopback:
                    return loopback
                else:
                    loopback = True
        except socket.gaierror:
            pass

    return loopback


def create_origin_only_middleware():
    @web.middleware
    async def origin_only_middleware(request: web.Request, handler):
        if 'Sec-Fetch-Site' in request.headers:
            sec_fetch_site = request.headers['Sec-Fetch-Site']
            if sec_fetch_site == 'cross-site':
                return web.Response(status=403)
        #this code is used to prevent the case where a random website can queue comfy workflows by making a POST to 127.0.0.1 which browsers don't prevent for some dumb reason.
        #in that case the Host and Origin hostnames won't match
        #I know the proper fix would be to add a cookie but this should take care of the problem in the meantime
        if 'Host' in request.headers and 'Origin' in request.headers:
            host = request.headers['Host']
            origin = request.headers['Origin']
            host_domain = host.lower()
            parsed = urllib.parse.urlparse(origin)
            origin_domain = parsed.netloc.lower()
            host_domain_parsed = urllib.parse.urlsplit('//' + host_domain)

            #limit the check to when the host domain is localhost, this makes it slightly less safe but should still prevent the exploit
            loopback = is_loopback(host_domain_parsed.hostname)

            if parsed.port is None: #if origin doesn't have a port strip it from the host to handle weird browsers, same for host
                host_domain = host_domain_parsed.hostname
            if host_domain_parsed.port is None:
                origin_domain = parsed.hostname

            if loopback and host_domain is not None and origin_domain is not None and len(host_domain) > 0 and len(origin_domain) > 0:
                if host_domain != origin_domain:
                    logging.warning("WARNING: request with non matching host and origin {} != {}, returning 403".format(host_domain, origin_domain))
                    return web.Response(status=403)

        if request.method == "OPTIONS":
            response = web.Response()
        else:
            response = await handler(request)

        return response

    return origin_only_middleware


def create_block_external_middleware():
    @web.middleware
    async def block_external_middleware(request: web.Request, handler):
        if request.method == "OPTIONS":
            # Pre-flight request. Reply successfully:
            response = web.Response()
        else:
            response = await handler(request)

        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' blob:; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self' data:; frame-src 'self'; object-src 'self';"
        return response

    return block_external_middleware


class PromptServer():
    def __init__(self, loop):
        PromptServer.instance = self

        self.user_manager = UserManager()
        self.model_file_manager = ModelFileManager()
        self.custom_node_manager = CustomNodeManager()
        self.subgraph_manager = SubgraphManager()
        self.node_replace_manager = NodeReplaceManager()
        self.internal_routes = InternalRoutes(self)
        self.supports = ["custom_nodes_from_web"]
        self.prompt_queue = execution.PromptQueue(self)
        self.loop = loop
        self.messages = asyncio.Queue()
        self.client_session:Optional[aiohttp.ClientSession] = None
        self.number = 0

        middlewares = [cache_control, deprecation_warning]
        if args.enable_compress_response_body:
            middlewares.append(compress_body)

        if args.enable_cors_header:
            middlewares.append(create_cors_middleware(args.enable_cors_header))
        else:
            middlewares.append(create_origin_only_middleware())

        if args.disable_api_nodes:
            middlewares.append(create_block_external_middleware())

        if args.enable_manager:
            middlewares.append(comfyui_manager.create_middleware())

        max_upload_size = round(args.max_upload_size * 1024 * 1024)
        self.app = web.Application(client_max_size=max_upload_size, middlewares=middlewares)
        self.sockets = dict()
        self.sockets_metadata = dict()
        self.web_root = (
            FrontendManager.init_frontend(args.front_end_version)
            if args.front_end_root is None
            else args.front_end_root
        )
        logging.info(f"[Prompt Server] web root: {self.web_root}")
        if args.enable_assets:
            register_assets_routes(self.app, self.user_manager)
        else:
            register_assets_routes(self.app)
            asset_seeder.disable()
        routes = web.RouteTableDef()
        self.routes = routes
        self.last_node_id = None
        self.client_id = None

        self.on_prompt_handlers = []

        @routes.get('/ws')
        async def websocket_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            sid = request.rel_url.query.get('clientId', '')
            if sid:
                # Reusing existing session, remove old
                self.sockets.pop(sid, None)
            else:
                sid = uuid.uuid4().hex

            # Store WebSocket for backward compatibility
            self.sockets[sid] = ws
            # Store metadata separately
            self.sockets_metadata[sid] = {"feature_flags": {}}

            try:
                # Send initial state to the new client
                await self.send("status", {"status": self.get_queue_info(), "sid": sid}, sid)
                # On reconnect if we are the currently executing client send the current node
                if self.client_id == sid and self.last_node_id is not None:
                    await self.send("executing", { "node": self.last_node_id }, sid)

                # Flag to track if we've received the first message
                first_message = True

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.ERROR:
                        logging.warning('ws connection closed with exception %s' % ws.exception())
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            data = json.loads(msg.data)
                            # Check if first message is feature flags
                            if first_message and data.get("type") == "feature_flags":
                                # Store client feature flags
                                client_flags = data.get("data", {})
                                self.sockets_metadata[sid]["feature_flags"] = client_flags

                                # Send server feature flags in response
                                await self.send(
                                    "feature_flags",
                                    feature_flags.get_server_features(),
                                    sid,
                                )

                                logging.debug(
                                    f"Feature flags negotiated for client {sid}: {client_flags}"
                                )
                            first_message = False
                        except json.JSONDecodeError:
                            logging.warning(
                                f"Invalid JSON received from client {sid}: {msg.data}"
                            )
                        except Exception as e:
                            logging.error(f"Error processing WebSocket message: {e}")
            finally:
                self.sockets.pop(sid, None)
                self.sockets_metadata.pop(sid, None)
            return ws

        @routes.get("/")
        async def get_root(request):
            response = web.FileResponse(os.path.join(self.web_root, "index.html"))
            response.headers['Cache-Control'] = 'no-store, must-revalidate'
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response

        @routes.get("/embeddings")
        def get_embeddings(request):
            embeddings = folder_paths.get_filename_list("embeddings")
            return web.json_response(list(map(lambda a: os.path.splitext(a)[0], embeddings)))

        @routes.get("/models")
        def list_model_types(request):
            model_types = list(folder_paths.folder_names_and_paths.keys())

            return web.json_response(model_types)

        @routes.get("/models/{folder}")
        async def get_models(request):
            folder = request.match_info.get("folder", None)
            if folder not in folder_paths.folder_names_and_paths:
                return web.Response(status=404)
            files = folder_paths.get_filename_list(folder)
            return web.json_response(files)

        @routes.get("/extensions")
        async def get_extensions(request):
            files = glob.glob(os.path.join(
                glob.escape(self.web_root), 'extensions/**/*.js'), recursive=True)

            extensions = list(map(lambda f: "/" + os.path.relpath(f, self.web_root).replace("\\", "/"), files))

            for name, dir in nodes.EXTENSION_WEB_DIRS.items():
                files = glob.glob(os.path.join(glob.escape(dir), '**/*.js'), recursive=True)
                extensions.extend(list(map(lambda f: "/extensions/" + urllib.parse.quote(
                    name) + "/" + os.path.relpath(f, dir).replace("\\", "/"), files)))

            return web.json_response(extensions)

        def get_dir_by_type(dir_type):
            if dir_type is None:
                dir_type = "input"

            if dir_type == "input":
                type_dir = folder_paths.get_input_directory()
            elif dir_type == "temp":
                type_dir = folder_paths.get_temp_directory()
            elif dir_type == "output":
                type_dir = folder_paths.get_output_directory()

            return type_dir, dir_type

        def compare_image_hash(filepath, image):
            hasher = node_helpers.hasher()

            # function to compare hashes of two images to see if it already exists, fix to #3465
            if os.path.exists(filepath):
                a = hasher()
                b = hasher()
                with open(filepath, "rb") as f:
                    a.update(f.read())
                    b.update(image.file.read())
                    image.file.seek(0)
                return a.hexdigest() == b.hexdigest()
            return False

        def image_upload(post, image_save_function=None):
            image = post.get("image")
            overwrite = post.get("overwrite")
            image_is_duplicate = False

            image_upload_type = post.get("type")
            upload_dir, image_upload_type = get_dir_by_type(image_upload_type)

            if image and image.file:
                filename = image.filename
                if not filename:
                    return web.Response(status=400)

                subfolder = post.get("subfolder", "")
                full_output_folder = os.path.join(upload_dir, os.path.normpath(subfolder))
                filepath = os.path.abspath(os.path.join(full_output_folder, filename))

                if os.path.commonpath((upload_dir, filepath)) != upload_dir:
                    return web.Response(status=400)

                if not os.path.exists(full_output_folder):
                    os.makedirs(full_output_folder)

                split = os.path.splitext(filename)

                if overwrite is not None and (overwrite == "true" or overwrite == "1"):
                    pass
                else:
                    i = 1
                    while os.path.exists(filepath):
                        if compare_image_hash(filepath, image): #compare hash to prevent saving of duplicates with same name, fix for #3465
                            image_is_duplicate = True
                            break
                        filename = f"{split[0]} ({i}){split[1]}"
                        filepath = os.path.join(full_output_folder, filename)
                        i += 1

                if not image_is_duplicate:
                    if image_save_function is not None:
                        image_save_function(image, post, filepath)
                    else:
                        with open(filepath, "wb") as f:
                            f.write(image.file.read())

                resp = {"name" : filename, "subfolder": subfolder, "type": image_upload_type}

                if args.enable_assets:
                    try:
                        tag = image_upload_type if image_upload_type in ("input", "output") else "input"
                        result = register_file_in_place(abs_path=filepath, name=filename, tags=[tag])
                        resp["asset"] = {
                            "id": result.ref.id,
                            "name": result.ref.name,
                            "asset_hash": result.asset.hash,
                            "size": result.asset.size_bytes,
                            "mime_type": result.asset.mime_type,
                            "tags": result.tags,
                        }
                    except Exception:
                        logging.warning("Failed to register uploaded image as asset", exc_info=True)

                return web.json_response(resp)
            else:
                return web.Response(status=400)

        @routes.post("/upload/image")
        async def upload_image(request):
            post = await request.post()
            return image_upload(post)


        @routes.post("/upload/mask")
        async def upload_mask(request):
            post = await request.post()

            def image_save_function(image, post, filepath):
                original_ref = json.loads(post.get("original_ref"))
                filename, output_dir = folder_paths.annotated_filepath(original_ref['filename'])

                if not filename:
                    return web.Response(status=400)

                # validation for security: prevent accessing arbitrary path
                if filename[0] == '/' or '..' in filename:
                    return web.Response(status=400)

                if output_dir is None:
                    type = original_ref.get("type", "output")
                    output_dir = folder_paths.get_directory_by_type(type)

                if output_dir is None:
                    return web.Response(status=400)

                if original_ref.get("subfolder", "") != "":
                    full_output_dir = os.path.join(output_dir, original_ref["subfolder"])
                    if os.path.commonpath((os.path.abspath(full_output_dir), output_dir)) != output_dir:
                        return web.Response(status=403)
                    output_dir = full_output_dir

                file = os.path.join(output_dir, filename)

                if os.path.isfile(file):
                    with Image.open(file) as original_pil:
                        metadata = PngInfo()
                        if hasattr(original_pil,'text'):
                            for key in original_pil.text:
                                metadata.add_text(key, original_pil.text[key])
                        original_pil = original_pil.convert('RGBA')
                        mask_pil = Image.open(image.file).convert('RGBA')

                        # alpha copy
                        new_alpha = mask_pil.getchannel('A')
                        original_pil.putalpha(new_alpha)
                        original_pil.save(filepath, compress_level=4, pnginfo=metadata)

            return image_upload(post, image_save_function)

        @routes.get("/view")
        async def view_image(request):
            if "filename" in request.rel_url.query:
                filename = request.rel_url.query["filename"]

                # The frontend's LoadImage combo widget uses asset_hash values
                # (e.g. "blake3:...") as widget values. When litegraph renders the
                # node preview, it constructs /view?filename=<asset_hash>, so this
                # endpoint must resolve blake3 hashes to their on-disk file paths.
                if filename.startswith("blake3:"):
                    owner_id = self.user_manager.get_request_user_id(request)
                    result = resolve_hash_to_path(filename, owner_id=owner_id)
                    if result is None:
                        return web.Response(status=404)
                    file, filename, resolved_content_type = result.abs_path, result.download_name, result.content_type
                else:
                    resolved_content_type = None
                    filename, output_dir = folder_paths.annotated_filepath(filename)

                    if not filename:
                        return web.Response(status=400)

                    # validation for security: prevent accessing arbitrary path
                    if filename[0] == '/' or '..' in filename:
                        return web.Response(status=400)

                    if output_dir is None:
                        type = request.rel_url.query.get("type", "output")
                        output_dir = folder_paths.get_directory_by_type(type)

                    if output_dir is None:
                        return web.Response(status=400)

                    if "subfolder" in request.rel_url.query:
                        full_output_dir = os.path.join(output_dir, request.rel_url.query["subfolder"])
                        if os.path.commonpath((os.path.abspath(full_output_dir), output_dir)) != output_dir:
                            return web.Response(status=403)
                        output_dir = full_output_dir

                    filename = os.path.basename(filename)
                    file = os.path.join(output_dir, filename)

                if os.path.isfile(file):
                    if 'preview' in request.rel_url.query:
                        with Image.open(file) as img:
                            preview_info = request.rel_url.query['preview'].split(';')
                            image_format = preview_info[0]
                            if image_format not in ['webp', 'jpeg'] or 'a' in request.rel_url.query.get('channel', ''):
                                image_format = 'webp'

                            quality = 90
                            if preview_info[-1].isdigit():
                                quality = int(preview_info[-1])

                            buffer = BytesIO()
                            if image_format in ['jpeg'] or request.rel_url.query.get('channel', '') == 'rgb':
                                img = img.convert("RGB")
                            img.save(buffer, format=image_format, quality=quality)
                            buffer.seek(0)

                            return web.Response(body=buffer.read(), content_type=f'image/{image_format}',
                                                headers={"Content-Disposition": f"filename=\"{filename}\""})

                    if 'channel' not in request.rel_url.query:
                        channel = 'rgba'
                    else:
                        channel = request.rel_url.query["channel"]

                    if channel == 'rgb':
                        with Image.open(file) as img:
                            if img.mode == "RGBA":
                                r, g, b, a = img.split()
                                new_img = Image.merge('RGB', (r, g, b))
                            else:
                                new_img = img.convert("RGB")

                            buffer = BytesIO()
                            new_img.save(buffer, format='PNG')
                            buffer.seek(0)

                            return web.Response(body=buffer.read(), content_type='image/png',
                                                headers={"Content-Disposition": f"filename=\"{filename}\""})

                    elif channel == 'a':
                        with Image.open(file) as img:
                            if img.mode == "RGBA":
                                _, _, _, a = img.split()
                            else:
                                a = Image.new('L', img.size, 255)

                            # alpha img
                            alpha_img = Image.new('RGBA', img.size)
                            alpha_img.putalpha(a)
                            alpha_buffer = BytesIO()
                            alpha_img.save(alpha_buffer, format='PNG')
                            alpha_buffer.seek(0)

                            return web.Response(body=alpha_buffer.read(), content_type='image/png',
                                                headers={"Content-Disposition": f"filename=\"{filename}\""})
                    else:
                        # Use the content type from asset resolution if available,
                        # otherwise guess from the filename.
                        content_type = (
                            resolved_content_type
                            or mimetypes.guess_type(filename)[0]
                            or 'application/octet-stream'
                        )

                        # For security, force certain mimetypes to download instead of display
                        if content_type in {'text/html', 'text/html-sandboxed', 'application/xhtml+xml', 'text/javascript', 'text/css'}:
                            content_type = 'application/octet-stream'  # Forces download

                        return web.FileResponse(
                            file,
                            headers={
                                "Content-Disposition": f"filename=\"{filename}\"",
                                "Content-Type": content_type
                            }
                        )

            return web.Response(status=404)

        @routes.get("/view_metadata/{folder_name}")
        async def view_metadata(request):
            folder_name = request.match_info.get("folder_name", None)
            if folder_name is None:
                return web.Response(status=404)
            if "filename" not in request.rel_url.query:
                return web.Response(status=404)

            filename = request.rel_url.query["filename"]
            if not filename.endswith(".safetensors"):
                return web.Response(status=404)

            safetensors_path = folder_paths.get_full_path(folder_name, filename)
            if safetensors_path is None:
                return web.Response(status=404)
            out = comfy.utils.safetensors_header(safetensors_path, max_size=1024*1024)
            if out is None:
                return web.Response(status=404)
            dt = json.loads(out)
            if "__metadata__" not in dt:
                return web.Response(status=404)
            return web.json_response(dt["__metadata__"])

        @routes.get("/system_stats")
        async def system_stats(request):
            primary_device = comfy.model_management.get_torch_device()
            cpu_device = comfy.model_management.torch.device("cpu")
            ram_total = comfy.model_management.get_total_memory(cpu_device)
            ram_free = comfy.model_management.get_free_memory(cpu_device)
            required_frontend_version = FrontendManager.get_required_frontend_version()
            installed_templates_version = FrontendManager.get_installed_templates_version()
            required_templates_version = FrontendManager.get_required_templates_version()
            comfy_package_versions = FrontendManager.get_comfy_package_versions()

            # Report every torch device visible to multigpu, with the primary
            # device first so existing clients that read devices[0] keep working.
            torch_devices = comfy.model_management.get_all_torch_devices()
            if primary_device in torch_devices:
                torch_devices = [primary_device] + [d for d in torch_devices if d != primary_device]
            else:
                torch_devices = [primary_device] + list(torch_devices)

            device_entries = []
            for d in torch_devices:
                vram_total, torch_vram_total = comfy.model_management.get_total_memory(d, torch_total_too=True)
                vram_free, torch_vram_free = comfy.model_management.get_free_memory(d, torch_free_too=True)
                device_entries.append({
                    "name": comfy.model_management.get_torch_device_name(d),
                    "type": d.type,
                    "index": d.index,
                    "vram_total": vram_total,
                    "vram_free": vram_free,
                    "torch_vram_total": torch_vram_total,
                    "torch_vram_free": torch_vram_free,
                })

            system_stats = {
                "system": {
                    "os": sys.platform,
                    "ram_total": ram_total,
                    "ram_free": ram_free,
                    "comfyui_version": __version__,
                    "required_frontend_version": required_frontend_version,
                    "installed_templates_version": installed_templates_version,
                    "required_templates_version": required_templates_version,
                    "comfy_package_versions": comfy_package_versions,
                    "python_version": sys.version,
                    "pytorch_version": comfy.model_management.torch_version,
                    "embedded_python": os.path.split(os.path.split(sys.executable)[0])[1] == "python_embeded",
                    "deploy_environment": get_deploy_environment(),
                    "argv": sys.argv
                },
                "devices": device_entries
            }
            return web.json_response(system_stats)

        @routes.get("/features")
        async def get_features(request):
            return web.json_response(feature_flags.get_server_features())

        @routes.get("/prompt")
        async def get_prompt(request):
            return web.json_response(self.get_queue_info())

        def node_info(node_class):
            obj_class = nodes.NODE_CLASS_MAPPINGS[node_class]
            if issubclass(obj_class, _ComfyNodeInternal):
                return obj_class.GET_NODE_INFO_V1()
            info = {}
            info['input'] = obj_class.INPUT_TYPES()
            info['input_order'] = {key: list(value.keys()) for (key, value) in obj_class.INPUT_TYPES().items()}
            info['is_input_list'] = getattr(obj_class, "INPUT_IS_LIST", False)
            info['output'] = obj_class.RETURN_TYPES
            info['output_is_list'] = obj_class.OUTPUT_IS_LIST if hasattr(obj_class, 'OUTPUT_IS_LIST') else [False] * len(obj_class.RETURN_TYPES)
            info['output_name'] = obj_class.RETURN_NAMES if hasattr(obj_class, 'RETURN_NAMES') else info['output']
            info['name'] = node_class
            info['display_name'] = nodes.NODE_DISPLAY_NAME_MAPPINGS[node_class] if node_class in nodes.NODE_DISPLAY_NAME_MAPPINGS.keys() else node_class
            info['description'] = obj_class.DESCRIPTION if hasattr(obj_class,'DESCRIPTION') else ''
            info['python_module'] = getattr(obj_class, "RELATIVE_PYTHON_MODULE", "nodes")
            info['category'] = 'sd'
            if hasattr(obj_class, 'OUTPUT_NODE') and obj_class.OUTPUT_NODE == True:
                info['output_node'] = True
            else:
                info['output_node'] = False

            if hasattr(obj_class, 'HAS_INTERMEDIATE_OUTPUT') and obj_class.HAS_INTERMEDIATE_OUTPUT == True:
                info['has_intermediate_output'] = True
            else:
                info['has_intermediate_output'] = False

            if hasattr(obj_class, 'CATEGORY'):
                info['category'] = obj_class.CATEGORY

            if hasattr(obj_class, 'OUTPUT_TOOLTIPS'):
                info['output_tooltips'] = obj_class.OUTPUT_TOOLTIPS

            if getattr(obj_class, "DEPRECATED", False):
                info['deprecated'] = True
            if getattr(obj_class, "EXPERIMENTAL", False):
                info['experimental'] = True
            if getattr(obj_class, "DEV_ONLY", False):
                info['dev_only'] = True

            if hasattr(obj_class, 'API_NODE'):
                info['api_node'] = obj_class.API_NODE

            info['search_aliases'] = getattr(obj_class, 'SEARCH_ALIASES', [])

            if hasattr(obj_class, 'ESSENTIALS_CATEGORY'):
                info['essentials_category'] = obj_class.ESSENTIALS_CATEGORY

            return info

        @routes.get("/object_info")
        async def get_object_info(request):
            asset_seeder.start(roots=("models", "input", "output"))
            with folder_paths.cache_helper:
                out = {}
                for x in nodes.NODE_CLASS_MAPPINGS:
                    try:
                        out[x] = node_info(x)
                    except Exception:
                        logging.error(f"[ERROR] An error occurred while retrieving information for the '{x}' node.")
                        logging.error(traceback.format_exc())
                return web.json_response(out)

        @routes.get("/object_info/{node_class}")
        async def get_object_info_node(request):
            node_class = request.match_info.get("node_class", None)
            out = {}
            if (node_class is not None) and (node_class in nodes.NODE_CLASS_MAPPINGS):
                out[node_class] = node_info(node_class)
            return web.json_response(out)

        @routes.get("/api/jobs")
        async def get_jobs(request):
            """List all jobs with filtering, sorting, and pagination.

            Query parameters:
                status: Filter by status (comma-separated): pending, in_progress, completed, failed
                workflow_id: Filter by workflow ID
                sort_by: Sort field: created_at (default), execution_duration
                sort_order: Sort direction: asc, desc (default)
                limit: Max items to return (positive integer)
                offset: Items to skip (non-negative integer, default 0)
            """
            query = request.rel_url.query

            status_param = query.get('status')
            workflow_id = query.get('workflow_id')
            sort_by = query.get('sort_by', 'created_at').lower()
            sort_order = query.get('sort_order', 'desc').lower()

            status_filter = None
            if status_param:
                status_filter = [s.strip().lower() for s in status_param.split(',') if s.strip()]
                invalid_statuses = [s for s in status_filter if s not in JobStatus.ALL]
                if invalid_statuses:
                    return web.json_response(
                        {"error": f"Invalid status value(s): {', '.join(invalid_statuses)}. Valid values: {', '.join(JobStatus.ALL)}"},
                        status=400
                    )

            if sort_by not in {'created_at', 'execution_duration'}:
                return web.json_response(
                    {"error": "sort_by must be 'created_at' or 'execution_duration'"},
                    status=400
                )

            if sort_order not in {'asc', 'desc'}:
                return web.json_response(
                    {"error": "sort_order must be 'asc' or 'desc'"},
                    status=400
                )

            limit = None

            # If limit is provided, validate that it is a positive integer, else continue without a limit
            if 'limit' in query:
                try:
                    limit = int(query.get('limit'))
                    if limit <= 0:
                        return web.json_response(
                            {"error": "limit must be a positive integer"},
                            status=400
                        )
                except (ValueError, TypeError):
                    return web.json_response(
                        {"error": "limit must be an integer"},
                        status=400
                    )

            offset = 0
            if 'offset' in query:
                try:
                    offset = int(query.get('offset'))
                    if offset < 0:
                        offset = 0
                except (ValueError, TypeError):
                    return web.json_response(
                        {"error": "offset must be an integer"},
                        status=400
                    )

            running, queued = self.prompt_queue.get_current_queue_volatile()
            history = self.prompt_queue.get_history()

            running = _remove_sensitive_from_queue(running)
            queued = _remove_sensitive_from_queue(queued)

            jobs, total = get_all_jobs(
                running, queued, history,
                status_filter=status_filter,
                workflow_id=workflow_id,
                sort_by=sort_by,
                sort_order=sort_order,
                limit=limit,
                offset=offset
            )

            has_more = (offset + len(jobs)) < total

            return web.json_response({
                'jobs': jobs,
                'pagination': {
                    'offset': offset,
                    'limit': limit,
                    'total': total,
                    'has_more': has_more
                }
            })

        @routes.get("/api/jobs/{job_id}")
        async def get_job_by_id(request):
            """Get a single job by ID."""
            job_id = request.match_info.get("job_id", None)
            if not job_id:
                return web.json_response(
                    {"error": "job_id is required"},
                    status=400
                )

            running, queued = self.prompt_queue.get_current_queue_volatile()
            history = self.prompt_queue.get_history(prompt_id=job_id)

            running = _remove_sensitive_from_queue(running)
            queued = _remove_sensitive_from_queue(queued)

            job = get_job(job_id, running, queued, history)
            if job is None:
                return web.json_response(
                    {"error": "Job not found"},
                    status=404
                )

            return web.json_response(job)

        def _cancel_job_by_id(job_id):
            """Cancel a single job by id using the queue's existing mechanics.

            Running jobs are interrupted (same mechanism as /interrupt); pending
            jobs are dequeued (same mechanism as /queue {"delete": [...]}).
            Already-finished or unknown ids are no-ops. State-agnostic.

            Returns True when a cancel was actually dispatched (running or
            pending job), False when the call was a no-op (terminal/unknown id).
            """
            running, queued = self.prompt_queue.get_current_queue()
            history = self.prompt_queue.get_history()

            def interrupt(prompt_id):
                logging.info(f"Cancelling running prompt {prompt_id}")
                # Atomic: only interrupts if the job is still the one running,
                # so a cancel can't land on a prompt that started in the gap
                # since the snapshot above. Returns whether it actually fired.
                return self.prompt_queue.interrupt_if_running(prompt_id)

            def dequeue(prompt_id):
                logging.info(f"Cancelling pending prompt {prompt_id}")
                return self.prompt_queue.delete_queue_item(lambda a: a[1] == prompt_id)

            classification = cancel_job(job_id, running, queued, history, interrupt, dequeue)
            return classification in (CANCEL_RUNNING, CANCEL_PENDING)

        @routes.post("/api/jobs/{job_id}/cancel")
        async def cancel_job_by_id(request):
            """Cancel a single job by id, regardless of state.

            Idempotent: cancelling a job that has already finished, or an id
            that is not known, returns 200 with {"cancelled": false} rather
            than an error.
            """
            job_id = request.match_info.get("job_id", None)
            if not job_id:
                return web.json_response(
                    {"error": "job_id is required"},
                    status=400
                )

            cancelled = _cancel_job_by_id(job_id)
            return web.json_response({"cancelled": cancelled})

        @routes.post("/api/jobs/cancel")
        async def cancel_jobs_batch(request):
            """Cancel a batch of jobs by id.

            Body: {"job_ids": ["<uuid>", ...]}

            Best-effort and idempotent: every well-formed id is cancelled if it
            is running or pending; ids that are already finished or unknown are
            no-ops, not errors. A batch of all no-ops still returns 200 with
            {"cancelled": false}. This matches the single-cancel endpoint and
            means "cancel all" still cancels the in-progress jobs even if some
            finished between the client's snapshot and the request. Malformed
            ids are still rejected up front with 400 (see below).
            """
            try:
                json_data = await request.json()
            except json.JSONDecodeError:
                return web.json_response(
                    {"error": "Request body must be valid JSON"},
                    status=400
                )

            job_ids = json_data.get("job_ids") if isinstance(json_data, dict) else None
            if not isinstance(job_ids, list):
                return web.json_response(
                    {"error": "job_ids must be a list"},
                    status=400
                )

            # Validate that every element is a well-formed job id before doing
            # anything else.  An unhashable element (e.g. a nested dict or list)
            # would cause a TypeError when used as a history dict key; a
            # non-string or non-UUID value is never a valid id.  Reject early
            # with 400 rather than letting the classify loop raise 500.
            invalid_ids = []
            for jid in job_ids:
                try:
                    validate_job_id(jid)
                except (ValueError, AttributeError):
                    invalid_ids.append(jid if isinstance(jid, str) else repr(jid))
            if invalid_ids:
                return web.json_response(
                    {"error": "job_ids contains invalid id(s)", "invalid_ids": invalid_ids},
                    status=400,
                )

            # Best-effort: cancel each id that is still running/pending; an id
            # that has finished or never existed is a no-op rather than a reason
            # to fail the whole batch.
            cancelled = False
            for jid in job_ids:
                if _cancel_job_by_id(jid):
                    cancelled = True

            return web.json_response({"cancelled": cancelled})

        @routes.get("/history")
        async def get_history(request):
            max_items = request.rel_url.query.get("max_items", None)
            if max_items is not None:
                max_items = int(max_items)

            offset = request.rel_url.query.get("offset", None)
            if offset is not None:
                offset = int(offset)
            else:
                offset = -1

            return web.json_response(self.prompt_queue.get_history(max_items=max_items, offset=offset))

        @routes.get("/history/{prompt_id}")
        async def get_history_prompt_id(request):
            prompt_id = request.match_info.get("prompt_id", None)
            return web.json_response(self.prompt_queue.get_history(prompt_id=prompt_id))

        @routes.get("/queue")
        async def get_queue(request):
            queue_info = {}
            current_queue = self.prompt_queue.get_current_queue_volatile()
            queue_info['queue_running'] = _remove_sensitive_from_queue(current_queue[0])
            queue_info['queue_pending'] = _remove_sensitive_from_queue(current_queue[1])
            return web.json_response(queue_info)

        @routes.post("/prompt")
        async def post_prompt(request):
            logging.info("got prompt")
            json_data =  await request.json()
            json_data = self.trigger_on_prompt(json_data)

            if "number" in json_data:
                number = float(json_data['number'])
            else:
                number = self.number
                if "front" in json_data:
                    if json_data['front']:
                        number = -number

                self.number += 1

            if "prompt" in json_data:
                prompt = json_data["prompt"]
                client_prompt_id = json_data.get("prompt_id")
                if client_prompt_id is None:
                    # Absent or explicit null: the server mints the id.
                    prompt_id = str(uuid.uuid4())
                else:
                    try:
                        prompt_id = validate_job_id(client_prompt_id)
                    except ValueError:
                        error = {
                            "type": "invalid_prompt_id",
                            "message": "prompt_id must be a valid UUID",
                            "details": "prompt_id must be a UUID string in canonical lowercase hyphenated form; omit it to let the server generate one",
                            "extra_info": {}
                        }
                        return web.json_response({"error": error, "node_errors": {}}, status=400)

                partial_execution_targets = None
                if "partial_execution_targets" in json_data:
                    partial_execution_targets = json_data["partial_execution_targets"]

                self.node_replace_manager.apply_replacements(prompt)

                valid = await execution.validate_prompt(prompt_id, prompt, partial_execution_targets)
                extra_data = {}
                if "extra_data" in json_data:
                    extra_data = json_data["extra_data"]

                if "client_id" in json_data:
                    extra_data["client_id"] = json_data["client_id"]

                if "comfy_usage_source" not in extra_data:
                    usage_source = request.headers.get("Comfy-Usage-Source")
                    if usage_source:
                        extra_data["comfy_usage_source"] = usage_source
                if valid[0]:
                    outputs_to_execute = valid[2]
                    sensitive = {}
                    for sensitive_val in execution.SENSITIVE_EXTRA_DATA_KEYS:
                        if sensitive_val in extra_data:
                            sensitive[sensitive_val] = extra_data.pop(sensitive_val)
                    extra_data["create_time"] = int(time.time() * 1000)  # timestamp in milliseconds
                    self.prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute, sensitive))
                    response = {"prompt_id": prompt_id, "number": number, "node_errors": valid[3]}
                    return web.json_response(response)
                else:
                    logging.warning("invalid prompt: {}".format(valid[1]))
                    return web.json_response({"error": valid[1], "node_errors": valid[3]}, status=400)
            else:
                error = {
                    "type": "no_prompt",
                    "message": "No prompt provided",
                    "details": "No prompt provided",
                    "extra_info": {}
                }
                return web.json_response({"error": error, "node_errors": {}}, status=400)

        @routes.post("/queue")
        async def post_queue(request):
            json_data =  await request.json()
            if "clear" in json_data:
                if json_data["clear"]:
                    self.prompt_queue.wipe_queue()
            if "delete" in json_data:
                to_delete = json_data['delete']
                for id_to_delete in to_delete:
                    delete_func = lambda a: a[1] == id_to_delete
                    self.prompt_queue.delete_queue_item(delete_func)

            return web.Response(status=200)

        @routes.post("/interrupt")
        async def post_interrupt(request):
            try:
                json_data = await request.json()
            except json.JSONDecodeError:
                json_data = {}

            # Check if a specific prompt_id was provided for targeted interruption
            prompt_id = json_data.get('prompt_id')
            if prompt_id:
                currently_running, _ = self.prompt_queue.get_current_queue()

                # Check if the prompt_id matches any currently running prompt
                should_interrupt = False
                for item in currently_running:
                    # item structure: (number, prompt_id, prompt, extra_data, outputs_to_execute)
                    if item[1] == prompt_id:
                        logging.info(f"Interrupting prompt {prompt_id}")
                        should_interrupt = True
                        break

                if should_interrupt:
                    nodes.interrupt_processing()
                else:
                    logging.info(f"Prompt {prompt_id} is not currently running, skipping interrupt")
            else:
                # No prompt_id provided, do a global interrupt
                logging.info("Global interrupt (no prompt_id specified)")
                nodes.interrupt_processing()

            return web.Response(status=200)

        @routes.post("/free")
        async def post_free(request):
            json_data = await request.json()
            unload_models = json_data.get("unload_models", False)
            free_memory = json_data.get("free_memory", False)
            if unload_models:
                self.prompt_queue.set_flag("unload_models", unload_models)
            if free_memory:
                self.prompt_queue.set_flag("free_memory", free_memory)
            return web.Response(status=200)

        @routes.post("/history")
        async def post_history(request):
            json_data =  await request.json()
            if "clear" in json_data:
                if json_data["clear"]:
                    self.prompt_queue.wipe_history()
            if "delete" in json_data:
                to_delete = json_data['delete']
                for id_to_delete in to_delete:
                    self.prompt_queue.delete_history_item(id_to_delete)

            return web.Response(status=200)

    async def setup(self):
        timeout = aiohttp.ClientTimeout(total=None) # no timeout
        self.client_session = aiohttp.ClientSession(timeout=timeout)

    def add_routes(self):
        self.user_manager.add_routes(self.routes)
        self.model_file_manager.add_routes(self.routes)
        self.custom_node_manager.add_routes(self.routes, self.app, nodes.LOADED_MODULE_DIRS.items())
        self.subgraph_manager.add_routes(self.routes, nodes.LOADED_MODULE_DIRS.items())
        self.node_replace_manager.add_routes(self.routes)
        self.app.add_subapp('/internal', self.internal_routes.get_app())

        # Prefix every route with /api for easier matching for delegation.
        # This is very useful for frontend dev server, which need to forward
        # everything except serving of static files.
        # Currently both the old endpoints without prefix and new endpoints with
        # prefix are supported.
        api_routes = web.RouteTableDef()
        for route in self.routes:
            # Custom nodes might add extra static routes. Only process non-static
            # routes to add /api prefix.
            if isinstance(route, web.RouteDef):
                api_routes.route(route.method, "/api" + route.path)(route.handler, **route.kwargs)
        self.app.add_routes(api_routes)
        self.app.add_routes(self.routes)

        # Add routes from web extensions.
        for name, dir in nodes.EXTENSION_WEB_DIRS.items():
            self.app.add_routes([web.static('/extensions/' + name, dir)])

        installed_templates_version = FrontendManager.get_installed_templates_version()
        use_legacy_templates = True
        if installed_templates_version:
            try:
                use_legacy_templates = (
                    parse_version(installed_templates_version)
                    < parse_version("0.3.0")
                )
            except Exception as exc:
                logging.warning(
                    "Unable to parse templates version '%s': %s",
                    installed_templates_version,
                    exc,
                )

        if use_legacy_templates:
            workflow_templates_path = FrontendManager.legacy_templates_path()
            if workflow_templates_path:
                self.app.add_routes([
                    web.static('/templates', workflow_templates_path)
                ])
        else:
            handler = FrontendManager.template_asset_handler()
            if handler:
                self.app.router.add_get("/templates/{path:.*}", handler)

        # Serve embedded documentation from the package
        embedded_docs_path = FrontendManager.embedded_docs_path()
        if embedded_docs_path:
            self.app.add_routes([
                web.static('/docs', embedded_docs_path)
            ])

        self.app.add_routes([
            web.static('/', self.web_root),
        ])

    def get_queue_info(self):
        prompt_info = {}
        exec_info = {}
        exec_info['queue_remaining'] = self.prompt_queue.get_tasks_remaining()
        prompt_info['exec_info'] = exec_info
        return prompt_info

    async def send(self, event, data, sid=None):
        if event == BinaryEventTypes.UNENCODED_PREVIEW_IMAGE:
            await self.send_image(data, sid=sid)
        elif event == BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA:
            # data is (preview_image, metadata)
            preview_image, metadata = data
            await self.send_image_with_metadata(preview_image, metadata, sid=sid)
        elif isinstance(data, (bytes, bytearray)):
            await self.send_bytes(event, data, sid)
        else:
            await self.send_json(event, data, sid)

    def encode_bytes(self, event, data):
        if not isinstance(event, int):
            raise RuntimeError(f"Binary event types must be integers, got {event}")

        packed = struct.pack(">I", event)
        message = bytearray(packed)
        message.extend(data)
        return message

    async def send_image(self, image_data, sid=None):
        image_type = image_data[0]
        image = image_data[1]
        max_size = image_data[2]
        if max_size is not None:
            if hasattr(Image, 'Resampling'):
                resampling = Image.Resampling.BILINEAR
            else:
                resampling = Image.Resampling.LANCZOS

            image = ImageOps.contain(image, (max_size, max_size), resampling)
        type_num = 1
        if image_type == "JPEG":
            type_num = 1
        elif image_type == "PNG":
            type_num = 2

        bytesIO = BytesIO()
        header = struct.pack(">I", type_num)
        bytesIO.write(header)
        image.save(bytesIO, format=image_type, quality=95, compress_level=1)
        preview_bytes = bytesIO.getvalue()
        await self.send_bytes(BinaryEventTypes.PREVIEW_IMAGE, preview_bytes, sid=sid)

    async def send_image_with_metadata(self, image_data, metadata=None, sid=None):
        image_type = image_data[0]
        image = image_data[1]
        max_size = image_data[2]
        if max_size is not None:
            if hasattr(Image, 'Resampling'):
                resampling = Image.Resampling.BILINEAR
            else:
                resampling = Image.Resampling.LANCZOS

            image = ImageOps.contain(image, (max_size, max_size), resampling)

        mimetype = "image/png" if image_type == "PNG" else "image/jpeg"

        # Prepare metadata
        if metadata is None:
            metadata = {}
        metadata["image_type"] = mimetype

        # Serialize metadata as JSON
        import json
        metadata_json = json.dumps(metadata).encode('utf-8')
        metadata_length = len(metadata_json)

        # Prepare image data
        bytesIO = BytesIO()
        image.save(bytesIO, format=image_type, quality=95, compress_level=1)
        image_bytes = bytesIO.getvalue()

        # Combine metadata and image
        combined_data = bytearray()
        combined_data.extend(struct.pack(">I", metadata_length))
        combined_data.extend(metadata_json)
        combined_data.extend(image_bytes)

        await self.send_bytes(BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA, combined_data, sid=sid)

    async def send_bytes(self, event, data, sid=None):
        message = self.encode_bytes(event, data)

        if sid is None:
            sockets = list(self.sockets.values())
            for ws in sockets:
                await send_socket_catch_exception(ws.send_bytes, message)
        elif sid in self.sockets:
            await send_socket_catch_exception(self.sockets[sid].send_bytes, message)

    async def send_json(self, event, data, sid=None):
        message = {"type": event, "data": data}

        if sid is None:
            sockets = list(self.sockets.values())
            for ws in sockets:
                await send_socket_catch_exception(ws.send_json, message)
        elif sid in self.sockets:
            await send_socket_catch_exception(self.sockets[sid].send_json, message)

    def send_sync(self, event, data, sid=None):
        self.loop.call_soon_threadsafe(
            self.messages.put_nowait, (event, data, sid))

    def queue_updated(self):
        self.send_sync("status", { "status": self.get_queue_info() })

    async def publish_loop(self):
        while True:
            msg = await self.messages.get()
            await self.send(*msg)

    async def start(self, address, port, verbose=True, call_on_start=None):
        await self.start_multi_address([(address, port)], call_on_start=call_on_start)

    async def start_multi_address(self, addresses, call_on_start=None, verbose=True):
        runner = web.AppRunner(self.app, access_log=None)
        await runner.setup()
        ssl_ctx = None
        scheme = "http"
        if args.tls_keyfile and args.tls_certfile:
            ssl_ctx = ssl.SSLContext(protocol=ssl.PROTOCOL_TLS_SERVER, verify_mode=ssl.CERT_NONE)
            ssl_ctx.load_cert_chain(certfile=args.tls_certfile,
                                keyfile=args.tls_keyfile)
            scheme = "https"

        if verbose:
            logging.info("Starting server\n")
            if args.debug_hang:
                logging.info(
                    f"{'-' * 80}\n"
                    "ComfyUI has been started in debug-hang mode. Run your workflow as normal up to\n"
                    "the point of the hang or freeze, then use ctrl-C in the cmd or controlling\n"
                    "terminal to dump the python backtraces for debugging. Please attach the extra\n"
                    "debug info to your bug report.\n"
                    f"{'-' * 80}"
                )
        for addr in addresses:
            address = addr[0]
            port = addr[1]
            site = web.TCPSite(runner, address, port, ssl_context=ssl_ctx)
            try:
                await site.start()
            except OSError as e:
                if e.errno == errno.EADDRINUSE:
                    logging.error(f"Port {port} is already in use on address {address}. Please close the other application or use a different port with --port.")
                    raise SystemExit(1)
                raise

            if not hasattr(self, 'address'):
                self.address = address #TODO: remove this
                self.port = port

            if ':' in address:
                address_print = "[{}]".format(address)
            else:
                address_print = address

            if verbose:
                logging.info("To see the GUI go to: {}://{}:{}".format(scheme, address_print, port))

        if call_on_start is not None:
            call_on_start(scheme, self.address, self.port)

    def add_on_prompt_handler(self, handler):
        self.on_prompt_handlers.append(handler)

    def trigger_on_prompt(self, json_data):
        for handler in self.on_prompt_handlers:
            try:
                json_data = handler(json_data)
            except Exception:
                logging.warning("[ERROR] An error occurred during the on_prompt_handler processing")
                logging.warning(traceback.format_exc())

        return json_data

    def send_progress_text(
        self, text: Union[bytes, bytearray, str], node_id: str, sid=None
    ):
        if isinstance(text, str):
            text = text.encode("utf-8")
        node_id_bytes = str(node_id).encode("utf-8")

        # Pack the node_id length as a 4-byte unsigned integer, followed by the node_id bytes
        message = struct.pack(">I", len(node_id_bytes)) + node_id_bytes + text

        self.send_sync(BinaryEventTypes.TEXT, message, sid)
