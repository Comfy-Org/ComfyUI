from aiohttp import web
from typing import Optional
from folder_paths import folder_names_and_paths, get_directory_by_type
from api_server.services.terminal_service import TerminalService
import app.logger
import os

class InternalRoutes:
    '''
    The top level web router for internal routes: /internal/*
    The endpoints here should NOT be depended upon. It is for ComfyUI frontend use only.
    Check README.md for more information.
    '''

    def __init__(self, prompt_server):
        self.routes: web.RouteTableDef = web.RouteTableDef()
        self._app: Optional[web.Application] = None
        self.prompt_server = prompt_server
        self.terminal_service = TerminalService(prompt_server)

    def setup_routes(self):
        @self.routes.get('/logs')
        async def get_logs(request):
            return web.json_response("".join([(l["t"] + " - " + l["m"]) for l in app.logger.get_logs()]))

        @self.routes.get('/logs/raw')
        async def get_raw_logs(request):
            self.terminal_service.update_size()
            return web.json_response({
                "entries": list(app.logger.get_logs()),
                "size": {"cols": self.terminal_service.cols, "rows": self.terminal_service.rows}
            })

        @self.routes.patch('/logs/subscribe')
        async def subscribe_logs(request):
            json_data = await request.json()
            client_id = json_data["clientId"]
            enabled = json_data["enabled"]
            if enabled:
                self.terminal_service.subscribe(client_id)
            else:
                self.terminal_service.unsubscribe(client_id)

            return web.Response(status=200)


        @self.routes.get('/folder_paths')
        async def get_folder_paths(request):
            response = {}
            for key in folder_names_and_paths:
                response[key] = folder_names_and_paths[key][0]
            return web.json_response(response)

        @self.routes.get('/files/{directory_type}')
        async def get_files(request: web.Request) -> web.Response:
            directory_type = request.match_info['directory_type']
            if directory_type not in ("output", "input", "temp"):
                return web.json_response({"error": "Invalid directory type"}, status=400)

            directory = get_directory_by_type(directory_type)

            def is_visible_file(entry: os.DirEntry) -> bool:
                """Filter out hidden files (e.g., .DS_Store on macOS)."""
                return entry.is_file() and not entry.name.startswith('.')

            sorted_files = sorted(
                (entry for entry in os.scandir(directory) if is_visible_file(entry)),
                key=lambda entry: -entry.stat().st_mtime
            )
            return web.json_response([f"{entry.name} [{directory_type}]" for entry in sorted_files], status=200)


    def get_app(self):
        if self._app is None:
            self._app = web.Application()
            self.setup_routes()
            self._app.add_routes(self.routes)
        return self._app
