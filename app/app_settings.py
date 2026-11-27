import os
import json
from aiohttp import web
import logging


class AppSettings():
    def __init__(self, user_manager):
        self.user_manager = user_manager

    def get_settings(self, request):
        try:
            file = self.user_manager.get_request_user_filepath(
                request,
                "comfy.settings.json"
            )
        except KeyError as e:
            logging.error("User settings not found.")
            raise web.HTTPUnauthorized() from e
        if file is None:
            raise web.HTTPForbidden()
        if os.path.isfile(file):
            try:
                with open(file) as f:
                    return json.load(f)
            except:
                logging.error(f"The user settings file is corrupted: {file}")
                return {}
        else:
            return {}

    def save_settings(self, request, settings):
        file = self.user_manager.get_request_user_filepath(
            request, "comfy.settings.json")
        if file is None:
            raise web.HTTPForbidden()
        with open(file, "w") as f:
            f.write(json.dumps(settings, indent=4))

    def add_routes(self, routes):
        @routes.get("/settings")
        async def get_settings(request):
            return web.json_response(self.get_settings(request))

        @routes.get("/settings/{id}")
        async def get_setting(request):
            value = None
            settings = self.get_settings(request)
            setting_id = request.match_info.get("id", None)
            if setting_id and setting_id in settings:
                value = settings[setting_id]
            return web.json_response(value)

        @routes.post("/settings")
        async def post_settings(request):
            settings = self.get_settings(request)
            new_settings = await request.json()
            self.save_settings(request, {**settings, **new_settings})
            return web.Response(status=200)

        @routes.post("/settings/{id}")
        async def post_setting(request):
            setting_id = request.match_info.get("id", None)
            if not setting_id:
                return web.Response(status=400)
            settings = self.get_settings(request)
            settings[setting_id] = await request.json()
            self.save_settings(request, settings)
            return web.Response(status=200)
