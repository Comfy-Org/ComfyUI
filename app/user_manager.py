import json
import os
import re
import uuid
import glob
import shutil
import logging
import tempfile
from aiohttp import web
from urllib import parse
from comfy.cli_args import args
import folder_paths
from .app_settings import AppSettings
from typing import TypedDict

default_user = "default"


class FileInfo(TypedDict):
    path: str
    size: int
    modified: int
    created: int


def get_file_info(path: str, relative_to: str) -> FileInfo:
    return {
        "path": os.path.relpath(path, relative_to).replace(os.sep, '/'),
        "size": os.path.getsize(path),
        "modified": int(os.path.getmtime(path) * 1000),
        "created": int(os.path.getctime(path) * 1000),
    }


class UserManager():
    def __init__(self):
        user_directory = folder_paths.get_user_directory()

        self.settings = AppSettings(self)
        if not os.path.exists(user_directory):
            os.makedirs(user_directory, exist_ok=True)
            if not args.multi_user:
                logging.warning("****** User settings have been changed to be stored on the server instead of browser storage. ******")
                logging.warning("****** For multi-user setups add the --multi-user CLI argument to enable multiple user profiles. ******")

        if args.multi_user:
            if os.path.isfile(self.get_users_file()):
                with open(self.get_users_file()) as f:
                    self.users = json.load(f)
            else:
                self.users = {}
        else:
            self.users = {"default": "default"}

    def get_users_file(self):
        return os.path.join(folder_paths.get_user_directory(), "users.json")

    def get_request_user_id(self, request):
        user = "default"
        if args.multi_user and "comfy-user" in request.headers:
            user = request.headers["comfy-user"]
            # Block System Users (use same error message to prevent probing)
            if user.startswith(folder_paths.SYSTEM_USER_PREFIX):
                raise KeyError("Unknown user: " + user)

        if user not in self.users:
            raise KeyError("Unknown user: " + user)

        return user

    def get_request_user_filepath(self, request, file, type="userdata", create_dir=True):
        if type == "userdata":
            root_dir = folder_paths.get_user_directory()
        else:
            raise KeyError("Unknown filepath type:" + type)

        user = self.get_request_user_id(request)
        user_root = folder_paths.get_public_user_directory(user)
        if user_root is None:
            return None
        path = user_root

        # prevent leaving /{type}
        if os.path.commonpath((root_dir, user_root)) != root_dir:
            return None

        if file is not None:
            # Check if filename is url encoded
            if "%" in file:
                file = parse.unquote(file)

            # prevent leaving /{type}/{user}
            path = os.path.abspath(os.path.join(user_root, file))
            if os.path.commonpath((user_root, path)) != user_root:
                return None

        parent = os.path.split(path)[0]

        if create_dir and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        return path

    def add_user(self, name):
        name = name.strip()
        if not name:
            raise ValueError("username not provided")
        if name.startswith(folder_paths.SYSTEM_USER_PREFIX):
            raise ValueError("System User prefix not allowed")
        user_id = re.sub("[^a-zA-Z0-9-_]+", '-', name)
        if user_id.startswith(folder_paths.SYSTEM_USER_PREFIX):
            raise ValueError("System User prefix not allowed")
        user_id = user_id + "_" + str(uuid.uuid4())

        self.users[user_id] = name

        with open(self.get_users_file(), "w") as f:
            json.dump(self.users, f)

        return user_id

    def add_routes(self, routes):
        self.settings.add_routes(routes)

        @routes.get("/users")
        async def get_users(request):
            if args.multi_user:
                return web.json_response({"storage": "server", "users": self.users})
            else:
                user_dir = self.get_request_user_filepath(request, None, create_dir=False)
                return web.json_response({
                    "storage": "server",
                    "migrated": os.path.exists(user_dir)
                })

        @routes.post("/users")
        async def post_users(request):
            body = await request.json()
            username = body["username"]
            if username in self.users.values():
                return web.json_response({"error": "Duplicate username."}, status=400)

            try:
                user_id = self.add_user(username)
            except ValueError as e:
                return web.json_response({"error": str(e)}, status=400)
            return web.json_response(user_id)

        @routes.get("/userdata")
        async def listuserdata(request):
            """
            List user data files in a specified directory.

            This endpoint allows listing files in a user's data directory, with options for recursion,
            full file information, and path splitting.

            Query Parameters:
            - dir (required): The directory to list files from.
            - recurse (optional): If "true", recursively list files in subdirectories.
            - full_info (optional): If "true", return detailed file information (path, size, modified time).
            - split (optional): If "true", split file paths into components (only applies when full_info is false).

            Returns:
            - 400: If 'dir' parameter is missing.
            - 403: If the requested path is not allowed.
            - 404: If the requested directory does not exist.
            - 200: JSON response with the list of files or file information.

            The response format depends on the query parameters:
            - Default: List of relative file paths.
            - full_info=true: List of dictionaries with file details.
            - split=true (and full_info=false): List of lists, each containing path components.
            """
            directory = request.rel_url.query.get('dir', '')
            if not directory:
                return web.Response(status=400, text="Directory not provided")

            path = self.get_request_user_filepath(request, directory)
            if not path:
                return web.Response(status=403, text="Invalid directory")

            if not os.path.exists(path):
                return web.Response(status=404, text="Directory not found")

            recurse = request.rel_url.query.get('recurse', '').lower() == "true"
            full_info = request.rel_url.query.get('full_info', '').lower() == "true"
            split_path = request.rel_url.query.get('split', '').lower() == "true"

            # Use different patterns based on whether we're recursing or not
            if recurse:
                pattern = os.path.join(glob.escape(path), '**', '*')
            else:
                pattern = os.path.join(glob.escape(path), '*')

            def process_full_path(full_path: str) -> FileInfo | str | list[str]:
                if full_info:
                    return get_file_info(full_path, path)

                rel_path = os.path.relpath(full_path, path).replace(os.sep, '/')
                if split_path:
                    return [rel_path] + rel_path.split('/')

                return rel_path

            results = [
                process_full_path(full_path)
                for full_path in glob.glob(pattern, recursive=recurse)
                if os.path.isfile(full_path)
            ]

            return web.json_response(results)

        @routes.get("/v2/userdata")
        async def list_userdata_v2(request):
            """
            List files and directories in a user's data directory.

            This endpoint provides a structured listing of contents within a specified
            subdirectory of the user's data storage.

            Query Parameters:
            - path (optional): The relative path within the user's data directory
                               to list. Defaults to the root ('').

            Returns:
            - 400: If the requested path is invalid, outside the user's data directory, or is not a directory.
            - 404: If the requested path does not exist.
            - 403: If the user is invalid.
            - 500: If there is an error reading the directory contents.
            - 200: JSON response containing a list of file and directory objects.
                   Each object includes:
                   - name: The name of the file or directory.
                   - type: 'file' or 'directory'.
                   - path: The relative path from the user's data root.
                   - size (for files): The size in bytes.
                   - modified (for files): The last modified timestamp (Unix epoch).
            """
            requested_rel_path = request.rel_url.query.get('path', '')

            # URL-decode the path parameter
            try:
                requested_rel_path = parse.unquote(requested_rel_path)
            except Exception as e:
                logging.warning(f"Failed to decode path parameter: {requested_rel_path}, Error: {e}")
                return web.Response(status=400, text="Invalid characters in path parameter")


            # Check user validity and get the absolute path for the requested directory
            try:
                 base_user_path = self.get_request_user_filepath(request, None, create_dir=False)

                 if requested_rel_path:
                     target_abs_path = self.get_request_user_filepath(request, requested_rel_path, create_dir=False)
                 else:
                     target_abs_path = base_user_path

            except KeyError as e:
                 # Invalid user detected by get_request_user_id inside get_request_user_filepath
                 logging.warning(f"Access denied for user: {e}")
                 return web.Response(status=403, text="Invalid user specified in request")


            if not target_abs_path:
                 # Path traversal or other issue detected by get_request_user_filepath
                 return web.Response(status=400, text="Invalid path requested")

            # Handle cases where the user directory or target path doesn't exist
            if not os.path.exists(target_abs_path):
                # Check if it's the base user directory that's missing (new user case)
                if target_abs_path == base_user_path:
                    # It's okay if the base user directory doesn't exist yet, return empty list
                     return web.json_response([])
                else:
                    # A specific subdirectory was requested but doesn't exist
                     return web.Response(status=404, text="Requested path not found")

            if not os.path.isdir(target_abs_path):
                 return web.Response(status=400, text="Requested path is not a directory")

            results = []
            try:
                for root, dirs, files in os.walk(target_abs_path, topdown=True):
                    # Process directories
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        rel_path = os.path.relpath(dir_path, base_user_path).replace(os.sep, '/')
                        results.append({
                            "name": dir_name,
                            "path": rel_path,
                            "type": "directory"
                        })

                    # Process files
                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        rel_path = os.path.relpath(file_path, base_user_path).replace(os.sep, '/')
                        entry_info = {
                            "name": file_name,
                            "path": rel_path,
                            "type": "file"
                        }
                        try:
                            stats = os.stat(file_path) # Use os.stat for potentially better performance with os.walk
                            entry_info["size"] = stats.st_size
                            entry_info["modified"] = stats.st_mtime
                        except OSError as stat_error:
                            logging.warning(f"Could not stat file {file_path}: {stat_error}")
                            pass # Include file with available info
                        results.append(entry_info)
            except OSError as e:
                logging.error(f"Error listing directory {target_abs_path}: {e}")
                return web.Response(status=500, text="Error reading directory contents")

            # Sort results alphabetically, directories first then files
            results.sort(key=lambda x: (x['type'] != 'directory', x['name'].lower()))

            return web.json_response(results)

        def get_user_data_path(request, check_exists = False, param = "file"):
            file = request.match_info.get(param, None)
            if not file:
                return web.Response(status=400)

            path = self.get_request_user_filepath(request, file)
            if not path:
                return web.Response(status=403)

            if check_exists and not os.path.exists(path):
                return web.Response(status=404)

            return path

        @routes.get("/userdata/{file}")
        async def getuserdata(request):
            path = get_user_data_path(request, check_exists=True)
            if not isinstance(path, str):
                return path

            return web.FileResponse(path)

        @routes.post("/userdata/{file}")
        async def post_userdata(request):
            """
            Upload or update a user data file.

            This endpoint handles file uploads to a user's data directory, with options for
            controlling overwrite behavior and response format.

            Query Parameters:
            - overwrite (optional): If "false", prevents overwriting existing files. Defaults to "true".
            - full_info (optional): If "true", returns detailed file information (path, size, modified time).
                                  If "false", returns only the relative file path.

            Path Parameters:
            - file: The target file path (URL encoded if necessary).

            Returns:
            - 400: If 'file' parameter is missing.
            - 403: If the requested path is not allowed.
            - 409: If overwrite=false and the file already exists.
            - 200: JSON response with either:
                  - Full file information (if full_info=true)
                  - Relative file path (if full_info=false)

            The request body should contain the raw file content to be written.
            """
            path = get_user_data_path(request)
            if not isinstance(path, str):
                return path

            overwrite = request.query.get("overwrite", 'true') != "false"
            full_info = request.query.get('full_info', 'false').lower() == "true"

            if not overwrite and os.path.exists(path):
                return web.Response(status=409, text="File already exists")

            try:
                body = await request.read()

                dir_name = os.path.dirname(path)
                fd, tmp_path = tempfile.mkstemp(dir=dir_name)
                try:
                    with os.fdopen(fd, "wb") as f:
                        f.write(body)
                    os.replace(tmp_path, path)
                except:
                    os.unlink(tmp_path)
                    raise
            except OSError as e:
                logging.warning(f"Error saving file '{path}': {e}")
                return web.Response(
                    status=400,
                    reason="Invalid filename. Please avoid special characters like :\\/*?\"<>|"
                )

            user_path = self.get_request_user_filepath(request, None)
            if full_info:
                resp = get_file_info(path, user_path)
            else:
                resp = os.path.relpath(path, user_path)

            return web.json_response(resp)

        @routes.delete("/userdata/{file}")
        async def delete_userdata(request):
            path = get_user_data_path(request, check_exists=True)
            if not isinstance(path, str):
                return path

            os.remove(path)

            return web.Response(status=204)

        @routes.post("/userdata/{file}/move/{dest}")
        async def move_userdata(request):
            """
            Move or rename a user data file.

            This endpoint handles moving or renaming files within a user's data directory, with options for
            controlling overwrite behavior and response format.

            Path Parameters:
            - file: The source file path (URL encoded if necessary)
            - dest: The destination file path (URL encoded if necessary)

            Query Parameters:
            - overwrite (optional): If "false", prevents overwriting existing files. Defaults to "true".
            - full_info (optional): If "true", returns detailed file information (path, size, modified time).
                                  If "false", returns only the relative file path.

            Returns:
            - 400: If either 'file' or 'dest' parameter is missing
            - 403: If either requested path is not allowed
            - 404: If the source file does not exist
            - 409: If overwrite=false and the destination file already exists
            - 200: JSON response with either:
                  - Full file information (if full_info=true)
                  - Relative file path (if full_info=false)
            """
            source = get_user_data_path(request, check_exists=True)
            if not isinstance(source, str):
                return source

            dest = get_user_data_path(request, check_exists=False, param="dest")
            if not isinstance(dest, str):
                return dest

            overwrite = request.query.get("overwrite", 'true') != "false"
            full_info = request.query.get('full_info', 'false').lower() == "true"

            if not overwrite and os.path.exists(dest):
                return web.Response(status=409, text="File already exists")

            logging.info(f"moving '{source}' -> '{dest}'")
            shutil.move(source, dest)

            user_path = self.get_request_user_filepath(request, None)
            if full_info:
                resp = get_file_info(dest, user_path)
            else:
                resp = os.path.relpath(dest, user_path)

            return web.json_response(resp)
