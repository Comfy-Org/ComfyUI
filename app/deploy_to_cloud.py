import asyncio
import configparser
import importlib.metadata
import json
import logging
import os
import platform
import re
import sys
import urllib.parse
from datetime import datetime, timezone

import aiohttp
from aiohttp import web

import folder_paths
from comfy.cli_args import args
from comfy_api import feature_flags
from app.comfy_api_env import builder_base_url, platform_base_url
from comfyui_version import __version__

HEADLESS_MARKER = "used_headlessly"
_marker_warning_logged = False


def headless_marker_path() -> str:
    return os.path.join(folder_paths.get_user_directory(), HEADLESS_MARKER)


def used_headlessly() -> bool:
    return os.path.exists(headless_marker_path())


def update_feature_flag() -> None:
    feature_flags.set_server_feature("deploy_to_cloud", not args.offline and used_headlessly())


def mark_used_headlessly() -> None:
    """Record, once and permanently, that a prompt was queued without a client_id."""
    global _marker_warning_logged
    path = headless_marker_path()
    if os.path.exists(path):
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "a").close()
    except OSError as e:
        if not _marker_warning_logged:
            _marker_warning_logged = True
            logging.warning("Could not write headless marker %s: %s", path, e)
        return
    update_feature_flag()


# Installation snapshot in Comfy-Desktop's "comfyui-desktop-2-snapshot" format.
# Local file reads only (no git binary, no network); bad or unreadable data drops
# the affected field or entry instead of raising.

_SHA = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
# tomllib is 3.11+ and ComfyUI supports 3.10, so name/version are read from the [project] table text.
_PROJECT_FIELD = re.compile(r"""^\s*(name|version)\s*=\s*(?:"([^"\n]*)"|'([^'\n]*)')""", re.MULTILINE)


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _pyproject_fields(path: str) -> dict[str, str]:
    """First name/version in [project]: from its header line up to the next line starting with `[`."""
    table = []
    inside = False
    for line in (_read_text(path) or "").splitlines():
        if inside and line.startswith("["):
            break
        if inside:
            table.append(line)
        elif line.strip() == "[project]":
            inside = True
    fields = {}
    for m in _PROJECT_FIELD.finditer("\n".join(table)):
        fields.setdefault(m.group(1), m.group(2) if m.group(2) is not None else m.group(3))
    return fields


def _git_dir(path: str) -> str | None:
    dot_git = os.path.join(path, ".git")
    if os.path.isdir(dot_git):
        return dot_git
    m = re.search(r"^gitdir:\s*(.+?)\s*$", _read_text(dot_git) or "", re.MULTILINE)
    return os.path.normpath(os.path.join(path, m.group(1))) if m else None


def _git_head(gitdir: str) -> str | None:
    head = (_read_text(os.path.join(gitdir, "HEAD")) or "").strip()
    if not head.startswith("ref:"):
        return head if _SHA.fullmatch(head) else None
    ref = head[4:].strip()
    parts = ref.split("/")
    if parts[0] != "refs" or ".." in parts:
        return None
    # simplification: a linked worktree's refs live under its `commondir`, which is not followed.
    sha = (_read_text(os.path.join(gitdir, *parts)) or "").strip()
    if _SHA.fullmatch(sha):
        return sha
    for line in (_read_text(os.path.join(gitdir, "packed-refs")) or "").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1] == ref and _SHA.fullmatch(fields[0]):
            return fields[0]
    return None


_GIT_ESCAPES = {"\\": "\\", '"': '"', "n": "\n", "t": "\t", "b": "\b"}


def _git_config_value(raw: str) -> str | None:
    """Decode a value the way git does: `"` toggles quoting, backslash escapes, an unquoted ; or # starts a comment.
    None for an unknown escape or an unterminated quote."""
    # configparser appends deeper-indented lines to the value; git reads them as separate entries.
    chars = iter(raw.partition("\n")[0])
    value, quoted = "", False
    for c in chars:
        if c == '"':
            quoted = not quoted
        elif c == "\\":
            c = _GIT_ESCAPES.get(next(chars, ""))
            if c is None:
                return None
            value += c
        elif c in ";#" and not quoted:
            break
        else:
            value += c
    return None if quoted else value.strip()


_REMOTE_SCHEMES = ("http", "https", "ssh", "git")
_SCP_LIKE = re.compile(r"[A-Za-z0-9._-]+@([A-Za-z0-9.-]+):[^/\\]\S*")


def _strip_userinfo(url: str) -> str | None:
    """http(s)/ssh/git URLs reduced to scheme/host/port/path, keeping a password-less ssh:// user (ssh://git@host/...);
    scp-like user@host:path only for a dotted host.
    None for everything else: remote helpers (`::`), file://, local paths, internal scp-like hosts."""
    if "::" in url:
        return None
    scp = _SCP_LIKE.fullmatch(url)
    if scp:
        return url if "." in scp.group(1) else None
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return None
    # An unencoded / ? or # in the password ends the netloc early and pushes the rest of the userinfo past it.
    if parts.scheme not in _REMOTE_SCHEMES or not parts.hostname or "@" in parts.path + parts.query + parts.fragment:
        return None
    user, _, host = parts.netloc.rpartition("@")
    if parts.scheme == "ssh" and re.fullmatch(r"[A-Za-z0-9._-]+", user):
        host = f"{user}@{host}"
    return urllib.parse.urlunsplit((parts.scheme, host, parts.path, "", ""))


def _git_origin_url(gitdir: str) -> str | None:
    config = configparser.ConfigParser(strict=False, interpolation=None)
    try:
        config.read(os.path.join(gitdir, "config"), encoding="utf-8")
        url = _git_config_value(config.get('remote "origin"', "url", fallback=""))
    except (configparser.Error, UnicodeDecodeError):
        return None
    return _strip_userinfo(url) if url else None


# Shipped inside the ComfyUI checkout itself, so not an installed node (Desktop CORE_FILE_NODES).
_CORE_FILE_NODES = {"websocket_image_save.py"}


def _scan_dir(path: str, name: str, enabled: bool) -> dict | None:
    try:
        entries = os.listdir(path)
    except OSError:  # unreadable or vanished directory
        return None
    if ".tracking" in entries:
        fields = _pyproject_fields(os.path.join(path, "pyproject.toml"))
        node = {"id": fields.get("name") or name, "type": "cnr", "dirName": name, "enabled": enabled}
        if fields.get("version"):
            node["version"] = fields["version"]
        return node
    node = {"id": name, "type": "git", "dirName": name, "enabled": enabled}
    gitdir = _git_dir(path) if ".git" in entries else None
    if gitdir is not None:
        url, commit = _git_origin_url(gitdir), _git_head(gitdir)
        if url:
            node["url"] = url
        if commit:
            node["commit"] = commit
        return node
    # An unmanaged directory is a node only with a pyproject.toml or top-level Python, so stray folders are skipped.
    if "pyproject.toml" in entries or any(e.endswith(".py") for e in entries):
        return node
    return None


def scan_custom_nodes(dirs: list[str]) -> list[dict]:
    """Desktop ScannedNode entries for each custom_nodes dir and the directories in its .disabled/ subfolder (enabled False).
    A directory named *.disabled is reported disabled and a *.py.disabled file is skipped, as ComfyUI's loader skips both."""
    nodes = []
    for base in dirs:
        for root, enabled in ((base, True), (os.path.join(base, ".disabled"), False)):
            try:
                names = sorted(os.listdir(root))
            except OSError:
                continue
            for name in names:
                path = os.path.join(root, name)
                if name.startswith(".") or name == "__pycache__":
                    continue
                if os.path.isdir(path):
                    node = _scan_dir(path, name, enabled and not name.endswith(".disabled"))
                    if node is not None:
                        nodes.append(node)
                elif enabled and name.endswith(".py") and name != "__init__.py" and name not in _CORE_FILE_NODES:
                    nodes.append({"id": name[:-3], "type": "file", "dirName": name, "enabled": True})
    return nodes


def count_custom_nodes() -> int:
    return sum(1 for n in scan_custom_nodes(folder_paths.get_folder_paths("custom_nodes")) if n["enabled"])


# Pathless stand-in for a local install: not a PEP 440 release, so comfy-builder's parsePins reports it unpinnable, and Desktop restore skips any value containing "://".
_LOCAL_REFERENCE = "file://local"


def _pip_version(dist: importlib.metadata.Distribution) -> str:
    """The version, or for a PEP 610 direct-URL install the reference Desktop's parsePipFreeze stores
    (`git+<url>@<commit>` for VCS, `<url>` for an archive) with credentials stripped; local or malformed -> _LOCAL_REFERENCE."""
    try:
        raw = dist.read_text("direct_url.json")
        if raw is None:
            return dist.version
        direct = json.loads(raw)
        url = _strip_userinfo(direct["url"])
        if url is None or "dir_info" in direct:
            return _LOCAL_REFERENCE
        if "vcs_info" in direct:
            return f"{direct['vcs_info']['vcs']}+{url}@{direct['vcs_info']['commit_id']}"
        return url
    except (ValueError, TypeError, KeyError):
        return _LOCAL_REFERENCE


def pip_packages() -> dict[str, str]:
    """This environment's distributions, like Desktop's `uv pip freeze --python <venv python>`: only sys.path
    entries under sys.prefix are read, so user site-packages and a venv's base interpreter are left out."""
    prefix = os.path.join(os.path.normcase(os.path.realpath(sys.prefix)), "")
    paths = [p for p in sys.path if os.path.join(os.path.normcase(os.path.realpath(p)), "").startswith(prefix)]
    packages = {}
    for dist in importlib.metadata.distributions(path=paths):
        name = dist.metadata.get("Name")
        if name:  # the first distribution of a name is the one imports resolve to
            packages.setdefault(name, _pip_version(dist))
    return packages


MODEL_EXTENSIONS = set(folder_paths.supported_pt_extensions) | {".gguf", ".onnx"}


def list_models() -> list[dict]:
    """Model files as folder name + /-separated relative path. Identical relative paths under two roots of one folder collapse to one entry."""
    return [
        {"directory": folder, "filename": filename.replace(os.sep, "/")}
        for folder in list(folder_paths.folder_names_and_paths)
        if folder not in ("custom_nodes", "configs")
        for filename in folder_paths.get_filename_list(folder)
        if os.path.splitext(filename)[1].lower() in MODEL_EXTENSIONS
    ]


def build_snapshot() -> dict:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    tag = "v" + __version__
    return {
        "type": "comfyui-desktop-2-snapshot",
        "version": 2,
        "exportedAt": now,
        "installationName": os.path.basename(os.path.abspath(folder_paths.base_path)),
        "snapshots": [{
            "version": 2,
            "createdAt": now,
            "trigger": "manual",
            "label": None,
            "comfyui": {"ref": tag, "commit": None, "releaseTag": tag, "variant": "comfyui-core", "baseTag": tag},
            "customNodes": scan_custom_nodes(folder_paths.get_folder_paths("custom_nodes")),
            "pipPackages": pip_packages(),
            "pythonVersion": platform.python_version(),
        }],
    }


def collect() -> tuple[dict, list[dict]]:
    return build_snapshot(), list_models()


# Routes. Registered on the app directly (not PromptServer.routes) so they exist only under /api.
# The bundle upload is ComfyUI's only outbound request here: explicit user action, never under --offline.

_API_KEY = re.compile(r"comfyui-[0-9a-f]{64}")
_UPLOAD_TIMEOUT = aiohttp.ClientTimeout(total=60)
# Builder status -> (our status, error code), after the builder_busy cases in bundle();
# anything not listed (and transport failures) -> 502 builder_unavailable.
_BUILDER_ERRORS = {
    401: (401, "unauthorized"),
    403: (403, "unauthorized"),
    413: (413, "too_large"),
    429: (429, "too_many_bundles"),
}


def _error(status: int, code: str, message: str | None = None) -> web.Response:
    body = {"error": code} if message is None else {"error": code, "message": message}
    return web.json_response(body, status=status)


def _credential(request: web.Request) -> tuple[str, str] | web.Response:
    """The single caller credential header to forward verbatim, or the error response.
    Only a non-empty Bearer Authorization counts; other schemes (e.g. a proxy's Basic auth) are ignored, never forwarded."""
    auth = request.headers.get("Authorization", "")
    bearer = auth.startswith("Bearer ") and bool(auth[7:].strip())
    key = request.headers.get("X-API-Key")
    if bearer and key is not None:
        return _error(400, "multiple_credentials")
    if key is not None:
        return ("X-API-Key", key) if _API_KEY.fullmatch(key) else _error(400, "invalid_api_key_format")
    if bearer:
        return ("Authorization", auth)
    return _error(401, "missing_credentials")


async def _bundle_fields(request: web.Request) -> dict | web.Response:
    """Optional workflow/name from the request body (null counts as absent)."""
    try:
        body = await request.json()
    except ValueError:
        return _error(400, "invalid_body", "body must be a JSON object")
    if not isinstance(body, dict):
        return _error(400, "invalid_body", "body must be a JSON object")
    workflow, name = body.get("workflow"), body.get("name")
    if workflow is not None and not isinstance(workflow, dict):
        return _error(400, "invalid_body", "workflow must be an object")
    if name is not None and not (isinstance(name, str) and len(name) <= 200):
        return _error(400, "invalid_body", "name must be a string of at most 200 characters")
    return {k: v for k, v in (("workflow", workflow), ("name", name)) if v is not None}


def _created(data) -> tuple | None:
    """(id, workspaceId, expiresAt) when all three are non-empty strings."""
    if not isinstance(data, dict):
        return None
    fields = (data.get("id"), data.get("workspaceId"), data.get("expiresAt"))
    return fields if all(isinstance(f, str) and f for f in fields) else None


class DeployToCloudRoutes:
    def __init__(self, prompt_server):
        self.prompt_server = prompt_server
        update_feature_flag()

    def add_routes(self, app: web.Application) -> None:
        app.add_routes([
            web.get("/api/deploy/summary", self.summary),
            web.post("/api/deploy/bundle", self.bundle),
        ])

    async def summary(self, request: web.Request) -> web.Response:
        if args.offline:
            return _error(403, "offline")
        custom_nodes = await asyncio.to_thread(count_custom_nodes)
        models = await asyncio.to_thread(list_models)
        return web.json_response({"custom_nodes": custom_nodes, "models": len(models)})

    async def bundle(self, request: web.Request) -> web.Response:
        if args.offline:
            return _error(403, "offline")
        credential = _credential(request)
        if isinstance(credential, web.Response):
            return credential
        fields = await _bundle_fields(request)
        if isinstance(fields, web.Response):
            return fields

        snapshot, models = await asyncio.to_thread(collect)
        payload = {"snapshot": snapshot, "models": models, **fields, "source": "comfyui"}
        try:
            async with self.prompt_server.client_session.post(
                builder_base_url() + "/v1/import-bundles",
                json=payload,
                headers=dict([credential]),
                timeout=_UPLOAD_TIMEOUT,
                allow_redirects=False,
            ) as resp:
                status = resp.status
                retry_after = resp.headers.get("Retry-After", "")
                try:
                    data = await resp.json(content_type=None) if status in (201, 400, 429, 503) else None
                except ValueError:  # a non-JSON 400 is still a rejection; a non-JSON 201 has no bundle
                    data = None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logging.warning("Deploy to cloud: builder request failed (%s)", type(e).__name__)
            return _error(502, "builder_unavailable")

        created = _created(data) if status == 201 else None
        if created is not None:
            bundle_id, workspace_id, expires_at = created
            logging.info("Deploy to cloud: uploaded import bundle %s", bundle_id)
            query = urllib.parse.urlencode({
                "step": "import", "bundle": bundle_id, "workspace": workspace_id, "bundle_workspace": workspace_id,
            })
            return web.json_response({
                "url": f"{platform_base_url()}/profile/builds/new?{query}",
                "bundle_id": bundle_id,
                "workspace_id": workspace_id,
                "expires_at": expires_at,
            })

        logging.warning("Deploy to cloud: builder returned status %d", status)
        if status == 400:
            message = data.get("message") if isinstance(data, dict) else None
            return _error(502, "builder_rejected", message[:500] if isinstance(message, str) else "")
        builder_code = data.get("error") if isinstance(data, dict) else None
        # Only IMPORT_BUNDLE_LIMIT is the per-user bundle cap; any other 429 (e.g. IMPORT_BUNDLE_IN_PROGRESS) is transient.
        if (status == 429 and builder_code != "IMPORT_BUNDLE_LIMIT") or (status == 503 and builder_code == "IMPORT_BUNDLE_BUSY"):
            headers = {"Retry-After": retry_after} if re.fullmatch(r"[0-9]+", retry_after) else None
            return web.json_response({"error": "builder_busy"}, status=503, headers=headers)
        mapped_status, code = _BUILDER_ERRORS.get(status, (502, "builder_unavailable"))
        return _error(mapped_status, code)
