import atexit
import json
import os
import subprocess
import sys

from aiohttp import web

from comfy.cli_args import args
from server import PromptServer

# Frontend extension only: no nodes and no outbound requests. Remote traffic goes through the
# gateway, a separate process started here so launching ComfyUI is the only step.
WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {}

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PACKAGE_DIR, "data")
SESSION_PATH = os.path.join(DATA_DIR, "session.json")

os.makedirs(DATA_DIR, exist_ok=True)
# If a gateway already holds the port, this one exits and the running one keeps serving.
gateway_process = subprocess.Popen(
    [sys.executable, "-m", "gateway", "--local-url", f"http://127.0.0.1:{args.port}"],
    cwd=PACKAGE_DIR,
    stdout=open(os.path.join(DATA_DIR, "gateway.log"), "a"),
    stderr=subprocess.STDOUT,
)
atexit.register(gateway_process.terminate)


@PromptServer.instance.routes.get("/remote_execution/session")
async def gateway_session(request):
    if not os.path.exists(SESSION_PATH):
        return web.json_response({"error": "The remote execution gateway is not running; see custom_nodes/comfyui_remote_execution/data/gateway.log."}, status=404)
    with open(SESSION_PATH, encoding="utf-8") as f:
        return web.json_response(json.load(f))
