import argparse
import asyncio
import json
import os
import secrets
from urllib.parse import urlsplit

from aiohttp import web

from gateway.app import Gateway

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def serve(gateway, port, session_path, origins):
    runner = web.AppRunner(gateway.make_app())
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", port).start()
        # Written only once the port is ours, so a second copy that fails to bind leaves the running one intact.
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump({"port": port, "token": gateway.token}, f)
        print(f"Remote execution gateway on http://127.0.0.1:{port} for {sorted(origins)}", flush=True)
        try:
            await asyncio.Event().wait()
        finally:
            os.remove(session_path)
    finally:
        await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Loopback gateway that runs the local ComfyUI editor against a remote ComfyUI.")
    parser.add_argument("--port", type=int, default=8190)
    parser.add_argument("--local-url", default="http://127.0.0.1:8188", help="The local ComfyUI server whose editor uses this gateway.")
    parser.add_argument("--ui-origin", action="append", default=[], help="Additional browser origin allowed to use the gateway.")
    parser.add_argument("--data-dir", default=os.path.join(PACKAGE_DIR, "data"))
    args = parser.parse_args()

    local = urlsplit(args.local_url)
    origins = {f"{local.scheme}://{local.netloc}", *args.ui_origin}
    if local.hostname in ("127.0.0.1", "localhost"):
        origins |= {f"{local.scheme}://127.0.0.1:{local.port}", f"{local.scheme}://localhost:{local.port}"}

    os.makedirs(args.data_dir, exist_ok=True)
    gateway = Gateway(args.local_url, secrets.token_urlsafe(32), origins, args.data_dir)
    # The local ComfyUI extension reads the session file to reach the gateway; see ../__init__.py.
    asyncio.run(serve(gateway, args.port, os.path.join(args.data_dir, "session.json"), origins))


if __name__ == "__main__":
    main()
