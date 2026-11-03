"""Loads the optional comfy-agent package and fronts it to ComfyUI.

ComfyUI owns a thin v1 contract and nothing more: it discovers the
``comfyui.agent`` "v1" entry point, creates the host, hands it the live aiohttp
app once (``register(app)``) so the host mounts its own routes on ComfyUI's
origin, starts it after the server is listening, stops it on shutdown, and
publishes the host's state as the ``agent`` feature flag plus an ``agent_status``
websocket message on every change. Everything else -- the agent process, its
storage, restarts, the proxy and the loopback/same-origin trust boundary --
lives in the package.

A v1 host must provide:
    register(app) -> None
    start(*, comfy_url: str, comfy_path: str, logger, on_change=None) -> None
    stop(timeout: float = ...) -> None
    status() -> {"state": "starting"|"ready"|"failed", "reason": str | None, ...}
"""

from __future__ import annotations

import importlib.metadata
import logging
import os

import folder_paths
from comfy_api import feature_flags

ENTRY_POINT_GROUP = "comfyui.agent"
PROTOCOL = "v1"

_host = None
# Set when loading or starting the host failed, so the flag reports why rather
# than leaving the agent looking switched off.
_failure = None
# feature_flag() can be read on every connection; log a raising status() once.
_status_logged = False


def emit_event(event, /, **fields) -> None:
    logging.info(" ".join([f"[agent-event] {event}", *(f"{k}={v}" for k, v in fields.items() if v is not None)]))


def load() -> bool:
    """Find and create the agent host. Reports why when it can't; nothing about
    the installed package can stop ComfyUI from starting."""
    global _host, _failure
    try:
        entry_points = {ep.name: ep for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)}
        if PROTOCOL not in entry_points:
            # Some comfy-agent is installed (other protocols present) -> "update";
            # nothing there -> "install".
            if entry_points:
                logging.warning(f"The installed comfy-agent offers agent protocols {sorted(entry_points)}; this ComfyUI needs {PROTOCOL}. Update comfy-agent to use the agent.")
                _failure = {"state": "failed", "reason": "outdated"}
                emit_event("agent_error", reason="outdated")
            else:
                req = os.path.join(os.path.dirname(os.path.abspath(folder_paths.__file__)), "agent_requirements.txt")
                logging.warning(f"`--enable-agent` was passed but the comfy-agent package is not installed, so the agent is disabled. Its requirements are in {req}.")
                _failure = {"state": "failed", "reason": "package_missing"}
                emit_event("package_missing")
            _publish_flag()
            return False
        host = entry_points[PROTOCOL].load()()
        if not all(callable(getattr(host, name, None)) for name in ("start", "stop", "status", "register")):
            raise TypeError(f"the {PROTOCOL} entry point returned {type(host).__name__}, not an agent host")
        _host = host
    except (Exception, SystemExit):  # a broken wheel may exit; that must not end ComfyUI's startup
        logging.exception("Could not load the local agent; it is disabled.")
        _failure = {"state": "failed", "reason": "unknown"}
        emit_event("agent_error", reason="unknown")
        _publish_flag()
        return False
    _publish_flag()  # reads status(), which disables a host that exits
    return _failure is None


def register(app) -> None:
    """Mount the agent's own API routes onto ComfyUI's aiohttp ``app``.

    ComfyUI calls this once while it registers its routes (before the router
    freezes); the host mounts /api/agent and /api/workflows onto the live app.
    No-op when the package did not load; a failure here disables the agent, not
    ComfyUI's startup."""
    global _failure
    if _host is None or _failure is not None:
        return
    try:
        _host.register(app)
    except (Exception, SystemExit):  # a broken wheel must not end ComfyUI's startup
        logging.exception("The local agent's routes could not be registered; it is disabled.")
        _failure = {"state": "failed", "reason": "unknown"}
        emit_event("agent_error", reason="unknown")
        _publish_flag()


def start(server, scheme: str, address: str, port: int) -> None:
    """Start the loaded host once the server is listening. Main thread only."""
    # register() may have failed after load(): _host stays set but _failure is
    # now recorded. Starting then would spawn an agent with no routes mounted
    # (claiming its data dir) while the flag reports failed, so bail out too.
    if _host is None or _failure is not None:
        return
    host = "127.0.0.1" if address in ("", "0.0.0.0") else "::1" if address == "::" else address
    comfy_url = f"{scheme}://[{host}]:{port}" if ":" in host else f"{scheme}://{host}:{port}"

    last_sent = None

    def on_change():
        # Called from the host's threads; last_sent is unsynchronized, so a race
        # costs at most one duplicate message. The host also reports changes the
        # published flag doesn't carry (e.g. its restart count), so only send
        # when the flag itself moves.
        nonlocal last_sent
        flag = _publish_flag()
        if flag != last_sent:
            last_sent = flag
            server.send_sync("agent_status", flag)

    def run():
        global _failure
        try:
            _host.start(comfy_url=comfy_url, comfy_path=os.path.dirname(os.path.abspath(folder_paths.__file__)), logger=logging.getLogger(), on_change=on_change)
        except (Exception, SystemExit):
            logging.exception("The local agent could not be started.")
            _failure = {"state": "failed", "reason": "spawn_failed"}
            emit_event("agent_error", reason="spawn_failed")
            on_change()

    # start() spawns and waits on the agent child, so keep it off the event loop.
    if server.loop is not None:
        server.loop.run_in_executor(None, run)
    else:
        run()


def stop() -> None:
    if _host is not None:
        try:
            _host.stop()
        except (Exception, SystemExit):
            logging.exception("The local agent could not be stopped.")


def _publish_flag():
    """Mirror the current agent flag into the shared server feature-flag dict so
    new connections read it via ``get_server_features()``; omitted when there is
    no value. Returns the flag so callers can detect transitions."""
    flag = feature_flag()
    if flag is None:
        feature_flags.SERVER_FEATURE_FLAGS.pop("agent", None)
    else:
        feature_flags.SERVER_FEATURE_FLAGS["agent"] = flag
    return flag


def feature_flag():
    """The `agent` feature flag: the host's state, and its reason when it has one."""
    global _failure, _status_logged
    if _failure is not None or _host is None:
        return _failure
    try:
        status = _host.status()
        flag = {"state": status["state"]}
        if status.get("reason"):
            flag["reason"] = status["reason"]
        return flag
    except SystemExit:  # a wheel that exits is broken, not briefly unhealthy, so disable it
        logging.exception("The local agent's status could not be read; it is disabled.")
        _failure = {"state": "failed", "reason": "unknown"}
        emit_event("agent_error", reason="unknown")
        return _failure
    except Exception:
        if not _status_logged:
            _status_logged = True
            logging.exception("The local agent's status could not be read.")
        return {"state": "failed", "reason": "unknown"}
