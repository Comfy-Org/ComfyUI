import ast
import asyncio
import logging
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="the proactor event loop is Windows only")


def _load_iocp_proactor():
    main_path = Path(__file__).resolve().parents[1] / "main.py"
    module = ast.parse(main_path.read_text(), filename=str(main_path))
    cls = next(node for node in ast.walk(module) if isinstance(node, ast.ClassDef) and node.name == "IocpProactor")
    compiled = compile(ast.Module(body=[cls], type_ignores=[]), filename=str(main_path), mode="exec")
    namespace = {"asyncio": asyncio, "logging": logging}
    exec(compiled, namespace)  # noqa: S102 - trusted AST extracted from main.py itself, not external input
    return namespace["IocpProactor"]


def _serve_after_failed_accept(monkeypatch, proactor, winerror):
    """Fails the server's first accept with winerror, then reports whether a client still gets served and what the loop's exception handler saw."""
    real_accept = asyncio.IocpProactor.accept
    calls = 0

    def flaky_accept(self, listener):
        nonlocal calls
        calls += 1
        if calls > 1:
            return real_accept(self, listener)
        failed = self._loop.create_future()
        failed.set_exception(OSError(None, "injected", None, winerror))
        return failed

    monkeypatch.setattr(asyncio.IocpProactor, "accept", flaky_accept)

    async def echo(reader, writer):
        writer.write(await reader.read(4))
        await writer.drain()
        writer.close()

    async def run():
        server = await asyncio.start_server(echo, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            await asyncio.sleep(0.1)
            reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), 5)
            writer.write(b"ping")
            reply = await asyncio.wait_for(reader.read(4), 5)
            writer.close()
            return reply == b"ping"
        except OSError:
            return False
        finally:
            server.close()
            await asyncio.wait_for(server.wait_closed(), 5)

    handled = []
    loop = asyncio.ProactorEventLoop(proactor)
    loop.set_exception_handler(lambda loop, context: handled.append(context["message"]))
    try:
        served = loop.run_until_complete(run())
    finally:
        loop.close()
    return served, handled


def test_stock_proactor_stops_listening_after_client_disconnects(monkeypatch):
    served, handled = _serve_after_failed_accept(monkeypatch, asyncio.IocpProactor(), 64)
    assert not served
    assert "Accept failed on a socket" in handled


def test_keeps_listening_after_client_disconnects(monkeypatch):
    served, handled = _serve_after_failed_accept(monkeypatch, _load_iocp_proactor()(), 64)
    assert served
    assert handled == []


def test_other_accept_errors_still_close_listener(monkeypatch):
    served, handled = _serve_after_failed_accept(monkeypatch, _load_iocp_proactor()(), 87)
    assert not served
    assert "Accept failed on a socket" in handled
