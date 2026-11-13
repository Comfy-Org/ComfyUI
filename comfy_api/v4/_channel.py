"""The socket between ComfyUI and one pack process, and how calls cross it.

A chain is one top-level request and every call nested in it, on either side. Each message
carries its chain. A call blocks the thread that makes it; while it waits, that thread serves
the calls of its own chain, so nested calls run on the threads they would run on in one process.

In ComfyUI a thread waiting on a pack also serves that pack's calls whose home is its own event
loop: without that, a node of the pack waiting on ComfyUI and ComfyUI waiting on the pack (for a
hook during sampling) would wait for each other. So, unlike in one process, a call one of the
pack's nodes makes can run inside a wait on the same pack, such as in the middle of a sampling
step that calls the pack's hook. Another pack's calls wait for the loop.

A message is one frame: the lengths of its parts, a pickled head, the body `_codec` made, and the
bytes of each tensor and array in the body. A block of `JOINED` bytes or more is written from the
sender's memory and read into the memory the receiver's tensor uses; smaller parts are copied
into one send, and a run of small blocks is read at once and copied apart. One thread sends a
whole frame before another sends.

Each side has one reader thread, which only reads messages and hands them on. Nothing else
runs on a thread of its own. Where `side.poll` is a number of seconds (in ComfyUI), the reader
also calls `side.tick` that often, whether messages come or not and also while one is arriving,
and ends the connection when nothing has come for that long and `side.ended()`: the other
process has ended, but a process it started may still hold its end of the socket.

Where `side.stall` is a number of seconds (in ComfyUI), a send ends the connection once the
other side has taken nothing for that long, so no thread of ComfyUI waits for ever on a pack
that has stopped reading; the reader then ends the pack's process. The reader itself never
waits to send, and ComfyUI's event loops send their requests from another thread, so neither
stops for a pack that takes nothing. A pack process waits on ComfyUI for as long as it takes.

What each side keeps, serves and starts is up to its `side` object: `_pack_host.PackProcess`
in ComfyUI, `_pack_main.Pack` in a pack process. Both have `peer`, `poll`, `stall`, `keep`, `forget`,
`hold`, `resolve`, `offers`, `remote`, `serve`, `schedule` and `stopped`; ComfyUI's also `tick`,
`ended` and `output`, and the pack process's `release` and `cancel`.
"""

import builtins
import contextlib
import contextvars
import functools
import itertools
import logging
import pickle
import queue
import selectors
import socket
import struct
import threading
import time
import traceback
import weakref
from collections.abc import Callable

import numpy
import torch

from comfy_api._exceptions import InterruptProcessingException

from . import _codec

_chains: contextvars.ContextVar[dict] = contextvars.ContextVar("chains", default={})
_local = threading.local()
_request_ids = itertools.count(1)
# A frame starts with the lengths of its head and body and its number of blocks, then each
# block's length.
_SIZES = struct.Struct("!QQQ")
_LENGTH = struct.Struct("!Q")
# With macOS's default of 8 KiB, a 256 MiB tensor crosses 13 times slower (measured).
BUFFER = 1 << 20
# macOS refuses one send or receive of 2 GiB or more.
PIECE = 256 << 20
# Parts of a frame under this size go in one system call with their neighbours: a copy of them
# costs less than a call.
JOINED = 64 << 10
# Each class lives while an exception of it does, so names the other side makes up do not pile up.
_error_classes: weakref.WeakValueDictionary[tuple[str, str], type] = (
    weakref.WeakValueDictionary()
)


class PackError(Exception):
    """Stands for an exception class of the other process that is not one of Python's own."""


class PackStopped(RuntimeError):
    """The other process ended while a call was waiting for it."""


class RemoteTraceback(Exception):
    """The traceback of an exception raised in the other process (its `__cause__`)."""

    def __str__(self) -> str:
        return self.args[0]


# What an exception's arguments are sent as; any others go as the exception's message.
_PLAIN = (str, int, float, bool, type(None))


def describe(error: BaseException) -> dict:
    cls = type(error)
    args = error.args
    if not all(type(arg) in _PLAIN for arg in args):
        try:
            args = (str(error),)
        except Exception:
            args = (f"{cls.__qualname__} (its message could not be made)",)
    return {
        "module": cls.__module__,
        "qualname": cls.__qualname__,
        "args": args,
        "traceback": "".join(traceback.format_exception(error)),
    }


def rebuild(description: dict) -> BaseException:
    """The exception `describe` described: Python's own class, ComfyUI's cancel, or a `PackError`.

    Python's own exceptions that are not an `Exception` (`SystemExit`, `KeyboardInterrupt`,
    `GeneratorExit`) become `PackError`s too: raised here, they would end the thread that waits.
    So do `StopIteration` and `StopAsyncIteration`, which a waiting coroutine cannot be given.
    """
    module, qualname, args = (
        description["module"],
        description["qualname"],
        description["args"],
    )
    text = (module, qualname, description["traceback"])
    if not (
        all(type(part) is str for part in text)
        and type(args) is tuple
        and all(type(arg) in _PLAIN for arg in args)
    ):
        # Not what `describe` sends: printing such an error could raise where it is reported.
        error = PackError("The other process sent an error that could not be read")
        error.__cause__ = RemoteTraceback("(no traceback)")
        return error
    cls = getattr(builtins, qualname, None) if module == "builtins" else None
    if (module, qualname) == (
        InterruptProcessingException.__module__,
        "InterruptProcessingException",
    ):
        cls = InterruptProcessingException
    elif not (
        isinstance(cls, type)
        and issubclass(cls, Exception)
        # An asyncio future refuses these two, and a generator would take them for its own end.
        and not issubclass(cls, (StopIteration, StopAsyncIteration))
    ):
        cls = _error_classes.get((module, qualname))
        if cls is None:
            name = qualname.rpartition(".")[2]
            attributes = {"__module__": module, "__qualname__": qualname}
            cls = _error_classes[module, qualname] = type(
                name, (PackError,), attributes
            )
    try:
        error = cls(*args)
    except TypeError:
        error = PackError(f"{qualname}: {', '.join(map(str, args))}")
    error.__cause__ = RemoteTraceback(description["traceback"])
    return error


def current_chain(side) -> int | None:
    """The chain of `side` this thread is in, if any."""
    return _chains.get().get(side)


def enter_chain(side, chain: int) -> contextvars.Token:
    return _chains.set({**_chains.get(), side: chain})


def leave_chain(token: contextvars.Token) -> None:
    _chains.reset(token)


def inbox(side) -> queue.SimpleQueue:
    """This thread's queue for `side`: the replies to its calls, and the calls it is to serve
    meanwhile. A thread waiting on one pack serves no other pack's calls."""
    if not hasattr(_local, "inboxes"):
        _local.inboxes, _local.replies = {}, {}
    return _local.inboxes.setdefault(side, queue.SimpleQueue())


def _drain(queued: queue.SimpleQueue) -> list:
    items = []
    with contextlib.suppress(queue.Empty):
        while True:
            items.append(queued.get_nowait())
    return items


def serve_queued(queued: queue.SimpleQueue) -> None:
    """Serve the calls queued for this thread that a wait on it has not served already."""
    for serve in _drain(queued):
        serve()


def write_frame(sock: socket.socket, head: bytes, body: bytes, blocks=()) -> None:
    """Write one frame. On a socket with a timeout, raises `TimeoutError` once the other side
    has taken nothing for that long; on one without, waits for as long as it takes."""
    lengths = b"".join(_LENGTH.pack(block.nbytes) for block in blocks)
    joined = bytearray(_SIZES.pack(len(head), len(body), len(blocks)) + lengths + head)
    for part in (body, *blocks):
        view = memoryview(part)
        if view.nbytes < JOINED:
            joined += view
            continue
        _write(sock, joined)
        joined = bytearray()
        _write(sock, view)
    _write(sock, joined)


def _write(sock: socket.socket, data) -> None:
    view = memoryview(data)
    while view:
        # One send at a time: with a timeout each waits only until some of it is taken, where
        # `sendall` would count the time for the whole.
        view = view[sock.send(view[:PIECE]) :]


def read_frame(
    sock: socket.socket, wait: Callable[[], object]
) -> tuple[memoryview, memoryview, list]:
    """Read one frame; `wait` is called before each read and returns once there is something
    to read."""
    head, body, count = _SIZES.unpack(_read(sock, _SIZES.size, wait))
    start = _read(sock, count * _LENGTH.size + head + body, wait)
    lengths = struct.unpack_from(f"!{count}Q", start)
    start = start[count * _LENGTH.size :]
    return start[:head], start[head:], _read_blocks(sock, lengths, wait)


def _read_blocks(sock: socket.socket, lengths: tuple[int, ...], wait) -> list:
    """Each block in memory of its own; a run of small blocks is read at once and then copied
    apart, so that each starts where its numbers are aligned."""
    blocks, run = [], []
    for length in (*lengths, None):
        if length is not None and length < JOINED:
            run.append(length)
            continue
        if run:
            data = _read(sock, sum(run), wait)
            blocks += [
                bytearray(data[end - n : end])
                for n, end in zip(run, itertools.accumulate(run))
            ]
            run = []
        if length is not None:
            blocks.append(_read(sock, length, wait))
    return blocks


def _read(sock: socket.socket, size: int, wait) -> memoryview:
    """`size` bytes in new memory, which is not touched before the bytes arrive: the size is
    what the other side said."""
    data = memoryview(numpy.empty(size, numpy.uint8))
    view = data
    while view:
        wait()
        received = sock.recv_into(view[:PIECE])
        if not received:
            raise EOFError
        view = view[received:]
    return data


def _writable(sock: socket.socket, timeout: float) -> bool:
    with selectors.DefaultSelector() as selector:
        selector.register(sock, selectors.EVENT_WRITE)
        return bool(selector.select(timeout))


class Channel:
    """One end of the socket: calls in both directions, served on the threads described above."""

    def __init__(self, sock: socket.socket, side):
        for option in (socket.SO_SNDBUF, socket.SO_RCVBUF):
            sock.setsockopt(socket.SOL_SOCKET, option, BUFFER)
        # A send then fails once the other side has taken nothing for `stall` seconds. The
        # reader reads only what has arrived, so the timeout does not end a read.
        sock.settimeout(side.stall)
        self.sock, self.side = sock, side
        self.lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.closed = False
        self.tick_due = time.monotonic()
        # request id -> called with the reply, on the reader thread
        self.waiters = {}
        # chain -> the inboxes of the threads waiting in it, innermost last
        self.waiting: dict[int, list[queue.SimpleQueue]] = {}
        # keys of the other side's objects this side no longer holds; sent with the next message
        self.releases = queue.SimpleQueue()

    def start(self, name: str) -> None:
        self.reader = threading.Thread(target=self._read, name=name, daemon=True)
        self.reader.start()

    def after_fork(self) -> None:
        """In a child forked from this process: close the child's copy of the socket, and replace
        the locks the parent's other threads may have held, so that what the child sends fails at
        once as if the other side had stopped."""
        self.lock, self.send_lock = threading.Lock(), threading.Lock()
        self.closed = True
        self.sock.close()

    def call(
        self,
        chain: int,
        target: tuple,
        args: tuple,
        kwargs: dict,
        lane: str = "exec",
        timeout: float | None = None,
    ):
        """Call the other side and block until it answers, serving this thread's calls meanwhile.

        After `timeout` seconds without an answer, raises `TimeoutError`; the answer is then
        dropped when it comes. With a `timeout`, it also raises `TimeoutError`, sending nothing,
        if the call cannot be sent in that time; the time it takes to send comes out of the
        time it waits for the answer. Refused on the reader thread, which alone could
        read the answer: an object freed there (its `__del__`) cannot call the other side.
        """
        if threading.current_thread() is self.reader:
            raise RuntimeError(
                f"{self.side.peer} cannot be called from its reader thread"
            )
        deadline = None if timeout is None else time.monotonic() + timeout
        request_id = next(_request_ids)
        body, blocks = _codec.dumps((args, kwargs), self.side, chain)
        waiting = inbox(self.side)
        with self.lock:
            if self.closed:
                raise PackStopped(f"{self.side.peer} has stopped")
            self.waiters[request_id] = waiting.put
            self.waiting.setdefault(chain, []).append(waiting)
        try:
            head = self._head(request_id, chain, target, lane)
            if not self._send(head, body, blocks, timeout):
                raise TimeoutError(
                    f"{self.side.peer} did not take the call in {timeout} s"
                )
            return self.outcome(self._wait(request_id, waiting, deadline))
        finally:
            self._done_waiting(request_id, chain, waiting)

    def _done_waiting(
        self, request_id: int, chain: int, waiting: queue.SimpleQueue
    ) -> None:
        with self.lock:
            if self.waiters.get(request_id) == waiting.put:
                del self.waiters[request_id]
            self.waiting[chain].remove(waiting)
            if not self.waiting[chain]:
                del self.waiting[chain]

    def _wait(
        self, request_id: int, waiting: queue.SimpleQueue, deadline: float | None
    ) -> dict:
        """The reply to `request_id`, serving the calls queued for this thread until it comes
        or, with a `deadline`, until then."""
        replies = _local.replies
        while request_id not in replies:
            left = None if deadline is None else deadline - time.monotonic()
            try:
                # The time is up also while calls keep coming to serve.
                if left is not None and left <= 0:
                    raise queue.Empty
                item = waiting.get(timeout=left)
            except queue.Empty:
                self._give_up(request_id, waiting)
                # The reader has taken the answer: it is on its way here.
                deadline = None
                continue
            if callable(item):
                item()
            else:
                replies[item["to"]] = item
        return replies.pop(request_id)

    def _give_up(self, request_id: int, waiting: queue.SimpleQueue) -> None:
        """Raise `TimeoutError`, unless the reader has taken the answer already."""
        with self.lock:
            if self.waiters.get(request_id) == waiting.put:
                self.waiters[request_id] = self.drop
                raise TimeoutError(f"{self.side.peer} did not answer in time")

    def prepare_call(
        self, chain: int, target: tuple, args: tuple, lane: str, deliver
    ) -> tuple[int, Callable[[], None]]:
        """Make a call to the other side that is not waited for: encode it on this thread, in
        its grad and inference modes, and return its id and the function that sends it, which
        another thread may run. `deliver` gets the reply, on the reader thread; the id is in
        `waiters` until the reply has come."""
        request_id = next(_request_ids)
        body, blocks = _codec.dumps((args, {}), self.side, chain)
        head = self._head(request_id, chain, target, lane)
        with self.lock:
            if self.closed:
                raise PackStopped(f"{self.side.peer} has stopped")
            self.waiters[request_id] = deliver

        def send() -> None:
            try:
                self._send(head, body, blocks)
            except PackStopped:
                with self.lock:
                    self.waiters.pop(request_id, None)
                raise

        return request_id, send

    def cancel(self, chain: int, lane: str) -> bool:
        """Ask the other side to cancel the request of `chain` it runs on `lane`; no reply.

        Never waits: returns False, sending nothing, while another thread sends or the socket has
        no room, as when the other side does not read.
        """
        return self._send({"cancel": chain, "lane": lane}, timeout=0)

    def notify(self, head: dict) -> None:
        """Send `head` to the other side, which does not answer it."""
        self._send(head)

    def release(self, key) -> None:
        """Tell the other side, with the next message, that this side no longer holds `key`.

        Safe to call from a finalizer, on any thread.
        """
        self.releases.put(key)

    def answer(self, call: dict) -> None:
        """Serve `call` on this thread, in the grad and inference modes of its caller, and reply."""
        self.prepare_answer(call)()

    def prepare_answer(self, call: dict) -> Callable[[], None]:
        """Serve `call` on this thread, in the grad and inference modes of its caller, and
        return the function that sends the reply, which another thread may run."""
        try:
            with (
                torch.inference_mode(call["inference"]),
                torch.set_grad_enabled(call["grad"]),
            ):
                args, kwargs = self.decode(call)
                value = self.side.serve(call["target"], args, kwargs)
        except BaseException as error:
            send = self._prepare_reply(call, error=error)
            # Any other exception that is not an Exception goes on, as it would in one process.
            if not isinstance(error, (Exception, InterruptProcessingException)):
                send()
                raise
            return send
        return self._prepare_reply(call, value)

    def reply(
        self,
        call: dict,
        value=None,
        error: BaseException | None = None,
        timeout: float | None = None,
    ) -> None:
        """Answer `call`. If the other side has stopped, there is nobody to answer: the reader
        tells whoever waits here. With a `timeout`, the answer is not sent if it cannot be in
        that time."""
        self._prepare_reply(call, value, error)(timeout)

    def _prepare_reply(
        self, call: dict, value=None, error: BaseException | None = None
    ) -> Callable[..., None]:
        """Encode the answer to `call` on this thread and return the function that sends it."""
        head = {"to": call["id"], "chain": call["chain"]}
        body, blocks = b"", []
        if error is None:
            try:
                body, blocks = _codec.dumps(value, self.side, call["chain"])
            except Exception as failure:
                error = failure
        if error is not None:
            head = {**head, "error": describe(error)}

        def send(timeout: float | None = None) -> None:
            with contextlib.suppress(PackStopped):
                self._send(head, body, blocks, timeout)

        return send

    def decode(self, message: dict):
        """The arguments of a call, or the value of a reply."""
        return _codec.loads(
            message["body"], self.side, message["chain"], message["blocks"]
        )

    def outcome(self, reply: dict):
        """The value of `reply`, or what it raises."""
        if "stopped" in reply:
            raise PackStopped(f"{self.side.peer} stopped")
        if "error" in reply:
            raise rebuild(reply["error"])
        return self.decode(reply)

    def drop(self, message: dict) -> None:
        """Let go of a message nobody uses: decoding it releases the other side's objects in it.
        Its value or error is of no use to anyone."""
        if message.get("body"):
            with contextlib.suppress(Exception):
                self.decode(message)

    @staticmethod
    def _head(request_id: int, chain: int, target: tuple, lane: str) -> dict:
        return {
            "id": request_id,
            "chain": chain,
            "target": target,
            "lane": lane,
            "grad": torch.is_grad_enabled(),
            "inference": torch.is_inference_mode_enabled(),
        }

    def _send(
        self,
        head: dict,
        body: bytes = b"",
        blocks=(),
        timeout: float | None = None,
    ) -> bool:
        """Send a frame whole. With a `timeout`, returns False, sending nothing, if in that time
        no other thread has finished sending or the socket has had no room; and what it sends
        is the frame alone, without the releases, whose number the other side decides."""
        deadline = None if timeout is None else time.monotonic() + timeout
        if not self.send_lock.acquire(timeout=-1 if timeout is None else timeout):
            return False
        try:
            if deadline is not None and not _writable(
                self.sock, max(0.0, deadline - time.monotonic())
            ):
                return False
            released = _drain(self.releases) if timeout is None else []
            if released:
                head = {**head, "release": released}
            write_frame(self.sock, pickle.dumps(head, protocol=5), body, blocks)
        except BaseException as error:
            # A frame may be cut short, so the socket is of no further use: end it, and the
            # reader fails every waiter.
            with contextlib.suppress(OSError):
                self.sock.shutdown(socket.SHUT_RDWR)
            if isinstance(error, OSError):
                raise PackStopped(f"{self.side.peer} has stopped") from error
            raise
        finally:
            self.send_lock.release()
        return True

    def _read(self) -> None:
        try:
            self._receive()
        except (EOFError, OSError):
            pass
        except Exception:
            logging.exception(f"{self.side.peer}: the connection ends after an error")
        # What is still sent now fails at once instead of filling a buffer nobody reads.
        with contextlib.suppress(OSError):
            self.sock.shutdown(socket.SHUT_RDWR)
        try:
            self.side.stopped()
        finally:
            # Also when ending the other process failed: nobody may wait for an answer.
            with self.lock:
                self.closed = True
                waiters, self.waiters = self.waiters, {}
            for request_id, waiter in waiters.items():
                waiter({"to": request_id, "stopped": True})

    def _receive(self) -> None:
        """Hand on each message until the connection ends; and in ComfyUI, tick meanwhile."""
        with selectors.DefaultSelector() as selector:
            selector.register(self.sock, selectors.EVENT_READ)
            wait = functools.partial(self._readable, selector)
            while self._readable(selector, between_messages=True):
                head, body, blocks = read_frame(self.sock, wait)
                message = _codec.loads(head, self.side)
                self._route({**message, "body": body, "blocks": blocks})

    def _readable(self, selector, between_messages: bool = False) -> bool:
        """Return once there is something to read, ticking meanwhile. When the other process
        has ended and nothing comes, return False between messages and raise `EOFError` inside
        one."""
        while not selector.select(self.side.poll):
            if self.side.ended():
                if between_messages:
                    return False
                raise EOFError
            self._tick()
        self._tick()
        return True

    def _tick(self) -> None:
        if self.side.poll is not None and time.monotonic() >= self.tick_due:
            self.side.tick()
            self.tick_due = time.monotonic() + self.side.poll

    def _route(self, message: dict) -> None:
        """Hand a reply to its caller, and a call to the thread waiting in its chain, or else to
        the side's `schedule`."""
        if "release" in message:
            self.side.release(message["release"])
        if "cancel" in message:
            self.side.cancel(message["cancel"], message["lane"])
            return
        if "output" in message:
            self.side.output(message)
            return
        with self.lock:
            if "to" in message:
                deliver = self.waiters.pop(message["to"], self.drop)
            else:
                stack = self.waiting.get(message["chain"])
                if stack:
                    stack[-1].put(functools.partial(self.answer, message))
                    return
                deliver = self.side.schedule
        deliver(message)
