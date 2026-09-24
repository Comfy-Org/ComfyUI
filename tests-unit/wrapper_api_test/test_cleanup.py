"""Tests for the worker cleanup endpoint (api_wrapper/cleanup.py).

The PromptServer is stubbed like tests-unit/server_test/test_wrapper_auth.py
does: a tiny aiohttp app gets the real DELETE route (registered by the same
function routes.py calls), the /api mirror server.py adds, and optionally the
real bearer auth middleware. The prompt queue is a plain fake with the
PromptQueue methods the endpoint uses, and ComfyUI's output/input/temp
directories are temporary folders.
"""

import os
import threading
import uuid

import pytest
from aiohttp import web

from api_wrapper import cleanup
from api_wrapper import progress as wrapper_progress
from middleware.wrapper_auth import create_bearer_auth_middleware

TOKEN = "s3cret-token"
GOOD = {"Authorization": f"Bearer {TOKEN}"}


class FakePromptQueue:
    def __init__(self):
        self.mutex = threading.RLock()
        self.currently_running = {}
        self.queue = []
        self.history = {}

    def get_history(self, prompt_id=None):
        if prompt_id is None:
            return dict(self.history)
        return {prompt_id: self.history[prompt_id]} if prompt_id in self.history else {}

    def delete_history_item(self, id_to_delete):
        self.history.pop(id_to_delete, None)


@pytest.fixture
def roots(tmp_path):
    made = cleanup.Roots(str(tmp_path / "output"), str(tmp_path / "input"), str(tmp_path / "temp"))
    for path in made:
        os.makedirs(path)
    return made


@pytest.fixture
def outside(tmp_path):
    path = tmp_path / "outside"
    path.mkdir()
    secret = path / "secret.txt"
    secret.write_text("keep me")
    return path


@pytest.fixture
def queue():
    return FakePromptQueue()


def write(path, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def make_app(prompt_queue, roots, auth_token=None):
    """The wrapper's DELETE route on a bare app, mirrored under /api the way
    server.py mirrors every route, next to a GET on the same path (routes.py
    has one) to prove the two coexist."""
    routes = web.RouteTableDef()

    @routes.get("/wrapper/jobs/{job_id}")
    async def get_job(request):
        return web.json_response({"get": request.match_info["job_id"]})

    cleanup.register_routes(routes, prompt_queue, roots_provider=lambda: roots)

    api_routes = web.RouteTableDef()
    for route in routes:
        api_routes.route(route.method, "/api" + route.path)(route.handler, **route.kwargs)
    middlewares = [create_bearer_auth_middleware(auth_token)] if auth_token else []
    app = web.Application(middlewares=middlewares)
    app.add_routes(api_routes)
    app.add_routes(routes)
    return app


def finished_job(queue, roots, status="success"):
    """What a finished wrapper job leaves behind: uploads in its own input
    folder, the output (and prompt sidecar) in its own output folder, a temp
    preview, and a history entry listing the outputs."""
    job_id = str(uuid.uuid4())
    job_dir = os.path.join("wrapper", job_id)
    files = {
        "image": write(os.path.join(roots.input, job_dir, "a1b2.png")),
        "audio": write(os.path.join(roots.input, job_dir, "c3d4.wav")),
        "video": write(os.path.join(roots.output, job_dir, "minimaxh3_00001_.mp4")),
        "sidecar": write(os.path.join(roots.output, job_dir, "minimaxh3_00001_.prompt.json")),
        "preview": write(os.path.join(roots.temp, "ComfyUI_temp_abcd_00001_.png")),
    }
    outputs = {
        "9": {"videos": [{"filename": "minimaxh3_00001_.mp4", "subfolder": job_dir, "type": "output"}],
              "animated": [True]},
        "10": {"images": [{"filename": "ComfyUI_temp_abcd_00001_.png", "subfolder": "", "type": "temp"}]},
    }
    queue.history[job_id] = {"prompt": (1, job_id, {}, {}, []), "outputs": outputs,
                             "status": {"status_str": status, "completed": status == "success", "messages": []}}
    return job_id, files


def bystanders(roots):
    """Files that belong to someone else and must survive every cleanup."""
    other = str(uuid.uuid4())
    return [
        write(os.path.join(roots.input, "example.png")),  # a UI upload
        write(os.path.join(roots.input, "wrapper", other, "e5f6.png")),  # another job's upload
        write(os.path.join(roots.output, "wrapper", other, "minimaxh3_00001_.mp4")),  # another job's output
        write(os.path.join(roots.output, "ComfyUI_00001_.png")),
        write(os.path.join(roots.temp, "ComfyUI_temp_other_00001_.png")),
    ]


@pytest.mark.asyncio
class TestDeleteJob:
    async def test_deletes_outputs_inputs_and_history(self, aiohttp_client, queue, roots):
        keep = bystanders(roots)
        job_id, files = finished_job(queue, roots)
        client = await aiohttp_client(make_app(queue, roots))

        response = await client.delete(f"/api/wrapper/jobs/{job_id}")
        assert response.status == 200
        assert await response.json() == {"deleted": {"outputs": 3, "inputs": 2, "history": True}, "refused": 0}
        for path in files.values():
            assert not os.path.lexists(path), path
        assert not os.path.exists(os.path.join(roots.input, "wrapper", job_id))
        assert not os.path.exists(os.path.join(roots.output, "wrapper", job_id))
        assert job_id not in queue.history
        for path in keep:
            assert os.path.exists(path), path

    async def test_unprefixed_route_and_get_still_answer(self, aiohttp_client, queue, roots):
        job_id, files = finished_job(queue, roots)
        client = await aiohttp_client(make_app(queue, roots))
        assert (await (await client.get(f"/api/wrapper/jobs/{job_id}")).json()) == {"get": job_id}

        response = await client.delete(f"/wrapper/jobs/{job_id}")
        assert response.status == 200
        assert (await response.json())["deleted"]["history"] is True

    async def test_history_outputs_outside_the_job_folder_are_found(self, aiohttp_client, queue, roots):
        # e.g. a job queued before outputs moved into wrapper/<job_id>/
        job_id = str(uuid.uuid4())
        video = write(os.path.join(roots.output, "wrapper", "minimaxh3_00007_.mp4"))
        sidecar = write(os.path.join(roots.output, "wrapper", "minimaxh3_00007_.prompt.json"))
        neighbour = write(os.path.join(roots.output, "wrapper", "minimaxh3_00008_.mp4"))
        queue.history[job_id] = {"outputs": {"9": {"videos": [
            {"filename": "minimaxh3_00007_.mp4", "subfolder": "wrapper", "type": "output"}]}}, "status": None}
        client = await aiohttp_client(make_app(queue, roots))

        body = await (await client.delete(f"/api/wrapper/jobs/{job_id}")).json()
        assert body == {"deleted": {"outputs": 2, "inputs": 0, "history": True}, "refused": 0}
        assert not os.path.exists(video) and not os.path.exists(sidecar)
        assert os.path.exists(neighbour)

    async def test_outputs_are_found_without_history(self, aiohttp_client, queue, roots):
        # The worker restarted between the render and the cleanup.
        job_id, files = finished_job(queue, roots)
        queue.history.clear()
        client = await aiohttp_client(make_app(queue, roots))

        body = await (await client.delete(f"/api/wrapper/jobs/{job_id}")).json()
        assert body == {"deleted": {"outputs": 2, "inputs": 2, "history": False}, "refused": 0}
        assert not os.path.exists(files["video"]) and not os.path.exists(files["image"])

    async def test_unknown_job_is_200_with_zeros(self, aiohttp_client, queue, roots):
        keep = bystanders(roots)
        client = await aiohttp_client(make_app(queue, roots))
        response = await client.delete(f"/api/wrapper/jobs/{uuid.uuid4()}")
        assert response.status == 200
        assert await response.json() == {"deleted": {"outputs": 0, "inputs": 0, "history": False}, "refused": 0}
        assert all(os.path.exists(path) for path in keep)

    async def test_second_call_is_200_with_zeros(self, aiohttp_client, queue, roots):
        job_id, _ = finished_job(queue, roots)
        client = await aiohttp_client(make_app(queue, roots))
        assert (await client.delete(f"/api/wrapper/jobs/{job_id}")).status == 200
        response = await client.delete(f"/api/wrapper/jobs/{job_id}")
        assert response.status == 200
        assert await response.json() == {"deleted": {"outputs": 0, "inputs": 0, "history": False}, "refused": 0}

    @pytest.mark.parametrize("job_id", ["not-a-uuid", "..", "%2e%2e%2f%2e%2e", str(uuid.uuid4()).upper(),
                                        str(uuid.uuid4()).replace("-", "")])
    async def test_non_canonical_id_is_400(self, aiohttp_client, queue, roots, job_id):
        keep = bystanders(roots)
        client = await aiohttp_client(make_app(queue, roots))
        response = await client.delete(f"/api/wrapper/jobs/{job_id}")
        assert response.status in (400, 404)  # an encoded '/' may not even route
        if response.status == 400:
            assert (await response.json())["error"]["message"] == "Invalid job id"
        assert all(os.path.exists(path) for path in keep)

    @pytest.mark.parametrize("where, status", [("running", "in_progress"), ("queued", "pending")])
    async def test_active_job_is_409_and_untouched(self, aiohttp_client, queue, roots, where, status):
        job_id, files = finished_job(queue, roots)
        entry = queue.history.pop(job_id)  # not finished yet
        item = (1, job_id, {}, {}, [])
        if where == "running":
            queue.currently_running[0] = item
        else:
            queue.queue.append(item)
        client = await aiohttp_client(make_app(queue, roots))

        response = await client.delete(f"/api/wrapper/jobs/{job_id}")
        assert response.status == 409
        body = await response.json()
        assert body["error"]["message"] == "Job still active"
        assert f"'{status}'" in body["error"]["details"]
        assert all(os.path.exists(path) for path in files.values())

        # Once it finishes, the same call cleans it.
        queue.currently_running.clear()
        queue.queue.clear()
        queue.history[job_id] = entry
        response = await client.delete(f"/api/wrapper/jobs/{job_id}")
        assert response.status == 200
        assert not any(os.path.exists(path) for path in files.values())

    async def test_open_generate_request_is_409(self, aiohttp_client, queue, roots):
        job_id, files = finished_job(queue, roots)
        client = await aiohttp_client(make_app(queue, roots))
        with cleanup.generate_scope(job_id, roots_provider=lambda: roots) as scope:
            response = await client.delete(f"/api/wrapper/jobs/{job_id}")  # still saving uploads
            assert response.status == 409
            assert "'pending'" in (await response.json())["error"]["details"]
            scope.queued = True  # e.g. now reading the output back to its caller
            response = await client.delete(f"/api/wrapper/jobs/{job_id}")
            assert response.status == 409
            assert "'in_progress'" in (await response.json())["error"]["details"]
        assert all(os.path.exists(path) for path in files.values())
        assert (await client.delete(f"/api/wrapper/jobs/{job_id}")).status == 200

    @pytest.mark.parametrize("history", ["failed", "cancelled_while_queued"])
    async def test_failed_or_cancelled_job_inputs_are_deleted(self, aiohttp_client, queue, roots, history):
        job_id, files = finished_job(queue, roots, status="error")
        queue.history[job_id]["outputs"] = {}
        os.remove(files["video"])
        os.remove(files["sidecar"])
        expected_history = True
        if history == "cancelled_while_queued":
            del queue.history[job_id]  # dequeued jobs never get a history entry
            expected_history = False
        client = await aiohttp_client(make_app(queue, roots))

        body = await (await client.delete(f"/api/wrapper/jobs/{job_id}")).json()
        assert body == {"deleted": {"outputs": 0, "inputs": 2, "history": expected_history}, "refused": 0}
        assert not os.path.exists(os.path.join(roots.input, "wrapper", job_id))

    async def test_forgets_remembered_progress(self, aiohttp_client, queue, roots):
        job_id, _ = finished_job(queue, roots)
        wrapper_progress._remember(job_id, 0.5)
        client = await aiohttp_client(make_app(queue, roots))
        await client.delete(f"/api/wrapper/jobs/{job_id}")
        assert job_id not in wrapper_progress._remembered


@pytest.mark.asyncio
class TestPathSafety:
    async def test_traversal_in_history_is_refused(self, aiohttp_client, queue, roots, outside):
        job_id, files = finished_job(queue, roots)
        queue.history[job_id]["outputs"]["11"] = {"images": [
            {"filename": "secret.txt", "subfolder": "../outside", "type": "output"},
            {"filename": "../outside/secret.txt", "subfolder": "", "type": "temp"},
            {"filename": str(outside / "secret.txt"), "subfolder": "", "type": "output"},
        ]}
        client = await aiohttp_client(make_app(queue, roots))

        response = await client.delete(f"/api/wrapper/jobs/{job_id}")
        assert response.status == 200
        body = await response.json()
        assert body["refused"] == 3
        assert body["deleted"] == {"outputs": 3, "inputs": 2, "history": True}  # the job itself is still cleaned
        assert (outside / "secret.txt").read_text() == "keep me"

    async def test_symlinked_subfolder_pointing_outside_is_refused(self, aiohttp_client, queue, roots, outside):
        job_id, _ = finished_job(queue, roots)
        os.symlink(outside, os.path.join(roots.output, "escape"))
        queue.history[job_id]["outputs"]["11"] = {"images": [
            {"filename": "secret.txt", "subfolder": "escape", "type": "output"}]}
        client = await aiohttp_client(make_app(queue, roots))

        body = await (await client.delete(f"/api/wrapper/jobs/{job_id}")).json()
        assert body["refused"] == 1
        assert (outside / "secret.txt").read_text() == "keep me"

    async def test_symlinked_file_pointing_outside_is_refused(self, aiohttp_client, queue, roots, outside):
        job_id, _ = finished_job(queue, roots)
        link = os.path.join(roots.output, "wrapper", "evil.png")
        os.symlink(outside / "secret.txt", link)
        queue.history[job_id]["outputs"]["11"] = {"images": [
            {"filename": "evil.png", "subfolder": "wrapper", "type": "output"}]}
        client = await aiohttp_client(make_app(queue, roots))

        body = await (await client.delete(f"/api/wrapper/jobs/{job_id}")).json()
        assert body["refused"] == 1
        assert (outside / "secret.txt").read_text() == "keep me"
        assert os.path.islink(link)  # refused, not followed

    async def test_symlinks_inside_the_job_folders_never_carry_the_delete_out(self, aiohttp_client, queue, roots, outside):
        job_id, files = finished_job(queue, roots)
        job_inputs = os.path.join(roots.input, "wrapper", job_id)
        os.symlink(outside / "secret.txt", os.path.join(job_inputs, "file-link.png"))
        os.symlink(outside, os.path.join(job_inputs, "dir-link"))
        client = await aiohttp_client(make_app(queue, roots))

        response = await client.delete(f"/api/wrapper/jobs/{job_id}")
        assert response.status == 200
        body = await response.json()
        assert body["refused"] == 2
        assert body["deleted"]["inputs"] == 2  # the two real uploads
        assert (outside / "secret.txt").read_text() == "keep me"
        assert not os.path.exists(files["image"])
        # The folder stays only because the refused links are still in it.
        assert sorted(os.listdir(job_inputs)) == ["dir-link", "file-link.png"]

    async def test_symlinked_job_folder_pointing_outside_is_refused(self, aiohttp_client, queue, roots, outside):
        job_id = str(uuid.uuid4())
        os.makedirs(os.path.join(roots.input, "wrapper"))
        os.symlink(outside, os.path.join(roots.input, "wrapper", job_id))
        client = await aiohttp_client(make_app(queue, roots))

        body = await (await client.delete(f"/api/wrapper/jobs/{job_id}")).json()
        assert body == {"deleted": {"outputs": 0, "inputs": 0, "history": False}, "refused": 1}
        assert (outside / "secret.txt").read_text() == "keep me"

    async def test_symlinked_wrapper_folder_pointing_outside_is_refused(self, aiohttp_client, queue, roots, outside):
        job_id = str(uuid.uuid4())
        write(str(outside / job_id / "upload.png"))
        os.symlink(outside, os.path.join(roots.input, "wrapper"))
        client = await aiohttp_client(make_app(queue, roots))

        body = await (await client.delete(f"/api/wrapper/jobs/{job_id}")).json()
        assert body["refused"] == 1
        assert os.path.exists(outside / job_id / "upload.png")

    async def test_symlink_to_a_file_inside_removes_only_the_link(self, aiohttp_client, queue, roots):
        job_id, _ = finished_job(queue, roots)
        target = write(os.path.join(roots.output, "ComfyUI_00001_.png"))
        link = os.path.join(roots.output, "wrapper", job_id, "alias.png")
        os.symlink(target, link)
        client = await aiohttp_client(make_app(queue, roots))

        body = await (await client.delete(f"/api/wrapper/jobs/{job_id}")).json()
        assert body["refused"] == 0
        assert not os.path.lexists(link)
        assert os.path.exists(target)

    async def test_symlinked_roots_are_resolved(self, aiohttp_client, queue, tmp_path):
        # A deployment may mount/symlink the whole output dir elsewhere.
        real = tmp_path / "big-disk"
        for name in ("output", "input", "temp"):
            (real / name).mkdir(parents=True)
            os.symlink(real / name, tmp_path / f"link-{name}")
        linked = cleanup.Roots(*(str(tmp_path / f"link-{name}") for name in ("output", "input", "temp")))
        job_id, files = finished_job(queue, linked)
        client = await aiohttp_client(make_app(queue, linked))

        body = await (await client.delete(f"/api/wrapper/jobs/{job_id}")).json()
        assert body == {"deleted": {"outputs": 3, "inputs": 2, "history": True}, "refused": 0}
        assert not any(os.path.exists(path) for path in files.values())

    async def test_input_typed_output_outside_the_job_folder_is_left_alone(self, aiohttp_client, queue, roots):
        job_id, _ = finished_job(queue, roots)
        shared = write(os.path.join(roots.input, "example.png"))
        queue.history[job_id]["outputs"]["11"] = {"images": [
            {"filename": "example.png", "subfolder": "", "type": "input"}]}
        client = await aiohttp_client(make_app(queue, roots))

        body = await (await client.delete(f"/api/wrapper/jobs/{job_id}")).json()
        assert body["refused"] == 1
        assert os.path.exists(shared)

    async def test_failed_delete_keeps_history_for_a_retry(self, aiohttp_client, queue, roots, monkeypatch):
        job_id, files = finished_job(queue, roots)
        real_unlink = os.unlink

        def flaky_unlink(path, *args, **kwargs):
            if str(path).endswith(".mp4"):
                raise PermissionError(13, "Permission denied", str(path))
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(os, "unlink", flaky_unlink)
        client = await aiohttp_client(make_app(queue, roots))
        response = await client.delete(f"/api/wrapper/jobs/{job_id}")
        assert response.status == 500
        body = await response.json()
        assert body["error"]["message"] == "Cleanup incomplete"
        assert body["deleted"]["history"] is False
        assert job_id in queue.history
        assert os.path.exists(files["video"])

        monkeypatch.setattr(os, "unlink", real_unlink)
        response = await client.delete(f"/api/wrapper/jobs/{job_id}")
        assert response.status == 200
        assert (await response.json())["deleted"] == {"outputs": 1, "inputs": 0, "history": True}
        assert not os.path.exists(files["video"])


@pytest.mark.asyncio
class TestAuth:
    async def test_bearer_required_when_auth_is_on(self, aiohttp_client, queue, roots):
        job_id, files = finished_job(queue, roots)
        client = await aiohttp_client(make_app(queue, roots, auth_token=TOKEN))

        for path in (f"/api/wrapper/jobs/{job_id}", f"/wrapper/jobs/{job_id}"):
            for headers in ({}, {"Authorization": "Bearer wrong"}):
                response = await client.delete(path, headers=headers)
                assert response.status == 401
                assert response.headers["WWW-Authenticate"] == "Bearer"
        assert all(os.path.exists(path) for path in files.values())
        assert job_id in queue.history

        response = await client.delete(f"/api/wrapper/jobs/{job_id}", headers=GOOD)
        assert response.status == 200
        assert (await response.json())["deleted"]["history"] is True


class TestGenerateScope:
    def test_uploads_of_a_job_that_never_queued_are_removed(self, roots):
        job_id = str(uuid.uuid4())
        upload = write(os.path.join(roots.input, "wrapper", job_id, "a1b2.png"))
        with pytest.raises(RuntimeError):
            with cleanup.generate_scope(job_id, roots_provider=lambda: roots):
                assert cleanup.is_open(job_id)
                raise RuntimeError("setup failed")
        assert not os.path.exists(os.path.dirname(upload))
        assert not cleanup.is_open(job_id)

    def test_uploads_of_a_queued_job_are_left_for_delete(self, roots):
        job_id = str(uuid.uuid4())
        upload = write(os.path.join(roots.input, "wrapper", job_id, "a1b2.png"))
        with cleanup.generate_scope(job_id, roots_provider=lambda: roots) as scope:
            scope.queued = True
        assert os.path.exists(upload)
        assert not cleanup.is_open(job_id)

    def test_job_subfolder(self):
        assert cleanup.job_subfolder("abc") == "wrapper/abc"
