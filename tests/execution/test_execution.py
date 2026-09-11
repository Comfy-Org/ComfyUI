from io import BytesIO
import numpy
from PIL import Image
import pytest
from pytest import fixture
import time
import torch
from typing import Union, Dict
import json
import subprocess
import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import urllib.request
import urllib.parse
import urllib.error
import os
from pathlib import PurePosixPath
from comfy_execution.graph_utils import GraphBuilder, Node
from app.assets.scanner_admission import _should_skip_extension
from app.assets.services.file_utils import list_files_recursively


ASSET_HEALTH_TIMEOUT_SECONDS = 120
ASSET_SEED_RETRY_ATTEMPTS = 10
ASSET_SEED_RETRY_DELAY_SECONDS = 3
ASSET_STATUS_POLL_DELAY_SECONDS = 1
ASSET_CAPTURE_TAIL_LINES = 80

# Static prefixes of app/assets exception-path messages; re-diff against logging.exception sites when the assets logging PR lands.
ASSET_FATAL_LOG_PREFIXES = (
    "get_asset failed for reference_id=",
    "upload_asset failed for tenant_id=",
    "update_asset failed for reference_id=",
    "delete_asset_reference failed for reference_id=",
    "add_tags_to_asset failed for reference_id=",
    "remove_tags_from_asset failed for reference_id=",
    "check_hash_exists failed for hash=",
    "Temp DB row wipe failed; skipping filesystem cleanup",
    "Asset startup maintenance failed",
    "Temp DB row wipe failed during shutdown",
    "Asset shutdown cleanup failed",
    "fast DB scan failed for ",
    "temp reference sync failed: ",
    "marking missing assets failed: ",
    "Failed to enrich ",
    "Asset scan failed",
    "Batch insert failed at offset ",
    "Failed to discard orphan content ",
    "Failed to register cached output: ",
    "Failed to register executed output: ",
    "Failed to register uploaded image as asset",
    "WARNING: blake3 package not installed",
)


def _asset_json_request(url, deadline, data=None):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AssertionError(
            f"Asset health check failure (i): deadline/transport failure before requesting {url}"
        )

    request = urllib.request.Request(url, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(request, timeout=max(0.01, remaining)) as response:
            status = response.status
            raw_payload = response.read()
    except urllib.error.HTTPError as error:
        status = error.code
        raw_payload = error.read()
    except (TimeoutError, ConnectionError, urllib.error.URLError) as error:
        raise AssertionError(
            f"Asset health check failure (i): deadline/transport failure requesting {url}: {error!r}"
        ) from error

    try:
        payload = json.loads(raw_payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        rendered_payload = raw_payload.decode("utf-8", errors="replace")
        raise AssertionError(
            "Asset health check failure (iv): malformed JSON/schema "
            f"from {url}: status={status}, payload={rendered_payload!r}"
        ) from error
    if not isinstance(payload, dict):
        raise AssertionError(
            "Asset health check failure (iv): malformed JSON/schema "
            f"from {url}: status={status}, payload={payload!r}"
        )
    return status, payload


def _seed_output_assets(base_url, deadline):
    request_data = json.dumps({"roots": ["output"]}).encode("utf-8")
    last_conflict_payload = None
    for attempt in range(1, ASSET_SEED_RETRY_ATTEMPTS + 1):
        if last_conflict_payload is not None and deadline - time.monotonic() <= 0:
            raise AssertionError(
                "Asset health check failure (ii): 409-retries-exhausted before the deadline; "
                f"last payload={last_conflict_payload!r}"
            )

        url = f"{base_url}/api/assets/seed?wait=true"
        status, payload = _asset_json_request(url, deadline, request_data)
        if status == 409:
            if payload.get("status") != "already_running":
                raise AssertionError(
                    "Asset health check failure (iv): malformed JSON/schema "
                    f"from {url}: status={status}, payload={payload!r}"
                )
            last_conflict_payload = payload
            if attempt == ASSET_SEED_RETRY_ATTEMPTS:
                raise AssertionError(
                    "Asset health check failure (ii): 409-retries-exhausted after "
                    f"{attempt} attempts; last payload={payload!r}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(
                    "Asset health check failure (ii): 409-retries-exhausted before the deadline; "
                    f"last payload={payload!r}"
                )
            time.sleep(min(ASSET_SEED_RETRY_DELAY_SECONDS, remaining))
            continue
        if status != 200:
            raise AssertionError(
                "Asset health check failure (iv): unexpected-HTTP-status "
                f"from {url}: status={status}, payload={payload!r}"
            )

        progress = payload.get("progress")
        errors = payload.get("errors")
        progress_fields = ("scanned", "total", "created", "skipped")
        if (
            payload.get("status") != "completed"
            or not isinstance(progress, dict)
            or any(type(progress.get(field)) is not int for field in progress_fields)
            or not isinstance(errors, list)
        ):
            raise AssertionError(
                "Asset health check failure (iv): malformed JSON/schema "
                f"from {url}: status={status}, payload={payload!r}"
            )
        if errors:
            raise AssertionError(
                f"Asset health check failure (iii): scan-completed-with-errors: errors={errors!r}"
            )
        return


def _wait_for_stable_asset_idle(base_url, deadline):
    stable_reads = 0
    last_payload = None
    url = f"{base_url}/api/assets/seed/status"
    while True:
        if deadline - time.monotonic() <= 0:
            raise AssertionError(
                "Asset health check failure (v): idle-stability-exhaustion; "
                f"last payload={last_payload!r}"
            )
        status, payload = _asset_json_request(url, deadline)
        progress = payload.get("progress")
        errors = payload.get("errors")
        if (
            status != 200
            or not isinstance(payload.get("state"), str)
            or not isinstance(errors, list)
            or (progress is not None and not isinstance(progress, dict))
        ):
            failure = "unexpected-HTTP-status" if status != 200 else "malformed JSON/schema"
            raise AssertionError(
                f"Asset health check failure (iv): {failure} from {url}: "
                f"status={status}, payload={payload!r}"
            )

        last_payload = payload
        if payload["state"].lower() == "idle" and errors == []:
            stable_reads += 1
            if stable_reads == 2:
                return
        else:
            stable_reads = 0

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(
                "Asset health check failure (v): idle-stability-exhaustion; "
                f"last payload={last_payload!r}"
            )
        time.sleep(min(ASSET_STATUS_POLL_DELAY_SECONDS, remaining))


def _normalize_asset_path(path):
    return PurePosixPath(path.replace("\\", "/")).as_posix()


def _fetch_output_asset_paths(base_url, deadline):
    asset_paths = set()
    after = None
    previous_cursor = None
    while True:
        query = {"tags_all": "output", "limit": "100"}
        if after is not None:
            query["after"] = after
        url = f"{base_url}/api/assets?{urllib.parse.urlencode(query)}"
        status, payload = _asset_json_request(url, deadline)
        assets = payload.get("assets")
        next_cursor = payload.get("next_cursor")
        if (
            status != 200
            or not isinstance(assets, list)
            or type(payload.get("total")) is not int
            or not isinstance(payload.get("has_more"), bool)
            or (next_cursor is not None and not isinstance(next_cursor, str))
        ):
            failure = "unexpected-HTTP-status" if status != 200 else "malformed JSON/schema"
            raise AssertionError(
                f"Asset health check failure (iv): {failure} from {url}: "
                f"status={status}, payload={payload!r}"
            )

        for asset in assets:
            if not isinstance(asset, dict):
                raise AssertionError(
                    "Asset health check failure (iv): malformed JSON/schema "
                    f"from {url}: asset={asset!r}, payload={payload!r}"
                )
            loader_path = asset.get("loader_path")
            if not isinstance(loader_path, str):
                raise AssertionError(
                    "Asset health check reconcile failure: asset has missing/non-string "
                    f"loader_path: asset={asset!r}"
                )
            asset_paths.add(_normalize_asset_path(loader_path))

        if next_cursor is None:
            if payload["has_more"]:
                raise AssertionError(
                    "Asset health check failure (iv): malformed JSON/schema "
                    f"from {url}: has_more=true without next_cursor, payload={payload!r}"
                )
            return asset_paths
        if next_cursor == previous_cursor:
            raise AssertionError(
                "Asset health check reconcile failure: pagination cursor made no progress: "
                f"cursor={next_cursor!r}, payload={payload!r}"
            )
        previous_cursor = next_cursor
        after = next_cursor


def _list_output_files_on_disk(output_dir):
    output_root = os.path.abspath(output_dir)
    disk_paths = set()
    for file_path in list_files_recursively(output_root):
        if _should_skip_extension(file_path):
            continue
        try:
            stat_result = os.stat(file_path, follow_symlinks=True)
        except OSError:
            continue
        if not stat_result.st_size:
            continue
        relative_path = os.path.relpath(file_path, output_root)
        disk_paths.add(_normalize_asset_path(relative_path))
    return disk_paths


def _assert_no_fatal_asset_logs(capture_path):
    server_output = capture_path.read_text(encoding="utf-8", errors="replace")
    matched_prefixes = [
        prefix for prefix in ASSET_FATAL_LOG_PREFIXES if prefix in server_output
    ]
    if matched_prefixes:
        raise AssertionError(
            "Asset health check fatal-log failure: "
            f"matched prefixes={matched_prefixes!r}"
        )


def _assert_assets_healthy(listen, port, output_dir, capture_path):
    deadline = time.monotonic() + ASSET_HEALTH_TIMEOUT_SECONDS
    base_url = f"http://{listen}:{port}"
    _seed_output_assets(base_url, deadline)
    _wait_for_stable_asset_idle(base_url, deadline)

    api_paths = _fetch_output_asset_paths(base_url, deadline)
    disk_paths = _list_output_files_on_disk(output_dir)
    if not api_paths or not disk_paths:
        raise AssertionError(
            "Asset health check reconcile failure: expected non-empty sets: "
            f"api_paths={sorted(api_paths)!r}, disk_paths={sorted(disk_paths)!r}"
        )
    disk_only = disk_paths - api_paths
    api_only = api_paths - disk_paths
    if disk_only or api_only:
        raise AssertionError(
            "Asset health check reconcile failure: "
            f"disk-has-but-API-lacks={sorted(disk_only)!r}, "
            f"API-has-but-disk-lacks={sorted(api_only)!r}"
        )
    _assert_no_fatal_asset_logs(capture_path)


def _capture_tail(capture_path):
    try:
        lines = capture_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        return f"<unable to read capture file: {error!r}>"
    return "\n".join(lines[-ASSET_CAPTURE_TAIL_LINES:])

def run_warmup(client, prefix="warmup"):
    """Run a simple workflow to warm up the server."""
    warmup_g = GraphBuilder(prefix=prefix)
    warmup_image = warmup_g.node("StubImage", content="BLACK", height=32, width=32, batch_size=1)
    warmup_g.node("PreviewImage", images=warmup_image.out(0))
    client.run(warmup_g)

class RunResult:
    def __init__(self, prompt_id: str):
        self.outputs: Dict[str,Dict] = {}
        self.runs: Dict[str,bool] = {}
        self.cached: Dict[str,bool] = {}
        self.prompt_id: str = prompt_id

    def get_output(self, node: Node):
        return self.outputs.get(node.id, None)

    def did_run(self, node: Node):
        return self.runs.get(node.id, False)

    def was_cached(self, node: Node):
        return self.cached.get(node.id, False)

    def was_executed(self, node: Node):
        """Returns True if node was either run or cached"""
        return self.did_run(node) or self.was_cached(node)

    def get_images(self, node: Node):
        output = self.get_output(node)
        if output is None:
            return []
        return output.get('image_objects', [])

    def get_prompt_id(self):
        return self.prompt_id

class ComfyClient:
    def __init__(self):
        self.test_name = ""

    def connect(self,
                    listen:str = '127.0.0.1',
                    port:Union[str,int] = 8188,
                    client_id: str = str(uuid.uuid4())
                    ):
        self.client_id = client_id
        self.server_address = f"{listen}:{port}"
        ws = websocket.WebSocket()
        ws.connect("ws://{}/ws?clientId={}".format(self.server_address, self.client_id))
        self.ws = ws

    def queue_prompt(self, prompt, partial_execution_targets=None):
        p = {"prompt": prompt, "client_id": self.client_id}
        if partial_execution_targets is not None:
            p["partial_execution_targets"] = partial_execution_targets
        data = json.dumps(p).encode('utf-8')
        req =  urllib.request.Request("http://{}/prompt".format(self.server_address), data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen("http://{}/view?{}".format(self.server_address, url_values)) as response:
            return response.read()

    def get_history(self, prompt_id):
        with urllib.request.urlopen("http://{}/history/{}".format(self.server_address, prompt_id)) as response:
            return json.loads(response.read())

    def get_all_history(self, max_items=None, offset=None):
        url = "http://{}/history".format(self.server_address)
        params = {}
        if max_items is not None:
            params["max_items"] = max_items
        if offset is not None:
            params["offset"] = offset

        if params:
            url_values = urllib.parse.urlencode(params)
            url = "{}?{}".format(url, url_values)

        with urllib.request.urlopen(url) as response:
            return json.loads(response.read())

    def get_jobs(self, status=None, limit=None, offset=None, sort_by=None, sort_order=None):
        url = "http://{}/api/jobs".format(self.server_address)
        params = {}
        if status is not None:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        if sort_by is not None:
            params["sort_by"] = sort_by
        if sort_order is not None:
            params["sort_order"] = sort_order

        if params:
            url_values = urllib.parse.urlencode(params)
            url = "{}?{}".format(url, url_values)

        with urllib.request.urlopen(url) as response:
            return json.loads(response.read())

    def get_job(self, job_id):
        url = "http://{}/api/jobs/{}".format(self.server_address, job_id)
        try:
            with urllib.request.urlopen(url) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    def set_test_name(self, name):
        self.test_name = name

    def run(self, graph, partial_execution_targets=None):
        prompt = graph.finalize()
        for node in graph.nodes.values():
            if node.class_type == 'SaveImage':
                node.inputs['filename_prefix'] = self.test_name

        prompt_id = self.queue_prompt(prompt, partial_execution_targets)['prompt_id']
        result = RunResult(prompt_id)
        while True:
            out = self.ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['prompt_id'] != prompt_id:
                        continue
                    if data['node'] is None:
                        break
                    result.runs[data['node']] = True
                elif message['type'] == 'execution_error':
                    raise Exception(message['data'])
                elif message['type'] == 'execution_cached':
                    if message['data']['prompt_id'] == prompt_id:
                        cached_nodes = message['data'].get('nodes', [])
                        for node_id in cached_nodes:
                            result.cached[node_id] = True

        history = self.get_history(prompt_id)[prompt_id]
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            result.outputs[node_id] = node_output
            images_output = []
            if 'images' in node_output:
                for image in node_output['images']:
                    image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                    image_obj = Image.open(BytesIO(image_data))
                    images_output.append(image_obj)
                node_output['image_objects'] = images_output

        return result

#
# Loop through these variables
#
@pytest.mark.execution
class TestExecution:
    #
    # Initialize server and client
    #
    @fixture(scope="class", autouse=True, params=[
        { "extra_args" : ["--cache-classic"], "should_cache_results" : True },
        { "extra_args" : ["--cache-lru", 100], "should_cache_results" : True },
        { "extra_args" : ["--cache-none"], "should_cache_results" : False },
        {"extra_args": ["--enable-assets"], "should_cache_results": True, "assets": True},
    ])
    def server(self, args_pytest, request, tmp_path_factory):
        # Start server
        assets_enabled = request.param.get("assets")
        pargs = [
            'python','main.py',
            '--output-directory', args_pytest["output_dir"],
            '--listen', args_pytest["listen"],
            '--port', str(args_pytest["port"]),
            '--extra-model-paths-config', 'tests/execution/extra_model_paths.yaml',
            '--cpu',
        ]
        pargs += [ str(param) for param in request.param["extra_args"] ]
        if assets_enabled:
            assets_tmp_dir = tmp_path_factory.mktemp("execution-assets")
            database_path = assets_tmp_dir / "assets.db"
            capture_path = assets_tmp_dir / "server.log"
            pargs += ["--database-url", f"sqlite:///{database_path}"]
        print("Running server with args:", pargs)  # noqa: T201
        if assets_enabled:
            capture_file = capture_path.open("a", encoding="utf-8")
            try:
                p = subprocess.Popen(
                    pargs, stdout=capture_file, stderr=subprocess.STDOUT
                )
            except OSError:
                capture_file.close()
                raise

            health_failure = None
            cleanup_failures = []
            try:
                yield request.param
                exit_code = p.poll()
                if exit_code is not None:
                    raise AssertionError(
                        "Asset health check server-dead failure: "
                        f"exit code={exit_code}"
                    )
                _assert_assets_healthy(
                    args_pytest["listen"],
                    args_pytest["port"],
                    args_pytest["output_dir"],
                    capture_path,
                )
            except AssertionError as error:
                health_failure = error
            finally:
                try:
                    p.kill()
                except OSError as error:
                    cleanup_failures.append(f"kill failed: {error!r}")
                finally:
                    try:
                        p.wait(timeout=10)
                    except (OSError, subprocess.TimeoutExpired) as error:
                        cleanup_failures.append(f"bounded reap failed: {error!r}")
                    finally:
                        try:
                            capture_file.close()
                        except OSError as error:
                            cleanup_failures.append(f"capture close failed: {error!r}")
                        finally:
                            torch.cuda.empty_cache()

            if cleanup_failures:
                print("Asset server cleanup warnings:", cleanup_failures)  # noqa: T201
            if health_failure is not None:
                capture_tail = _capture_tail(capture_path)
                raise AssertionError(
                    f"{health_failure}\nServer output tail:\n{capture_tail}"
                ) from health_failure
            return

        p = subprocess.Popen(pargs)
        yield request.param
        p.kill()
        torch.cuda.empty_cache()

    def start_client(self, listen:str, port:int):
        # Start client
        comfy_client = ComfyClient()
        # Connect to server (with retries)
        n_tries = 5
        for i in range(n_tries):
            time.sleep(4)
            try:
                comfy_client.connect(listen=listen, port=port)
            except ConnectionRefusedError as e:
                print(e)  # noqa: T201
                print(f"({i+1}/{n_tries}) Retrying...")  # noqa: T201
            else:
                break
        return comfy_client

    @fixture(scope="class", autouse=True)
    def shared_client(self, args_pytest, server):
        client = self.start_client(args_pytest["listen"], args_pytest["port"])
        yield client
        del client
        torch.cuda.empty_cache()

    @fixture
    def client(self, shared_client, request):
        shared_client.set_test_name(f"execution[{request.node.name}]")
        yield shared_client

    @fixture
    def builder(self, request):
        yield GraphBuilder(prefix=request.node.name)

    def test_lazy_input(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        mask = g.node("StubMask", value=0.0, height=512, width=512, batch_size=1)

        lazy_mix = g.node("TestLazyMixImages", image1=input1.out(0), image2=input2.out(0), mask=mask.out(0))
        output = g.node("SaveImage", images=lazy_mix.out(0))
        result = client.run(g)

        result_image = result.get_images(output)[0]
        assert numpy.array(result_image).any() == 0, "Image should be black"
        assert result.did_run(input1)
        assert not result.did_run(input2)
        assert result.did_run(mask)
        assert result.did_run(lazy_mix)

    def test_full_cache(self, client: ComfyClient, builder: GraphBuilder, server):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="NOISE", height=512, width=512, batch_size=1)
        mask = g.node("StubMask", value=0.5, height=512, width=512, batch_size=1)

        lazy_mix = g.node("TestLazyMixImages", image1=input1.out(0), image2=input2.out(0), mask=mask.out(0))
        g.node("SaveImage", images=lazy_mix.out(0))

        client.run(g)
        result2 = client.run(g)
        for node_id, node in g.nodes.items():
            if server["should_cache_results"]:
                assert not result2.did_run(node), f"Node {node_id} ran, but should have been cached"
            else:
                assert result2.did_run(node), f"Node {node_id} was cached, but should have been run"

    def test_partial_cache(self, client: ComfyClient, builder: GraphBuilder, server):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="NOISE", height=512, width=512, batch_size=1)
        mask = g.node("StubMask", value=0.5, height=512, width=512, batch_size=1)

        lazy_mix = g.node("TestLazyMixImages", image1=input1.out(0), image2=input2.out(0), mask=mask.out(0))
        g.node("SaveImage", images=lazy_mix.out(0))

        client.run(g)
        mask.inputs['value'] = 0.4
        result2 = client.run(g)
        if server["should_cache_results"]:
            assert not result2.did_run(input1), "Input1 should have been cached"
            assert not result2.did_run(input2), "Input2 should have been cached"
        else:
            assert result2.did_run(input1), "Input1 should have been rerun"
            assert result2.did_run(input2), "Input2 should have been rerun"

    def test_error(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        # Different size of the two images
        input2 = g.node("StubImage", content="NOISE", height=256, width=256, batch_size=1)
        mask = g.node("StubMask", value=0.5, height=512, width=512, batch_size=1)

        lazy_mix = g.node("TestLazyMixImages", image1=input1.out(0), image2=input2.out(0), mask=mask.out(0))
        g.node("SaveImage", images=lazy_mix.out(0))

        try:
            client.run(g)
            assert False, "Should have raised an error"
        except Exception as e:
            assert 'prompt_id' in e.args[0], f"Did not get back a proper error message: {e}"

    @pytest.mark.parametrize("test_value, expect_error", [
        (5, True),
        ("foo", True),
        (5.0, False),
    ])
    def test_validation_error_literal(self, test_value, expect_error, client: ComfyClient, builder: GraphBuilder):
        g = builder
        validation1 = g.node("TestCustomValidation1", input1=test_value, input2=3.0)
        g.node("SaveImage", images=validation1.out(0))

        if expect_error:
            with pytest.raises(urllib.error.HTTPError):
                client.run(g)
        else:
            client.run(g)

    @pytest.mark.parametrize("test_type, test_value", [
        ("StubInt", 5),
        ("StubMask", 5.0)
    ])
    def test_validation_error_edge1(self, test_type, test_value, client: ComfyClient, builder: GraphBuilder):
        g = builder
        stub = g.node(test_type, value=test_value)
        validation1 = g.node("TestCustomValidation1", input1=stub.out(0), input2=3.0)
        g.node("SaveImage", images=validation1.out(0))

        with pytest.raises(urllib.error.HTTPError):
            client.run(g)

    @pytest.mark.parametrize("test_type, test_value, expect_error", [
        ("StubInt", 5, True),
        ("StubFloat", 5.0, False)
    ])
    def test_validation_error_edge2(self, test_type, test_value, expect_error, client: ComfyClient, builder: GraphBuilder):
        g = builder
        stub = g.node(test_type, value=test_value)
        validation2 = g.node("TestCustomValidation2", input1=stub.out(0), input2=3.0)
        g.node("SaveImage", images=validation2.out(0))

        if expect_error:
            with pytest.raises(urllib.error.HTTPError):
                client.run(g)
        else:
            client.run(g)

    @pytest.mark.parametrize("test_type, test_value, expect_error", [
        ("StubInt", 5, True),
        ("StubFloat", 5.0, False)
    ])
    def test_validation_error_edge3(self, test_type, test_value, expect_error, client: ComfyClient, builder: GraphBuilder):
        g = builder
        stub = g.node(test_type, value=test_value)
        validation3 = g.node("TestCustomValidation3", input1=stub.out(0), input2=3.0)
        g.node("SaveImage", images=validation3.out(0))

        if expect_error:
            with pytest.raises(urllib.error.HTTPError):
                client.run(g)
        else:
            client.run(g)

    @pytest.mark.parametrize("test_type, test_value, expect_error", [
        ("StubInt", 5, True),
        ("StubFloat", 5.0, False)
    ])
    def test_validation_error_edge4(self, test_type, test_value, expect_error, client: ComfyClient, builder: GraphBuilder):
        g = builder
        stub = g.node(test_type, value=test_value)
        validation4 = g.node("TestCustomValidation4", input1=stub.out(0), input2=3.0)
        g.node("SaveImage", images=validation4.out(0))

        if expect_error:
            with pytest.raises(urllib.error.HTTPError):
                client.run(g)
        else:
            client.run(g)

    @pytest.mark.parametrize("test_value1, test_value2, expect_error", [
        (0.0, 0.5, False),
        (0.0, 5.0, False),
        (0.0, 7.0, True)
    ])
    def test_validation_error_kwargs(self, test_value1, test_value2, expect_error, client: ComfyClient, builder: GraphBuilder):
        g = builder
        validation5 = g.node("TestCustomValidation5", input1=test_value1, input2=test_value2)
        g.node("SaveImage", images=validation5.out(0))

        if expect_error:
            with pytest.raises(urllib.error.HTTPError):
                client.run(g)
        else:
            client.run(g)

    def test_cycle_error(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        mask = g.node("StubMask", value=0.5, height=512, width=512, batch_size=1)

        lazy_mix1 = g.node("TestLazyMixImages", image1=input1.out(0), mask=mask.out(0))
        lazy_mix2 = g.node("TestLazyMixImages", image1=lazy_mix1.out(0), image2=input2.out(0), mask=mask.out(0))
        g.node("SaveImage", images=lazy_mix2.out(0))

        # When the cycle exists on initial submission, it should raise a validation error
        with pytest.raises(urllib.error.HTTPError):
            client.run(g)

    def test_dynamic_cycle_error(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        generator = g.node("TestDynamicDependencyCycle", input1=input1.out(0), input2=input2.out(0))
        g.node("SaveImage", images=generator.out(0))

        # When the cycle is in a graph that is generated dynamically, it should raise a runtime error
        try:
            client.run(g)
            assert False, "Should have raised an error"
        except Exception as e:
            assert 'prompt_id' in e.args[0], f"Did not get back a proper error message: {e}"
            assert e.args[0]['node_id'] == generator.id, "Error should have been on the generator node"

    def test_missing_node_error(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", id="removeme", content="WHITE", height=512, width=512, batch_size=1)
        input3 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        mask = g.node("StubMask", value=0.5, height=512, width=512, batch_size=1)
        mix1 = g.node("TestLazyMixImages", image1=input1.out(0), image2=input2.out(0), mask=mask.out(0))
        mix2 = g.node("TestLazyMixImages", image1=input1.out(0), image2=input3.out(0), mask=mask.out(0))
        # We have multiple outputs. The first is invalid, but the second is valid
        g.node("SaveImage", images=mix1.out(0))
        g.node("SaveImage", images=mix2.out(0))
        g.remove_node("removeme")

        client.run(g)

        # Add back in the missing node to make sure the error doesn't break the server
        input2 = g.node("StubImage", id="removeme", content="WHITE", height=512, width=512, batch_size=1)
        client.run(g)

    def test_custom_is_changed(self, client: ComfyClient, builder: GraphBuilder, server):
        g = builder
        # Creating the nodes in this specific order previously caused a bug
        save = g.node("SaveImage")
        is_changed = g.node("TestCustomIsChanged", should_change=False)
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)

        save.set_input('images', is_changed.out(0))
        is_changed.set_input('image', input1.out(0))

        result1 = client.run(g)
        result2 = client.run(g)
        is_changed.set_input('should_change', True)
        result3 = client.run(g)
        result4 = client.run(g)
        assert result1.did_run(is_changed), "is_changed should have been run"
        if server["should_cache_results"]:
            assert not result2.did_run(is_changed), "is_changed should have been cached"
        else:
            assert result2.did_run(is_changed), "is_changed should have been re-run"
        assert result3.did_run(is_changed), "is_changed should have been re-run"
        assert result4.did_run(is_changed), "is_changed should not have been cached"

    def test_undeclared_inputs(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        input3 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input4 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        average = g.node("TestVariadicAverage", input1=input1.out(0), input2=input2.out(0), input3=input3.out(0), input4=input4.out(0))
        output = g.node("SaveImage", images=average.out(0))

        result = client.run(g)
        result_image = result.get_images(output)[0]
        expected = 255 // 4
        assert numpy.array(result_image).min() == expected and numpy.array(result_image).max() == expected, "Image should be grey"

    def test_for_loop(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        iterations = 4
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        is_changed = g.node("TestCustomIsChanged", should_change=True, image=input2.out(0))
        for_open = g.node("TestForLoopOpen", remaining=iterations, initial_value1=is_changed.out(0))
        average = g.node("TestVariadicAverage", input1=input1.out(0), input2=for_open.out(2))
        for_close = g.node("TestForLoopClose", flow_control=for_open.out(0), initial_value1=average.out(0))
        output = g.node("SaveImage", images=for_close.out(0))

        for iterations in range(1, 5):
            for_open.set_input('remaining', iterations)
            result = client.run(g)
            result_image = result.get_images(output)[0]
            expected = 255 // (2 ** iterations)
            assert numpy.array(result_image).min() == expected and numpy.array(result_image).max() == expected, "Image should be grey"
            assert result.did_run(is_changed)

    def test_mixed_expansion_returns(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        val_list = g.node("TestMakeListNode", value1=0.1, value2=0.2, value3=0.3)
        mixed = g.node("TestMixedExpansionReturns", input1=val_list.out(0))
        output_dynamic = g.node("SaveImage", images=mixed.out(0))
        output_literal = g.node("SaveImage", images=mixed.out(1))

        result = client.run(g)
        images_dynamic = result.get_images(output_dynamic)
        assert len(images_dynamic) == 3, "Should have 2 images"
        assert numpy.array(images_dynamic[0]).min() == 25 and numpy.array(images_dynamic[0]).max() == 25, "First image should be 0.1"
        assert numpy.array(images_dynamic[1]).min() == 51 and numpy.array(images_dynamic[1]).max() == 51, "Second image should be 0.2"
        assert numpy.array(images_dynamic[2]).min() == 76 and numpy.array(images_dynamic[2]).max() == 76, "Third image should be 0.3"

        images_literal = result.get_images(output_literal)
        assert len(images_literal) == 3, "Should have 2 images"
        for i in range(3):
            assert numpy.array(images_literal[i]).min() == 255 and numpy.array(images_literal[i]).max() == 255, "All images should be white"

    def test_mixed_lazy_results(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        val_list = g.node("TestMakeListNode", value1=0.0, value2=0.5, value3=1.0)
        mask = g.node("StubMask", value=val_list.out(0), height=512, width=512, batch_size=1)
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        mix = g.node("TestLazyMixImages", image1=input1.out(0), image2=input2.out(0), mask=mask.out(0))
        rebatch = g.node("RebatchImages", images=mix.out(0), batch_size=3)
        output = g.node("SaveImage", images=rebatch.out(0))

        result = client.run(g)
        images = result.get_images(output)
        assert len(images) == 3, "Should have 3 image"
        assert numpy.array(images[0]).min() == 0 and numpy.array(images[0]).max() == 0, "First image should be 0.0"
        assert numpy.array(images[1]).min() == 127 and numpy.array(images[1]).max() == 127, "Second image should be 0.5"
        assert numpy.array(images[2]).min() == 255 and numpy.array(images[2]).max() == 255, "Third image should be 1.0"

    def test_output_reuse(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)

        output1 = g.node("SaveImage", images=input1.out(0))
        output2 = g.node("SaveImage", images=input1.out(0))

        result = client.run(g)
        images1 = result.get_images(output1)
        images2 = result.get_images(output2)
        assert len(images1) == 1, "Should have 1 image"
        assert len(images2) == 1, "Should have 1 image"

    # This tests that only constant outputs are used in the call to `IS_CHANGED`
    def test_is_changed_with_outputs(self, client: ComfyClient, builder: GraphBuilder, server):
        g = builder
        input1 = g.node("StubConstantImage", value=0.5, height=512, width=512, batch_size=1)
        test_node = g.node("TestIsChangedWithConstants", image=input1.out(0), value=0.5)

        output = g.node("PreviewImage", images=test_node.out(0))

        result = client.run(g)
        images = result.get_images(output)
        assert len(images) == 1, "Should have 1 image"
        assert numpy.array(images[0]).min() == 63 and numpy.array(images[0]).max() == 63, "Image should have value 0.25"

        result = client.run(g)
        images = result.get_images(output)
        assert len(images) == 1, "Should have 1 image"
        assert numpy.array(images[0]).min() == 63 and numpy.array(images[0]).max() == 63, "Image should have value 0.25"
        if server["should_cache_results"]:
            assert not result.did_run(test_node), "The execution should have been cached"
        else:
            assert result.did_run(test_node), "The execution should have been re-run"


    def test_parallel_sleep_nodes(self, client: ComfyClient, builder: GraphBuilder, skip_timing_checks):
        # Warmup execution to ensure server is fully initialized
        run_warmup(client)

        g = builder
        image = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)

        # Create sleep nodes for each duration
        sleep_node1 = g.node("TestSleep", value=image.out(0), seconds=2.9)
        sleep_node2 = g.node("TestSleep", value=image.out(0), seconds=3.1)
        sleep_node3 = g.node("TestSleep", value=image.out(0), seconds=3.0)

        # Add outputs to verify the execution
        _output1 = g.node("PreviewImage", images=sleep_node1.out(0))
        _output2 = g.node("PreviewImage", images=sleep_node2.out(0))
        _output3 = g.node("PreviewImage", images=sleep_node3.out(0))

        start_time = time.time()
        result = client.run(g)
        elapsed_time = time.time() - start_time

        # The test should take around 3.0 seconds (the longest sleep duration)
        # plus some overhead, but definitely less than the sum of all sleeps (9.0s)
        if not skip_timing_checks:
            assert elapsed_time < 8.9, f"Parallel execution took {elapsed_time}s, expected less than 8.9s"

        # Verify that all nodes executed
        assert result.did_run(sleep_node1), "Sleep node 1 should have run"
        assert result.did_run(sleep_node2), "Sleep node 2 should have run"
        assert result.did_run(sleep_node3), "Sleep node 3 should have run"

    def test_parallel_sleep_expansion(self, client: ComfyClient, builder: GraphBuilder, skip_timing_checks):
        # Warmup execution to ensure server is fully initialized
        run_warmup(client)

        g = builder
        # Create input images with different values
        image1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        image2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        image3 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)

        # Create a TestParallelSleep node that expands into multiple TestSleep nodes
        parallel_sleep = g.node("TestParallelSleep",
                                image1=image1.out(0),
                                image2=image2.out(0),
                                image3=image3.out(0),
                                sleep1=4.8,
                                sleep2=4.9,
                                sleep3=5.0)
        output = g.node("SaveImage", images=parallel_sleep.out(0))

        start_time = time.time()
        result = client.run(g)
        elapsed_time = time.time() - start_time

        # Similar to the previous test, expect parallel execution of the sleep nodes
        # which should complete in less than the sum of all sleeps
        # Lots of leeway here since Windows CI is slow
        if not skip_timing_checks:
            assert elapsed_time < 13.0, f"Expansion execution took {elapsed_time}s"

        # Verify the parallel sleep node executed
        assert result.did_run(parallel_sleep), "ParallelSleep node should have run"

        # Verify we get an image as output (blend of the three input images)
        result_images = result.get_images(output)
        assert len(result_images) == 1, "Should have 1 image"
        # Average pixel value should be around 170 (255 * 2 // 3)
        avg_value = numpy.array(result_images[0]).mean()
        assert avg_value == 170, f"Image average value {avg_value} should be 170"

    # This tests that nodes with OUTPUT_IS_LIST function correctly when they receive an ExecutionBlocker
    # as input. We also test that when that list (containing an ExecutionBlocker) is passed to a node,
    # only that one entry in the list is blocked.
    def test_execution_block_list_output(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        image1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        image2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        image3 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        image_list = g.node("TestMakeListNode", value1=image1.out(0), value2=image2.out(0), value3=image3.out(0))
        int1 = g.node("StubInt", value=1)
        int2 = g.node("StubInt", value=2)
        int3 = g.node("StubInt", value=3)
        int_list = g.node("TestMakeListNode", value1=int1.out(0), value2=int2.out(0), value3=int3.out(0))
        compare = g.node("TestIntConditions", a=int_list.out(0), b=2, operation="==")
        blocker = g.node("TestExecutionBlocker", input=image_list.out(0), block=compare.out(0), verbose=False)

        list_output = g.node("TestMakeListNode", value1=blocker.out(0))
        output = g.node("PreviewImage", images=list_output.out(0))

        result = client.run(g)
        assert result.did_run(output), "The execution should have run"
        images = result.get_images(output)
        assert len(images) == 2, "Should have 2 images"
        assert numpy.array(images[0]).min() == 0 and numpy.array(images[0]).max() == 0, "First image should be black"
        assert numpy.array(images[1]).min() == 0 and numpy.array(images[1]).max() == 0, "Second image should also be black"

    # Output nodes included in the partial execution list are executed
    def test_partial_execution_included_outputs(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)

        # Create two separate output nodes
        output1 = g.node("SaveImage", images=input1.out(0))
        output2 = g.node("SaveImage", images=input2.out(0))

        # Run with partial execution targeting only output1
        result = client.run(g, partial_execution_targets=[output1.id])

        assert result.was_executed(input1), "Input1 should have been executed (run or cached)"
        assert result.was_executed(output1), "Output1 should have been executed (run or cached)"
        assert not result.did_run(input2), "Input2 should not have run"
        assert not result.did_run(output2), "Output2 should not have run"

        # Verify only output1 produced results
        assert len(result.get_images(output1)) == 1, "Output1 should have produced an image"
        assert len(result.get_images(output2)) == 0, "Output2 should not have produced an image"

    # Output nodes NOT included in the partial execution list are NOT executed
    def test_partial_execution_excluded_outputs(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        input3 = g.node("StubImage", content="NOISE", height=512, width=512, batch_size=1)

        # Create three output nodes
        output1 = g.node("SaveImage", images=input1.out(0))
        output2 = g.node("SaveImage", images=input2.out(0))
        output3 = g.node("SaveImage", images=input3.out(0))

        # Run with partial execution targeting only output1 and output3
        result = client.run(g, partial_execution_targets=[output1.id, output3.id])

        assert result.was_executed(input1), "Input1 should have been executed"
        assert result.was_executed(input3), "Input3 should have been executed"
        assert result.was_executed(output1), "Output1 should have been executed"
        assert result.was_executed(output3), "Output3 should have been executed"
        assert not result.did_run(input2), "Input2 should not have run"
        assert not result.did_run(output2), "Output2 should not have run"

    # Output nodes NOT in list ARE executed if necessary for nodes that are in the list
    def test_partial_execution_dependencies(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)

        # Create a processing chain with an OUTPUT_NODE that has socket outputs
        output_with_socket = g.node("TestOutputNodeWithSocketOutput", image=input1.out(0), value=2.0)

        # Create another node that depends on the output_with_socket
        dependent_node = g.node("TestLazyMixImages",
                                image1=output_with_socket.out(0),
                                image2=input1.out(0),
                                mask=g.node("StubMask", value=0.5, height=512, width=512, batch_size=1).out(0))

        # Create the final output
        final_output = g.node("SaveImage", images=dependent_node.out(0))

        # Run with partial execution targeting only the final output
        result = client.run(g, partial_execution_targets=[final_output.id])

        # All nodes should have been executed because they're dependencies
        assert result.was_executed(input1), "Input1 should have been executed"
        assert result.was_executed(output_with_socket), "Output with socket should have been executed (dependency)"
        assert result.was_executed(dependent_node), "Dependent node should have been executed"
        assert result.was_executed(final_output), "Final output should have been executed"

    # Lazy execution works with partial execution
    def test_partial_execution_with_lazy_nodes(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        input3 = g.node("StubImage", content="NOISE", height=512, width=512, batch_size=1)

        # Create masks that will trigger different lazy execution paths
        mask1 = g.node("StubMask", value=0.0, height=512, width=512, batch_size=1)  # Will only need image1
        mask2 = g.node("StubMask", value=0.5, height=512, width=512, batch_size=1)  # Will need both images

        # Create two lazy mix nodes
        lazy_mix1 = g.node("TestLazyMixImages", image1=input1.out(0), image2=input2.out(0), mask=mask1.out(0))
        lazy_mix2 = g.node("TestLazyMixImages", image1=input2.out(0), image2=input3.out(0), mask=mask2.out(0))

        output1 = g.node("SaveImage", images=lazy_mix1.out(0))
        output2 = g.node("SaveImage", images=lazy_mix2.out(0))

        # Run with partial execution targeting only output1
        result = client.run(g, partial_execution_targets=[output1.id])

        # For output1 path - only input1 should run due to lazy evaluation (mask=0.0)
        assert result.was_executed(input1), "Input1 should have been executed"
        assert not result.did_run(input2), "Input2 should not have run (lazy evaluation)"
        assert result.was_executed(mask1), "Mask1 should have been executed"
        assert result.was_executed(lazy_mix1), "Lazy mix1 should have been executed"
        assert result.was_executed(output1), "Output1 should have been executed"

        # Nothing from output2 path should run
        assert not result.did_run(input3), "Input3 should not have run"
        assert not result.did_run(mask2), "Mask2 should not have run"
        assert not result.did_run(lazy_mix2), "Lazy mix2 should not have run"
        assert not result.did_run(output2), "Output2 should not have run"

    # Multiple OUTPUT_NODEs with dependencies
    def test_partial_execution_multiple_output_nodes(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)

        # Create a chain of OUTPUT_NODEs
        output_node1 = g.node("TestOutputNodeWithSocketOutput", image=input1.out(0), value=1.5)
        output_node2 = g.node("TestOutputNodeWithSocketOutput", image=output_node1.out(0), value=2.0)

        # Create regular output nodes
        save1 = g.node("SaveImage", images=output_node1.out(0))
        save2 = g.node("SaveImage", images=output_node2.out(0))
        save3 = g.node("SaveImage", images=input2.out(0))

        # Run targeting only save2
        result = client.run(g, partial_execution_targets=[save2.id])

        # Should run: input1, output_node1, output_node2, save2
        assert result.was_executed(input1), "Input1 should have been executed"
        assert result.was_executed(output_node1), "Output node 1 should have been executed (dependency)"
        assert result.was_executed(output_node2), "Output node 2 should have been executed (dependency)"
        assert result.was_executed(save2), "Save2 should have been executed"

        # Should NOT run: input2, save1, save3
        assert not result.did_run(input2), "Input2 should not have run"
        assert not result.did_run(save1), "Save1 should not have run"
        assert not result.did_run(save3), "Save3 should not have run"

    # Empty partial execution list (should execute nothing)
    def test_partial_execution_empty_list(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        _output1 = g.node("SaveImage", images=input1.out(0))

        # Run with empty partial execution list
        try:
            _result = client.run(g, partial_execution_targets=[])
            # Should get an error because no outputs are selected
            assert False, "Should have raised an error for empty partial execution list"
        except urllib.error.HTTPError:
            pass  # Expected behavior

    def test_cached_outputs_in_job_without_client_id(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        image = g.node("StubImage", content="BLACK", height=32, width=32, batch_size=1)
        output = g.node("SaveImage", images=image.out(0))

        # Prime the cache with a normal run.
        client.run(g)

        # Resubmit anonymously (no client_id) so output nodes are cache hits with no websocket client.
        data = json.dumps({"prompt": g.finalize()}).encode('utf-8')
        req = urllib.request.Request(f"http://{client.server_address}/prompt", data=data)
        prompt_id = json.loads(urllib.request.urlopen(req).read())['prompt_id']

        for _ in range(100):
            job = client.get_job(prompt_id)
            if job is not None and job['status'] not in ('pending', 'in_progress'):
                break
            time.sleep(0.1)
        else:
            raise AssertionError("Prompt did not complete in time")

        assert job['status'] == 'completed'
        assert output.id in job['outputs'], "Cached outputs must appear in job outputs without a client_id"

    def _create_history_item(self, client, builder):
        g = GraphBuilder(prefix="offset_test")
        input_node = g.node(
            "StubImage", content="BLACK", height=32, width=32, batch_size=1
        )
        g.node("SaveImage", images=input_node.out(0))
        return client.run(g)

    def test_offset_returns_different_items_than_beginning_of_history(
        self, client: ComfyClient, builder: GraphBuilder
    ):
        """Test that offset skips items at the beginning"""
        for _ in range(5):
            self._create_history_item(client, builder)

        first_two = client.get_all_history(max_items=2, offset=0)
        next_two = client.get_all_history(max_items=2, offset=2)

        assert set(first_two.keys()).isdisjoint(
            set(next_two.keys())
        ), "Offset should skip initial items"

    def test_offset_beyond_history_length_returns_empty(
        self, client: ComfyClient, builder: GraphBuilder
    ):
        """Test offset larger than total history returns empty result"""
        self._create_history_item(client, builder)

        result = client.get_all_history(offset=100)
        assert len(result) == 0, "Large offset should return no items"

    def test_offset_at_exact_history_length_returns_empty(
        self, client: ComfyClient, builder: GraphBuilder
    ):
        """Test offset equal to history length returns empty"""
        for _ in range(3):
            self._create_history_item(client, builder)

        all_history = client.get_all_history()
        result = client.get_all_history(offset=len(all_history))
        assert len(result) == 0, "Offset at history length should return empty"

    def test_offset_zero_equals_no_offset_parameter(
        self, client: ComfyClient, builder: GraphBuilder
    ):
        """Test offset=0 behaves same as omitting offset"""
        self._create_history_item(client, builder)

        with_zero = client.get_all_history(offset=0)
        without_offset = client.get_all_history()

        assert with_zero == without_offset, "offset=0 should equal no offset"

    def test_offset_without_max_items_skips_from_beginning(
        self, client: ComfyClient, builder: GraphBuilder
    ):
        """Test offset alone (no max_items) returns remaining items"""
        for _ in range(4):
            self._create_history_item(client, builder)

        all_items = client.get_all_history()
        offset_items = client.get_all_history(offset=2)

        assert (
            len(offset_items) == len(all_items) - 2
        ), "Offset should skip specified number of items"

    def test_offset_with_max_items_returns_correct_window(
        self, client: ComfyClient, builder: GraphBuilder
    ):
        """Test offset + max_items returns correct slice of history"""
        for _ in range(6):
            self._create_history_item(client, builder)

        window = client.get_all_history(max_items=2, offset=1)
        assert len(window) <= 2, "Should respect max_items limit"

    def test_offset_near_end_returns_remaining_items_only(
        self, client: ComfyClient, builder: GraphBuilder
    ):
        """Test offset near end of history returns only remaining items"""
        for _ in range(3):
            self._create_history_item(client, builder)

        all_history = client.get_all_history()
        # Offset to near the end
        result = client.get_all_history(max_items=5, offset=len(all_history) - 1)

        assert len(result) <= 1, "Should return at most 1 item when offset is near end"

    # Jobs API tests
    def test_jobs_api_job_structure(
        self, client: ComfyClient, builder: GraphBuilder
    ):
        """Test that job objects have required fields"""
        self._create_history_item(client, builder)

        jobs_response = client.get_jobs(status="completed", limit=1)
        assert len(jobs_response["jobs"]) > 0, "Should have at least one job"

        job = jobs_response["jobs"][0]
        assert "id" in job, "Job should have id"
        assert "status" in job, "Job should have status"
        assert "create_time" in job, "Job should have create_time"
        assert "outputs_count" in job, "Job should have outputs_count"
        assert "preview_output" in job, "Job should have preview_output"

    def test_jobs_api_preview_output_structure(
        self, client: ComfyClient, builder: GraphBuilder
    ):
        """Test that preview_output has correct structure"""
        self._create_history_item(client, builder)

        jobs_response = client.get_jobs(status="completed", limit=1)
        job = jobs_response["jobs"][0]

        if job["preview_output"] is not None:
            preview = job["preview_output"]
            assert "filename" in preview, "Preview should have filename"
            assert "nodeId" in preview, "Preview should have nodeId"
            assert "mediaType" in preview, "Preview should have mediaType"

    def test_jobs_api_pagination(
        self, client: ComfyClient, builder: GraphBuilder
    ):
        """Test jobs API pagination"""
        for _ in range(5):
            self._create_history_item(client, builder)

        first_page = client.get_jobs(limit=2, offset=0)
        second_page = client.get_jobs(limit=2, offset=2)

        assert len(first_page["jobs"]) <= 2, "First page should have at most 2 jobs"
        assert len(second_page["jobs"]) <= 2, "Second page should have at most 2 jobs"

        first_ids = {j["id"] for j in first_page["jobs"]}
        second_ids = {j["id"] for j in second_page["jobs"]}
        assert first_ids.isdisjoint(second_ids), "Pages should have different jobs"

    def test_jobs_api_sorting(
        self, client: ComfyClient, builder: GraphBuilder
    ):
        """Test jobs API sorting"""
        for _ in range(3):
            self._create_history_item(client, builder)

        desc_jobs = client.get_jobs(sort_order="desc")
        asc_jobs = client.get_jobs(sort_order="asc")

        if len(desc_jobs["jobs"]) >= 2:
            desc_times = [j["create_time"] for j in desc_jobs["jobs"] if j["create_time"]]
            asc_times = [j["create_time"] for j in asc_jobs["jobs"] if j["create_time"]]
            if len(desc_times) >= 2:
                assert desc_times == sorted(desc_times, reverse=True), "Desc should be newest first"
            if len(asc_times) >= 2:
                assert asc_times == sorted(asc_times), "Asc should be oldest first"

    def test_jobs_api_status_filter(
        self, client: ComfyClient, builder: GraphBuilder
    ):
        """Test jobs API status filtering"""
        self._create_history_item(client, builder)

        completed_jobs = client.get_jobs(status="completed")
        assert len(completed_jobs["jobs"]) > 0, "Should have completed jobs from history"

        for job in completed_jobs["jobs"]:
            assert job["status"] == "completed", "Should only return completed jobs"

        # Pending jobs are transient - just verify filter doesn't error
        pending_jobs = client.get_jobs(status="pending")
        for job in pending_jobs["jobs"]:
            assert job["status"] == "pending", "Should only return pending jobs"

    def test_get_job_by_id(
        self, client: ComfyClient, builder: GraphBuilder
    ):
        """Test getting a single job by ID"""
        result = self._create_history_item(client, builder)
        prompt_id = result.get_prompt_id()

        job = client.get_job(prompt_id)
        assert job is not None, "Should find the job"
        assert job["id"] == prompt_id, "Job ID should match"
        assert "outputs" in job, "Single job should include outputs"

    def test_get_job_not_found(
        self, client: ComfyClient, builder: GraphBuilder
    ):
        """Test getting a non-existent job returns 404"""
        job = client.get_job("nonexistent-job-id")
        assert job is None, "Non-existent job should return None"
