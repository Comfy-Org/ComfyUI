"""An API-format prompt saved before an optional input existed still has to run."""

import subprocess
import time

import numpy
import pytest
import torch
from pytest import fixture

from comfy_execution.graph_utils import GraphBuilder
from tests.execution.test_execution import ComfyClient


@pytest.mark.execution
class TestOmittedOptionalInput:
    @fixture(scope="class", autouse=True)
    def _server(self, args_pytest):
        pargs = [
            'python', 'main.py',
            '--output-directory', args_pytest["output_dir"],
            '--listen', args_pytest["listen"],
            '--port', str(args_pytest["port"]),
            '--extra-model-paths-config', 'tests/execution/extra_model_paths.yaml',
            '--cpu',
        ]
        p = subprocess.Popen(pargs)
        yield
        p.kill()
        torch.cuda.empty_cache()

    @fixture(scope="class", autouse=True)
    def shared_client(self, args_pytest, _server):
        client = ComfyClient()
        n_tries = 5
        for i in range(n_tries):
            time.sleep(4)
            try:
                client.connect(listen=args_pytest["listen"], port=args_pytest["port"])
                break
            except ConnectionRefusedError:
                if i == n_tries - 1:
                    raise
        yield client
        del client
        torch.cuda.empty_cache()

    @fixture
    def client(self, shared_client, request):
        shared_client.set_test_name(f"omitted_optional_input[{request.node.name}]")
        yield shared_client

    @fixture
    def builder(self, request):
        yield GraphBuilder(prefix=request.node.name)

    def test_omitted_input_gets_its_schema_default(self, client: ComfyClient, builder: GraphBuilder):
        """`scale` has a default in the schema and none in the entry function."""
        g = builder
        image = g.node("StubImage", content="WHITE", height=64, width=64, batch_size=1)
        scaled = g.node("TestOmittedOptionalInput", image=image.out(0))
        output = g.node("SaveImage", images=scaled.out(0))

        result = client.run(g)

        result_image = numpy.array(result.get_images(output)[0])
        assert result_image.min() == 127 and result_image.max() == 127, "Image should be scaled by the default 0.5"

    def test_provided_input_still_wins(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        image = g.node("StubImage", content="WHITE", height=64, width=64, batch_size=1)
        scaled = g.node("TestOmittedOptionalInput", image=image.out(0), scale=1.0)
        output = g.node("SaveImage", images=scaled.out(0))

        result = client.run(g)

        result_image = numpy.array(result.get_images(output)[0])
        assert result_image.min() == 255 and result_image.max() == 255, "Image should keep its own scale"

    def test_connected_nullable_input_is_still_delivered(self, client: ComfyClient, builder: GraphBuilder):
        """`mask` has no default in the schema, so it stays a plain nullable socket."""
        g = builder
        image = g.node("StubImage", content="WHITE", height=64, width=64, batch_size=1)
        mask = g.node("StubMask", value=0.0, height=64, width=64, batch_size=1)
        scaled = g.node("TestOmittedOptionalInput", image=image.out(0), mask=mask.out(0))
        output = g.node("SaveImage", images=scaled.out(0))

        result = client.run(g)

        result_image = numpy.array(result.get_images(output)[0])
        assert result_image.min() == 0 and result_image.max() == 0, "Image should be masked to black"
