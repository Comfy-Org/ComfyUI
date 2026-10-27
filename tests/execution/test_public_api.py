"""
Tests for public ComfyAPI and ComfyAPISync functions.

These tests verify that the public API methods work correctly in both sync and async contexts,
ensuring that the sync wrapper generation (via get_type_hints() in async_to_sync.py) correctly
handles string annotations from 'from __future__ import annotations'.
"""

import pytest
import time
import subprocess
import torch
from pytest import fixture
from comfy_execution.graph_utils import GraphBuilder
from tests.execution.test_execution import ComfyClient


@pytest.mark.execution
class TestPublicAPI:
    """Test suite for public ComfyAPI and ComfyAPISync methods."""

    @fixture(scope="class", autouse=True)
    def _server(self, args_pytest):
        """Start ComfyUI server for testing."""
        pargs = [
            'python', 'main.py',
            '--output-directory', args_pytest["output_dir"],
            '--listen', args_pytest["listen"],
            '--port', str(args_pytest["port"]),
            '--extra-model-paths-config', 'tests/execution/extra_model_paths.yaml',
            '--cpu',
            '--disable-assets',
        ]
        p = subprocess.Popen(pargs)
        yield
        p.kill()
        torch.cuda.empty_cache()

    @fixture(scope="class", autouse=True)
    def shared_client(self, args_pytest, _server):
        """Create shared client with connection retry."""
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
        """Set test name for each test."""
        shared_client.set_test_name(f"public_api[{request.node.name}]")
        yield shared_client

    @fixture
    def builder(self, request):
        """Create GraphBuilder for each test."""
        yield GraphBuilder(prefix=request.node.name)

    def test_sync_progress_update_executes(self, client: ComfyClient, builder: GraphBuilder):
        """Test that TestSyncProgressUpdate executes without errors.

        This test validates that api_sync.execution.set_progress() works correctly,
        which is the primary code path fixed by adding get_type_hints() to async_to_sync.py.
        """
        g = builder
        image = g.node("StubImage", content="BLACK", height=256, width=256, batch_size=1)

        # Use TestSyncProgressUpdate with short sleep
        progress_node = g.node("TestSyncProgressUpdate",
                              value=image.out(0),
                              sleep_seconds=0.5)
        output = g.node("SaveImage", images=progress_node.out(0))

        # Execute workflow
        result = client.run(g)

        # Verify execution
        assert result.did_run(progress_node), "Progress node should have executed"
        assert result.did_run(output), "Output node should have executed"

        # Verify output
        images = result.get_images(output)
        assert len(images) == 1, "Should have produced 1 image"

    def test_async_progress_update_executes(self, client: ComfyClient, builder: GraphBuilder):
        """Test that TestAsyncProgressUpdate executes without errors.

        This test validates that await api.execution.set_progress() works correctly
        in async contexts.
        """
        g = builder
        image = g.node("StubImage", content="WHITE", height=256, width=256, batch_size=1)

        # Use TestAsyncProgressUpdate with short sleep
        progress_node = g.node("TestAsyncProgressUpdate",
                              value=image.out(0),
                              sleep_seconds=0.5)
        output = g.node("SaveImage", images=progress_node.out(0))

        # Execute workflow
        result = client.run(g)

        # Verify execution
        assert result.did_run(progress_node), "Async progress node should have executed"
        assert result.did_run(output), "Output node should have executed"

        # Verify output
        images = result.get_images(output)
        assert len(images) == 1, "Should have produced 1 image"

    def test_sync_and_async_progress_together(self, client: ComfyClient, builder: GraphBuilder):
        """Test both sync and async progress updates in same workflow.

        This test ensures that both ComfyAPISync and ComfyAPI can coexist and work
        correctly in the same workflow execution.
        """
        g = builder
        image1 = g.node("StubImage", content="BLACK", height=256, width=256, batch_size=1)
        image2 = g.node("StubImage", content="WHITE", height=256, width=256, batch_size=1)

        # Use both types of progress nodes
        sync_progress = g.node("TestSyncProgressUpdate",
                              value=image1.out(0),
                              sleep_seconds=0.3)
        async_progress = g.node("TestAsyncProgressUpdate",
                               value=image2.out(0),
                               sleep_seconds=0.3)

        # Create outputs
        output1 = g.node("SaveImage", images=sync_progress.out(0))
        output2 = g.node("SaveImage", images=async_progress.out(0))

        # Execute workflow
        result = client.run(g)

        # Both should execute successfully
        assert result.did_run(sync_progress), "Sync progress node should have executed"
        assert result.did_run(async_progress), "Async progress node should have executed"
        assert result.did_run(output1), "First output node should have executed"
        assert result.did_run(output2), "Second output node should have executed"

        # Verify outputs
        images1 = result.get_images(output1)
        images2 = result.get_images(output2)
        assert len(images1) == 1, "Should have produced 1 image from sync node"
        assert len(images2) == 1, "Should have produced 1 image from async node"
