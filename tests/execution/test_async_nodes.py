import pytest
import time
import torch
import urllib.error
import numpy as np
import subprocess

from pytest import fixture
from comfy_execution.graph_utils import GraphBuilder
from tests.execution.test_execution import ComfyClient, run_warmup


@pytest.mark.execution
class TestAsyncNodes:
    @fixture(scope="class", autouse=True, params=[
        (False, 0),
        (True, 100),
    ])
    def _server(self, args_pytest, request):
        pargs = [
            'python','main.py',
            '--output-directory', args_pytest["output_dir"],
            '--listen', args_pytest["listen"],
            '--port', str(args_pytest["port"]),
            '--extra-model-paths-config', 'tests/execution/extra_model_paths.yaml',
            '--cpu',
        ]
        use_lru, lru_size = request.param
        if use_lru:
            pargs += ['--cache-lru', str(lru_size)]
        else:
            pargs += ['--cache-classic']
        # Running server with args: pargs
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
            except ConnectionRefusedError:
                # Retrying...
                pass
            else:
                break
        yield client
        del client
        torch.cuda.empty_cache()

    @fixture
    def client(self, shared_client, request):
        shared_client.set_test_name(f"async_nodes[{request.node.name}]")
        yield shared_client

    @fixture
    def builder(self, request):
        yield GraphBuilder(prefix=request.node.name)

    # Happy Path Tests

    def test_basic_async_execution(self, client: ComfyClient, builder: GraphBuilder):
        """Test that a basic async node executes correctly."""
        g = builder
        image = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        sleep_node = g.node("TestSleep", value=image.out(0), seconds=0.1)
        output = g.node("SaveImage", images=sleep_node.out(0))

        result = client.run(g)

        # Verify execution completed
        assert result.did_run(sleep_node), "Async sleep node should have executed"
        assert result.did_run(output), "Output node should have executed"

        # Verify the image passed through correctly
        result_images = result.get_images(output)
        assert len(result_images) == 1, "Should have 1 image"
        assert np.array(result_images[0]).min() == 0 and np.array(result_images[0]).max() == 0, "Image should be black"

    def test_multiple_async_parallel_execution(self, client: ComfyClient, builder: GraphBuilder, skip_timing_checks):
        """Test that multiple async nodes execute in parallel."""
        # Warmup execution to ensure server is fully initialized
        run_warmup(client)

        g = builder
        image = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)

        # Create multiple async sleep nodes with different durations
        sleep1 = g.node("TestSleep", value=image.out(0), seconds=0.3)
        sleep2 = g.node("TestSleep", value=image.out(0), seconds=0.4)
        sleep3 = g.node("TestSleep", value=image.out(0), seconds=0.5)

        # Add outputs for each
        _output1 = g.node("PreviewImage", images=sleep1.out(0))
        _output2 = g.node("PreviewImage", images=sleep2.out(0))
        _output3 = g.node("PreviewImage", images=sleep3.out(0))

        start_time = time.time()
        result = client.run(g)
        elapsed_time = time.time() - start_time

        # Should take ~0.5s (max duration) not 1.2s (sum of durations)
        if not skip_timing_checks:
            assert elapsed_time < 0.8, f"Parallel execution took {elapsed_time}s, expected < 0.8s"

        # Verify all nodes executed
        assert result.did_run(sleep1) and result.did_run(sleep2) and result.did_run(sleep3)

    def test_async_with_dependencies(self, client: ComfyClient, builder: GraphBuilder):
        """Test async nodes with proper dependency handling."""
        g = builder
        image1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        image2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)

        # Chain of async operations
        sleep1 = g.node("TestSleep", value=image1.out(0), seconds=0.2)
        sleep2 = g.node("TestSleep", value=image2.out(0), seconds=0.2)

        # Average depends on both async results
        average = g.node("TestVariadicAverage", input1=sleep1.out(0), input2=sleep2.out(0))
        output = g.node("SaveImage", images=average.out(0))

        result = client.run(g)

        # Verify execution order
        assert result.did_run(sleep1) and result.did_run(sleep2)
        assert result.did_run(average) and result.did_run(output)

        # Verify averaged result
        result_images = result.get_images(output)
        avg_value = np.array(result_images[0]).mean()
        assert abs(avg_value - 127.5) < 1, f"Average value {avg_value} should be ~127.5"

    def test_async_validate_inputs(self, client: ComfyClient, builder: GraphBuilder):
        """Test async VALIDATE_INPUTS function."""
        g = builder
        # Create a test node with async validation
        validation_node = g.node("TestAsyncValidation", value=5.0, threshold=10.0)
        g.node("SaveImage", images=validation_node.out(0))

        # Should pass validation
        result = client.run(g)
        assert result.did_run(validation_node)

        # Test validation failure
        validation_node.inputs['threshold'] = 3.0  # Will fail since value > threshold
        with pytest.raises(urllib.error.HTTPError):
            client.run(g)

    def test_async_lazy_evaluation(self, client: ComfyClient, builder: GraphBuilder, skip_timing_checks):
        """Test async nodes with lazy evaluation."""
        # Warmup execution to ensure server is fully initialized
        run_warmup(client, prefix="warmup_lazy")

        g = builder
        input1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        input2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        mask = g.node("StubMask", value=0.0, height=512, width=512, batch_size=1)

        # Create async nodes that will be evaluated lazily
        sleep1 = g.node("TestSleep", value=input1.out(0), seconds=0.3)
        sleep2 = g.node("TestSleep", value=input2.out(0), seconds=0.3)

        # Use lazy mix that only needs sleep1 (mask=0.0)
        lazy_mix = g.node("TestLazyMixImages", image1=sleep1.out(0), image2=sleep2.out(0), mask=mask.out(0))
        g.node("SaveImage", images=lazy_mix.out(0))

        start_time = time.time()
        result = client.run(g)
        elapsed_time = time.time() - start_time

        # Should only execute sleep1, not sleep2
        if not skip_timing_checks:
            assert elapsed_time < 0.5, f"Should skip sleep2, took {elapsed_time}s"
        assert result.did_run(sleep1), "Sleep1 should have executed"
        assert not result.did_run(sleep2), "Sleep2 should have been skipped"

    def test_async_check_lazy_status(self, client: ComfyClient, builder: GraphBuilder):
        """Test async check_lazy_status function."""
        g = builder
        # Create a node with async check_lazy_status
        lazy_node = g.node("TestAsyncLazyCheck",
                          input1="value1",
                          input2="value2",
                          condition=True)
        g.node("SaveImage", images=lazy_node.out(0))

        result = client.run(g)
        assert result.did_run(lazy_node)

    # Error Handling Tests

    def test_async_execution_error(self, client: ComfyClient, builder: GraphBuilder):
        """Test that async execution errors are properly handled."""
        g = builder
        image = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        # Create an async node that will error
        error_node = g.node("TestAsyncError", value=image.out(0), error_after=0.1)
        g.node("SaveImage", images=error_node.out(0))

        try:
            client.run(g)
            assert False, "Should have raised an error"
        except Exception as e:
            assert 'prompt_id' in e.args[0], f"Did not get proper error message: {e}"
            assert e.args[0]['node_id'] == error_node.id, "Error should be from async error node"

    def test_async_validation_error(self, client: ComfyClient, builder: GraphBuilder):
        """Test async validation error handling."""
        g = builder
        # Node with async validation that will fail
        validation_node = g.node("TestAsyncValidationError", value=15.0, max_value=10.0)
        g.node("SaveImage", images=validation_node.out(0))

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            client.run(g)
        # Verify it's a validation error
        assert exc_info.value.code == 400

    def test_async_timeout_handling(self, client: ComfyClient, builder: GraphBuilder):
        """Test handling of async operations that timeout."""
        g = builder
        image = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        # Very long sleep that would timeout
        timeout_node = g.node("TestAsyncTimeout", value=image.out(0), timeout=0.5, operation_time=2.0)
        g.node("SaveImage", images=timeout_node.out(0))

        try:
            client.run(g)
            assert False, "Should have raised a timeout error"
        except Exception as e:
            assert 'timeout' in str(e).lower(), f"Expected timeout error, got: {e}"

    def test_concurrent_async_error_recovery(self, client: ComfyClient, builder: GraphBuilder):
        """Test that workflow can recover after async errors."""
        g = builder
        image = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)

        # First run with error
        error_node = g.node("TestAsyncError", value=image.out(0), error_after=0.1)
        g.node("SaveImage", images=error_node.out(0))

        try:
            client.run(g)
        except Exception:
            pass  # Expected

        # Second run should succeed
        g2 = GraphBuilder(prefix="recovery_test")
        image2 = g2.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        sleep_node = g2.node("TestSleep", value=image2.out(0), seconds=0.1)
        g2.node("SaveImage", images=sleep_node.out(0))

        result = client.run(g2)
        assert result.did_run(sleep_node), "Should be able to run after error"

    def test_sync_error_during_async_execution(self, client: ComfyClient, builder: GraphBuilder):
        """Test handling when sync node errors while async node is executing."""
        g = builder
        image = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)

        # Async node that takes time
        sleep_node = g.node("TestSleep", value=image.out(0), seconds=0.5)

        # Sync node that will error immediately
        error_node = g.node("TestSyncError", value=image.out(0))

        # Both feed into output
        g.node("PreviewImage", images=sleep_node.out(0))
        g.node("PreviewImage", images=error_node.out(0))

        try:
            client.run(g)
            assert False, "Should have raised an error"
        except Exception as e:
            # Verify the sync error was caught even though async was running
            assert 'prompt_id' in e.args[0]

    def test_async_sibling_completes_after_error(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        image = g.node("StubImage", content="BLACK", height=32, width=32, batch_size=1)
        error_node = g.node("TestAsyncError", value=image.out(0), error_after=0.05)
        sleep_node = g.node("TestSleep", value=image.out(0), seconds=0.1)
        g.node("PreviewImage", images=error_node.out(0))
        successful_output = g.node("SaveImage", images=sleep_node.out(0))

        result = client.run(g, node_failure_policy="continue_independent")

        assert result.did_run(error_node)
        assert result.did_run(sleep_node)
        assert result.was_executed(successful_output)
        assert len(result.get_images(successful_output)) == 1
        assert result.execution_success['completion_status'] == 'partial_success'
        assert result.node_errors[0]['node_id'] == error_node.id

    def test_async_sibling_completes_after_multiple_errors(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        image = g.node("StubImage", content="BLACK", height=32, width=32, batch_size=1)
        error1 = g.node("TestAsyncError", value=image.out(0), error_after=0.02)
        error2 = g.node("TestAsyncError", value=image.out(0), error_after=0.04)
        sleep_node = g.node("TestSleep", value=image.out(0), seconds=0.06)
        g.node("PreviewImage", images=error1.out(0))
        g.node("PreviewImage", images=error2.out(0))
        successful_output = g.node("SaveImage", images=sleep_node.out(0))

        result = client.run(g, node_failure_policy="continue_independent")

        assert {error['node_id'] for error in result.node_errors} == {error1.id, error2.id}
        assert result.did_run(sleep_node)
        assert result.was_executed(successful_output)
        assert result.execution_success['completion_status'] == 'partial_success'
        assert result.execution_success['execution_error_count'] == 2

    # Edge Cases

    def test_async_with_execution_blocker(self, client: ComfyClient, builder: GraphBuilder):
        """Test async nodes with execution blockers."""
        g = builder
        image1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        image2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)

        # Async sleep nodes
        sleep1 = g.node("TestSleep", value=image1.out(0), seconds=0.2)
        sleep2 = g.node("TestSleep", value=image2.out(0), seconds=0.2)

        # Create list of images
        image_list = g.node("TestMakeListNode", value1=sleep1.out(0), value2=sleep2.out(0))

        # Create list of blocking conditions - [False, True] to block only the second item
        int1 = g.node("StubInt", value=1)
        int2 = g.node("StubInt", value=2)
        block_list = g.node("TestMakeListNode", value1=int1.out(0), value2=int2.out(0))

        # Compare each value against 2, so first is False (1 != 2) and second is True (2 == 2)
        compare = g.node("TestIntConditions", a=block_list.out(0), b=2, operation="==")

        # Block based on the comparison results
        blocker = g.node("TestExecutionBlocker", input=image_list.out(0), block=compare.out(0), verbose=False)

        output = g.node("PreviewImage", images=blocker.out(0))

        result = client.run(g)
        images = result.get_images(output)
        assert len(images) == 1, "Should have blocked second image"

    def test_async_caching_behavior(self, client: ComfyClient, builder: GraphBuilder, skip_timing_checks):
        """Test that async nodes are properly cached."""
        # Warmup execution to ensure server is fully initialized
        run_warmup(client, prefix="warmup_cache")

        g = builder
        image = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        sleep_node = g.node("TestSleep", value=image.out(0), seconds=0.2)
        g.node("SaveImage", images=sleep_node.out(0))

        # First run
        result1 = client.run(g)
        assert result1.did_run(sleep_node), "Should run first time"

        # Second run - should be cached
        start_time = time.time()
        result2 = client.run(g)
        elapsed_time = time.time() - start_time

        assert not result2.did_run(sleep_node), "Should be cached"
        if not skip_timing_checks:
            assert elapsed_time < 0.1, f"Cached run took {elapsed_time}s, should be instant"

    def test_async_with_dynamic_prompts(self, client: ComfyClient, builder: GraphBuilder, skip_timing_checks):
        """Test async nodes within dynamically generated prompts."""
        # Warmup execution to ensure server is fully initialized
        run_warmup(client, prefix="warmup_dynamic")

        g = builder
        image1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        image2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)

        # Node that generates async nodes dynamically
        dynamic_async = g.node("TestDynamicAsyncGeneration",
                              image1=image1.out(0),
                              image2=image2.out(0),
                              num_async_nodes=5,
                              sleep_duration=0.4)
        g.node("SaveImage", images=dynamic_async.out(0))

        start_time = time.time()
        result = client.run(g)
        elapsed_time = time.time() - start_time

        # Should execute async nodes in parallel within dynamic prompt
        if not skip_timing_checks:
            assert elapsed_time < 1.0, f"Dynamic async execution took {elapsed_time}s"
        assert result.did_run(dynamic_async)

    def test_async_resource_cleanup(self, client: ComfyClient, builder: GraphBuilder):
        """Test that async resources are properly cleaned up."""
        g = builder
        image = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)

        # Create multiple async nodes that use resources
        resource_nodes = []
        for i in range(5):
            node = g.node("TestAsyncResourceUser",
                         value=image.out(0),
                         resource_id=f"resource_{i}",
                         duration=0.1)
            resource_nodes.append(node)
            g.node("PreviewImage", images=node.out(0))

        result = client.run(g)

        # Verify all nodes executed
        for node in resource_nodes:
            assert result.did_run(node)

        # Run again to ensure resources were cleaned up
        result2 = client.run(g)
        # Should be cached but not error due to resource conflicts
        for node in resource_nodes:
            assert not result2.did_run(node), "Should be cached"

    def test_async_cancellation(self, client: ComfyClient, builder: GraphBuilder):
        """Test cancellation of async operations."""
        # This would require implementing cancellation in the client
        # For now, we'll test that long-running async operations can be interrupted
        pass  # TODO: Implement when cancellation API is available

    def test_mixed_sync_async_execution(self, client: ComfyClient, builder: GraphBuilder):
        """Test workflows with both sync and async nodes."""
        g = builder
        image1 = g.node("StubImage", content="BLACK", height=512, width=512, batch_size=1)
        image2 = g.node("StubImage", content="WHITE", height=512, width=512, batch_size=1)
        mask = g.node("StubMask", value=0.5, height=512, width=512, batch_size=1)

        # Mix of sync and async operations
        # Sync: lazy mix images
        sync_op1 = g.node("TestLazyMixImages", image1=image1.out(0), image2=image2.out(0), mask=mask.out(0))
        # Async: sleep
        async_op1 = g.node("TestSleep", value=sync_op1.out(0), seconds=0.2)
        # Sync: custom validation
        sync_op2 = g.node("TestCustomValidation1", input1=async_op1.out(0), input2=0.5)
        # Async: sleep again
        async_op2 = g.node("TestSleep", value=sync_op2.out(0), seconds=0.2)

        output = g.node("SaveImage", images=async_op2.out(0))

        result = client.run(g)

        # Verify all nodes executed in correct order
        assert result.did_run(sync_op1)
        assert result.did_run(async_op1)
        assert result.did_run(sync_op2)
        assert result.did_run(async_op2)

        # Image should be a mix of black and white (gray)
        result_images = result.get_images(output)
        avg_value = np.array(result_images[0]).mean()
        assert abs(avg_value - 63.75) < 5, f"Average value {avg_value} should be ~63.75"
