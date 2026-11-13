"""Tests for --disable-subgraph-caching and the per-node "no_cache" prompt key.

The frontend flattens UI subgraphs into colon-joined node ids ("12:5" is node 5
inside the subgraph instance with node id 12), so these tests fake subgraph
membership with explicit colon ids. See mark_subgraph_internal_nodes_no_cache
in execution.py for the marking rules being tested.
"""
import pytest
import subprocess
import time

from pytest import fixture
from comfy_execution.graph_utils import GraphBuilder
from tests.execution.test_execution import ComfyClient, run_warmup


class GraphWithNoCacheKeys:
    """Wraps a GraphBuilder so finalize() stamps "no_cache" on chosen nodes."""
    def __init__(self, graph, no_cache_ids):
        self.graph = graph
        self.nodes = graph.nodes
        self.no_cache_ids = no_cache_ids

    def finalize(self):
        prompt = self.graph.finalize()
        for node_id in self.no_cache_ids:
            prompt[node_id]["no_cache"] = True
        return prompt


@pytest.mark.execution
class TestSubgraphInternalCache:
    @fixture(scope="class", autouse=True, params=[
        [],
        ["--cache-classic"],
    ])
    def _server(self, args_pytest, request):
        pargs = [
            'python','main.py',
            '--output-directory', args_pytest["output_dir"],
            '--listen', args_pytest["listen"],
            '--port', str(args_pytest["port"]),
            '--extra-model-paths-config', 'tests/execution/extra_model_paths.yaml',
            '--cpu',
            '--disable-subgraph-caching',
        ]
        pargs += request.param
        print("Running server with args:", pargs)  # noqa: T201
        p = subprocess.Popen(pargs)
        yield
        p.kill()

    @fixture(scope="class", autouse=True)
    def shared_client(self, args_pytest, _server):
        client = ComfyClient()
        n_tries = 5
        for i in range(n_tries):
            time.sleep(4)
            try:
                client.connect(listen=args_pytest["listen"], port=args_pytest["port"])
            except ConnectionRefusedError as e:
                print(e)  # noqa: T201
                print(f"({i+1}/{n_tries}) Retrying...")  # noqa: T201
            else:
                break
        run_warmup(client)
        yield client
        del client

    @fixture
    def client(self, shared_client, request):
        shared_client.set_test_name(f"subgraph_cache[{request.node.name}]")
        yield shared_client

    @fixture
    def builder(self, request):
        yield GraphBuilder(prefix=request.node.name)

    def build_subgraph_chain(self, g):
        """Two top-level sources feeding a two-stage mix chain inside a faked
        subgraph, with the second stage's output consumed at the top level."""
        src = g.node("StubImage", id="src", content="BLACK", height=32, width=32, batch_size=1)
        white = g.node("StubImage", id="white", content="WHITE", height=32, width=32, batch_size=1)
        stage1_mask = g.node("StubMask", id="sg:stage1_mask", value=0.5, height=32, width=32, batch_size=1)
        boundary_mask = g.node("StubMask", id="sg:boundary_mask", value=0.5, height=32, width=32, batch_size=1)
        stage1 = g.node("TestLazyMixImages", id="sg:stage1", image1=src.out(0), image2=white.out(0), mask=stage1_mask.out(0))
        boundary = g.node("TestLazyMixImages", id="sg:boundary", image1=stage1.out(0), image2=white.out(0), mask=boundary_mask.out(0))
        top_preview = g.node("PreviewImage", id="top_preview", images=boundary.out(0))
        return src, white, stage1_mask, boundary_mask, stage1, boundary, top_preview

    def test_unchanged_requeue_is_noop_without_caching_internals(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        src, white, stage1_mask, boundary_mask, stage1, boundary, top_preview = self.build_subgraph_chain(g)

        client.run(g)
        result = client.run(g)

        for node in (src, white, stage1_mask, boundary_mask, stage1, boundary, top_preview):
            assert not result.did_run(node), f"{node.id} should not re-run on unchanged re-queue"
        assert result.was_cached(boundary), "boundary node's value leaves the subgraph, must stay cached"
        assert result.was_cached(src)
        assert result.was_cached(white)
        assert not result.was_cached(stage1), "subgraph-internal node must not be in the persistent cache"
        assert not result.was_cached(stage1_mask), "subgraph-internal node must not be in the persistent cache"

    def test_output_node_inside_subgraph_stays_cached(self, client: ComfyClient, builder: GraphBuilder):
        # Also regression-tests the ExecutionList change: a cached output node
        # must not pull its uncached in-subgraph ancestors into execution.
        g = builder
        _, _, _, _, stage1, _, _ = self.build_subgraph_chain(g)
        inner_preview = g.node("PreviewImage", id="sg:preview", images=stage1.out(0))

        client.run(g)
        result = client.run(g)

        assert result.was_cached(inner_preview), "output nodes inside subgraphs stay cached so re-queues stay no-ops"
        assert not result.did_run(inner_preview)
        assert not result.did_run(stage1), "cached output node must not drag its uncached ancestor into execution"
        assert not result.was_cached(stage1)

    def test_internal_edit_reruns_chain_from_inside(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        src, white, stage1_mask, _, stage1, boundary, top_preview = self.build_subgraph_chain(g)

        client.run(g)
        stage1_mask.inputs["value"] = 0.7
        result = client.run(g)

        assert result.did_run(stage1_mask)
        assert result.did_run(stage1)
        assert result.did_run(boundary)
        assert result.did_run(top_preview)
        assert result.was_cached(src)
        assert result.was_cached(white)

    def test_uncached_internal_recomputes_when_downstream_invalidated(self, client: ComfyClient, builder: GraphBuilder):
        # The documented tradeoff: stage1 is unchanged, but its output was not
        # kept, so invalidating only the boundary stage recomputes stage1 too.
        g = builder
        src, white, _, boundary_mask, stage1, boundary, _ = self.build_subgraph_chain(g)

        client.run(g)
        boundary_mask.inputs["value"] = 0.9
        result = client.run(g)

        assert result.did_run(boundary)
        assert result.did_run(stage1), "uncached internal must recompute to feed the invalidated boundary node"
        assert result.was_cached(src)
        assert result.was_cached(white)

    def test_explicit_no_cache_key_without_subgraph(self, client: ComfyClient, builder: GraphBuilder):
        g = builder
        src = g.node("StubImage", id="src", content="BLACK", height=32, width=32, batch_size=1)
        white = g.node("StubImage", id="white", content="WHITE", height=32, width=32, batch_size=1)
        mask = g.node("StubMask", id="mask", value=0.5, height=32, width=32, batch_size=1)
        middle = g.node("TestLazyMixImages", id="middle", image1=src.out(0), image2=white.out(0), mask=mask.out(0))
        final = g.node("TestLazyMixImages", id="final", image1=middle.out(0), image2=white.out(0), mask=mask.out(0))
        preview = g.node("PreviewImage", id="preview", images=final.out(0))
        wrapped = GraphWithNoCacheKeys(g, [middle.id])

        client.run(wrapped)
        result = client.run(wrapped)

        for node in (src, white, mask, middle, final, preview):
            assert not result.did_run(node), f"{node.id} should not re-run on unchanged re-queue"
        assert not result.was_cached(middle), "node with no_cache key must not be in the persistent cache"
        assert result.was_cached(final)
