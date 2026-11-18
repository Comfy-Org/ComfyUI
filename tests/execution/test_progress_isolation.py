"""Test that progress updates are properly isolated between WebSocket clients."""

import json
import pytest
import time
import threading
import uuid
import websocket
from typing import List, Dict, Any
from comfy_execution.graph_utils import GraphBuilder
from tests.execution.test_execution import ComfyClient


class ProgressTracker:
    """Tracks progress messages received by a WebSocket client."""

    def __init__(self, client_id: str):
        self.client_id = client_id
        self.progress_messages: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def add_message(self, message: Dict[str, Any]):
        """Thread-safe addition of progress messages."""
        with self.lock:
            self.progress_messages.append(message)

    def get_messages_for_prompt(self, prompt_id: str) -> List[Dict[str, Any]]:
        """Get all progress messages for a specific prompt_id."""
        with self.lock:
            return [
                msg for msg in self.progress_messages
                if msg.get('data', {}).get('prompt_id') == prompt_id
            ]

    def has_cross_contamination(self, own_prompt_id: str) -> bool:
        """Check if this client received progress for other prompts."""
        with self.lock:
            for msg in self.progress_messages:
                msg_prompt_id = msg.get('data', {}).get('prompt_id')
                if msg_prompt_id and msg_prompt_id != own_prompt_id:
                    return True
            return False


class IsolatedClient(ComfyClient):
    """Extended ComfyClient that tracks all WebSocket messages."""

    def __init__(self):
        super().__init__()
        self.progress_tracker = None
        self.all_messages: List[Dict[str, Any]] = []

    def connect(self, listen='127.0.0.1', port=8188, client_id=None):
        """Connect with a specific client_id and set up message tracking."""
        if client_id is None:
            client_id = str(uuid.uuid4())
        super().connect(listen, port, client_id)
        self.progress_tracker = ProgressTracker(client_id)

    def listen_for_messages(self, duration: float = 5.0):
        """Listen for WebSocket messages for a specified duration."""
        end_time = time.time() + duration
        self.ws.settimeout(0.5)  # Non-blocking with timeout

        while time.time() < end_time:
            try:
                out = self.ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    self.all_messages.append(message)

                    # Track progress_state messages
                    if message.get('type') == 'progress_state':
                        self.progress_tracker.add_message(message)
            except websocket.WebSocketTimeoutException:
                continue
            except Exception:
                # Log error silently in test context
                break


@pytest.mark.execution
class TestProgressIsolation:
    """Test suite for verifying progress update isolation between clients."""

    @pytest.fixture(scope="class", autouse=True)
    def _server(self, args_pytest):
        """Start the ComfyUI server for testing."""
        import subprocess
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

    def start_client_with_retry(self, listen: str, port: int, client_id: str = None):
        """Start client with connection retries."""
        client = IsolatedClient()
        # Connect to server (with retries)
        n_tries = 5
        for i in range(n_tries):
            time.sleep(4)
            try:
                client.connect(listen, port, client_id)
                return client
            except ConnectionRefusedError as e:
                print(e)  # noqa: T201
                print(f"({i+1}/{n_tries}) Retrying...")  # noqa: T201
        raise ConnectionRefusedError(f"Failed to connect after {n_tries} attempts")

    def test_progress_isolation_between_clients(self, args_pytest):
        """Test that progress updates are isolated between different clients."""
        listen = args_pytest["listen"]
        port = args_pytest["port"]

        # Create two separate clients with unique IDs
        client_a_id = "client_a_" + str(uuid.uuid4())
        client_b_id = "client_b_" + str(uuid.uuid4())

        try:
            # Connect both clients with retries
            client_a = self.start_client_with_retry(listen, port, client_a_id)
            client_b = self.start_client_with_retry(listen, port, client_b_id)

            # Create simple workflows for both clients
            graph_a = GraphBuilder(prefix="client_a")
            image_a = graph_a.node("StubImage", content="BLACK", height=256, width=256, batch_size=1)
            graph_a.node("PreviewImage", images=image_a.out(0))

            graph_b = GraphBuilder(prefix="client_b")
            image_b = graph_b.node("StubImage", content="WHITE", height=256, width=256, batch_size=1)
            graph_b.node("PreviewImage", images=image_b.out(0))

            # Submit workflows from both clients
            prompt_a = graph_a.finalize()
            prompt_b = graph_b.finalize()

            response_a = client_a.queue_prompt(prompt_a)
            prompt_id_a = response_a['prompt_id']

            response_b = client_b.queue_prompt(prompt_b)
            prompt_id_b = response_b['prompt_id']

            # Start threads to listen for messages on both clients
            def listen_client_a():
                client_a.listen_for_messages(duration=10.0)

            def listen_client_b():
                client_b.listen_for_messages(duration=10.0)

            thread_a = threading.Thread(target=listen_client_a)
            thread_b = threading.Thread(target=listen_client_b)

            thread_a.start()
            thread_b.start()

            # Wait for threads to complete
            thread_a.join()
            thread_b.join()

            # Verify isolation
            # Client A should only receive progress for prompt_id_a
            assert not client_a.progress_tracker.has_cross_contamination(prompt_id_a), \
                f"Client A received progress updates for other clients' workflows. " \
                f"Expected only {prompt_id_a}, but got messages for multiple prompts."

            # Client B should only receive progress for prompt_id_b
            assert not client_b.progress_tracker.has_cross_contamination(prompt_id_b), \
                f"Client B received progress updates for other clients' workflows. " \
                f"Expected only {prompt_id_b}, but got messages for multiple prompts."

            # Verify each client received their own progress updates
            client_a_messages = client_a.progress_tracker.get_messages_for_prompt(prompt_id_a)
            client_b_messages = client_b.progress_tracker.get_messages_for_prompt(prompt_id_b)

            assert len(client_a_messages) > 0, \
                "Client A did not receive any progress updates for its own workflow"
            assert len(client_b_messages) > 0, \
                "Client B did not receive any progress updates for its own workflow"

            # Ensure no cross-contamination
            client_a_other = client_a.progress_tracker.get_messages_for_prompt(prompt_id_b)
            client_b_other = client_b.progress_tracker.get_messages_for_prompt(prompt_id_a)

            assert len(client_a_other) == 0, \
                f"Client A incorrectly received {len(client_a_other)} progress updates for Client B's workflow"
            assert len(client_b_other) == 0, \
                f"Client B incorrectly received {len(client_b_other)} progress updates for Client A's workflow"

        finally:
            # Clean up connections
            if hasattr(client_a, 'ws'):
                client_a.ws.close()
            if hasattr(client_b, 'ws'):
                client_b.ws.close()

    def test_progress_with_missing_client_id(self, args_pytest):
        """Test that progress updates handle missing client_id gracefully."""
        listen = args_pytest["listen"]
        port = args_pytest["port"]

        try:
            # Connect client with retries
            client = self.start_client_with_retry(listen, port)

            # Create a simple workflow
            graph = GraphBuilder(prefix="test_missing_id")
            image = graph.node("StubImage", content="BLACK", height=128, width=128, batch_size=1)
            graph.node("PreviewImage", images=image.out(0))

            # Submit workflow
            prompt = graph.finalize()
            response = client.queue_prompt(prompt)
            prompt_id = response['prompt_id']

            # Listen for messages
            client.listen_for_messages(duration=5.0)

            # Should still receive progress updates for own workflow
            messages = client.progress_tracker.get_messages_for_prompt(prompt_id)
            assert len(messages) > 0, \
                "Client did not receive progress updates even though it initiated the workflow"

        finally:
            if hasattr(client, 'ws'):
                client.ws.close()

