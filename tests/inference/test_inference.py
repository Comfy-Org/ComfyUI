from copy import deepcopy
from io import BytesIO
import numpy
import os
from PIL import Image
import pytest
from pytest import fixture
import time
import torch
from typing import Union
import json
import subprocess
import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import urllib.request
import urllib.parse


from comfy.samplers import KSampler

"""
These tests generate and save images through a range of parameters
"""

class ComfyGraph:
    def __init__(self,
                 graph: dict,
                 sampler_nodes: list[str],
                 ):
        self.graph = graph
        self.sampler_nodes = sampler_nodes

    def set_prompt(self, prompt, negative_prompt=None):
        # Sets the prompt for the sampler nodes (eg. base and refiner)
        for node in self.sampler_nodes:
            prompt_node = self.graph[node]['inputs']['positive'][0]
            self.graph[prompt_node]['inputs']['text'] = prompt
            if negative_prompt:
                negative_prompt_node = self.graph[node]['inputs']['negative'][0]
                self.graph[negative_prompt_node]['inputs']['text'] = negative_prompt

    def set_sampler_name(self, sampler_name:str, ):
        # sets the sampler name for the sampler nodes (eg. base and refiner)
        for node in self.sampler_nodes:
            self.graph[node]['inputs']['sampler_name'] = sampler_name

    def set_scheduler(self, scheduler:str):
        # sets the sampler name for the sampler nodes (eg. base and refiner)
        for node in self.sampler_nodes:
            self.graph[node]['inputs']['scheduler'] = scheduler

    def set_filename_prefix(self, prefix:str):
        # sets the filename prefix for the save nodes
        for node in self.graph:
            if self.graph[node]['class_type'] == 'SaveImage':
                self.graph[node]['inputs']['filename_prefix'] = prefix


class ComfyClient:
    # From examples/websockets_api_example.py

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

    def queue_prompt(self, prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
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

    def get_images(self, graph, save=True):
        prompt = graph
        if not save:
            # Replace save nodes with preview nodes
            prompt_str = json.dumps(prompt)
            prompt_str = prompt_str.replace('SaveImage', 'PreviewImage')
            prompt = json.loads(prompt_str)

        prompt_id = self.queue_prompt(prompt)['prompt_id']
        output_images = {}
        while True:
            out = self.ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break #Execution is done
            else:
                continue #previews are binary data

        history = self.get_history(prompt_id)[prompt_id]
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            images_output = []
            if 'images' in node_output:
                for image in node_output['images']:
                    image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

        return output_images

#
# Initialize graphs
#
default_graph_file = 'tests/inference/graphs/default_graph_sdxl1_0.json'
with open(default_graph_file, 'r') as file:
    default_graph = json.loads(file.read())
DEFAULT_COMFY_GRAPH = ComfyGraph(graph=default_graph, sampler_nodes=['10','14'])
DEFAULT_COMFY_GRAPH_ID = os.path.splitext(os.path.basename(default_graph_file))[0]

#
# Loop through these variables
#
comfy_graph_list = [DEFAULT_COMFY_GRAPH]
comfy_graph_ids = [DEFAULT_COMFY_GRAPH_ID]
prompt_list = [
    'a painting of a cat',
]

sampler_list = KSampler.SAMPLERS
scheduler_list = KSampler.SCHEDULERS

@pytest.mark.inference
@pytest.mark.parametrize("sampler", sampler_list)
@pytest.mark.parametrize("scheduler", scheduler_list)
@pytest.mark.parametrize("prompt", prompt_list)
class TestInference:
    #
    # Initialize server and client
    #
    @fixture(scope="class", autouse=True)
    def _server(self, args_pytest):
        # Start server
        p = subprocess.Popen([
                'python','main.py',
                '--output-directory', args_pytest["output_dir"],
                '--listen', args_pytest["listen"],
                '--port', str(args_pytest["port"]),
                '--disable-assets',
                ])
        yield
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

    #
    # Client and graph fixtures with server warmup
    #
    # Returns a "_client_graph", which is client-graph pair corresponding to an initialized server
    # The "graph" is the default graph
    @fixture(scope="class", params=comfy_graph_list, ids=comfy_graph_ids, autouse=True)
    def _client_graph(self, request, args_pytest, _server) -> (ComfyClient, ComfyGraph):
        comfy_graph = request.param

        # Start client
        comfy_client = self.start_client(args_pytest["listen"], args_pytest["port"])

        # Warm up pipeline
        comfy_client.get_images(graph=comfy_graph.graph, save=False)

        yield comfy_client, comfy_graph
        del comfy_client
        del comfy_graph
        torch.cuda.empty_cache()

    @fixture
    def client(self, _client_graph):
        client = _client_graph[0]
        yield client

    @fixture
    def comfy_graph(self, _client_graph):
        # avoid mutating the graph
        graph = deepcopy(_client_graph[1])
        yield graph

    def test_comfy(
        self,
        client,
        comfy_graph,
        sampler,
        scheduler,
        prompt,
        request
    ):
        test_info = request.node.name
        comfy_graph.set_filename_prefix(test_info)
        # Settings for comfy graph
        comfy_graph.set_sampler_name(sampler)
        comfy_graph.set_scheduler(scheduler)
        comfy_graph.set_prompt(prompt)

        # Generate
        images = client.get_images(comfy_graph.graph)

        assert len(images) != 0, "No images generated"
        # assert all images are not blank
        for images_output in images.values():
            for image_data in images_output:
                pil_image = Image.open(BytesIO(image_data))
                assert numpy.array(pil_image).any() != 0, "Image is blank"


