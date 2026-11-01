import copy
import json
import os

import safetensors
import safetensors.torch
import torch

import comfy.memory_management
import comfy.model_management
import comfy.model_patcher
import comfy.model_prefetch
import comfy.storage
import comfy.utils

METADATA_KEY = "comfy_model_state"


class ModelState:
    """A writable model's values as CPU tensors with its generation metadata, passed between nodes by value.

    Loading copies the values into the model's own backing, so a state's tensors
    are never written in place and copies share them. Flushing reads the loaded
    model's values back and drops it; a node that loads a state flushes or
    releases it before the state reaches another node.
    """

    def __init__(self, tensors=None, metadata=None):
        self.tensors = {} if tensors is None else tensors
        self.metadata = {} if metadata is None else metadata
        self.patcher = None

    @classmethod
    def open(cls, path, create="if_missing"):
        """Read the state file at path; a missing file is an empty state unless create is never, and always starts empty."""
        if create != "always" and os.path.exists(path):
            with safetensors.safe_open(path, "pt") as file:
                metadata = file.metadata() or {}
                if METADATA_KEY not in metadata:
                    raise ValueError(f"Not a model state file: {path}")
                return cls({name: file.get_tensor(name).clone() for name in file.keys()}, json.loads(metadata[METADATA_KEY]))
        if create == "never":
            raise FileNotFoundError(f"Model state file does not exist: {path}")
        return cls()

    def save(self, path):
        if self.tensors is None:
            raise RuntimeError("The model state was released without being written back; rerun the node that produced it")
        safetensors.torch.save_file(self.tensors, path, {METADATA_KEY: json.dumps(self.metadata)})

    def copy(self):
        if self.tensors is None:
            raise RuntimeError("The model state was released without being written back; rerun the node that produced it")
        return ModelState(dict(self.tensors), copy.deepcopy(self.metadata))

    def load(self, model, device):
        """Load model with this state's values, or zeros, and return it ready to run."""
        if comfy.memory_management.aimdo_enabled and not comfy.model_management.is_device_cpu(device):
            # Dynamic VRAM pages the model from an anonymous file and writes it back there.
            tensors, _ = comfy.utils.create_safetensors(model.state_dict())
            for name, value in self.tensors.items():
                comfy.memory_management.write_tensor_file_slice(tensors[name], value)
            model.load_state_dict(tensors, assign=True)
            self.patcher = comfy.model_patcher.ModelPatcherDynamic(model, device, torch.device("cpu"), fast_disk=comfy.storage.anonymous_fast_disk())
        else:
            model.to_empty(device="cpu")
            for name, value in model.state_dict().items():
                if self.tensors:
                    value.copy_(self.tensors[name])
                else:
                    value.zero_()
            self.patcher = comfy.model_patcher.ModelPatcher(model, device, torch.device("cpu"))
        comfy.model_management.load_models_gpu([self.patcher], force_full_load=not self.patcher.is_dynamic())
        if self.patcher.is_dynamic():
            self.patcher.set_in_use_by_current_prompt(True)
        return model

    def flush(self):
        """Take the loaded model's values and drop it."""
        if self.patcher is None:
            return
        # The last forward may still be using the allocation graph's pages.
        comfy.model_management.synchronize()
        comfy.model_prefetch.cleanup_prefetch_queues()
        state_dict = self.patcher.model.state_dict()
        if self.patcher.is_dynamic():
            self.patcher.flush_writes()
            self.tensors = {name: torch.empty(value.shape, dtype=value.dtype) for name, value in state_dict.items()}
            for name, value in state_dict.items():
                if not comfy.memory_management.read_tensor_file_slice_into(value, self.tensors[name]):
                    raise RuntimeError(f"Could not read model state tensor {name}")
        else:
            self.tensors = {name: value.cpu() for name, value in state_dict.items()}
        self.patcher = None

    def release(self):
        """Drop a still loaded model, discarding its values; the metadata may already describe them, so the state is unusable."""
        if self.patcher is None:
            return
        if self.patcher.is_dynamic():
            self.patcher.discard_writes()
        self.patcher = None
        self.tensors = None
