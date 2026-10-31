import json
import os
import shutil

import torch

import comfy.memory_management
import comfy.model_management
import comfy.model_patcher
import comfy.model_prefetch
import comfy.storage
import comfy.utils

METADATA_KEY = "comfy_model_state"
HEADER_RESERVE = 1 << 20


class ModelState:
    """A writable model's state over a safetensors file.

    The file is created when the first model loads into it, and the loaded
    model is reused by every later load while this object survives. Without a
    path the file is anonymous.
    """

    def __init__(self, path=None, create="if_missing"):
        self.path = path
        self.file = None
        self.metadata = {}
        self.patcher = None
        if path is None or create == "always":
            return
        if os.path.exists(path):
            self.file = open(path, "r+b", buffering=0)
            metadata = comfy.utils.open_safetensors(self.file)[1].get("__metadata__", {})
            if METADATA_KEY not in metadata:
                raise ValueError(f"Not a model state file: {path}")
            self.metadata = json.loads(metadata[METADATA_KEY])
        elif create == "never":
            raise FileNotFoundError(f"Model state file does not exist: {path}")

    def load(self, model, device):
        """Load model from this state, creating the file on first use, and return the model to run."""
        if self.patcher is None:
            geometry = model.state_dict()
            if self.file is not None:
                tensors = comfy.utils.open_safetensors(self.file)[0]
            else:
                if self.path is not None:
                    self.file = open(self.path, "w+b", buffering=0)
                tensors, self.file = comfy.utils.create_safetensors(geometry, {METADATA_KEY: "{}"}, file=self.file, header_size=HEADER_RESERVE)
            if comfy.memory_management.aimdo_enabled and not comfy.model_management.is_device_cpu(device):
                model.load_state_dict(tensors, assign=True)
                fast_disk = comfy.storage.anonymous_fast_disk() if self.path is None else comfy.storage.model_fast_disk([self.path])
                self.patcher = comfy.model_patcher.ModelPatcherDynamic(model, device, torch.device("cpu"), fast_disk=fast_disk)
            else:
                model.to_empty(device="cpu")
                for name, value in model.state_dict().items():
                    if not comfy.memory_management.read_tensor_file_slice_into(tensors[name], value):
                        raise RuntimeError(f"Could not read model state tensor {name}")
                self.patcher = comfy.model_patcher.ModelPatcher(model, device, torch.device("cpu"))
        comfy.model_management.load_models_gpu([self.patcher], force_full_load=not self.patcher.is_dynamic())
        if self.patcher.is_dynamic():
            self.patcher.set_in_use_by_current_prompt(True)
        return self.patcher.model

    def flush(self):
        """Write the loaded values and metadata back into the file."""
        if self.patcher is None:
            return
        # The last forward may still be using the allocation graph's pages.
        comfy.model_management.synchronize()
        comfy.model_prefetch.cleanup_prefetch_queues()
        tensors, header = comfy.utils.open_safetensors(self.file)
        if self.patcher.is_dynamic():
            self.patcher.flush_writes()
        else:
            for name, value in self.patcher.model.state_dict().items():
                comfy.memory_management.write_tensor_file_slice(tensors[name], value)
        header["__metadata__"] = {METADATA_KEY: json.dumps(self.metadata)}
        comfy.utils.rewrite_safetensors_header(self.file, header)

    def save(self):
        self.flush()
        os.fsync(self.file.fileno())

    def copy(self, path):
        """Save, copy the file to path and return the copy, opened."""
        self.save()
        with open(path, "wb") as destination:
            size = os.fstat(self.file.fileno()).st_size
            try:
                copied = 0
                while copied < size:
                    copied += os.copy_file_range(self.file.fileno(), destination.fileno(), size - copied, copied, copied)
            except (AttributeError, OSError):
                self.file.seek(0)
                destination.seek(0)
                destination.truncate()
                shutil.copyfileobj(self.file, destination)
        return ModelState(path, create="never")

    def release(self):
        """Drop the loaded model without writing anything back."""
        if self.patcher is not None and self.patcher.is_dynamic():
            self.patcher.discard_writes()
        self.patcher = None
