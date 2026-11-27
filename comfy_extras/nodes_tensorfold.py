import json
from pathlib import Path

import comfy.model_management
import comfy.utils
import folder_paths
from comfy.tensorfold.device import subprocess_environment
from comfy.tensorfold.moss import sample_moss
from comfy.tensorfold.runtime import load_runtime, manifest_fingerprint
from comfy_api.latest import ComfyExtension, io


def runtime_manifest():
    return Path(folder_paths.models_dir) / "tensorfold" / "runtime.json"


def checkpoints():
    return [name for name in folder_paths.get_filename_list("diffusion_models") if name.lower().endswith(".safetensors")]


def fingerprint(checkpoint_name):
    path = folder_paths.get_full_path("diffusion_models", checkpoint_name)
    metadata = Path(path).stat() if path is not None else None
    return (manifest_fingerprint(runtime_manifest()),
            (metadata.st_size, metadata.st_mtime_ns) if metadata is not None else "missing")


class TensorFoldMOSSImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TensorFoldMOSSImage",
            display_name="TensorFold MOSS Image",
            category="sampling/tensorfold",
            description="Samples a BF16 MOSS-Image checkpoint with TensorFold CUDA. Uses the MOSS V2 shift-5 schedule, guidance 3 and negative skip layer 22.",
            inputs=[
                io.Combo.Input("checkpoint_name", options=checkpoints(), tooltip="Prepared BF16 MOSS transformer. Use prepare_moss.py for the official fused-QKV file."),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Int.Input("width", default=1024, min=16, max=4096, step=16),
                io.Int.Input("height", default=1024, min=16, max=4096, step=16),
                io.Int.Input("steps", default=50, min=1, max=1000),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
            ],
            outputs=[io.Latent.Output(), io.String.Output(display_name="receipt")],
            is_experimental=True,
        )

    @classmethod
    def fingerprint_inputs(cls, checkpoint_name, **kwargs):
        return fingerprint(checkpoint_name)

    @classmethod
    def execute(cls, checkpoint_name, positive, negative, width, height, steps, seed):
        runtime = load_runtime(runtime_manifest())
        device = comfy.model_management.get_torch_device()
        environment = subprocess_environment(device)
        checkpoint = Path(folder_paths.get_full_path_or_raise("diffusion_models", checkpoint_name))
        comfy.model_management.throw_exception_if_processing_interrupted()
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        progress = comfy.utils.ProgressBar(steps)
        latent, receipt = sample_moss(
            checkpoint, runtime.artifact, positive, negative, width, height, steps, seed,
            Path(folder_paths.get_temp_directory()),
            comfy.model_management.throw_exception_if_processing_interrupted,
            progress.update_absolute, environment=environment,
        )
        receipt.update(engine_version=runtime.engine_version, source_revision=runtime.source_revision,
                       source_patch_sha256=runtime.source_patch_sha256, artifact_sha256=runtime.digest)
        return io.NodeOutput(latent, json.dumps(receipt, separators=(",", ":"), allow_nan=False))


class TensorFoldExtension(ComfyExtension):
    async def get_node_list(self):
        return [TensorFoldMOSSImage]


async def comfy_entrypoint():
    return TensorFoldExtension()
