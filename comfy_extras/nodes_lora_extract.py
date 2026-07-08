import torch
import comfy.model_management
import comfy.utils
import folder_paths
import os
import logging
from enum import Enum
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from tqdm.auto import trange

CLAMP_QUANTILE = 0.99

def extract_lora(diff, rank):
    conv2d = (len(diff.shape) == 4)
    kernel_size = None if not conv2d else diff.size()[2:4]
    conv2d_3x3 = conv2d and kernel_size != (1, 1)
    out_dim, in_dim = diff.size()[0:2]
    rank = min(rank, in_dim, out_dim)

    if conv2d:
        if conv2d_3x3:
            diff = diff.flatten(start_dim=1)
        else:
            diff = diff.squeeze()


    U, S, Vh = torch.linalg.svd(diff.float())
    U = U[:, :rank]
    S = S[:rank]
    U = U @ torch.diag(S)
    Vh = Vh[:rank, :]

    dist = torch.cat([U.flatten(), Vh.flatten()])
    hi_val = torch.quantile(dist, CLAMP_QUANTILE)
    low_val = -hi_val

    U = U.clamp(low_val, hi_val)
    Vh = Vh.clamp(low_val, hi_val)
    if conv2d:
        U = U.reshape(out_dim, rank, 1, 1)
        Vh = Vh.reshape(rank, in_dim, kernel_size[0], kernel_size[1])
    return (U, Vh)

class LORAType(Enum):
    STANDARD = 0
    FULL_DIFF = 1

LORA_TYPES = {"standard": LORAType.STANDARD,
              "full_diff": LORAType.FULL_DIFF}

def calc_lora_model(model_diff, rank, prefix_model, prefix_lora, output_sd, lora_type, bias_diff=False):
    comfy.model_management.load_models_gpu([model_diff])
    sd = model_diff.model_state_dict(filter_prefix=prefix_model)

    sd_keys = list(sd.keys())
    for index in trange(len(sd_keys), unit="weight"):
        k = sd_keys[index]
        op_keys = sd_keys[index].rsplit('.', 1)
        if len(op_keys) < 2 or op_keys[1] not in ["weight", "bias"] or (op_keys[1] == "bias" and not bias_diff):
            continue
        op = comfy.utils.get_attr(model_diff.model, op_keys[0])
        if hasattr(op, "comfy_cast_weights") and not getattr(op, "comfy_patched_weights", False):
            weight_diff = model_diff.patch_weight_to_device(k, model_diff.load_device, return_weight=True)
        else:
            weight_diff = sd[k]

        if op_keys[1] == "weight":
            if lora_type == LORAType.STANDARD:
                if weight_diff.ndim < 2:
                    if bias_diff:
                        output_sd["{}{}.diff".format(prefix_lora, k[len(prefix_model):-7])] = weight_diff.contiguous().half().cpu()
                    continue
                try:
                    out = extract_lora(weight_diff, rank)
                    output_sd["{}{}.lora_up.weight".format(prefix_lora, k[len(prefix_model):-7])] = out[0].contiguous().half().cpu()
                    output_sd["{}{}.lora_down.weight".format(prefix_lora, k[len(prefix_model):-7])] = out[1].contiguous().half().cpu()
                except:
                    logging.warning("Could not generate lora weights for key {}, is the weight difference a zero?".format(k))
            elif lora_type == LORAType.FULL_DIFF:
                output_sd["{}{}.diff".format(prefix_lora, k[len(prefix_model):-7])] = weight_diff.contiguous().half().cpu()

        elif bias_diff and op_keys[1] == "bias":
            output_sd["{}{}.diff_b".format(prefix_lora, k[len(prefix_model):-5])] = weight_diff.contiguous().half().cpu()
    return output_sd

class LoraSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoraSave",
            search_aliases=["export lora"],
            display_name="Extract and Save Lora",
            category="experimental",
            inputs=[
                io.String.Input("filename_prefix", default="loras/ComfyUI_extracted_lora"),
                io.Int.Input("rank", default=8, min=1, max=4096, step=1, advanced=True),
                io.Combo.Input("lora_type", options=tuple(LORA_TYPES.keys()), advanced=True),
                io.Boolean.Input("bias_diff", default=True, advanced=True),
                io.Model.Input(
                    "model_diff",
                    tooltip="The ModelSubtract output to be converted to a lora.",
                    optional=True,
                ),
                io.Clip.Input(
                  "text_encoder_diff",
                    tooltip="The CLIPSubtract output to be converted to a lora.",
                    optional=True,
                ),
            ],
            is_experimental=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, filename_prefix, rank, lora_type, bias_diff, model_diff=None, text_encoder_diff=None) -> io.NodeOutput:
        if model_diff is None and text_encoder_diff is None:
            return io.NodeOutput()

        lora_type = LORA_TYPES.get(lora_type)
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())

        output_sd = {}
        if model_diff is not None:
            output_sd = calc_lora_model(model_diff, rank, "diffusion_model.", "diffusion_model.", output_sd, lora_type, bias_diff=bias_diff)
        if text_encoder_diff is not None:
            output_sd = calc_lora_model(text_encoder_diff.patcher, rank, "", "text_encoders.", output_sd, lora_type, bias_diff=bias_diff)

        output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

        comfy.utils.save_torch_file(output_sd, output_checkpoint, metadata=None)
        return io.NodeOutput()


class LoraSaveExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LoraSave,
        ]


async def comfy_entrypoint() -> LoraSaveExtension:
    return LoraSaveExtension()
