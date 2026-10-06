import json
import os

import safetensors.torch
import torch
from typing_extensions import override

import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io


class CLIPTextEncodeControlnet(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CLIPTextEncodeControlnet",
            display_name="CLIP Text Encode (Controlnet)",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.Conditioning.Input("conditioning"),
                io.String.Input("text", multiline=True, dynamic_prompts=True),
            ],
            outputs=[io.Conditioning.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clip, conditioning, text) -> io.NodeOutput:
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]['cross_attn_controlnet'] = cond
            n[1]['pooled_output_controlnet'] = pooled
            c.append(n)
        return io.NodeOutput(c)

class T5TokenizerOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T5TokenizerOptions",
            display_name="T5 Tokenizer Options",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.Int.Input("min_padding", default=0, min=0, max=10000, step=1),
                io.Int.Input("min_length", default=0, min=0, max=10000, step=1),
            ],
            outputs=[io.Clip.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clip, min_padding, min_length) -> io.NodeOutput:
        clip = clip.clone()
        for t5_type in ["t5xxl", "pile_t5xl", "t5base", "mt5xl", "umt5xxl"]:
            clip.set_tokenizer_option("{}_min_padding".format(t5_type), min_padding)
            clip.set_tokenizer_option("{}_min_length".format(t5_type), min_length)

        return io.NodeOutput(clip)


class ConditioningLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ConditioningLoader",
            display_name="Load Conditioning",
            category="model/loaders",
            description="Loads a conditioning saved with Save Conditioning, or any safetensors file with a 'conditioning' tensor, from the embeddings folder.",
            inputs=[
                io.Combo.Input("conditioning_name", options=folder_paths.get_filename_list("embeddings")),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, conditioning_name) -> io.NodeOutput:
        sd, metadata = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("embeddings", conditioning_name), safe_load=True, return_metadata=True)
        cond = sd.pop("conditioning")
        options = json.loads((metadata or {}).get("conditioning_options", "{}"))
        lists = {}
        for k, v in sd.items():
            name, _, index = k.rpartition(".")
            if index.isdigit():
                lists.setdefault(name, {})[int(index)] = v
            else:
                options[k] = v
        options.update({k: [v[i] for i in sorted(v)] for k, v in lists.items()})
        return io.NodeOutput([[cond, options]])


class SaveConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SaveConditioning",
            search_aliases=["save conditioning", "export conditioning", "cache text encoder output"],
            display_name="Save Conditioning",
            category="model/conditioning",
            description="Saves a conditioning to the output folder as safetensors; move the file to models/embeddings to load it with Load Conditioning, e.g. to skip the text encoder.",
            is_output_node=True,
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.String.Input("filename_prefix", default="conditioning/ComfyUI"),
            ],
            outputs=[io.Conditioning.Output(display_name="conditioning")],
        )

    @classmethod
    def execute(cls, conditioning, filename_prefix) -> io.NodeOutput:
        if len(conditioning) != 1:
            raise ValueError("Save Conditioning supports a single conditioning entry, save it before combining.")
        cond, options = conditioning[0]
        sd = {"conditioning": cond}
        values = {}
        for k, v in options.items():
            if isinstance(v, torch.Tensor):
                sd[k] = v
            elif isinstance(v, (list, tuple)) and all(isinstance(t, torch.Tensor) for t in v):
                sd.update({f"{k}.{i}": t for i, t in enumerate(v)})
            elif isinstance(v, (bool, int, float, str)):
                values[k] = v
            elif v is not None:
                raise ValueError(f"Conditioning option '{k}' ({type(v).__name__}) can't be saved.")
        full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        safetensors.torch.save_file({k: v.detach().to("cpu", copy=True).contiguous() for k, v in sd.items()},
                                    os.path.join(full_output_folder, f"{filename}_{counter:05}_.safetensors"),
                                    metadata={"conditioning_options": json.dumps(values)})
        return io.NodeOutput(conditioning)


class CondExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CLIPTextEncodeControlnet,
            T5TokenizerOptions,
            ConditioningLoader,
            SaveConditioning,
        ]


async def comfy_entrypoint() -> CondExtension:
    return CondExtension()
