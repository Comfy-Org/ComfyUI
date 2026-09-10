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
            description="Loads a precomputed conditioning from the embeddings folder: the 'conditioning' tensor of a safetensors file, other tensors in it become conditioning options.",
            inputs=[
                io.Combo.Input("conditioning_name", options=folder_paths.get_filename_list("embeddings")),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, conditioning_name) -> io.NodeOutput:
        sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("embeddings", conditioning_name), safe_load=True)
        cond = sd.pop("conditioning")
        return io.NodeOutput([[cond, sd]])


class CondExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CLIPTextEncodeControlnet,
            T5TokenizerOptions,
            ConditioningLoader,
        ]


async def comfy_entrypoint() -> CondExtension:
    return CondExtension()
