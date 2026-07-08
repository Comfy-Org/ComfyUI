import folder_paths
import comfy.sd
import comfy.model_management
import nodes
import torch
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_slg import SkipLayerGuidanceDiT


class TripleCLIPLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TripleCLIPLoader",
            display_name="Load CLIP (Triple)",
            category="model/loaders",
            description="Recipes:\nsd3: clip-l, clip-g, t5",
            inputs=[
                io.Combo.Input("clip_name1", options=folder_paths.get_filename_list("text_encoders")),
                io.Combo.Input("clip_name2", options=folder_paths.get_filename_list("text_encoders")),
                io.Combo.Input("clip_name3", options=folder_paths.get_filename_list("text_encoders")),
            ],
            outputs=[
                io.Clip.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip_name1, clip_name2, clip_name3) -> io.NodeOutput:
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
        clip_path3 = folder_paths.get_full_path_or_raise("text_encoders", clip_name3)
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2, clip_path3], embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return io.NodeOutput(clip)

    load_clip = execute  # TODO: remove


class EmptySD3LatentImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptySD3LatentImage",
            category="model/latent/stable diffusion",
            inputs=[
                io.Int.Input("width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, width, height, batch_size=1) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 16, height // 8, width // 8], device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
        return io.NodeOutput({"samples": latent, "downscale_ratio_spacial": 8})

    generate = execute  # TODO: remove


class CLIPTextEncodeSD3(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPTextEncodeSD3",
            search_aliases=["sd3 prompt"],
            display_name="CLIP Text Encode (SD3)",
            category="model/conditioning/stable diffusion",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("clip_l", multiline=True, dynamic_prompts=True),
                io.String.Input("clip_g", multiline=True, dynamic_prompts=True),
                io.String.Input("t5xxl", multiline=True, dynamic_prompts=True),
                io.Combo.Input("empty_padding", options=["none", "empty_prompt"], advanced=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, clip_l, clip_g, t5xxl, empty_padding) -> io.NodeOutput:
        no_padding = empty_padding == "none"

        tokens = clip.tokenize(clip_g)
        if len(clip_g) == 0 and no_padding:
            tokens["g"] = []

        if len(clip_l) == 0 and no_padding:
            tokens["l"] = []
        else:
            tokens["l"] = clip.tokenize(clip_l)["l"]

        if len(t5xxl) == 0 and no_padding:
            tokens["t5xxl"] =  []
        else:
            tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens))

    encode = execute  # TODO: remove


class ControlNetApplySD3(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ControlNetApplySD3",
            display_name="Apply Controlnet with VAE",
            category="model/conditioning/controlnet",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.ControlNet.Input("control_net"),
                io.Vae.Input("vae"),
                io.Image.Input("image"),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.001),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ],
            is_deprecated=True,
        )

    @classmethod
    def execute(cls, positive, negative, control_net, image, strength, start_percent, end_percent, vae=None) -> io.NodeOutput:
        if strength == 0:
            return io.NodeOutput(positive, negative)

        control_hint = image.movedim(-1, 1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent),
                                                             vae=vae, extra_concat=[])
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return io.NodeOutput(out[0], out[1])

    apply_controlnet = execute  # TODO: remove


class SkipLayerGuidanceSD3(io.ComfyNode):
    '''
    Enhance guidance towards detailed dtructure by having another set of CFG negative with skipped layers.
    Inspired by Perturbed Attention Guidance (https://arxiv.org/abs/2403.17377)
    Experimental implementation by Dango233@StabilityAI.
    '''

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SkipLayerGuidanceSD3",
            category="advanced/guidance",
            description="Generic version of SkipLayerGuidance node that can be used on every DiT model.",
            inputs=[
                io.Model.Input("model"),
                io.String.Input("layers", default="7, 8, 9", multiline=False, advanced=True),
                io.Float.Input("scale", default=3.0, min=0.0, max=10.0, step=0.1),
                io.Float.Input("start_percent", default=0.01, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Float.Input("end_percent", default=0.15, min=0.0, max=1.0, step=0.001, advanced=True),
            ],
            outputs=[
                io.Model.Output(),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, layers, scale, start_percent, end_percent) -> io.NodeOutput:
        return SkipLayerGuidanceDiT().execute(model=model, scale=scale, start_percent=start_percent, end_percent=end_percent, double_layers=layers)

    skip_guidance_sd3 = execute  # TODO: remove


class SD3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TripleCLIPLoader,
            EmptySD3LatentImage,
            CLIPTextEncodeSD3,
            ControlNetApplySD3,
            SkipLayerGuidanceSD3,
        ]


async def comfy_entrypoint() -> SD3Extension:
    return SD3Extension()
