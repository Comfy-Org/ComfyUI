from typing_extensions import override
import torch

from comfy_api.latest import ComfyExtension, io


class RenormCFG(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RenormCFG",
            category="model/patch",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("cfg_trunc", default=100, min=0.0, max=100.0, step=0.01, advanced=True),
                io.Float.Input("renorm_cfg", default=1.0, min=0.0, max=100.0, step=0.01, advanced=True),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, cfg_trunc, renorm_cfg) -> io.NodeOutput:
        def renorm_cfg_func(args):
            cond_denoised = args["cond_denoised"]
            uncond_denoised = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            timestep = args["timestep"]
            x_orig = args["input"]
            in_channels = model.model.diffusion_model.in_channels

            if timestep[0] < cfg_trunc:
                cond_eps, uncond_eps = cond_denoised[:, :in_channels], uncond_denoised[:, :in_channels]
                cond_rest, _ = cond_denoised[:, in_channels:], uncond_denoised[:, in_channels:]
                half_eps = uncond_eps + cond_scale * (cond_eps - uncond_eps)
                half_rest = cond_rest

                if float(renorm_cfg) > 0.0:
                    ori_pos_norm = torch.linalg.vector_norm(cond_eps
                            , dim=tuple(range(1, len(cond_eps.shape))), keepdim=True
                    )
                    max_new_norm = ori_pos_norm * float(renorm_cfg)
                    new_pos_norm = torch.linalg.vector_norm(
                            half_eps, dim=tuple(range(1, len(half_eps.shape))), keepdim=True
                        )
                    if new_pos_norm >= max_new_norm:
                        half_eps = half_eps * (max_new_norm / new_pos_norm)
            else:
                cond_eps, uncond_eps = cond_denoised[:, :in_channels], uncond_denoised[:, :in_channels]
                cond_rest, _ = cond_denoised[:, in_channels:], uncond_denoised[:, in_channels:]
                half_eps = cond_eps
                half_rest = cond_rest

            cfg_result = torch.cat([half_eps, half_rest], dim=1)

            # cfg_result = uncond_denoised + (cond_denoised - uncond_denoised) * cond_scale

            return x_orig - cfg_result

        m = model.clone()
        m.set_model_sampler_cfg_function(renorm_cfg_func)
        return io.NodeOutput(m)


class CLIPTextEncodeLumina2(io.ComfyNode):
    SYSTEM_PROMPT = {
        "superior": "You are an assistant designed to generate superior images with the superior "\
            "degree of image-text alignment based on textual prompts or user prompts.",
        "alignment": "You are an assistant designed to generate high-quality images with the "\
            "highest degree of image-text alignment based on textual prompts."
    }
    SYSTEM_PROMPT_TIP = "Lumina2 provide two types of system prompts:" \
        "Superior: You are an assistant designed to generate superior images with the superior "\
        "degree of image-text alignment based on textual prompts or user prompts. "\
        "Alignment: You are an assistant designed to generate high-quality images with the highest "\
        "degree of image-text alignment based on textual prompts."
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPTextEncodeLumina2",
            search_aliases=["lumina prompt"],
            display_name="CLIP Text Encode (Lumina 2)",
            category="model/conditioning/lumina",
            description="Encodes a system prompt and a user prompt using a CLIP model into an embedding "
                        "that can be used to guide the diffusion model towards generating specific images.",
            inputs=[
                io.Combo.Input(
                    "system_prompt",
                    options=list(cls.SYSTEM_PROMPT.keys()),
                    tooltip=cls.SYSTEM_PROMPT_TIP,
                ),
                io.String.Input(
                    "user_prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="The text to be encoded.",
                ),
                io.Clip.Input("clip", tooltip="The CLIP model used for encoding the text."),
            ],
            outputs=[
                io.Conditioning.Output(
                    tooltip="A conditioning containing the embedded text used to guide the diffusion model.",
                ),
            ],
        )

    @classmethod
    def execute(cls, clip, user_prompt, system_prompt) -> io.NodeOutput:
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")
        system_prompt = cls.SYSTEM_PROMPT[system_prompt]
        prompt = f'{system_prompt} <Prompt Start> {user_prompt}'
        tokens = clip.tokenize(prompt)
        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens))


class Lumina2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CLIPTextEncodeLumina2,
            RenormCFG,
        ]


async def comfy_entrypoint() -> Lumina2Extension:
    return Lumina2Extension()
