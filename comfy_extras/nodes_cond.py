import torch
from typing_extensions import override

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


def _vary(tensor: torch.Tensor, noise: torch.Tensor, strength: float) -> torch.Tensor:
    eps = 1e-8
    norm = tensor.norm(dim=-1, keepdim=True)
    varied = tensor + strength * noise * (norm / (noise.norm(dim=-1, keepdim=True) + eps))
    return varied / (varied.norm(dim=-1, keepdim=True) + eps) * norm


class ConditioningVariation(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ConditioningVariation",
            display_name="Conditioning Variation",
            category="model/conditioning/transform",
            description="Nudges a conditioning with seeded noise to explore variations of a prompt "
            "without changing the sampler seed. Noise is scaled per-token and the result is "
            "renormalized, so strength controls how far the variation drifts in content without "
            "changing the prompt's overall strength.",
            search_aliases=["variation seed", "prompt variation", "vary conditioning", "vary prompt"],
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Float.Input(
                    "strength",
                    default=0.1,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="How far to nudge. 0 = unchanged; ~0.1 gives close variations; "
                    "higher drifts further from the prompt.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xffffffffffffffff,
                    control_after_generate=True,
                    tooltip="Variation seed — change it to get a different variation.",
                ),
            ],
            outputs=[io.Conditioning.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, conditioning, strength, seed) -> io.NodeOutput:
        if strength == 0.0:
            return io.NodeOutput(conditioning)
        generator = torch.Generator().manual_seed(seed)
        c = []
        for t in conditioning:
            noise = torch.randn(t[0].shape, generator=generator).to(t[0])
            n = [_vary(t[0], noise, strength), t[1].copy()]
            pooled = n[1].get("pooled_output", None)
            if pooled is not None:
                pnoise = torch.randn(pooled.shape, generator=generator).to(pooled)
                n[1]["pooled_output"] = _vary(pooled, pnoise, strength)
            c.append(n)
        return io.NodeOutput(c)


class CondExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CLIPTextEncodeControlnet,
            T5TokenizerOptions,
            ConditioningVariation,
        ]


async def comfy_entrypoint() -> CondExtension:
    return CondExtension()
