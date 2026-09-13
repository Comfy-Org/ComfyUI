import torch
from typing_extensions import override

import comfy.model_management
from comfy.text_encoders.yue2 import FRAMES_PER_SECOND
from comfy_api.latest import ComfyExtension, io


class YuE2GenerateABC(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2GenerateABC",
            display_name="YuE2 Generate ABC",
            category="model/conditioning/yue2",
            description="Generates the ABC notation of a song from style and lyrics. Connect abc to YuE2 Generate Music node.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("style", multiline=True, dynamic_prompts=True),
                io.String.Input("lyrics", multiline=True, dynamic_prompts=True),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Combo.Input("mode", options=["full", "melody"], tooltip="full: generates melody and chords; melody: generates melody only, recommended for covers."),
                io.Int.Input("max_abc_tokens", default=8192, min=1, max=20000, advanced=True),
                io.Float.Input("temperature", default=0.7, min=0.0, max=5.0, step=0.05, advanced=True),
                io.Float.Input("top_p", default=0.9, min=0.01, max=1.0, step=0.01, advanced=True),
                io.Int.Input("top_k", default=30, min=1, max=32768, advanced=True),
                io.Float.Input("repetition_penalty", default=1.005, min=0.01, max=10.0, step=0.005, advanced=True),
                io.Int.Input("penalty_window", default=100, min=1, max=20000, advanced=True, tooltip="Number of recent ABC tokens used to penalize repetition."),
            ],
            outputs=[io.String.Output(display_name="abc")],
        )

    @classmethod
    def execute(cls, clip, style, lyrics, seed, mode, max_abc_tokens, temperature=0.7, top_p=0.9, top_k=30, repetition_penalty=1.005, penalty_window=100):
        tokens = clip.tokenize(style, lyrics=lyrics, cot=mode, seed=seed, max_tokens=max_abc_tokens, penalty_window=penalty_window)
        ids = clip.generate(tokens, max_length=max_abc_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                            repetition_penalty=repetition_penalty, seed=seed)
        return io.NodeOutput(clip.decode(ids))


class YuE2GenerateMusic(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YuE2GenerateMusic",
            display_name="YuE2 Generate Music",
            category="model/conditioning/yue2",
            description="Generates music tokens and acoustic conditioning from style, lyrics, and an ABC notation. Provide the generated seconds to the Empty YuE2 Latent Audio node. An empty ABC input ignores the selected mode.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("style", multiline=True, dynamic_prompts=True),
                io.String.Input("lyrics", multiline=True, dynamic_prompts=True),
                io.String.Input("abc", default="", multiline=True, tooltip="Connect the ABC generator or supply an edited score. Leave empty to use off mode automatically."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Combo.Input("mode", options=["full", "melody"], tooltip="full: generates melody and chords; melody: generates melody only, recommended for covers."),
                io.Float.Input("max_duration", default=360.0, min=0.04, max=900.0, step=0.04, tooltip="Maximum duration in seconds. Automatically reduced for long prompts; generation can stop earlier."),
                io.Float.Input("temperature", default=1.0, min=0.0, max=5.0, step=0.05, advanced=True),
                io.Float.Input("top_p", default=0.95, min=0.01, max=1.0, step=0.01, advanced=True),
                io.Int.Input("top_k", default=100, min=1, max=32768, advanced=True),
                io.Float.Input("repetition_penalty", default=1.2, min=0.01, max=10.0, step=0.01, advanced=True),
            ],
            outputs=[io.Conditioning.Output(), io.Float.Output(display_name="seconds")],
        )

    @classmethod
    def execute(cls, clip, style, lyrics, seed, mode, max_duration, temperature, top_p, top_k, repetition_penalty, abc=""):
        if not abc.strip():
            mode = "off"
        tokens = clip.tokenize(style, lyrics=lyrics, cot=mode, seed=seed, abc=abc,
                               max_tokens=max(1, round(max_duration * FRAMES_PER_SECOND)),
                               temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning, conditioning[0][1]["yue2_frames"] / FRAMES_PER_SECOND)


class EmptyYuE2LatentAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyYuE2LatentAudio",
            display_name="Empty YuE2 Latent Audio",
            category="model/latent/yue2",
            inputs=[
                io.Float.Input("seconds", default=120.0, min=0.04, max=1000.0, step=0.04),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, seconds, batch_size):
        latent = torch.zeros((batch_size, 64, max(1, round(seconds * FRAMES_PER_SECOND))),
                             device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
        return io.NodeOutput({"samples": latent, "type": "audio", "downscale_ratio_temporal": 1920})


class YuE2Extension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [YuE2GenerateABC, YuE2GenerateMusic, EmptyYuE2LatentAudio]


async def comfy_entrypoint():
    return YuE2Extension()
