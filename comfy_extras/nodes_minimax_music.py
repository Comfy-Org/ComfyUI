import torch
from typing_extensions import override

import comfy.model_management
from comfy.ldm.minimax_music.ar import AUDIO_FRAMES_PER_SECOND, CFG_SCALE, CFG_TOP_K, C0_VOCAB_SIZE, MAX_AUDIO_FRAMES
from comfy.ldm.minimax_music.dit import latent_length
from comfy_api.latest import ComfyExtension, io


class MiniMaxMusic3TextEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxMusic3TextEncode",
            display_name="MiniMax Music3 Text Encode",
            category="model/conditioning/minimax music",
            description="Uses a MiniMax Music3 CLIP model to generate the acoustic conditioning sequence.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("caption", multiline=True, dynamic_prompts=True),
                io.String.Input("lyrics", multiline=True, dynamic_prompts=True),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Float.Input("max_duration", default=120.0, min=0.04, max=MAX_AUDIO_FRAMES / AUDIO_FRAMES_PER_SECOND, step=0.04, tooltip="Maximum duration in seconds; the model can end the song earlier."),
                io.Float.Input("cfg_scale", default=CFG_SCALE, min=0.0, max=100.0, step=0.1, round=0.01, advanced=True),
                io.Int.Input("top_k", default=CFG_TOP_K, min=1, max=C0_VOCAB_SIZE, advanced=True),
            ],
            outputs=[
                io.Conditioning.Output(),
                io.Float.Output(display_name="seconds"),
            ],
        )

    @classmethod
    def execute(cls, clip, caption, lyrics, seed, max_duration, cfg_scale, top_k):
        max_audio_frames = min(MAX_AUDIO_FRAMES, max(1, round(max_duration * AUDIO_FRAMES_PER_SECOND)))
        tokens = clip.tokenize(caption, lyrics=lyrics, seed=seed, max_audio_frames=max_audio_frames, cfg_scale=cfg_scale, top_k=top_k)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        for cond in conditioning:
            hidden = cond[0]
            cond[1]["conditioning_scale"] = torch.ones((hidden.shape[0], 1, 1), device=hidden.device, dtype=hidden.dtype)
        return io.NodeOutput(conditioning, conditioning[0][0].shape[1] / AUDIO_FRAMES_PER_SECOND)


class EmptyMiniMaxMusic3LatentAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyMiniMaxMusic3LatentAudio",
            display_name="Empty MiniMax Music3 Latent Audio",
            category="model/latent/minimax music",
            description="Creates an empty MiniMax Music3 audio latent for the requested duration.",
            inputs=[
                io.Float.Input("seconds", default=120.0, min=0.04, max=MAX_AUDIO_FRAMES / AUDIO_FRAMES_PER_SECOND, step=0.04),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, seconds, batch_size):
        audio_frames = min(MAX_AUDIO_FRAMES, max(1, round(seconds * AUDIO_FRAMES_PER_SECOND)))
        latent = torch.zeros(
            (batch_size, 128, latent_length(audio_frames)),
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        )
        return io.NodeOutput({"samples": latent, "type": "audio", "downscale_ratio_temporal": 512})


class MiniMaxMusic3Extension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [MiniMaxMusic3TextEncode, EmptyMiniMaxMusic3LatentAudio]


async def comfy_entrypoint():
    return MiniMaxMusic3Extension()
