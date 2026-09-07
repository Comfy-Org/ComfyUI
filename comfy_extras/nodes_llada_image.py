import torch
from tqdm.auto import trange
from typing_extensions import override

import comfy.samplers
import comfy.k_diffusion.sampling
import comfy.model_management
import comfy.utils
import node_helpers
import nodes
from comfy.text_encoders.llada_image import LLaDAImageTEModel
from comfy_api.latest import ComfyExtension, io


def llada_image_sigmas(steps, variant):
    if variant == "base":
        schedule = torch.linspace(0.001, 1.0, steps + 1, dtype=torch.float64)[:-1]
        schedule = (1.0 - (1.0 - schedule.pow(1.17)).pow(0.8)).pow(1.1)
        sigmas = 1.0 - schedule
    elif variant == "turbo":
        sigmas = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)[:-1]
        sigmas = 3.0 * sigmas / (1.0 + 2.0 * sigmas)
    else:
        raise ValueError(f"unsupported LLaDA-Image variant: {variant}")
    return torch.cat((sigmas.to(torch.float32), torch.zeros(1, dtype=torch.float32)))


def sample_llada_image_turbo(
    model, x, sigmas, extra_args=None, callback=None, disable=None
):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = comfy.k_diffusion.sampling.default_noise_sampler(
        x, seed=extra_args.get("seed")
    )
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )

        sigma_next = sigmas[i + 1]
        if sigma_next == 0:
            x = denoised
        else:
            noise = noise_sampler(sigmas[i], sigma_next)
            x = (1.0 - sigma_next) * denoised + sigma_next * noise
    return x


class LLaDAImageScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LLaDAImageScheduler",
            display_name="LLaDA-Image Scheduler",
            category="model/sampling/schedulers",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input(
                    "steps",
                    default=0,
                    min=0,
                    max=1000,
                    tooltip="0 selects the checkpoint default: Base 50, Turbo 4.",
                ),
            ],
            outputs=[io.Sigmas.Output()],
        )

    @classmethod
    def execute(cls, model, steps) -> io.NodeOutput:
        model_sampling = model.get_model_object("model_sampling")
        variant = model_sampling.llada_image_variant
        if steps == 0:
            steps = 50 if variant == "base" else 4
        return io.NodeOutput(llada_image_sigmas(steps, variant))


class SamplerLLaDAImageTurbo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerLLaDAImageTurbo",
            display_name="Sampler LLaDA-Image Turbo",
            category="model/sampling/samplers",
            description=(
                "Use the stochastic flow sampler enabled by the official Turbo "
                "checkpoint. Pair the LLaDA-Image schedule with the standard Euler "
                "sampler for the official stochastic_sampling=false alternative."
            ),
            inputs=[],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        return io.NodeOutput(comfy.samplers.KSAMPLER(sample_llada_image_turbo))


def _encode_prompts(clip, prompt, negative_prompt):
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))
    return positive, negative


def _set_semantic_conditioning(
    positive, negative, semantic_features, source_latents=None
):
    semantic_mask = torch.ones(
        semantic_features.shape[:2], dtype=torch.bool, device=semantic_features.device
    )
    empty_semantic = semantic_features.new_zeros(
        semantic_features.shape[0], 0, semantic_features.shape[-1]
    )
    empty_mask = semantic_mask[:, :0]
    positive_values = {
        "semantic_features": semantic_features,
        "semantic_mask": semantic_mask,
    }
    negative_values = {
        "semantic_features": empty_semantic,
        "semantic_mask": empty_mask,
    }
    if source_latents is not None:
        positive_values["source_latents"] = source_latents
        negative_values["source_latents"] = source_latents
    positive = node_helpers.conditioning_set_values(positive, positive_values)
    negative = node_helpers.conditioning_set_values(negative, negative_values)
    return positive, negative


def _load_llada_clip(clip, vq_sequence_length=0):
    if not isinstance(clip.cond_stage_model, LLaDAImageTEModel):
        raise ValueError("The connected CLIP is not an LLaDA-Image AIO text encoder")
    if vq_sequence_length:
        memory_required = clip.cond_stage_model.vq_memory_estimation(vq_sequence_length)
        comfy.model_management.load_models_gpu([clip.patcher], memory_required=memory_required)
    else:
        clip.load_model()
    device = clip.patcher.load_device
    clip.cond_stage_model.set_clip_options({"execution_device": device})
    return clip.cond_stage_model.llada2, device


class LLaDAImageVQConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LLaDAImageVQConditioning",
            display_name="LLaDA-Image VQ Conditioning",
            category="model/conditioning/llada image",
            description="Generate discrete image tokens with LLaDA2, then attach their SigVQ features to diffusion conditioning.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input(
                    "negative_prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    advanced=True,
                ),
                io.Int.Input(
                    "width", default=1024, min=64, max=nodes.MAX_RESOLUTION, step=16
                ),
                io.Int.Input(
                    "height", default=1024, min=64, max=nodes.MAX_RESOLUTION, step=16
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, negative_prompt, width, height) -> io.NodeOutput:
        if width % 16 or height % 16:
            raise ValueError("LLaDA-Image VQ width and height must be divisible by 16")
        positive, negative = _encode_prompts(clip, prompt, negative_prompt)
        input_ids, unconditional_ids, vq_height, vq_width = clip.tokenizer.tokenize_vq(
            prompt, height, width
        )
        image_token_count = vq_height * vq_width
        model, device = _load_llada_clip(clip, len(input_ids) + image_token_count)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(
            0
        )
        unconditional_ids = torch.tensor(
            unconditional_ids, dtype=torch.long, device=device
        )
        token_ids = model.generate_vq_tokens(
            input_ids, unconditional_ids, image_token_count, cfg_scale=2.0
        )
        if token_ids.shape[1] != image_token_count:
            raise ValueError(
                f"LLaDA2 generated {token_ids.shape[1]} VQ tokens, expected {image_token_count}"
            )
        codebook_size = model.sigvq.prior_token_embedding.num_embeddings
        if torch.any((token_ids < 0) | (token_ids >= codebook_size)):
            raise ValueError("LLaDA2 generated token IDs outside the SigVQ codebook")
        semantic_features, _ = model.encode_sigvq(token_ids=token_ids)
        semantic_features = semantic_features.to(
            comfy.model_management.intermediate_device()
        )
        positive, negative = _set_semantic_conditioning(
            positive, negative, semantic_features
        )
        return io.NodeOutput(positive, negative)


class LLaDAImageEditConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LLaDAImageEditConditioning",
            display_name="LLaDA-Image Edit Conditioning",
            category="model/conditioning/llada image",
            description="Encode a source image with the AIO SigVQ and VAE for LLaDA-Image editing.",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Image.Input("image"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input(
                    "negative_prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    advanced=True,
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, vae, image, prompt, negative_prompt="") -> io.NodeOutput:
        height = image.shape[1] // 32 * 32
        width = image.shape[2] // 32 * 32
        if height < 32 or width < 32:
            raise ValueError("LLaDA-Image editing requires an image at least 32x32")
        samples = image.movedim(-1, 1)
        if samples.shape[-2:] != (height, width):
            samples = comfy.utils.common_upscale(
                samples, width, height, "lanczos", "disabled"
            )
        samples = samples[:, :3]
        resized_image = samples.movedim(1, -1)
        source_latents = vae.encode(resized_image)

        positive, negative = _encode_prompts(clip, prompt, negative_prompt)
        model, device = _load_llada_clip(clip)
        sigvq_pixels = torch.nn.functional.interpolate(
            samples.float() * 2.0 - 1.0,
            size=(height // 2, width // 2),
            mode="bilinear",
            align_corners=False,
        )
        sigvq_pixels = sigvq_pixels.to(device=device, dtype=model.dtype)
        semantic_features, _ = model.encode_sigvq(pixel_values=sigvq_pixels)
        intermediate_device = comfy.model_management.intermediate_device()
        semantic_features = semantic_features.to(intermediate_device)
        source_latents = source_latents.to(intermediate_device)
        positive, negative = _set_semantic_conditioning(
            positive, negative, semantic_features, source_latents
        )
        return io.NodeOutput(
            positive,
            negative,
            {"samples": torch.zeros_like(source_latents)},
        )


class LLaDAImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LLaDAImageScheduler,
            SamplerLLaDAImageTurbo,
            LLaDAImageVQConditioning,
            LLaDAImageEditConditioning,
        ]


async def comfy_entrypoint() -> LLaDAImageExtension:
    return LLaDAImageExtension()
