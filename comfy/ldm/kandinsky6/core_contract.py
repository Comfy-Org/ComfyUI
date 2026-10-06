"""Kandinsky 6 release contract for the ComfyUI adapter.

Maintain these constants alongside the released checkpoint and adapter tests.
This module intentionally contains no checkpoint paths and imports no K6
runtime code, so ComfyUI can inspect node contracts without loading models.
"""

from typing import Final

PROMPT_TEMPLATE: Final[str] = '<|im_start|>system\nYou are a promt engineer. Describe the video in detail.\nDescribe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.\nDescribe the location of the video, main characters or objects and their action.\nDescribe the dynamism of the video and presented actions.\nName the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.\nDescribe the visual effects, postprocessing and transitions if they are presented in the video.\nPay attention to the order of key actions shown in the scene.<|im_end|>\n<|im_start|>user\n{}<|im_end|>'
PROMPT_CROP_START: Final[int] = 129
PROMPT_MAX_LENGTH: Final[int] = 1024
PROMPT_TOKENIZER_MAX_LENGTH: Final[int] = 1153

GENERATION_DEFAULTS: Final[dict[str, object]] = {'fps': 24.0,
 'height': 480,
 'image_divisibility': 16,
 'latent_frames': 31,
 'max_area': 414720,
 'sample_frames': 121,
 'scale_factor': [1.0, 2.0, 2.0],
 'video_latent_channels': 16,
 'visual_cond_scheme': 'tail_cond_first_frame',
 'width': 864}
SCHEDULER_DEFAULTS: Final[dict[str, object]] = {'guidance_weight': 5.0, 'num_steps': 50, 'scheduler_scale': 5.0}
AUDIO_DEFAULTS: Final[dict[str, object]] = {'latent_channels': 40,
 'latent_downsample_factor': 1024,
 'mode': '44k',
 'normalization_mode': 'clip',
 'sample_rate': 44100,
 'scaling_factor': 0.5302}
DIT_CONFIG: Final[dict[str, object]] = {'audio_freqs_scaling': 0.144,
 'axes_dims': (32, 48, 48),
 'axes_dims_a': (32, 48, 48),
 'ca_rope': True,
 'cross_gates': True,
 'ff_dim': 16384,
 'ff_dim_a': 7168,
 'fix_modulation': True,
 'in_audio_dim': 40,
 'in_text_dim': 3584,
 'in_text_dim2': 768,
 'in_visual_dim': 16,
 'is_multimodal': True,
 'model_dim': 4096,
 'model_dim_a': 2048,
 'num_text_blocks': 4,
 'num_visual_blocks': 60,
 'out_audio_dim': 40,
 'out_visual_dim': 16,
 'patch_size': (1, 2, 2),
 'text_token_padding': True,
 'time_dim': 1024,
 'time_dim_a': 1024,
 'visual_cond': True,
 'visual_token_type_num_embeddings': 2}
LATENT_DEFAULTS: Final[dict[str, object]] = {'audio_channels': 40,
 'audio_downsample_factor': 1024,
 'audio_sample_rate': 44100,
 'video_channels': 16,
 'video_spatial_compression_factor': 8,
 'video_temporal_compression_factor': 4}

__all__ = [
    "AUDIO_DEFAULTS",
    "DIT_CONFIG",
    "GENERATION_DEFAULTS",
    "LATENT_DEFAULTS",
    "PROMPT_CROP_START",
    "PROMPT_MAX_LENGTH",
    "PROMPT_TEMPLATE",
    "PROMPT_TOKENIZER_MAX_LENGTH",
    "SCHEDULER_DEFAULTS",
]
