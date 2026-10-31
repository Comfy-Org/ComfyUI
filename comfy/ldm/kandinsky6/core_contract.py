"""Kandinsky 6 release contract for the ComfyUI adapter.

Maintain these constants alongside the released checkpoint and adapter tests.
This module intentionally contains no checkpoint paths and imports no K6
runtime code, so ComfyUI can inspect node contracts without loading models.
"""

PROMPT_TEMPLATE = '<|im_start|>system\nYou are a promt engineer. Describe the video in detail.\nDescribe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.\nDescribe the location of the video, main characters or objects and their action.\nDescribe the dynamism of the video and presented actions.\nName the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.\nDescribe the visual effects, postprocessing and transitions if they are presented in the video.\nPay attention to the order of key actions shown in the scene.<|im_end|>\n<|im_start|>user\n{}<|im_end|>'
PROMPT_TOKENIZER_MAX_LENGTH = 1153

GENERATION_DEFAULTS = {'fps': 24.0,
 'height': 480,
 'sample_frames': 121,
 'scale_factor': [1.0, 2.0, 2.0],
 'width': 864}
SCHEDULER_DEFAULTS = {'scheduler_scale': 5.0}
AUDIO_DEFAULTS = {'scaling_factor': 0.5302}
DIT_CONFIG = {'audio_freqs_scaling': 0.144,
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
 'model_dim': 4096,
 'model_dim_a': 2048,
 'num_text_blocks': 4,
 'num_visual_blocks': 60,
 'out_visual_dim': 16,
 'patch_size': (1, 2, 2),
 'time_dim': 1024,
 'time_dim_a': 1024,
 'visual_token_type_num_embeddings': 2}
LATENT_DEFAULTS = {'audio_channels': 40,
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
    "PROMPT_TEMPLATE",
    "PROMPT_TOKENIZER_MAX_LENGTH",
    "SCHEDULER_DEFAULTS",
]
