"""SeedVR2 constants."""

# Temporal chunk-size law: the sampler's activation wall is linear in
# T_latent * pixel area (17-cell resolution sweep + T bisection, RTX 5090, 3b fp16):
#   max_latent_frames = (free_GiB - RESERVED - K*SIGMA) / (GIB_PER_MPX_FRAME * megapixels)
# RESERVED covers model staging plus fixed CUDA/torch overhead; SIGMA is the measured
# run-to-run spread of the wall; K=4 trades ~10% smaller chunks for ~1e-5 OOM odds.
SEEDVR2_CHUNK_GIB_PER_MPX_FRAME = 0.55
SEEDVR2_CHUNK_RESERVED_GIB = 8.5
SEEDVR2_CHUNK_SIGMA_GIB = 0.55
SEEDVR2_CHUNK_SIGMA_K = 4

SEEDVR2_7B_VID_DIM = 3072
SEEDVR2_OOM_BACKOFF_DIVISOR = 2
SEEDVR2_DTYPE_BYTES_FLOOR = 4
SEEDVR2_7B_MLP_CHUNK = 8192
SEEDVR2_ROPE_PARTIAL_CHUNK_TOKENS = 4096  # partial-RoPE application token-chunk.
SEEDVR2_LATENT_CHANNELS = 16

SEEDVR2_COLOR_MEM_HEADROOM = 0.75
SEEDVR2_LAB_SCALE_MULTIPLIER = 13
SEEDVR2_WAVELET_SCALE_MULTIPLIER = 10  # per-frame byte multiplier, wavelet path.
SEEDVR2_ADAIN_SCALE_MULTIPLIER = 6

BYTEDANCE_VAE_SCALING_FACTOR = 0.9152   # configs_3b/main.yaml:57.
BYTEDANCE_VAE_SHIFTING_FACTOR = 0.0
BYTEDANCE_VAE_CONV_MEM_GIB = 0.5
BYTEDANCE_VAE_NORM_MEM_GIB = 0.5
BYTEDANCE_LOGVAR_CLAMP_MIN = -30.0      # video_vae_v3/modules/types.py:28.
BYTEDANCE_LOGVAR_CLAMP_MAX = 20.0       # video_vae_v3/modules/types.py:28.
BYTEDANCE_GN_CHUNKS_FP16 = 4            # causal_inflation_lib.py:351 (GroupNorm chunk count, fp16).
BYTEDANCE_GN_CHUNKS_FP32 = 2            # causal_inflation_lib.py:351 (GroupNorm chunk count, fp32).
BYTEDANCE_BLOCK_OUT_CHANNELS = (128, 256, 512, 512)  # s8_c16_t4_inflation_sd3.yaml:7-11.
BYTEDANCE_SLICING_SAMPLE_MIN = 4        # s8_c16_t4_inflation_sd3.yaml:22 (slicing_sample_min_size).
BYTEDANCE_VAE_TEMPORAL_DOWNSAMPLE = 4   # infer.py:230 (temporal_downsample_factor); the 4n+1 factor.
BYTEDANCE_VAE_SPATIAL_DOWNSAMPLE = 8    # infer.py:231 (spatial_downsample_factor).
BYTEDANCE_720P_REF_AREA = 45 * 80       # dit_v2/window.py:32 (720p reference area for window scaling).
BYTEDANCE_MAX_TEMPORAL_WINDOW = 30      # dit_v2/window.py:35 (max temporal window frames).
BYTEDANCE_ROPE_MAX_FREQ = 256           # dit_v2/rope.py:31 (pixel-RoPE max frequency).
BYTEDANCE_SINUSOIDAL_DIM = 256          # dit_3b/nadit.py:120 (timestep sinusoidal embed dim).

ROPE_THETA = 10000   # RoPE base; Su et al., "RoFormer", arXiv:2104.09864.

CIELAB_DELTA = 6.0 / 29.0          # CIE 15 (delta).
CIELAB_KAPPA = (29.0 / 3.0) ** 3   # CIE 15 (kappa).
D65_WHITE_X = 0.95047              # CIE D65 standard illuminant Xn (Yn = 1).
D65_WHITE_Z = 1.08883              # CIE D65 standard illuminant Zn.
WAVELET_DECOMP_LEVELS = 5          # wavelet color-fix decomposition depth (GIMP/Krita; StableSR).
