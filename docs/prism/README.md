# Prism native audio/video support

Native loading of the released Prism Alpha joint checkpoint, Wan UMT5 conditioning, normal KSampler sampling, and native video/audio VAE decoding. No custom node package or diffusers dependency is required.

## Models

| File | ComfyUI folder | Notes |
| --- | --- | --- |
| `Prism/diffusion_pytorch_model.safetensors` | `models/diffusion_models` | Alpha BF16 joint checkpoint |
| `prism_alpha_video_fp8_scaled.safetensors` | `models/diffusion_models` | Video-block row-scaled E4M3FN, `.prism_scale`; weight-only high-precision computation |
| `prism_alpha_video_int8_convrot.safetensors` | `models/diffusion_models` | Native `int8_tensorwise`, `convrot=true`, group size 256 |
| `prism_umt5_xxl_bf16.safetensors` | `models/text_encoders` | Load with the existing CLIP Loader, type `wan` |
| `Wan2_1_VAE_bf16.safetensors` | `models/vae` | Native video VAE |
| `prism_dac_audio_vae.safetensors` | `models/vae` | Continuous 48 kHz mono DAC, 128 latent channels, hop 960 |

The two quantized checkpoints change only 800 video-block Linear weights. Audio, bridges, embeddings, convolutions, norms, and output heads retain their original precision. The legacy row-scaled FP8 path does not implement FP8 GEMM or LoRA. Native INT8 ConvRot uses the existing mixed-precision operations and Comfy Kitchen dispatch.

## One shared workflow

Import **`prism_native_test_workflow.json`** and replace the Load Image input with your reference. The original test reference image is not distributed here. Select one of the three main checkpoints in the normal UNET Loader; leave `weight_dtype=default`. Other nodes remain unchanged.

This workflow was extracted from `nativebf16test.mp4`. Metadata in all three provided outputs confirms the same settings: 81 frames, 848x480, 24 fps, seed 666, 30 steps, CFG 5, Euler/simple, video shift 9, audio shift 7. Results are 3.375 seconds with generated audio. Differences in output quality are not claimed to be zero.

`PrismPrepareAV` produces standard conditioning and a joint latent. `ModelSamplingAV` maps the video and audio schedules to the stock sampler. `PrismAttention` sets attention options through model patches. Decoding uses VAELoader, VAEDecode and VAEDecodeAudio; the explicit original audio sample count is preserved through AV latent separation.

The sparse modes need Triton. `dense` uses the selected native attention backend. `prism_sparse_tail_safe` preserves sparse attention except for real queries in an incomplete final temporal block, which receive full attention. This prevents approximation-induced tail collapse observed in local reproduction, but is not a blanket quality guarantee.

## Validation scope

- Full BF16 and legacy FP8 strict CPU loading, normal/lazy construction; 3735 original parameter keys.
- Full native INT8 ConvRot loading with 800 quantized layers and serialization roundtrip.
- Selected BF16 core blocks/bridges/towers compared with the reference implementation; legacy FP8 blocks compared with explicit reconstruction.
- Stock KSampler interface checks with a small test model at 49 and 81 frames (not a full-model quality benchmark).
- Three user-provided real generated videos: BF16, FP8, INT8 ConvRot. The user explicitly accepted FP8 quality. The BF16 and INT8 outputs are supplied for comparison, not a general parity claim.

The video assets are linked in the upstream PR. GPU speedup, arbitrary samplers, ControlNet and exhaustive seed/length/device coverage have not been established. INT8/FP8 filenames denote checkpoint storage formats, not equivalent numerical results or end-to-end speed guarantees.

## Provenance

Architecture and sparse kernels: Tencent-Hunyuan/Prism (MIT with the listed third-party licenses). Continuous audio codec: Prism and descript-audio-codec (MIT). The copied license/attribution files are retained next to the corresponding implementations. Wan and UMT5 components retain their own original licenses. This change does not download or upload models or contact any network service from core inference code.

## Attention precision and portability

Without a `PrismAttention` override, the model uses `dense` attention and does not import Triton. The supplied workflow explicitly selects `prism_sparse_tail_safe`; that optional path still requires CUDA and Triton.

On devices supporting FP64, adjacent-pair rotary arithmetic retains the upstream complex multiplication and single final rounding, with sequence chunks limiting temporary allocation size. Devices without FP64 use real FP32 rotation matrices and the shared ComfyUI RoPE dispatcher. Frequency tables are created on the target device once per tower forward and are not retained between executions. The FP32 fallback is not claimed to be numerically identical to the FP64 reference.
