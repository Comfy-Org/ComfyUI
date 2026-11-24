# Duration-stable noise for LTX video

The two example workflows use the same Q5 GGUF, CPU text encoder, both prompts, fixed seed 10000001, Euler CFG++, and eight sampler steps at 480×288. They take video latents with shape [B,C,T,H,W].

Copy `ltx_duration_noise.py` into the active ComfyUI `custom_nodes/` directory, copy the two JSON files into `user/default/workflows/`, and restart ComfyUI once. The examples require ComfyUI-GGUF and the LTX-2.5 weights named in the loaders.

1. Open `LTX25_Q5_T2V_DurationNoise.json`. Render with length 9, then change only length to 17 or 25. Keep the seed fixed. Use video lengths of the form 8n+1.
2. To anchor the subject, open `LTX25_Q5_I2V_DurationNoise.json`, upload your chosen first frame in Load Image, and use that same image at each length. The default conditioning strength is 1.0.
3. For the original noise behavior, reopen your existing workflow with RandomNoise.

The node draws fixed blocks of two latent time positions using ComfyUI’s CPU noise generator and crops the final block. The first block matches stock RandomNoise for nine video frames. With the same seed, batch indices, resolution and dtype, extending the clip preserves every overlapping initial noise value. Euler CFG++ also avoids the additional ancestral noise draws. The denoiser processes time positions jointly, so decoded frames and character identity can still change with duration.
