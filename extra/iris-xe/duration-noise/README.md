# Keep a text-to-video shot when changing its length

Install `ltx_duration_noise.py` and `ltx_text_continuation.py` in the active ComfyUI `custom_nodes/` directory. Copy `LTX25_Q5_T2V_DurationNoise.json` into `user/default/workflows/`. Restart once and reopen the workflow. The filename is retained for existing installations. The T2V example defaults to the Q4_K_M model, 480×288, seed 20000001, Euler CFG++, both prompts, and eight steps.

On the configured machines, stop the previous ComfyUI process after the queue finishes, then start the replacement process. The wrapper delegates to the existing XPU launcher, adding `--cpu-vae` on the local hp.

```bash
# hp8
bash ~/Project/opensource/github/ai-video3/iris-xe-9f/native-mixed/run-t2v.sh hp8

# hp (local checkout)
bash ~/Project/opensource/github/agent-offline/ltx-video/homdx-fork/ComfyUI/extra/iris-xe/duration-noise/run-t2v.sh hp
```

Render 9 with **unit = frames**. In the same workflow change only **total_length** to 17, 25, 225 or any desired count. **unit = seconds** interprets that number as the total duration at the selected output fps. At 24 fps a 9-second preview has 216 frames and a 10-second video has 240 frames.

The first window is generated from text. Later windows condition 16 new frames on the preceding two sampled latent positions using a zero denoise mask on that context. Decoding discards the repeated context frames. A partially exported window is saved in full. New windows have distinct seeds derived from the fixed shot seed. The completed RGB frames and boundary latents are saved atomically in `output/ltx-text-shots/<shot key>/` and reused across requests and server restarts. Keep this directory when moving the installation. A new seed, prompt, model file identity, resolution, or sampling configuration starts a separate shot. Keep frame rate and other inputs fixed when extending. Old MP4s made by the earlier variable-shape workflow do not automatically populate this cache.

Export encodes the requested prefix in order. Previously saved frames stay fixed. The new frames are predictions and can develop identity, motion, or background drift over many windows. At 480×288 RGB checkpoints need about 91 MiB for 9 seconds and 2140 MiB for 225 seconds (24 fps, plus window rounding and latent state). A long request can be stopped and repeated to reuse all completed windows.

The independent I2V example still uses its original two-position noise blocks. Its instructions and settings are separate from the text-only workflow.

From the fork root, run the checks with the ComfyUI Python:

```sh
python extra/iris-xe/duration-noise/check_noise.py --cpu
python extra/iris-xe/duration-noise/check_extension.py --cpu
```

The second check exercises real H.264 encoding and decoded-frame comparisons with small stand-ins for the expensive sampler and decoder.
