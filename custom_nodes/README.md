# MiniMax H3 Flow Scheduler & Sampler for ComfyUI

A dedicated **1 Scheduler** and **1 Sampler** pair engineered specifically for **MiniMax H3** (video + synchronized audio DiT) in the **6–12 step sweet spot**.

---

## What Nodes Are Provided?

1. **`MiniMax H3 Flow Scheduler (6-12 Steps)`** (`MiniMaxH3Scheduler`):
   - Preserves MiniMax H3's native flow shift (video shift 12.0 / audio shift 3.0).
   - In stock `simple`, 8 steps collapses from **0.632 straight to 0.0** in the final step, starving fine details and causing audio distortion.
   - This scheduler smoothly pulls terminal sigma down to **~0.15–0.25**, providing balanced steps for prompt following, facial structure, micro-textures, and audio clarity without wavy distortion.
   - Output: `SIGMAS` (feeds `SamplerCustomAdvanced` or `SamplerCustom`).
   - Also registers `h3_flow_optimal` and `h3_flow_turbo` directly into `BasicScheduler` and `KSampler` dropdowns.

2. **`MiniMax H3 Flow Sampler (Flow-2M)`** (`MiniMaxH3Sampler`):
   - **Flow-Matching 2nd-Order Multistep (Flow-2M)**: 1 NFE per step (maximum speed on the 33B model).
   - **Bounded Extrapolation ($c_{\max} = 0.75$, $\text{damp} = 0.15$)**: Tracks curved flow trajectories without the ringing, over-sharpening, or wavy motion artifacts of unbounded 2nd-order solvers.
   - **Deterministic $\eta = 0.0$**: Protects MiniMax H3's packed stereo audio stream from hiss, crackling, or flutter.
   - Output: `SAMPLER` (feeds `SamplerCustomAdvanced` or `SamplerCustom`).
   - Also registers `minimax_h3_flow2m` directly into `KSamplerSelect` and `KSampler` dropdowns!

---

## Do I Need a Turbo LoRA for 6–12 Steps?

### Recommendation
- **For 6–8 Steps (Best speed & crispness)**:
  **Yes, use a Turbo LoRA!**
  MiniMax H3 is a massive 33B parameter model trained natively for ~16–25 steps. At 6–8 steps, pairing it with the Turbo LoRA gives the best prompt adherence, motion fluidity, and crisp details.
  - **Recommended Checkpoint**: **`minimax_h3_turbo_v4_step600_ema.safetensors`** (already in your `models/loras/` folder).
    - `v4-600` has superior micro-detail, skin texture, and resolved the plastic look of older v1 checkpoints.
    - While 4 steps can occasionally have motion smear during intense action, **running at 6–8 steps completely eliminates the motion smear** while remaining blazing fast!
  - **Alternative (for i2v 8-step)**: `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`.
  - **CRITICAL**: With any Turbo LoRA, **keep CFG / Guidance at 1.0**. Turbo LoRAs are distilled at CFG=1.0; using CFG > 1.0 causes burnout, contrast blowout, and degrades prompt following.

- **For 10–12 Steps (Pure Base Model, No LoRA)**:
  - You can run the **base model directly without any Turbo LoRA**!
  - Select mode: `base_model_10-16_steps` on the scheduler.
  - Set CFG to `2.0`–`3.0`.
  - At 10–12 steps with `MiniMax H3 Flow Sampler`, the base model resolves clean coherent video and sharp audio.

---

## Recommended Workflow Settings

### Setup 1: Turbo LoRA (6–8 Steps) — Best Overall
- **LoRA**: `minimax_h3_turbo_v4_step600_ema.safetensors` (strength `1.0`)
- **Scheduler**: `MiniMax H3 Flow Scheduler`
  - `steps`: `6` to `8`
  - `mode`: `6-12_steps_optimal` (or `turbo_lora_6-8_steps`)
- **Sampler**: `MiniMax H3 Flow Sampler`
  - `damp`: `0.15`
  - `max_extrap`: `0.75`
- **Guider / CFG**: `1.0`

### Setup 2: Base Model (10–12 Steps) — No LoRA
- **Scheduler**: `MiniMax H3 Flow Scheduler`
  - `steps`: `10` to `12`
  - `mode`: `6-12_steps_optimal`
- **Sampler**: `MiniMax H3 Flow Sampler`
  - `damp`: `0.15`
  - `max_extrap`: `0.75`
- **CFG / Guidance**: `2.0` to `3.0`

---

## Wiring in ComfyUI

```text
[Load Diffusion Model] ──> [MiniMax-H3 Turbo LoRA] ──> [SamplerCustomAdvanced]
                                                             ▲    ▲    ▲
[MiniMax H3 Flow Scheduler] ────────────── (SIGMAS) ─────────┘    │    │
[MiniMax H3 Flow Sampler]   ───────────── (SAMPLER) ──────────────┘    │
[BasicGuider (cfg=1.0)]     ────────────── (GUIDER) ───────────────────┘
```
*Or select `minimax_h3_flow2m` in the standard `KSamplerSelect` dropdown!*
