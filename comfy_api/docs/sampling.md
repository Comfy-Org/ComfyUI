# Sampling

Most advanced packs change how a model samples: a different CFG combination, edits to attention, a schedule
of their own, a wrapper around the sampling loop. In V1 they did it by calling `ModelPatcher` methods or by
replacing functions in `comfy.samplers` and `comfy.k_diffusion.sampling`. In V4 each of those is a call that
returns a patched copy of the model, or a registration on the pack's `ComfyExtension`.

## How hooks work

A hook is a function your node hands to the engine, which calls it while sampling:

```py
def my_cfg(args):
    return args["uncond"] + (args["cond"] - args["uncond"]) * args["cond_scale"]

patched = await eng.model.set_cfg(model, my_cfg)
return io.NodeOutput(patched)
```

Three rules apply to every hook, and they're why the same hook works in every mode:

1. **It gets tensors and plain data.** Its arguments are the same as V1 hooks got, except that engine objects
   (the model, for one) are removed. Under pack isolation the hook runs in the pack's process and is called
   from ComfyUI's on every step, so only data can cross.
2. **It can't call `eng`.** Sampling is already inside the engine. Get anything you need from `eng` in
   `execute`, before handing the hook over, and capture it:

   ```py
   sigma_end = await eng.model.percent_to_sigma(model, end_percent)

   def my_cfg(args):
       if args["sigma"][0] < sigma_end:
           return args["uncond"] + (args["cond"] - args["uncond"]) * args["cond_scale"]
       return args["uncond"] + (args["cond"] - args["uncond"]) * args["cond_scale"] * 0.5
   ```

3. **It isn't `async`.** It runs inside the sampler, on every step.

The hook is stored with the returned copy. The model your node received is unchanged, so two nodes can patch
the same model differently.

## Guidance hooks

| Call | The hook | Runs |
|---|---|---|
| `eng.model.set_cfg(model, fn)` | `fn(args) -> noise prediction`; `args` has `cond`, `uncond`, `cond_scale`, `sigma`, `input`... | Instead of the CFG combination |
| `eng.model.set_pre_cfg(model, fn)` | `fn(args) -> list`; `args` has `conds_out` (the model outputs for cond and uncond), `cond_scale`, `sigma`... | Before CFG; returns the new `conds_out` |
| `eng.model.set_post_cfg(model, fn)` | `fn(args) -> denoised`; `args` has `denoised`, `cond_denoised`, `uncond_denoised`... | After CFG; several run in order |
| `eng.model.set_denoise_mask(model, fn)` | `fn(sigma, denoise_mask, extra_options) -> mask` | Each step, for inpainting |

`set_post_cfg` and `set_pre_cfg` are where guidance methods such as PAG and SAG do their work.

## Attention and block patches

| Call | The hook |
|---|---|
| `eng.model.set_attn1_patch(model, fn)` | `fn(q, k, v, extra_options) -> (q, k, v)` before self-attention |
| `eng.model.set_attn2_patch(model, fn)` | `fn(q, context, value, extra_options) -> (q, context, value)` before cross-attention |
| `eng.model.set_attn1_output_patch`, `set_attn2_output_patch` | `fn(out, extra_options) -> out` on attention's output |
| `eng.model.set_input_block_patch(model, fn)` | `fn(h, transformer_options) -> h` on input block features |
| `eng.model.set_input_block_patch_after_skip(model, fn)` | the same, after the skip connection |
| `eng.model.set_output_block_patch(model, fn)` | `fn(h, hsp, transformer_options) -> (h, hsp)` |
| `eng.model.set_middle_block_after_patch(model, fn)` | `fn(args) -> dict`; `args` has `h`, `x`, `emb`, `context`, `timesteps`, `transformer_options`; the returned `"h"` replaces the features |

`extra_options` and `transformer_options` carry the block, its index, the number of heads, `sigmas` and
`cond_or_uncond`, so a patch can act on some blocks or steps only. A value your patch needs at sampling time
can travel in `transformer_options`:

```py
m = await eng.model.set_option(model, ["transformer_options", "my_pack_scale"], scale)
m = await eng.model.set_attn1_patch(m, my_patch)   # reads extra_options["my_pack_scale"]
```

To switch to one of ComfyUI's attention implementations rather than editing attention yourself:

```py
m = await eng.model.set_attention(model, "sage")     # "pytorch", "sage", "flash", "xformers"...
```

## Wrappers

A wrapper runs around a whole stage of sampling and calls the stage itself. It's the V4 form of replacing a
method in `comfy.samplers` to do something before or after it.

```py
def outer_sample(executor, *args, **kwargs):
    prepare()
    result = executor(*args, **kwargs)
    clean_up()
    return result

m = await eng.model.add_wrapper(model, "outer_sample", outer_sample)
```

The stages, outermost first: `"outer_sample"`, `"prepare_sampling"`, `"sampler_sample"`, `"predict_noise"`,
`"calc_cond_batch"`, `"apply_model"`, `"diffusion_model"`. What the engine hands a wrapper to call (the
`executor`, a model function, a callback) is an engine function: call it like any function, with tensors,
plain data and what you were given. Engine objects in its arguments are opaque values: pass them on.

`eng.model.set_unet_wrapper(model, fn)` wraps every model call:
`fn(apply_model, args)` returns the model output, usually
`apply_model(args["input"], args["timestep"], **args["c"])` with changes before or after. Wrappers from
different nodes compose: a wrapper already on the model runs inside the new one.

## Samplers and schedulers

**A named sampler** with options:

```py
sampler = await eng.sampler.named("gradient_estimation", {"ge_gamma": 2.0})
```

**A sampler a node builds,** whose loop is the pack's own code:

```py
def my_sampler(model, x, sigmas, extra_args, callback=None, disable=None):
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        x = x + (x - denoised) / sigmas[i] * (sigmas[i + 1] - sigmas[i])
    return x

sampler = await eng.sampler.custom(my_sampler)
return io.NodeOutput(sampler)        # a SAMPLER output, for SamplerCustom and friends
```

`model` is an engine function the loop calls to denoise; `callback` reports each step, which drives previews
and progress. Pass `model_sampling=True` to also get the model's sigma table and conversion functions.

**Samplers and schedulers in ComfyUI's lists,** so they appear in KSampler's dropdowns from startup, are
registered on the pack's `ComfyExtension`:

```py
class MyPack(ComfyExtension):
    async def get_samplers(self):
        return {"my_sampler": my_sampler}

    async def get_schedulers(self):
        return {"my_schedule": my_schedule}       # fn(model_sampling, steps) -> sigmas
```

A scheduler gets `model_sampling` as plain data and engine functions: `sigma_min`, `sigma_max`, `sigmas`, and
`sigma(t)`, `timestep(s)`, `percent_to_sigma(p)`.

## Running a sampler from a node

A node that needs to sample, such as a detailer or an upscaler, runs ComfyUI's sampling node:

```py
(samples,) = await eng.node.run("KSampler", {...})
(output, denoised) = await eng.node.run("SamplerCustom", {
    "model": model, "add_noise": True, "noise_seed": seed, "cfg": cfg, "positive": positive,
    "negative": negative, "sampler": sampler, "sigmas": sigmas, "latent_image": latent})
```

That uses exactly the sampling ComfyUI's own nodes use, so results match a workflow that wires the same nodes
by hand.

## When the hooks aren't enough

Some packs need more than hooks: a sampling loop that holds engine state across steps, a new model
architecture, or patches to the model's modules. Those belong in an [engine extension](engine-extensions.md),
which runs next to the engine. For everything else, the calls on this page cover what V1 packs did by
patching: see [Migrating a pack](migrating.md#monkey-patching).
