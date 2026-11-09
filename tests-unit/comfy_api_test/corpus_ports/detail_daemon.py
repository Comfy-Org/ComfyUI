"""ComfyUI-Detail-Daemon's LyingSigmaSampler (detail_daemon_node.py), ported to the V2 API.

The original reaches into the engine (``model.inner_model.inner_model.model_sampling``) for
``percent_to_sigma``, and into the wrapped SAMPLER for ``sampler_function``. The port gets both from
``eng.sampler.custom``: ``model_sampling=True`` and a SAMPLER option, which arrives as an engine function.
"""
from comfy_api.index import eng, io


def lying_sigma_sampler(model, x, sigmas, extra_args, callback=None, disable=None, *, model_sampling,
                        lss_wrapped_sampler, lss_dishonesty_factor, lss_startend_percent, **kwargs):
    start_percent, end_percent = lss_startend_percent
    start_sigma = round(model_sampling["percent_to_sigma"](start_percent), 4)
    end_sigma = round(model_sampling["percent_to_sigma"](end_percent), 4)

    def model_wrapper(x, sigma, **extra_args):
        sigma_float = float(sigma.max().detach().cpu())
        if end_sigma <= sigma_float <= start_sigma:
            sigma = sigma * (1.0 + lss_dishonesty_factor)
        return model(x, sigma, **extra_args)

    return lss_wrapped_sampler(model_wrapper, x, sigmas, extra_args=extra_args, callback=callback,
                               disable=disable, **kwargs)


class LyingSigmaSamplerNode(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LyingSigmaSampler", category="sampling/custom_sampling",
            inputs=[io.Sampler.Input("sampler"),
                    io.Float.Input("dishonesty_factor", default=-0.05, min=-0.999, step=0.01),
                    io.Float.Input("start_percent", default=0.1, min=0.0, max=1.0, step=0.01, optional=True),
                    io.Float.Input("end_percent", default=0.9, min=0.0, max=1.0, step=0.01, optional=True)],
            outputs=[io.Sampler.Output()])

    @classmethod
    async def execute(cls, sampler, dishonesty_factor, start_percent=0.0, end_percent=1.0):
        return io.NodeOutput(await eng.sampler.custom(lying_sigma_sampler, {
            "lss_wrapped_sampler": sampler,
            "lss_dishonesty_factor": dishonesty_factor,
            "lss_startend_percent": (start_percent, end_percent),
        }, model_sampling=True))
