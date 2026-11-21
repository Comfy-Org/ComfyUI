import comfy.model_patcher
import comfy.samplers
import re
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class SkipLayerGuidanceDiT(io.ComfyNode):
    '''
    Enhance guidance towards detailed dtructure by having another set of CFG negative with skipped layers.
    Inspired by Perturbed Attention Guidance (https://arxiv.org/abs/2403.17377)
    Original experimental implementation for SD3 by Dango233@StabilityAI.
    '''

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SkipLayerGuidanceDiT",
            category="advanced/guidance",
            description="Generic version of SkipLayerGuidance node that can be used on every DiT model.",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.String.Input("double_layers", default="7, 8, 9", advanced=True),
                io.String.Input("single_layers", default="7, 8, 9", advanced=True),
                io.Float.Input("scale", default=3.0, min=0.0, max=10.0, step=0.1),
                io.Float.Input("start_percent", default=0.01, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Float.Input("end_percent", default=0.15, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Float.Input("rescaling_scale", default=0.0, min=0.0, max=10.0, step=0.01),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, scale, start_percent, end_percent, double_layers="", single_layers="", rescaling_scale=0) -> io.NodeOutput:
        # check if layer is comma separated integers
        def skip(args, extra_args):
            return args

        model_sampling = model.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(start_percent)
        sigma_end = model_sampling.percent_to_sigma(end_percent)

        double_layers = re.findall(r'\d+', double_layers)
        double_layers = [int(i) for i in double_layers]

        single_layers = re.findall(r'\d+', single_layers)
        single_layers = [int(i) for i in single_layers]

        if len(double_layers) == 0 and len(single_layers) == 0:
            return io.NodeOutput(model)

        def post_cfg_function(args):
            model = args["model"]
            cond_pred = args["cond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            x = args["input"]
            model_options = args["model_options"].copy()

            for layer in double_layers:
                model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, skip, "dit", "double_block", layer)

            for layer in single_layers:
                model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, skip, "dit", "single_block", layer)

            model_sampling.percent_to_sigma(start_percent)

            sigma_ = sigma[0].item()
            if scale > 0 and sigma_ > sigma_end and sigma_ <= sigma_start:
                (slg,) = comfy.samplers.calc_cond_batch(model, [cond], x, sigma, model_options)
                cfg_result = cfg_result + (cond_pred - slg) * scale
                if rescaling_scale != 0:
                    factor = cond_pred.std() / cfg_result.std()
                    factor = rescaling_scale * factor + (1 - rescaling_scale)
                    cfg_result *= factor

            return cfg_result

        m = model.clone()
        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return io.NodeOutput(m)

    skip_guidance = execute  # TODO: remove


class SkipLayerGuidanceDiTSimple(io.ComfyNode):
    '''
    Simple version of the SkipLayerGuidanceDiT node that only modifies the uncond pass.
    '''
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SkipLayerGuidanceDiTSimple",
            category="advanced/guidance",
            description="Simple version of the SkipLayerGuidanceDiT node that only modifies the uncond pass.",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.String.Input("double_layers", default="7, 8, 9", advanced=True),
                io.String.Input("single_layers", default="7, 8, 9", advanced=True),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.001, advanced=True),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, start_percent, end_percent, double_layers="", single_layers="") -> io.NodeOutput:
        def skip(args, extra_args):
            return args

        model_sampling = model.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(start_percent)
        sigma_end = model_sampling.percent_to_sigma(end_percent)

        double_layers = re.findall(r'\d+', double_layers)
        double_layers = [int(i) for i in double_layers]

        single_layers = re.findall(r'\d+', single_layers)
        single_layers = [int(i) for i in single_layers]

        if len(double_layers) == 0 and len(single_layers) == 0:
            return io.NodeOutput(model)

        def calc_cond_batch_function(args):
            x = args["input"]
            model = args["model"]
            conds = args["conds"]
            sigma = args["sigma"]

            model_options = args["model_options"]
            slg_model_options = model_options.copy()

            for layer in double_layers:
                slg_model_options = comfy.model_patcher.set_model_options_patch_replace(slg_model_options, skip, "dit", "double_block", layer)

            for layer in single_layers:
                slg_model_options = comfy.model_patcher.set_model_options_patch_replace(slg_model_options, skip, "dit", "single_block", layer)

            cond, uncond = conds
            sigma_ = sigma[0].item()
            if sigma_ > sigma_end and sigma_ <= sigma_start and uncond is not None:
                cond_out, _ = comfy.samplers.calc_cond_batch(model, [cond, None], x, sigma, model_options)
                _, uncond_out = comfy.samplers.calc_cond_batch(model, [None, uncond], x, sigma, slg_model_options)
                out = [cond_out, uncond_out]
            else:
                out = comfy.samplers.calc_cond_batch(model, conds, x, sigma, model_options)

            return out

        m = model.clone()
        m.set_model_sampler_calc_cond_batch_function(calc_cond_batch_function)

        return io.NodeOutput(m)

    skip_guidance = execute  # TODO: remove


class SkipLayerGuidanceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SkipLayerGuidanceDiT,
            SkipLayerGuidanceDiTSimple,
        ]


async def comfy_entrypoint() -> SkipLayerGuidanceExtension:
    return SkipLayerGuidanceExtension()
