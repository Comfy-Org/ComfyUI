import comfy.utils
import folder_paths
import torch
import logging
from comfy_api.latest import IO, ComfyExtension
from typing_extensions import override


def load_hypernetwork_patch(path, strength):
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    activation_func = sd.get('activation_func', 'linear')
    is_layer_norm = sd.get('is_layer_norm', False)
    use_dropout = sd.get('use_dropout', False)
    activate_output = sd.get('activate_output', False)
    last_layer_dropout = sd.get('last_layer_dropout', False)

    valid_activation = {
        "linear": torch.nn.Identity,
        "relu": torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
        "swish": torch.nn.Hardswish,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
        "softsign": torch.nn.Softsign,
        "mish": torch.nn.Mish,
    }

    if activation_func not in valid_activation:
        logging.error("Unsupported Hypernetwork format, if you report it I might implement it. {}   {} {} {} {} {}".format(path, activation_func, is_layer_norm, use_dropout, activate_output, last_layer_dropout))
        return None

    out = {}

    for d in sd:
        try:
            dim = int(d)
        except:
            continue

        output = []
        for index in [0, 1]:
            attn_weights = sd[dim][index]
            keys = attn_weights.keys()

            linears = filter(lambda a: a.endswith(".weight"), keys)
            linears = list(map(lambda a: a[:-len(".weight")], linears))
            layers = []

            i = 0
            while i < len(linears):
                lin_name = linears[i]
                last_layer = (i == (len(linears) - 1))
                penultimate_layer = (i == (len(linears) - 2))

                lin_weight = attn_weights['{}.weight'.format(lin_name)]
                lin_bias = attn_weights['{}.bias'.format(lin_name)]
                layer = torch.nn.Linear(lin_weight.shape[1], lin_weight.shape[0])
                layer.load_state_dict({"weight": lin_weight, "bias": lin_bias})
                layers.append(layer)
                if activation_func != "linear":
                    if (not last_layer) or (activate_output):
                        layers.append(valid_activation[activation_func]())
                if is_layer_norm:
                    i += 1
                    ln_name = linears[i]
                    ln_weight = attn_weights['{}.weight'.format(ln_name)]
                    ln_bias = attn_weights['{}.bias'.format(ln_name)]
                    ln = torch.nn.LayerNorm(ln_weight.shape[0])
                    ln.load_state_dict({"weight": ln_weight, "bias": ln_bias})
                    layers.append(ln)
                if use_dropout:
                    if (not last_layer) and (not penultimate_layer or last_layer_dropout):
                        layers.append(torch.nn.Dropout(p=0.3))
                i += 1

            output.append(torch.nn.Sequential(*layers))
        out[dim] = torch.nn.ModuleList(output)

    class hypernetwork_patch:
        def __init__(self, hypernet, strength):
            self.hypernet = hypernet
            self.strength = strength
        def __call__(self, q, k, v, extra_options):
            dim = k.shape[-1]
            if dim in self.hypernet:
                hn = self.hypernet[dim]
                k = k + hn[0](k) * self.strength
                v = v + hn[1](v) * self.strength

            return q, k, v

        def to(self, device):
            for d in self.hypernet.keys():
                self.hypernet[d] = self.hypernet[d].to(device)
            return self

    return hypernetwork_patch(out, strength)

class HypernetworkLoader(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HypernetworkLoader",
            display_name="Load Hypernetwork",
            category="model/loaders",
            inputs=[
                IO.Model.Input("model"),
                IO.Combo.Input("hypernetwork_name", options=folder_paths.get_filename_list("hypernetworks")),
                IO.Float.Input("strength", default=1.0, min=-10.0, max=10.0, step=0.01),
            ],
            outputs=[
                IO.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, hypernetwork_name, strength) -> IO.NodeOutput:
        hypernetwork_path = folder_paths.get_full_path_or_raise("hypernetworks", hypernetwork_name)
        model_hypernetwork = model.clone()
        patch = load_hypernetwork_patch(hypernetwork_path, strength)
        if patch is not None:
            model_hypernetwork.set_model_attn1_patch(patch)
            model_hypernetwork.set_model_attn2_patch(patch)
        return IO.NodeOutput(model_hypernetwork)

    load_hypernetwork = execute  # TODO: remove


class HyperNetworkExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            HypernetworkLoader,
        ]


async def comfy_entrypoint() -> HyperNetworkExtension:
    return HyperNetworkExtension()
