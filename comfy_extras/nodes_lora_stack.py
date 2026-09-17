from typing_extensions import override

import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io


def _load_lora_file(lora_name: str):
    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
    return comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)


def _lora_template() -> list[io.Input]:
    return [
        io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras"),
                       tooltip="The name of the LoRA file to apply."),
        io.Float.Input("strength", default=1.0, min=-100.0, max=100.0, step=0.01,
                       tooltip="How strongly to apply this LoRA. 0 = off, negative inverts the effect."),
    ]


class LoadLoraModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadLoraModel",
            display_name="Load LoRA (Model)",
            search_aliases=["lora", "load lora", "apply lora", "lora model", "lora stack"],
            category="model/loaders",
            description="Apply a stack of LoRAs to a diffusion model. Add one row per LoRA; "
                        "each row picks a LoRA file and its strength.",
            inputs=[
                io.Model.Input("model", tooltip="The diffusion model the LoRAs will be applied to."),
                io.DynamicGroup.Input(
                    "loras",
                    template=_lora_template(),
                    min=1,
                    max=50,
                    tooltip="Each row applies one LoRA to the model.",
                    group_name="LoRA",
                ),
            ],
            outputs=[io.Model.Output(tooltip="The modified diffusion model.")],
        )

    @classmethod
    def execute(cls, model, loras: list[dict]) -> io.NodeOutput:
        for row in loras:
            lora_name = row.get("lora_name")
            strength = row["strength"]
            if not lora_name or strength == 0:
                continue
            lora, metadata = _load_lora_file(lora_name)
            model, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0, lora_metadata=metadata)
        return io.NodeOutput(model)


class LoadLoraTextEncoder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadLoraTextEncoder",
            display_name="Load LoRA (Text Encoder)",
            search_aliases=["lora", "load lora", "apply lora", "clip lora", "lora stack"],
            category="model/loaders",
            description="Apply a stack of LoRAs to a CLIP text encoder. Add one row per LoRA; "
                        "each row picks a LoRA file and its strength.",
            inputs=[
                io.Clip.Input("clip", tooltip="The CLIP text encoder the LoRAs will be applied to."),
                io.DynamicGroup.Input(
                    "loras",
                    template=_lora_template(),
                    min=1,
                    max=50,
                    tooltip="Each row applies one LoRA to the text encoder.",
                    group_name="LoRA",
                ),
            ],
            outputs=[io.Clip.Output(tooltip="The modified CLIP text encoder.")],
        )

    @classmethod
    def execute(cls, clip, loras: list[dict]) -> io.NodeOutput:
        for row in loras:
            lora_name = row.get("lora_name")
            strength = row["strength"]
            if not lora_name or strength == 0:
                continue
            lora, metadata = _load_lora_file(lora_name)
            _, clip = comfy.sd.load_lora_for_models(None, clip, lora, 0, strength, lora_metadata=metadata)
        return io.NodeOutput(clip)


class LoraStackExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LoadLoraModel,
            LoadLoraTextEncoder,
        ]


async def comfy_entrypoint() -> LoraStackExtension:
    return LoraStackExtension()
