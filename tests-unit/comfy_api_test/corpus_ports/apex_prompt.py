"""ComfyUI-Apex-Artist's ApexPromptPreset (apex_prompt.py), ported to the V2 API.

The original keeps its presets in ``prompt_presets.json`` next to its own code, written on first use
and read every run. The port keeps them in ``eng.store`` (scope "pack"), seeded the same way. The
preset logic is the original's, unchanged: it's taken from the corpus copy, not rewritten.
"""
import json
import os
import random
import re
import threading
from typing import Any, Dict, List, Optional

from comfy_api.index import eng, io

from ._corpus import corpus_file, original  # a submodule: a pack process doesn't run __init__

_Apex = original(corpus_file("comfyui-apex-artist", "apex_prompt.py"), ["ApexPromptPreset"],
                 {"os": os, "json": json, "random": random, "re": re, "threading": threading, "Dict": Dict,
                  "List": List, "Optional": Optional, "Any": Any, "__file__": os.devnull})["ApexPromptPreset"]
CATEGORIES = ("Apex Environment", "Apex Lighting", "Apex Style", "Apex Camera Lens")


def _choices(category):
    return ["Disabled", "Random"] + _Apex.get_all_presets_in_category(category)


class ApexPromptPresetV2(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ApexPromptPreset", display_name="Apex Prompt", category="Apex Artist/Text",
            inputs=[io.String.Input("input_text", multiline=True, default=""),
                    io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                    io.Combo.Input("environment_preset", options=_choices("Apex Environment"), default="Disabled", optional=True),
                    io.Combo.Input("lighting_preset", options=_choices("Apex Lighting"), default="Disabled", optional=True),
                    io.Combo.Input("style_preset", options=_choices("Apex Style"), default="Disabled", optional=True),
                    io.Combo.Input("camera_lens_preset", options=_choices("Apex Camera Lens"), default="Disabled", optional=True)],
            outputs=[io.String.Output(display_name=n) for n in
                     ("combined_prompt", "environment_text", "lighting_text", "style_text", "camera_lens_text")])

    @classmethod
    async def execute(cls, input_text, seed, environment_preset="Disabled", lighting_preset="Disabled",
                      style_preset="Disabled", camera_lens_preset="Disabled"):
        presets = await eng.store.get("presets")
        if presets is None:  # first use: the defaults, saved, as the original writes its file
            presets = _Apex.get_default_presets()
            await eng.store.set("presets", presets)
        apex = _Apex.__new__(_Apex)
        apex.presets = presets
        return io.NodeOutput(*apex.combine_prompts(input_text, seed, environment_preset, lighting_preset,
                                                   style_preset, camera_lens_preset))
