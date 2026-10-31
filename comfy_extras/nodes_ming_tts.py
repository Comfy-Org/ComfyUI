import json

from comfy.ldm.ming_tts.runtime import generate, speaker_embedding
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_audio import VAEEncodeAudio


class MingTTSSample(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MingTTSSample",
            display_name="Ming TTS Sample",
            category="audio/ming tts",
            description="Autoregressive speech sampling with the Ming flow model. Connect the result to VAE Decode Audio.",
            inputs=[
                io.Clip.Input("clip"),
                io.Model.Input("model"),
                io.String.Input("text", multiline=True, tooltip="Words to speak. For Tiny dialogue, start every line with speaker_1:, speaker_2:, etc."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("max_steps", default=1000, min=1, max=4096, tooltip="Total audio-patch limit for the whole output, including all dialogue turns."),
                io.Float.Input("cfg", default=2.0, min=0.0, max=100.0, step=0.1),
                io.Float.Input("temperature", default=0.0, min=0.0, max=100.0, step=0.1),
                io.String.Input("instruction", multiline=True, default="", optional=True, tooltip="Optional voice or delivery description. Uses each model's released instruction format automatically. Tiny instruction tasks omit the language line."),
                io.Combo.Input("language", options=["English", "Chinese", "German", "Spanish", "French", "Korean", "Japanese", "Auto"], default="Auto", optional=True, tooltip="Language hint for the supplied text; does not translate it. Tiny ignores this setting when instruction is nonempty, matching the reference. Auto lets Tiny infer Chinese or English and leaves the 0.5B language prompt unset."),
                io.Autogrow.Input("speaker", optional=True, template=io.Autogrow.TemplateNames(
                    input=io.Custom("MING_SPEAKER").Input("speaker"),
                    names=[f"speaker_{i}" for i in range(1, 101)], min=0,
                ), tooltip="Connect Ming Speaker Encode outputs in order. Multiple speakers require Tiny. VAE and reference text are optional."),
                io.Float.Input("sigma", default=0.25, min=0.0, max=100.0, step=0.01, optional=True, advanced=True),
                io.Boolean.Input("separate_turns", default=True, optional=True, tooltip="Generate speaker turns and blank-line-separated paragraphs in chunks, splitting long passages at chunk_length. Tiny voice descriptions reuse the first chunk's voice when no speaker is connected. max_steps still limits the whole output. Disable for joint generation; multiple speakers then need reference context."),
                io.Int.Input("chunk_length", default=60, min=1, max=4096, optional=True, advanced=True, tooltip="When separate_turns is on, target this many words or CJK characters per chunk. Prefer sentences, then clauses or word boundaries. Larger chunks give more continuous delivery; lower this if words are skipped. This is a text limit, not a duration in seconds."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, model, text, seed, max_steps, cfg, temperature, speaker: io.Autogrow.Type = None, language="Auto", **kwargs):
        speakers = []
        if speaker:
            names = [f"speaker_{i}" for i in range(1, len(speaker) + 1)]
            if set(speaker) != set(names):
                raise ValueError("Connect speakers consecutively, starting at speaker_1.")
            speakers = [speaker[name] for name in names]
        latents, _ = generate(
            clip, model, text, seed=seed, max_steps=max_steps, cfg=cfg, temperature=temperature,
            speakers=speakers, language="" if language == "Auto" else language, **kwargs,
        )
        return io.NodeOutput({"samples": latents, "type": "audio", "downscale_ratio_temporal": 3528})


class MingSpeakerEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MingSpeakerEncode",
            display_name="Ming Speaker Encode",
            category="audio/ming tts",
            inputs=[
                io.Clip.Input("clip"),
                io.Audio.Input("audio"),
                io.Vae.Input("vae", optional=True, tooltip="Optional. With reference_text, also encode the recording as audio context. Without both, encode voice identity only."),
                io.String.Input("reference_text", multiline=True, default="", optional=True, tooltip="Optional transcript of this recording. Used only when a VAE is connected; do not add speaker labels."),
            ],
            outputs=[io.Custom("MING_SPEAKER").Output()],
        )

    @classmethod
    def execute(cls, clip, audio, vae=None, reference_text=""):
        speaker = {"embedding": speaker_embedding(clip, audio)}
        if speaker["embedding"].shape != (1, 192):
            raise ValueError("Use one recording per Ming Speaker Encode node.")
        if vae is not None and reference_text.strip():
            speaker["reference_latents"] = VAEEncodeAudio.execute(vae, audio)[0]["samples"]
            speaker["reference_text"] = reference_text.strip()
        return io.NodeOutput(speaker)


class MingTTSInstruction(io.ComfyNode):
    SPEED = {"Slow": "慢速", "Medium": "中速", "Fast": "快速"}
    LEVEL = {"Low": "低", "Medium": "中", "High": "高"}
    EMOTION = {"Neutral": "中性", "Happy": "高兴", "Sad": "悲伤", "Angry": "愤怒", "Surprised": "惊讶", "Fearful": "恐惧", "Disgusted": "厌恶"}
    DIALECT = {"Cantonese": "广粤话", "Sichuan": "川渝话"}

    @classmethod
    def define_schema(cls):
        variants = []
        for variant in ("Tiny", "0.5B"):
            inputs = [io.DynamicCombo.Input("task", options=[
                io.DynamicCombo.Option("basic", [
                    io.Combo.Input("speed", options=list(cls.SPEED), default="Medium"),
                    io.Combo.Input("pitch", options=list(cls.LEVEL), default="Medium"),
                    io.Combo.Input("volume", options=list(cls.LEVEL), default="Medium"),
                ]),
                io.DynamicCombo.Option("emotion", [io.Combo.Input("emotion", options=list(cls.EMOTION), default="Neutral")]),
                io.DynamicCombo.Option("dialect", [io.Combo.Input("dialect", options=list(cls.DIALECT), default="Cantonese")]),
                io.DynamicCombo.Option("voice_design", [io.String.Input("style", default="A calm, clear voice.", multiline=True, tooltip="Describe the desired voice.")]),
                io.DynamicCombo.Option("ip", [io.String.Input("ip", default="US_Emma", tooltip="Reference voice or character name, such as US_Emma.")]),
                io.DynamicCombo.Option("custom", [
                    io.Combo.Input("speed", options=["Unspecified", *cls.SPEED], default="Unspecified"),
                    io.Combo.Input("pitch", options=["Unspecified", *cls.LEVEL], default="Unspecified"),
                    io.Combo.Input("volume", options=["Unspecified", *cls.LEVEL], default="Unspecified"),
                    io.Combo.Input("emotion", options=["Unspecified", *cls.EMOTION], default="Unspecified"),
                    io.Combo.Input("dialect", options=["Unspecified", *cls.DIALECT], default="Unspecified"),
                    io.String.Input("style", default="", multiline=True, tooltip="Describe the desired voice. Leave empty to keep this unspecified."),
                    io.String.Input("ip", default="", tooltip="Reference voice or character name. Leave empty to keep this unspecified."),
                ]),
            ], tooltip="Choose a reference task, or custom to combine controls.")]
            if variant == "0.5B":
                inputs.append(io.Combo.Input("instruction_schema", options=["Auto", "Chinese", "English"], default="Auto", advanced=True,
                                             tooltip="JSON field names, independent of the spoken language. Auto uses the reference defaults."))
            variants.append(io.DynamicCombo.Option(variant, inputs))
        return io.Schema(
            node_id="MingTTSInstruction",
            display_name="Ming TTS Instruction",
            category="audio/ming tts",
            description="Build the reference instruction JSON. Connect the output to Ming TTS Sample's instruction input.",
            inputs=[
                io.DynamicCombo.Input("model_variant", options=variants, tooltip="Match the checkpoint used by the sampler."),
            ],
            outputs=[io.String.Output(display_name="instruction")],
        )

    @classmethod
    def execute(cls, model_variant: io.DynamicCombo.Type):
        task = model_variant["task"]
        kind = task["task"]
        tiny = model_variant["model_variant"] == "Tiny"
        dialects = cls.DIALECT if tiny else {"Cantonese": "cantonese", "Sichuan": "chuanyu dialect"}
        if kind == "custom":
            choices = {"speed": cls.SPEED, "pitch": cls.LEVEL, "volume": cls.LEVEL, "emotion": cls.EMOTION, "dialect": dialects}
            controls = {key: None if task[key] == "Unspecified" else options[task[key]] for key, options in choices.items()}
            for key in ("style", "ip"):
                value = task[key]
                controls[key] = (value.strip() if tiny else value) if value.strip() else None
        elif kind == "basic":
            controls = {"speed": cls.SPEED[task["speed"]], "pitch": cls.LEVEL[task["pitch"]], "volume": cls.LEVEL[task["volume"]]}
        elif kind == "emotion":
            controls = {"emotion": cls.EMOTION[task["emotion"]]}
        elif kind == "dialect":
            controls = {"dialect": dialects[task["dialect"]]}
        elif kind == "voice_design":
            controls = {"style": task["style"].strip() if tiny else task["style"]}
        else:
            controls = {"ip": task["ip"].strip() if tiny else task["ip"]}
        # Tiny's emotion task uses the 0.5B field order; its other tasks do not.
        fields = ("id", "speaker", "dialect", "style", "speed", "pitch", "volume", "emotion")
        if tiny and kind != "emotion":
            fields = ("id", "speaker", "dialect", "emotion", "style", "speed", "pitch", "volume")
        labels = {"id": "序号", "speaker": "说话人", "dialect": "方言", "style": "风格", "speed": "语速", "pitch": "基频", "volume": "音量", "emotion": "情感"}
        schema = "Chinese" if tiny else model_variant["instruction_schema"]
        english = schema == "English" or (schema == "Auto" and kind == "dialect")
        values = {"id": 1, "speaker": "speaker_1", **controls}
        item = {key if english else labels[key]: values.get(key) for key in fields}
        item["BGM"] = dict.fromkeys(("Genre", "Mood", "Instrument", "Theme", "ENV", "SNR"))
        item["ip" if english else "IP"] = controls.get("ip")
        return io.NodeOutput(json.dumps({"audio_sequence": [item]}, ensure_ascii=False))


class MingTTSExtension(ComfyExtension):
    async def get_node_list(self):
        return [MingTTSSample, MingSpeakerEncode, MingTTSInstruction]


async def comfy_entrypoint():
    return MingTTSExtension()
