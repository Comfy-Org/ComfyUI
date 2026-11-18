import json
import logging
import re
from decimal import Decimal

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


PAN2_CAPTION_KEYS = ["video_style", "lighting", "perspective_view", "subjects_lifecycle", "init_frame_visual_description", "events", "quality"]
PAN2_CAMERA_MOTIONS = {"static", "pan_left", "pan_right", "tilt_up", "tilt_down", "track_left", "track_right", "dolly_forward", "dolly_backward",
                       "dolly_up", "dolly_down", "zoom_in", "zoom_out", "roll_clockwise", "roll_counterclockwise", "unstable"}
PAN2_SUBJECT_ID = re.compile(r"<\|Subject[1-9][0-9]*\|>")
PAN2_MAX_CAPTION_TOKENS = 1536  # caption tokens the PAN 2 reference pipeline encodes


def pan2_interval(text):
    if not isinstance(text, str) or not re.fullmatch(r"\d+(\.\d)? to \d+(\.\d)?", text):
        raise ValueError(f"bad time range {text!r}")
    start, end = (Decimal(t) for t in text.split(" to "))
    if end <= start:
        raise ValueError(f"time range {text!r} is empty")
    return start, end


def pan2_keys(obj, keys, what):
    if not isinstance(obj, dict) or list(obj) != keys:
        raise ValueError(f"{what} needs the keys {keys} in that order")


def pan2_check_caption(caption, duration):
    # the structure PAN 2's caption checker (prompt_tools.py in the PAN 2 repo) accepts
    try:
        c = json.loads(caption)
    except json.JSONDecodeError as e:
        raise ValueError(f"caption is not valid JSON ({e})")
    pan2_keys(c, PAN2_CAPTION_KEYS, "caption")
    subjects = set()
    for i, s in enumerate(c["subjects_lifecycle"], 1):
        pan2_keys(s, ["subject_id", "display_name", "subject_attribute"], "subject")
        if s["subject_id"] != f"<|Subject{i}|>":
            raise ValueError("subject ids must be <|Subject1|>, <|Subject2|>, ... in order")
        subjects.add(s["subject_id"])
    for k in ("video_style", "lighting", "perspective_view", "init_frame_visual_description"):
        if not isinstance(c[k], str) or not c[k].strip() or PAN2_SUBJECT_ID.search(c[k]):
            raise ValueError(f"{k} must be text without subject ids")
    t = Decimal(0)
    for i, e in enumerate(c["events"]):
        pan2_keys(e, ["event_index", "time_range_sec", "event_description", "camera_motion", "contained_actions"], f"event {i}")
        start, end = pan2_interval(e["time_range_sec"])
        if e["event_index"] != i or start != t:
            raise ValueError(f"event {i} ({e['time_range_sec']}) does not continue from {t} s")
        if PAN2_SUBJECT_ID.search(e["event_description"]):
            raise ValueError(f"event {i}: subject ids belong only in contained_actions")
        cam_t = start
        for cam in e["camera_motion"]:
            pan2_keys(cam, ["time_range_sec", "camera_motion"], f"event {i} camera_motion entry")
            cam_start, cam_t_next = pan2_interval(cam["time_range_sec"])
            moves = cam["camera_motion"]
            if cam_start != cam_t or not moves or not set(moves) <= PAN2_CAMERA_MOTIONS or len(set(moves)) != len(moves) or ("static" in moves and len(moves) > 1):
                raise ValueError(f"event {i}: bad camera_motion {cam['time_range_sec']} {moves}")
            cam_t = cam_t_next
        if cam_t != end:
            raise ValueError(f"event {i}: camera_motion does not cover the event")
        for a in e["contained_actions"]:
            pan2_keys(a, ["time_range_sec", "action"], f"event {i} action")
            a_start, a_end = pan2_interval(a["time_range_sec"])
            ids = PAN2_SUBJECT_ID.findall(a["action"])
            if not start <= a_start < a_end <= end or not ids or not a["action"].startswith(ids[0]) or not set(ids) <= subjects:
                raise ValueError(f"event {i}: action {a['time_range_sec']} must lie in the event and start with a declared subject id")
        t = end
    if t != duration:
        raise ValueError(f"events cover 0 to {t} s but the video is {duration} s")
    pan2_keys(c["quality"], ["video_quality", "text_logos_overlays", "artifacts"], "quality")
    if c["quality"]["artifacts"] is not None:
        raise ValueError("quality.artifacts must be null")


class PAN2PromptCheck(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PAN2PromptCheck",
            display_name="PAN 2 Prompt Check",
            category="text",
            description="Checks a PAN 2 caption from the prompt rewriter against PAN 2's caption format, the video length and the caption token limit.",
            inputs=[
                io.String.Input("caption", force_input=True),
                io.Int.Input("length", default=121, min=1, max=16384, step=4, tooltip="Video length in frames, the same as the latent's."),
                io.Combo.Input("on_problem", options=["report", "stop"], tooltip="report: output ok = False and continue. stop: raise an error."),
                io.Clip.Input("clip", optional=True, tooltip="The PAN 2 text encoder, to count the caption's tokens."),
            ],
            outputs=[
                io.String.Output(display_name="caption"),
                io.Boolean.Output(display_name="ok"),
                io.String.Output(display_name="report"),
            ],
            search_aliases=["pan2", "prompt check", "validate caption"],
        )

    @classmethod
    def execute(cls, caption, length, on_problem, clip=None) -> io.NodeOutput:
        duration = (Decimal(length - 1) / 24).quantize(Decimal("0.1"))
        report = {"duration_sec": float(duration), "problems": []}
        try:
            pan2_check_caption(caption, duration)
        except ValueError as e:
            report["problems"].append(str(e))
        if clip is not None:
            ids = [t[0] for t in next(iter(clip.tokenize(caption).values()))[0]]
            # the video model gets the tokens after <|im_start|>user\n, as in comfy/text_encoders/pan2.py
            start = next((i + 3 for i in range(len(ids) - 2) if ids[i:i + 3] == [248045, 846, 198]), 0)
            report["tokens"] = len(ids) - start
            if report["tokens"] > PAN2_MAX_CAPTION_TOKENS:
                report["problems"].append(f"caption is {report['tokens']} tokens, PAN 2 was trained on up to {PAN2_MAX_CAPTION_TOKENS}")
        ok = not report["problems"]
        if not ok:
            if on_problem == "stop":
                raise ValueError("PAN 2 Prompt Check: " + "; ".join(report["problems"]))
            logging.warning("PAN 2 Prompt Check: " + "; ".join(report["problems"]))
        return io.NodeOutput(caption, ok, json.dumps(report, indent=1))


class PAN2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PAN2PromptCheck]


async def comfy_entrypoint() -> PAN2Extension:
    return PAN2Extension()
