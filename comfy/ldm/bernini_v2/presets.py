"""Published Bernini v2 task defaults from the official demo."""

from __future__ import annotations

TASK_PRESETS = {
    "t2i": {
        "steps": 50,
        "planning_steps": 25,
        "vit_denoising_steps": 5,
        "max_media_size": 842,
        "omega_video": 1.0,
        "omega_image": 1.0,
        "omega_text": 4.0,
        "omega_target": 0.5,
        "omega_scale": 1.0,
    },
    "i2i": {
        "steps": 40,
        "planning_steps": 25,
        "vit_denoising_steps": 5,
        "max_media_size": 842,
        "omega_video": 1.25,
        "omega_image": 1.25,
        "omega_text": 4.0,
        "omega_target": 0.5,
        "omega_scale": 0.75,
    },
    "t2v": {
        "steps": 50,
        "planning_steps": 50,
        "vit_denoising_steps": 1,
        "max_media_size": 842,
        "omega_video": 1.0,
        "omega_image": 1.0,
        "omega_text": 5.0,
        "omega_target": 1.2,
        "omega_scale": 1.0,
    },
    "v2v": {
        "steps": 40,
        "planning_steps": 50,
        "vit_denoising_steps": 1,
        "max_media_size": 848,
        "omega_video": 1.25,
        "omega_image": 1.25,
        "omega_text": 4.0,
        "omega_target": 1.2,
        "omega_scale": 0.75,
    },
    "r2v": {
        "steps": 40,
        "planning_steps": 50,
        "vit_denoising_steps": 1,
        "max_media_size": 842,
        "omega_video": 1.0,
        "omega_image": 3.0,
        "omega_text": 4.5,
        "omega_target": 1.5,
        "omega_scale": 0.75,
    },
    "rv2v": {
        "steps": 40,
        "planning_steps": 50,
        "vit_denoising_steps": 1,
        "max_media_size": 848,
        "omega_video": 1.5,
        "omega_image": 3.0,
        "omega_text": 3.6,
        "omega_target": 1.5,
        "omega_scale": 0.5,
    },
}


def task_preset(task: str) -> dict[str, float | int]:
    try:
        return TASK_PRESETS[task].copy()
    except KeyError as error:
        raise ValueError(f"unsupported Bernini v2 task {task!r}") from error
