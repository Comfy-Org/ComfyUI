import torch

from comfy import supported_models


def test_sam3_process_clip_state_dict_without_unet_pass():
    config = supported_models.SAM3({"image_model": "SAM3"})
    sd = {
        "detector.backbone.language_backbone.encoder.token_embedding.weight": torch.zeros(2, 2),
        "detector.backbone.language_backbone.resizer.weight": torch.zeros(2, 2),
        "detector.backbone.vision_backbone.weight": torch.zeros(2, 2),
    }
    clip_sd = config.process_clip_state_dict(sd)
    assert any("token_embedding" in k for k in clip_sd)
    assert not any(k.startswith("resizer.") for k in clip_sd)
    assert "detector.backbone.vision_backbone.weight" in sd
