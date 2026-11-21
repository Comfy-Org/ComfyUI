import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch


def test_text_detection_uses_presence_score():
    source = Path(__file__).resolve().parents[2] / "comfy_extras/nodes_sam3.py"

    class Detector:
        presence = -8.0

        def __call__(self, frame, **kwargs):
            return {
                "boxes": torch.tensor([[[0., 0., 2., 2.]]]),
                "scores": torch.tensor([[4.]]),
                "masks": torch.ones(1, 1, 2, 2),
                "presence": torch.tensor([[self.presence]]),
            }

    detector = Detector()
    model = SimpleNamespace(model=SimpleNamespace(get_dtype=lambda: torch.float32, diffusion_model=detector))
    management = SimpleNamespace(load_model_gpu=lambda model: None, get_torch_device=lambda: "cpu", intermediate_device=lambda: "cpu")
    comfy = SimpleNamespace(model_management=management, utils=SimpleNamespace(
        common_upscale=lambda image, *args, **kwargs: image,
        ProgressBar=lambda size: SimpleNamespace(update=lambda value: None),
    ))
    io = SimpleNamespace(ComfyNode=object, Custom=lambda name: object(), NodeOutput=lambda *values: values)
    comfy_module = ModuleType("comfy")
    comfy_module.model_management = management
    comfy_module.utils = comfy.utils
    comfy_api = ModuleType("comfy_api")
    latest = ModuleType("comfy_api.latest")
    latest.ComfyExtension = object
    latest.io = io
    latest.ui = SimpleNamespace()
    modules = {
        "comfy": comfy_module,
        "comfy.model_management": management,
        "comfy.utils": comfy.utils,
        "comfy_api": comfy_api,
        "comfy_api.latest": latest,
        "folder_paths": ModuleType("folder_paths"),
        "av": ModuleType("av"),
    }
    spec = importlib.util.spec_from_file_location("nodes_sam3_presence_test_module", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
        module._extract_text_prompts = lambda conditioning, device, dtype: [(torch.empty(1), None, 1)]
        module._refine_mask = lambda model, image, mask, box, h, w, device, dtype, iterations: (mask > 0).float().unsqueeze(0)
        image = torch.zeros(1, 2, 2, 3)

        masks, boxes = module.SAM3_Detect.execute(model, image, conditioning=[object()], refine_iterations=0)
        assert masks.count_nonzero() == 0
        assert boxes == [[]]

        detector.presence = 8.0
        masks, boxes = module.SAM3_Detect.execute(model, image, conditioning=[object()], refine_iterations=0)
        assert masks.count_nonzero() == 4
        assert len(boxes[0]) == 1
        torch.testing.assert_close(
            torch.tensor(boxes[0][0]["score"]),
            torch.sigmoid(torch.tensor(4.0)) * torch.sigmoid(torch.tensor(8.0)),
        )
