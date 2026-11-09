from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.text_encoders.qwen3vl import Qwen3VLClipModel


@pytest.mark.parametrize("penalize_prompt", [False, True])
def test_generation_preserves_prompt_penalties_and_beam_options(penalize_prompt):
    calls = []
    embeds = torch.ones(1, 3, 4)
    output = torch.tensor([[12, 34, 56]])

    class Transformer:
        def build_image_inputs(self, values, info):
            return "positions", "visual-mask", "deepstack"

        def generate(self, *positional, **options):
            calls.append(options)
            return output

    model = SimpleNamespace(
        process_tokens=lambda tokens, device: (embeds, None, None, {}),
        execution_device=torch.device("cpu"),
        transformer=Transformer(),
        penalize_prompt=penalize_prompt,
    )
    tokens = {"qwen3vl_8b": [[(12, 1.0), ({"type": "image"}, 1.0), (34, 1.0)]]}
    actual = Qwen3VLClipModel.generate(
        model, tokens, do_sample=False, max_length=4, temperature=1.0,
        top_k=1, top_p=1.0, min_p=0.0, repetition_penalty=1.2, seed=7,
        presence_penalty=0.25, num_beams=3,
    )

    assert actual is output
    assert len(calls) == 1
    assert calls[0]["initial_tokens"] == ([12, 34] if penalize_prompt else [])
    assert calls[0]["num_beams"] == 3
    assert calls[0]["presence_penalty"] == 0.25
    assert calls[0]["position_ids"] == "positions"
    assert calls[0]["visual_pos_masks"] == "visual-mask"
    assert calls[0]["deepstack_embeds"] == "deepstack"
