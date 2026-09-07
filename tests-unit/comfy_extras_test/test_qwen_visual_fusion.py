import pytest
import torch

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
if not torch.cuda.is_available():
    cli_args.cpu = True

try:
    from comfy.text_encoders.krea2 import KREA2_TEMPLATE, Krea2Tokenizer
    from comfy_extras.nodes_cond import CLIPTextEncodeImageFusion, _flatten_images, _fuse_conditionings, _resize_visual_tokens, _spatial_fusion_mask, _visual_grid, _visual_token_span
finally:
    cli_args.cpu = prior_cpu


def _tokens(image_position=1, suffix=1):
    pairs = [(1, 1.0)] * image_position
    pairs.append(({"type": "image", "data": torch.zeros(1, 32, 32, 3)}, 1.0))
    pairs.extend([(2, 1.0)] * suffix)
    return {"qwen3vl_4b": [pairs]}


def test_checkerboard_mask_multiple_sources():
    mask = _spatial_fusion_mask(2, 3, 3, "spatial-checkerboard", 2, 0.5, "cpu")
    assert mask.tolist() == [0, 1, 2, 1, 2, 0]


def test_block_interleave_mask():
    mask = _spatial_fusion_mask(4, 4, 2, "spatial-block-interleave", 2, 0.5, "cpu")
    assert mask.reshape(4, 4).tolist() == [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
    ]


def test_dither_mask_honors_seed_and_two_source_ratio():
    first = _spatial_fusion_mask(4, 4, 2, "spatial-dither-random", 2, 0.5, "cpu", 7)
    second = _spatial_fusion_mask(4, 4, 2, "spatial-dither-random", 2, 0.5, "cpu", 7)
    changed = _spatial_fusion_mask(4, 4, 2, "spatial-dither-random", 2, 0.5, "cpu", 8)
    assert torch.equal(first, second)
    assert not torch.equal(first, changed)
    assert _spatial_fusion_mask(2, 2, 2, "spatial-dither-random", 2, 1.0, "cpu").tolist() == [0, 0, 0, 0]
    assert _spatial_fusion_mask(2, 2, 2, "spatial-dither-random", 2, 0.0, "cpu").tolist() == [1, 1, 1, 1]


def test_dither_ratio_selects_first_source_or_remaining_checkerboard():
    assert _spatial_fusion_mask(2, 3, 4, "spatial-dither-random", 2, 1.0, "cpu").tolist() == [0, 0, 0, 0, 0, 0]
    assert _spatial_fusion_mask(2, 3, 4, "spatial-dither-random", 2, 0.0, "cpu").tolist() == [1, 2, 3, 2, 3, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_dither_mask_is_seeded_on_cuda():
    first = _spatial_fusion_mask(4, 4, 3, "spatial-dither-random", 2, 0.5, "cuda", 7)
    second = _spatial_fusion_mask(4, 4, 3, "spatial-dither-random", 2, 0.5, "cuda", 7)
    assert torch.equal(first, second)


def test_visual_span_accounts_for_stripped_prefix():
    tokens = _tokens(image_position=3, suffix=4)
    assert _visual_token_span(tokens, cond_length=9, visual_tokens=4) == (1, 5)


def test_krea_conditioning_template_preserves_visual_tokens():
    tokenizer = Krea2Tokenizer()
    image = torch.zeros(1, 64, 64, 3)

    for image_count in (1, 2, 3):
        tokens = tokenizer.tokenize_with_weights("test prompt", images=[image] * image_count)
        token_pairs = tokens["qwen3vl_4b"][0]
        image_positions = [i for i, pair in enumerate(token_pairs) if isinstance(pair[0], dict)]
        im_start_positions = [i for i, pair in enumerate(token_pairs) if pair[0] == 151644]
        template_end = im_start_positions[1] + 3

        assert len(image_positions) == image_count
        assert token_pairs[im_start_positions[1] + 1][0] == 872
        assert token_pairs[im_start_positions[1] + 2][0] == 198
        assert template_end <= image_positions[0]

        if image_count == 1:
            cond_length = len(token_pairs) - 1 + 4 - template_end
            assert _visual_token_span(tokens, cond_length, 4) == (image_positions[0] - template_end, image_positions[0] - template_end + 4)


def test_krea_preformatted_and_text_generation_templates_are_unchanged():
    tokenizer = Krea2Tokenizer()
    image = torch.zeros(1, 64, 64, 3)
    image_prompt = "<|vision_start|><|image_pad|><|vision_end|>test prompt"

    preformatted = tokenizer.tokenize_with_weights(KREA2_TEMPLATE.format(image_prompt), images=[image])["qwen3vl_4b"][0]
    generated = tokenizer.tokenize_with_weights("test prompt", image=image, thinking=False)["qwen3vl_4b"][0]

    assert [pair[0] for pair in preformatted[:3]] == [151644, 8948, 198]
    assert [pair[0] for pair in generated[:3]] == [151644, 872, 198]


def test_krea_tokenization_runs_through_image_fusion_node():
    class FakeKreaClip:
        def __init__(self):
            self.tokenizer = Krea2Tokenizer()

        def tokenize(self, text, images):
            return self.tokenizer.tokenize_with_weights(text, images=images)

        def encode_from_tokens_scheduled(self, tokens):
            token_pairs = tokens["qwen3vl_4b"][0]
            im_start_positions = [i for i, pair in enumerate(token_pairs) if pair[0] == 151644]
            values = []
            for pair in token_pairs[im_start_positions[1] + 3:]:
                if isinstance(pair[0], dict):
                    values.extend([pair[0]["data"].mean()] * 4)
                else:
                    values.append(torch.tensor(0.0))
            return [[torch.stack(values).reshape(1, -1, 1), {}]]

    images = {
        "image_1": torch.zeros(1, 64, 64, 3),
        "image_2": torch.ones(1, 64, 64, 3),
    }

    conditioning = CLIPTextEncodeImageFusion.execute(FakeKreaClip(), "test prompt", images, "spatial-checkerboard").args[0]

    assert set(conditioning[0][0][:, 1:5].flatten().tolist()) == {0.0, 1.0}


def test_fusion_replaces_only_visual_tokens_and_preserves_dtype_and_metadata():
    tokens = [_tokens(), _tokens()]
    first = torch.tensor([[[10], [10], [10], [10], [10], [20]]], dtype=torch.float16)
    second = torch.tensor([[[30], [30], [30], [30], [30], [40]]], dtype=torch.float16)
    metadata = {"pooled_output": torch.tensor([1.0]), "marker": "first"}
    conditionings = [
        [[first, metadata]],
        [[second, {"pooled_output": torch.tensor([2.0])}]],
    ]

    fused = _fuse_conditionings(conditionings, tokens, [(2, 2), (2, 2)], "spatial-checkerboard", 2, 0.5)
    output, output_metadata = fused[0]

    assert output.dtype == torch.float16
    assert output.flatten().tolist() == [10, 10, 30, 30, 10, 20]
    assert output_metadata == metadata
    assert output_metadata is not metadata


def test_dither_seed_changes_fused_conditioning():
    tokens = [_tokens(), _tokens()]
    conditionings = [
        [[torch.zeros((1, 6, 1)), {}]],
        [[torch.ones((1, 6, 1)), {}]],
    ]

    first = _fuse_conditionings(conditionings, tokens, [(2, 2), (2, 2)], "spatial-dither-random", 2, 0.5, 7)[0][0]
    second = _fuse_conditionings(conditionings, tokens, [(2, 2), (2, 2)], "spatial-dither-random", 2, 0.5, 8)[0][0]

    assert not torch.equal(first, second)


def test_flatten_images_uses_numeric_input_order_and_splits_batches():
    images = {
        "image_10": torch.full((1, 2, 2, 3), 10.0),
        "image_2": torch.stack([torch.full((2, 2, 3), 2.0), torch.full((2, 2, 3), 3.0)]),
        "image_1": torch.full((1, 2, 2, 3), 1.0),
    }

    sources = _flatten_images(images)
    assert [source[0, 0, 0, 0].item() for source in sources] == [1.0, 2.0, 3.0, 10.0]
    images["image_2"][0, 0, 0, 0] = 99.0
    assert sources[1][0, 0, 0, 0].item() == 2.0


def test_visual_tokens_interpolate_in_two_dimensions_and_restore_dtype():
    visual = torch.tensor([[[0.0], [2.0]]], dtype=torch.float16)

    resized = _resize_visual_tokens(visual, (2, 1), (2, 2))

    assert resized.dtype == torch.float16
    assert resized.flatten().tolist() == [0.0, 0.0, 2.0, 2.0]


def test_fusion_interpolates_to_first_image_grid():
    first = torch.zeros((1, 6, 1), dtype=torch.float16)
    second = torch.tensor([[[0.0], [0.0], [2.0], [0.0]]], dtype=torch.float16)
    conditionings = [[[first, {}]], [[second, {}]]]

    fused = _fuse_conditionings(conditionings, [_tokens(), _tokens()], [(2, 2), (2, 1)], "spatial-dither-random", 2, 0.0)[0][0]

    assert fused.shape == first.shape
    assert fused.flatten().tolist() == [0.0, 0.0, 0.0, 2.0, 2.0, 0.0]


def test_fusion_rejects_different_text_layouts():
    conditionings = [
        [[torch.zeros((1, 6, 1)), {}]],
        [[torch.zeros((1, 7, 1)), {}]],
    ]

    with pytest.raises(ValueError, match="different text token layouts"):
        _fuse_conditionings(conditionings, [_tokens(), _tokens(suffix=2)], [(2, 2), (2, 2)], "spatial-checkerboard", 2, 0.5)


def test_node_preserves_images_uses_tokenizer_template_and_returns_fused_conditioning():
    seen_shapes = []

    class FakeClip:
        def tokenize(self, text, images):
            assert text == "test prompt"
            seen_shapes.append(images[0].shape)
            pairs = [
                (1, 1.0),
                ({"type": "image", "data": images[0]}, 1.0),
                (2, 1.0),
            ]
            return {"qwen3vl_4b": [pairs]}

        def encode_from_tokens_scheduled(self, tokens):
            image = next(pair[0]["data"] for pair in tokens["qwen3vl_4b"][0] if isinstance(pair[0], dict))
            value = image.mean()
            height, width = _visual_grid(image)
            return [[torch.full((1, height * width + 2, 1), value, dtype=torch.float16), {"source": float(value)}]]

    images = {"image_1": torch.zeros(1, 32, 64, 4), "image_2": torch.ones(1, 64, 32, 4)}
    result = CLIPTextEncodeImageFusion.execute(
        FakeClip(),
        "test prompt",
        images,
        "spatial-dither-random",
        seed=7,
    )
    changed_seed = CLIPTextEncodeImageFusion.execute(
        FakeClip(),
        "test prompt",
        images,
        "spatial-dither-random",
        seed=8,
    )
    conditioning = result.args[0]
    output, metadata = conditioning[0]

    assert seen_shapes == [torch.Size([1, 32, 64, 3]), torch.Size([1, 64, 32, 3])] * 2
    assert output.dtype == torch.float16
    assert output.shape == (1, 8, 1)
    assert output[:, 0].item() == 0.0
    assert output[:, -1].item() == 0.0
    assert set(output[:, 1:-1].flatten().tolist()) == {0.0, 1.0}
    assert metadata == {"source": 0.0}
    assert not torch.equal(output, changed_seed.args[0][0][0])


def test_node_exposes_generic_interface_without_vae():
    schema = CLIPTextEncodeImageFusion.define_schema()
    inputs = {value.id: value for value in schema.inputs}
    assert schema.node_id == "CLIPTextEncodeImageFusion"
    assert schema.category == "model/conditioning"
    assert "text" in inputs
    assert "vae" not in inputs
    assert inputs["seed"].default == 0
    assert inputs["seed"].control_after_generate is True
