import torch

from comfy.cli_args import args

args.cpu = True

from comfy.ldm.minimax.model import PackedLayout
from comfy_extras import nodes_minimax_h3


class FakeClip:
    def tokenize(self, prompt, minimax_ref_items=None):
        self.prompt = prompt
        self.ref_items = minimax_ref_items
        return {}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.empty(1), {}]]


class FakeVAE:
    def __init__(self):
        self.images = []

    def encode(self, image):
        self.images.append(image)
        return torch.empty(1, 24, 1, 2, 2)


def test_text_encoder_only_reference_wraps_image():
    image = torch.rand(1, 32, 32, 3)

    reference = nodes_minimax_h3.MiniMaxH3TextEncoderOnlyReference.execute(image).result[0]

    assert list(reference) == ["image"]
    assert reference["image"] is image


def test_reference_input_accepts_images_and_marked_references():
    schema = nodes_minimax_h3.MiniMaxH3ReferenceToVideo.define_schema()
    schema.finalize()
    schema.validate()
    ref_images = schema.get_v1_info(nodes_minimax_h3.MiniMaxH3ReferenceToVideo).input["optional"]["ref_images"]

    assert ref_images[1]["template"]["input"]["required"]["ref_image"][0] == "IMAGE,MINIMAX_H3_REFERENCE_IMAGE"


def test_marked_reference_skips_vae_but_still_reaches_text_encoder(monkeypatch):
    monkeypatch.setattr(nodes_minimax_h3, "_empty_av_latent", lambda width, height, length: ({"samples": torch.empty(0)}, 5))
    monkeypatch.setattr(nodes_minimax_h3, "_resize", lambda image, width, height, crop: image)
    plain_image = torch.rand(1, 32, 32, 3)
    marked_image = torch.rand(1, 32, 32, 3)
    marked_reference = nodes_minimax_h3.MiniMaxH3TextEncoderOnlyReference.execute(marked_image).result[0]
    clip = FakeClip()
    vae = FakeVAE()

    output = nodes_minimax_h3.MiniMaxH3ReferenceToVideo.execute(
        clip=clip,
        vae=vae,
        audio_vae=None,
        prompt="test",
        width=32,
        height=32,
        length=5,
        ref_images={"ref_image_0": marked_reference, "ref_image_1": plain_image},
    )

    assert clip.prompt == "test"
    assert len(clip.ref_items) == 2
    assert torch.equal(clip.ref_items[0]["data"], marked_image)
    assert torch.equal(clip.ref_items[1]["data"], plain_image)
    assert len(vae.images) == 1
    assert torch.equal(vae.images[0], plain_image)
    refs = output.result[0][0][1]["minimax_refs"]
    assert len(refs) == 1
    assert refs[0]["kind"] == "image"
    assert refs[0]["picture_index"] == 1


def test_vae_reference_keeps_position_after_marked_reference():
    refs = [{"kind": "image", "picture_index": 1, "latent_h": 2, "latent_w": 2}]

    layout = PackedLayout(text_len=8, latent_t=1, latent_h=2, latent_w=2, audio_t=1, refs=refs)

    ref_start = next(start for start, _, kind in layout.segments if kind == "ref_img")
    assert layout.position_ids[ref_start, 0] == 9


def test_no_vae_keeps_all_references_text_encoder_only(monkeypatch):
    monkeypatch.setattr(nodes_minimax_h3, "_empty_av_latent", lambda width, height, length: ({"samples": torch.empty(0)}, 5))
    monkeypatch.setattr(nodes_minimax_h3, "_resize", lambda image, width, height, crop: image)
    plain_image = torch.rand(1, 32, 32, 3)
    marked_image = torch.rand(1, 32, 32, 3)
    marked_reference = nodes_minimax_h3.MiniMaxH3TextEncoderOnlyReference.execute(marked_image).result[0]
    clip = FakeClip()

    output = nodes_minimax_h3.MiniMaxH3ReferenceToVideo.execute(
        clip=clip,
        vae=None,
        audio_vae=None,
        prompt="test",
        width=32,
        height=32,
        length=5,
        ref_images={"ref_image_0": plain_image, "ref_image_1": marked_reference},
    )

    assert len(clip.ref_items) == 2
    assert "minimax_refs" not in output.result[0][0][1]
