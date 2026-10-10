import unittest
from unittest.mock import patch

import torch

import comfy.sd


class TestMiniMaxH3AudioVAECrop(unittest.TestCase):
    def test_h3_audio_vae_preserves_unaligned_input_length(self):
        device = torch.device("cpu")
        state_dict = {
            "pre_block.attn.zero_k_bias": torch.zeros(1),
        }

        class DummyPatcher:
            def __init__(self, *args, **kwargs):
                pass

            def is_dynamic(self):
                return False

        with (
            patch(
                "comfy.sd.comfy.ldm.minimax.audio_vae.MiniMaxH3AudioVAE",
                return_value=torch.nn.Identity(),
            ),
            patch(
                "comfy.sd.comfy.model_patcher.CoreModelPatcher",
                DummyPatcher,
            ),
            patch(
                "comfy.sd.model_management.vae_offload_device",
                return_value=device,
            ),
            patch(
                "comfy.sd.model_management.intermediate_device",
                return_value=device,
            ),
        ):
            vae = comfy.sd.VAE(
                sd=state_dict,
                device=device,
                dtype=torch.float32,
            )

        self.assertFalse(vae.crop_input)

        audio = torch.zeros((1, 437333, 2))
        prepared = vae.vae_encode_crop_pixels(audio)

        self.assertEqual(prepared.shape, audio.shape)


if __name__ == "__main__":
    unittest.main()
