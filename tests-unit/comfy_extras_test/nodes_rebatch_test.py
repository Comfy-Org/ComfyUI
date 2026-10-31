from unittest.mock import patch, MagicMock

import torch

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_server = MagicMock()

with patch.dict("sys.modules", {"nodes": mock_nodes, "server": mock_server}):
    from comfy_extras.nodes_rebatch import LatentRebatch


class TestLatentRebatchGetBatch:
    def test_default_mask_matches_pixel_resolution(self):
        # a latent without a noise_mask gets an all-ones mask at samples * 8
        latents = [{"samples": torch.zeros(1, 4, 16, 16)}]
        _, mask, _ = LatentRebatch.get_batch(latents, 0, 0)
        assert mask.shape == (1, 1, 128, 128)

    def test_matching_mask_is_kept(self):
        latents = [{"samples": torch.zeros(1, 4, 16, 16), "noise_mask": torch.ones(1, 1, 128, 128)}]
        _, mask, _ = LatentRebatch.get_batch(latents, 0, 0)
        assert mask.shape == (1, 1, 128, 128)

    def test_mismatched_mask_is_resized_to_pixel_resolution(self):
        # SetLatentNoiseMask stores masks without resizing, so a mask can arrive
        # at a resolution that does not match samples * 8 and must be scaled up.
        latents = [{"samples": torch.zeros(1, 4, 16, 16), "noise_mask": torch.ones(1, 1, 48, 48)}]
        _, mask, _ = LatentRebatch.get_batch(latents, 0, 0)
        assert mask.shape == (1, 1, 128, 128)

    def test_mismatched_height_only_is_resized(self):
        latents = [{"samples": torch.zeros(1, 4, 16, 16), "noise_mask": torch.ones(1, 1, 48, 128)}]
        _, mask, _ = LatentRebatch.get_batch(latents, 0, 0)
        assert mask.shape == (1, 1, 128, 128)

    def test_batches_with_mismatched_mask_can_be_concatenated(self):
        # a resized mask must line up with another latent's default mask so the
        # two batches can be concatenated instead of raising a size mismatch.
        with_mask = [{"samples": torch.zeros(1, 4, 16, 16), "noise_mask": torch.ones(1, 1, 48, 48)}]
        without_mask = [{"samples": torch.zeros(1, 4, 16, 16)}]
        batch_a = LatentRebatch.get_batch(with_mask, 0, 0)
        batch_b = LatentRebatch.get_batch(without_mask, 0, 1)
        samples, mask, _ = LatentRebatch.cat_batch(batch_a, batch_b)
        assert samples.shape == (2, 4, 16, 16)
        assert mask.shape == (2, 1, 128, 128)
