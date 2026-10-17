import sys
import torch
from unittest.mock import MagicMock

# Mock nodes module to prevent CUDA initialization during import
mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

# Mock server module for PromptServer
mock_server = MagicMock()

# Stub only `nodes` and `server`, and put back only those two keys.
#
# `patch.dict("sys.modules", ...)` restores the mapping to its state on entry,
# which also *evicts* every module this import pulled in for the first time. A
# later import of one of those re-initialises an already-loaded C extension --
# numpy raises "cannot load module more than once per process" -- aborting the
# whole pytest run rather than failing one test.
_MISSING = object()
_stubs = {"nodes": mock_nodes, "server": mock_server}
_originals = {name: sys.modules.get(name, _MISSING) for name in _stubs}
sys.modules.update(_stubs)
try:
    from comfy_extras.nodes_images import ImageStitch
finally:
    for _name, _original in _originals.items():
        if _original is _MISSING:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _original


class TestImageStitch:

    def create_test_image(self, batch_size=1, height=64, width=64, channels=3):
        """Helper to create test images with specific dimensions"""
        return torch.rand(batch_size, height, width, channels)

    def test_no_image2_passthrough(self):
        """Test that when image2 is None, image1 is returned unchanged"""
        node = ImageStitch()
        image1 = self.create_test_image()

        result = node.stitch(image1, "right", True, 0, "white", image2=None)

        assert len(result.result) == 1
        assert torch.equal(result[0], image1)

    def test_basic_horizontal_stitch_right(self):
        """Test basic horizontal stitching to the right"""
        node = ImageStitch()
        image1 = self.create_test_image(height=32, width=32)
        image2 = self.create_test_image(height=32, width=24)

        result = node.stitch(image1, "right", False, 0, "white", image2)

        assert result[0].shape == (1, 32, 56, 3)  # 32 + 24 width

    def test_basic_horizontal_stitch_left(self):
        """Test basic horizontal stitching to the left"""
        node = ImageStitch()
        image1 = self.create_test_image(height=32, width=32)
        image2 = self.create_test_image(height=32, width=24)

        result = node.stitch(image1, "left", False, 0, "white", image2)

        assert result[0].shape == (1, 32, 56, 3)  # 24 + 32 width

    def test_basic_vertical_stitch_down(self):
        """Test basic vertical stitching downward"""
        node = ImageStitch()
        image1 = self.create_test_image(height=32, width=32)
        image2 = self.create_test_image(height=24, width=32)

        result = node.stitch(image1, "down", False, 0, "white", image2)

        assert result[0].shape == (1, 56, 32, 3)  # 32 + 24 height

    def test_basic_vertical_stitch_up(self):
        """Test basic vertical stitching upward"""
        node = ImageStitch()
        image1 = self.create_test_image(height=32, width=32)
        image2 = self.create_test_image(height=24, width=32)

        result = node.stitch(image1, "up", False, 0, "white", image2)

        assert result[0].shape == (1, 56, 32, 3)  # 24 + 32 height

    def test_size_matching_horizontal(self):
        """Test size matching for horizontal concatenation"""
        node = ImageStitch()
        image1 = self.create_test_image(height=64, width=64)
        image2 = self.create_test_image(height=32, width=32)  # Different aspect ratio

        result = node.stitch(image1, "right", True, 0, "white", image2)

        # image2 should be resized to match image1's height (64) with preserved aspect ratio
        expected_width = 64 + 64  # original + resized (32*64/32 = 64)
        assert result[0].shape == (1, 64, expected_width, 3)

    def test_size_matching_vertical(self):
        """Test size matching for vertical concatenation"""
        node = ImageStitch()
        image1 = self.create_test_image(height=64, width=64)
        image2 = self.create_test_image(height=32, width=32)

        result = node.stitch(image1, "down", True, 0, "white", image2)

        # image2 should be resized to match image1's width (64) with preserved aspect ratio
        expected_height = 64 + 64  # original + resized (32*64/32 = 64)
        assert result[0].shape == (1, expected_height, 64, 3)

    def test_padding_for_mismatched_heights_horizontal(self):
        """Test padding when heights don't match in horizontal concatenation"""
        node = ImageStitch()
        image1 = self.create_test_image(height=64, width=32)
        image2 = self.create_test_image(height=48, width=24)  # Shorter height

        result = node.stitch(image1, "right", False, 0, "white", image2)

        # Both images should be padded to height 64
        assert result[0].shape == (1, 64, 56, 3)  # 32 + 24 width, max(64,48) height

    def test_padding_for_mismatched_widths_vertical(self):
        """Test padding when widths don't match in vertical concatenation"""
        node = ImageStitch()
        image1 = self.create_test_image(height=32, width=64)
        image2 = self.create_test_image(height=24, width=48)  # Narrower width

        result = node.stitch(image1, "down", False, 0, "white", image2)

        # Both images should be padded to width 64
        assert result[0].shape == (1, 56, 64, 3)  # 32 + 24 height, max(64,48) width

    def test_spacing_horizontal(self):
        """Test spacing addition in horizontal concatenation"""
        node = ImageStitch()
        image1 = self.create_test_image(height=32, width=32)
        image2 = self.create_test_image(height=32, width=24)
        spacing_width = 16

        result = node.stitch(image1, "right", False, spacing_width, "white", image2)

        # Expected width: 32 + 16 (spacing) + 24 = 72
        assert result[0].shape == (1, 32, 72, 3)

    def test_spacing_vertical(self):
        """Test spacing addition in vertical concatenation"""
        node = ImageStitch()
        image1 = self.create_test_image(height=32, width=32)
        image2 = self.create_test_image(height=24, width=32)
        spacing_width = 16

        result = node.stitch(image1, "down", False, spacing_width, "white", image2)

        # Expected height: 32 + 16 (spacing) + 24 = 72
        assert result[0].shape == (1, 72, 32, 3)

    def test_spacing_color_values(self):
        """Test that spacing colors are applied correctly"""
        node = ImageStitch()
        image1 = self.create_test_image(height=32, width=32)
        image2 = self.create_test_image(height=32, width=32)

        # Test white spacing
        result_white = node.stitch(image1, "right", False, 16, "white", image2)
        # Check that spacing region contains white values (close to 1.0)
        spacing_region = result_white[0][:, :, 32:48, :]  # Middle 16 pixels
        assert torch.all(spacing_region >= 0.9)  # Should be close to white

        # Test black spacing
        result_black = node.stitch(image1, "right", False, 16, "black", image2)
        spacing_region = result_black[0][:, :, 32:48, :]
        assert torch.all(spacing_region <= 0.1)  # Should be close to black

    def test_odd_spacing_width_made_even(self):
        """Test that odd spacing widths are made even"""
        node = ImageStitch()
        image1 = self.create_test_image(height=32, width=32)
        image2 = self.create_test_image(height=32, width=32)

        # Use odd spacing width
        result = node.stitch(image1, "right", False, 15, "white", image2)

        # Should be made even (16), so total width = 32 + 16 + 32 = 80
        assert result[0].shape == (1, 32, 80, 3)

    def test_batch_size_matching(self):
        """Test that different batch sizes are handled correctly"""
        node = ImageStitch()
        image1 = self.create_test_image(batch_size=2, height=32, width=32)
        image2 = self.create_test_image(batch_size=1, height=32, width=32)

        result = node.stitch(image1, "right", False, 0, "white", image2)

        # Should match larger batch size
        assert result[0].shape == (2, 32, 64, 3)

    def test_channel_matching_rgb_to_rgba(self):
        """Test that channel differences are handled (RGB + alpha)"""
        node = ImageStitch()
        image1 = self.create_test_image(channels=3)  # RGB
        image2 = self.create_test_image(channels=4)  # RGBA

        result = node.stitch(image1, "right", False, 0, "white", image2)

        # Should have 4 channels (RGBA)
        assert result[0].shape[-1] == 4

    def test_channel_matching_rgba_to_rgb(self):
        """Test that channel differences are handled (RGBA + RGB)"""
        node = ImageStitch()
        image1 = self.create_test_image(channels=4)  # RGBA
        image2 = self.create_test_image(channels=3)  # RGB

        result = node.stitch(image1, "right", False, 0, "white", image2)

        # Should have 4 channels (RGBA)
        assert result[0].shape[-1] == 4

    def test_all_color_options(self):
        """Test all available color options"""
        node = ImageStitch()
        image1 = self.create_test_image(height=32, width=32)
        image2 = self.create_test_image(height=32, width=32)

        colors = ["white", "black", "red", "green", "blue"]

        for color in colors:
            result = node.stitch(image1, "right", False, 16, color, image2)
            assert result[0].shape == (1, 32, 80, 3)  # Basic shape check

    def test_all_directions(self):
        """Test all direction options"""
        node = ImageStitch()
        image1 = self.create_test_image(height=32, width=32)
        image2 = self.create_test_image(height=32, width=32)

        directions = ["right", "left", "up", "down"]

        for direction in directions:
            result = node.stitch(image1, direction, False, 0, "white", image2)
            assert result[0].shape == (1, 32, 64, 3) if direction in ["right", "left"] else (1, 64, 32, 3)

    def test_batch_size_channel_spacing_integration(self):
        """Test integration of batch matching, channel matching, size matching, and spacings"""
        node = ImageStitch()
        image1 = self.create_test_image(batch_size=2, height=64, width=48, channels=3)
        image2 = self.create_test_image(batch_size=1, height=32, width=32, channels=4)

        result = node.stitch(image1, "right", True, 8, "red", image2)

        # Should handle: batch matching, size matching, channel matching, spacing
        assert result[0].shape[0] == 2  # Batch size matched
        assert result[0].shape[-1] == 4  # Channels matched to max
        assert result[0].shape[1] == 64  # Height from image1 (size matching)
        # Width should be: 48 + 8 (spacing) + resized_image2_width
        expected_image2_width = int(64 * (32/32))  # Resized to height 64
        expected_total_width = 48 + 8 + expected_image2_width
        assert result[0].shape[2] == expected_total_width

