"""Tests for LoadImage handling of an empty (unset) file selection."""

import pytest
import torch

from comfy.cli_args import args as cli_args

_HAS_ACCELERATOR = (
    torch.cuda.is_available()
    or torch.backends.mps.is_available()
    or (hasattr(torch, "xpu") and torch.xpu.is_available())
)

if not _HAS_ACCELERATOR:
    cli_args.cpu = True

from nodes import LoadImage, _LOADIMAGE_NO_SELECTION_MSG  # noqa: E402

EMPTY_SELECTIONS = ["", " ", "\t"]


class TestLoadImageEmptySelection:
    @pytest.mark.parametrize("image", EMPTY_SELECTIONS)
    def test_validate_inputs_reports_actionable_message(self, image):
        result = LoadImage.VALIDATE_INPUTS(image)
        assert result == _LOADIMAGE_NO_SELECTION_MSG

    @pytest.mark.parametrize("image", EMPTY_SELECTIONS)
    def test_is_changed_is_safe(self, image):
        assert LoadImage.IS_CHANGED(image) == ""

    @pytest.mark.parametrize("image", EMPTY_SELECTIONS)
    def test_load_image_raises_actionable_error(self, image):
        with pytest.raises(ValueError, match="no image file selected"):
            LoadImage().load_image(image)

    def test_validate_inputs_still_rejects_missing_file(self):
        result = LoadImage.VALIDATE_INPUTS("definitely_not_a_real_file.png")
        assert result.startswith("Invalid image file")
