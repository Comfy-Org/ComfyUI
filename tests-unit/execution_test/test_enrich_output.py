"""Tests for enrich_output_with_assets in comfy_execution/asset_enrichment.py."""
import os
import types
import unittest
from unittest.mock import MagicMock, patch


def _make_args(enable_assets: bool):
    a = types.SimpleNamespace()
    a.enable_assets = enable_assets
    return a


def _make_register_result(ref_id="ref-id-2"):
    result = MagicMock()
    result.ref.id = ref_id
    return result


# Platform-appropriate absolute base. tempfile.gettempdir() returns C:\... on
# Windows and /tmp on POSIX, so containment via commonpath behaves naturally.
_DEFAULT_BASE = os.path.join(__import__("tempfile").gettempdir(), "asset-enrichment-test-base")


def _mocked_modules(*, enable_assets=True, register_file_in_place=None, directory=_DEFAULT_BASE):
    return {
        "comfy.cli_args": MagicMock(args=_make_args(enable_assets)),
        "folder_paths": MagicMock(get_directory_by_type=MagicMock(return_value=directory)),
        "app.assets.services.ingest": MagicMock(
            register_file_in_place=register_file_in_place or MagicMock(return_value=_make_register_result()),
            DependencyMissingError=type("DependencyMissingError", (Exception,), {}),
        ),
    }


def _call(output_ui, *, enable_assets=True, file_exists=True, register_result=None, directory=_DEFAULT_BASE):
    register_mock = MagicMock(return_value=register_result or _make_register_result())
    mocked = _mocked_modules(
        enable_assets=enable_assets,
        register_file_in_place=register_mock,
        directory=directory,
    )

    # Only os.path.isfile is patched — abspath/join must run natively so the
    # containment check sees real platform paths.
    with patch.dict("sys.modules", mocked), \
         patch("os.path.isfile", return_value=file_exists):
        import importlib
        import comfy_execution.asset_enrichment as mod
        importlib.reload(mod)
        return mod.enrich_output_with_assets(output_ui)


class TestEnrichOutputWithAssets(unittest.TestCase):

    def test_disabled_returns_unchanged(self):
        output = {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}
        result = _call(output, enable_assets=False)
        self.assertNotIn("id", result["images"][0])

    def test_non_list_value_passed_through(self):
        output = {"text": "hello"}
        result = _call(output)
        self.assertEqual(result["text"], "hello")

    def test_entry_without_filename_unchanged(self):
        output = {"latent": [{"subfolder": "", "type": "output"}]}
        result = _call(output)
        self.assertNotIn("id", result["latent"][0])

    def test_entry_without_type_unchanged(self):
        output = {"data": [{"filename": "a.png", "subfolder": ""}]}
        result = _call(output)
        self.assertNotIn("id", result["data"][0])

    def test_file_not_on_disk_unchanged(self):
        output = {"images": [{"filename": "missing.png", "subfolder": "", "type": "output"}]}
        result = _call(output, file_exists=False)
        self.assertNotIn("id", result["images"][0])

    def test_unknown_type_returns_none_directory_unchanged(self):
        output = {"images": [{"filename": "a.png", "subfolder": "", "type": "unknown"}]}
        result = _call(output, directory=None)
        self.assertNotIn("id", result["images"][0])

    def test_register_injects_only_id(self):
        reg = _make_register_result(ref_id="inline-ref")
        output = {"images": [{"filename": "new.png", "subfolder": "", "type": "output"}]}
        result = _call(output, register_result=reg)
        img = result["images"][0]
        self.assertEqual(img["id"], "inline-ref")
        # Only id is injected — no asset_hash, name, preview_url, size
        self.assertNotIn("asset_hash", img)
        self.assertNotIn("name", img)
        self.assertNotIn("preview_url", img)
        self.assertNotIn("size", img)

    def test_register_called_per_entry(self):
        register_mock = MagicMock(return_value=_make_register_result())
        mocked = _mocked_modules(register_file_in_place=register_mock)
        output = {
            "images": [
                {"filename": "a.png", "subfolder": "", "type": "output"},
                {"filename": "b.png", "subfolder": "", "type": "output"},
            ]
        }

        with patch.dict("sys.modules", mocked), \
             patch("os.path.isfile", return_value=True):
            import importlib
            import comfy_execution.asset_enrichment as mod
            importlib.reload(mod)
            mod.enrich_output_with_assets(output)

        self.assertEqual(register_mock.call_count, 2)

    def test_original_entry_not_mutated(self):
        orig = {"filename": "a.png", "subfolder": "", "type": "output"}
        output = {"images": [orig]}
        _call(output)
        self.assertNotIn("id", orig)

    def test_enrichment_error_does_not_block_sibling_entries(self):
        call_count = [0]
        good_reg = _make_register_result(ref_id="good-ref")

        def register_side_effect(abs_path, name, tags):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("boom")
            return good_reg

        mocked = _mocked_modules(register_file_in_place=register_side_effect)

        output = {
            "images": [
                {"filename": "bad.png", "subfolder": "", "type": "output"},
                {"filename": "good.png", "subfolder": "", "type": "output"},
            ]
        }

        with patch.dict("sys.modules", mocked), \
             patch("os.path.isfile", return_value=True):
            import importlib
            import comfy_execution.asset_enrichment as mod
            importlib.reload(mod)
            result = mod.enrich_output_with_assets(output)

        imgs = result["images"]
        self.assertNotIn("id", imgs[0])
        self.assertEqual(imgs[1]["id"], "good-ref")

    def test_multiple_output_keys_all_enriched(self):
        output = {
            "images": [{"filename": "a.png", "subfolder": "", "type": "output"}],
            "videos": [{"filename": "b.mp4", "subfolder": "", "type": "output"}],
        }
        result = _call(output)
        self.assertIn("id", result["images"][0])
        self.assertIn("id", result["videos"][0])

    def test_none_entry_in_list_unchanged(self):
        output = {"images": [None, {"filename": "a.png", "subfolder": "", "type": "output"}]}
        result = _call(output)
        self.assertIsNone(result["images"][0])
        self.assertIn("id", result["images"][1])

    def test_path_traversal_subfolder_skipped(self):
        register_mock = MagicMock(return_value=_make_register_result())
        mocked = _mocked_modules(register_file_in_place=register_mock)

        output = {"images": [{"filename": "passwd", "subfolder": "../../etc", "type": "output"}]}

        # Do NOT patch os.path.abspath — real resolution is required for the containment check.
        with patch.dict("sys.modules", mocked), \
             patch("os.path.isfile", return_value=True):
            import importlib
            import comfy_execution.asset_enrichment as mod
            importlib.reload(mod)
            result = mod.enrich_output_with_assets(output)

        self.assertNotIn("id", result["images"][0])
        register_mock.assert_not_called()

    def test_absolute_filename_skipped(self):
        register_mock = MagicMock(return_value=_make_register_result())
        mocked = _mocked_modules(register_file_in_place=register_mock)

        # Absolute filename — os.path.join discards earlier components when a later one is absolute.
        absolute_filename = os.path.abspath(os.sep + "etc" + os.sep + "passwd")
        output = {"images": [{"filename": absolute_filename, "subfolder": "", "type": "output"}]}

        with patch.dict("sys.modules", mocked), \
             patch("os.path.isfile", return_value=True):
            import importlib
            import comfy_execution.asset_enrichment as mod
            importlib.reload(mod)
            result = mod.enrich_output_with_assets(output)

        self.assertNotIn("id", result["images"][0])
        register_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
