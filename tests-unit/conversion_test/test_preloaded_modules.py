"""``import PIL`` then ``PIL.X``: only real Pillow submodules are imported, never exception classes or constants."""
import os as _os
import sys as _sys
_CONVERSION = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "comfy_api", "conversion")
_sys.path.insert(0, _CONVERSION)
_os.environ.setdefault("COMFY_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_os.environ["PYTHONPATH"] = _os.pathsep.join(p for p in (_CONVERSION, _os.environ["COMFY_ROOT"], _os.environ.get("PYTHONPATH", "")) if p)
import unittest

import convert


class PillowModuleTests(unittest.TestCase):
    def rewritten(self, source):
        return convert._explicit_submodules(source, 'fixture.py')

    def test_exception_is_not_imported_as_a_module(self):
        source = 'import PIL\ntry:\n    raise PIL.UnidentifiedImageError("fixture")\nexcept PIL.UnidentifiedImageError:\n    handled = True\n'
        result = self.rewritten(source)
        self.assertNotIn('import PIL.UnidentifiedImageError', result)
        scope = {}
        exec(compile(result, 'trusted_fixture.py', 'exec'), scope)  # noqa: S102  a trusted fixture
        self.assertTrue(scope['handled'])

    def test_actual_image_module_is_loaded_in_a_fresh_process(self):
        source = 'import PIL\nimage = PIL.Image.new("RGB", (2, 3))\n'
        result = self.rewritten(source)
        self.assertIn('import PIL.Image\n', result)
        scope = {}
        exec(compile(result, 'trusted_fixture.py', 'exec'), scope)  # noqa: S102  a trusted fixture
        self.assertEqual(scope['image'].size, (2, 3))

    def test_mixed_module_and_exception_keeps_both_behaviors(self):
        source = 'import PIL\ntry:\n    PIL.Image.open("/nonexistent/trusted-fixture.png")\nexcept (FileNotFoundError, PIL.UnidentifiedImageError):\n    handled = True\n'
        result = self.rewritten(source)
        self.assertIn('import PIL.Image\n', result)
        self.assertNotIn('import PIL.UnidentifiedImageError', result)
        scope = {}
        exec(compile(result, 'trusted_fixture.py', 'exec'), scope)  # noqa: S102  a trusted fixture
        self.assertTrue(scope['handled'])


if __name__ == '__main__':
    unittest.main()
