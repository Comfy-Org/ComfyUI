"""The generated IO facade is an exact copy of Core's pure IO enum: names, values, constructors and comparisons."""
import os as _os
import sys as _sys
_CONVERSION = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "comfy_api", "conversion")
_sys.path.insert(0, _CONVERSION)
_os.environ.setdefault("COMFY_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_os.environ["PYTHONPATH"] = _os.pathsep.join(p for p in (_CONVERSION, _os.environ["COMFY_ROOT"], _os.environ.get("PYTHONPATH", "")) if p)
import ast
import enum
import os
from pathlib import Path
import unittest

import convert


class IOEnumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(os.environ['COMFY_ROOT']) / 'comfy/comfy_types/node_typing.py'
        definitions = [n for n in ast.parse(source.read_text()).body
                       if isinstance(n, ast.ClassDef) and n.name in ('StrEnum', 'IO')]
        scope = {'Enum': enum.Enum}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), scope)  # noqa: S102  Core's own source
        cls.original = scope['IO']
        generated = {}
        exec(compile(convert._io_shim(), 'trusted_enum_copy.py', 'exec'), generated)  # noqa: S102  the converter's copy of it
        cls.generated = generated['IO']

    def test_all_members_names_values_and_constructors_match(self):
        self.assertEqual(list(self.original.__members__), list(self.generated.__members__))
        for name, original in self.original.__members__.items():
            got = self.generated[name]
            self.assertEqual((got.name, got.value, str(got)), (original.name, original.value, str(original)))
            self.assertIs(self.generated(original.value), got)

    def test_wildcard_and_multi_type_comparisons_match(self):
        for old in self.original:
            new = self.generated[old.name]
            for other in ('*', 'MODEL', 'INT', 'FLOAT,INT', 'INT,FLOAT', 'MISSING', None, 5):
                self.assertEqual(old == other, new == other, (old.name, other))
                self.assertEqual(old != other, new != other, (old.name, other))

    def test_unknown_enum_names_and_values_raise(self):
        with self.assertRaises(AttributeError):
            self.generated.DOES_NOT_EXIST
        with self.assertRaises(ValueError):
            self.generated('DOES_NOT_EXIST')


if __name__ == '__main__':
    unittest.main()
