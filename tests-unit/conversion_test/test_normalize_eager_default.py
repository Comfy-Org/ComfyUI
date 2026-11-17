"""Trusted fixture regression for getattr's eager default evaluation."""
import ast
import tempfile
import types
import unittest
from pathlib import Path

import normalize


class FixtureModule:
    def __init__(self, source, path):
        self.source = source
        self.tree = ast.parse(source, path)
        self.aliases = {"fixture": "fixture"}
        self.shadowed = set()
        self.lines = source.splitlines(keepends=True)
        self.offsets = [0]
        for line in self.lines:
            self.offsets.append(self.offsets[-1] + len(line))

    def pos(self, line, col):
        return self.offsets[line - 1] + col

    def span(self, node):
        return self.pos(node.lineno, node.col_offset), self.pos(node.end_lineno, node.end_col_offset)

    def resolve(self, node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name) and node.id in self.aliases:
            return '.'.join([self.aliases[node.id], *reversed(parts)])
        return None


class ProbeDefaults(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        Path(self.temp.name, 'fixture.py').write_text('feature = 42\n', encoding='utf-8')
        self.index = normalize.ComfyIndex(self.temp.name)
        self.fixture = types.SimpleNamespace(feature=42)

    def rewrite(self, expression):
        source = 'result = ' + expression + '\n'
        return normalize.resolve_probes(source, 'trusted.py', self.index, FixtureModule)[0]

    def run_source(self, source, default):
        namespace = {'fixture': self.fixture, 'default': default}
        exec(compile(source, 'trusted.py', 'exec'), namespace)
        return namespace['result']

    def test_computed_default_runs_even_when_attribute_exists(self):
        calls = []
        result = self.run_source(self.rewrite('getattr(fixture, "feature", default())'),
                                 lambda: calls.append('called'))
        self.assertEqual(result, 42)
        self.assertEqual(calls, ['called'])

    def test_computed_default_exception_propagates(self):
        def default():
            raise RuntimeError('default evaluated')
        with self.assertRaisesRegex(RuntimeError, 'default evaluated'):
            self.run_source(self.rewrite('getattr(fixture, "feature", default())'), default)

    def test_name_default_exception_propagates(self):
        source = self.rewrite('getattr(fixture, "feature", undefined_default)')
        with self.assertRaises(NameError):
            self.run_source(source, None)

    def test_literal_default_still_normalizes(self):
        source = self.rewrite('getattr(fixture, "feature", None)')
        self.assertEqual(source, 'result = fixture.feature\n')
        self.assertEqual(self.run_source(source, None), 42)

    def test_missing_attribute_uses_computed_default_once(self):
        calls = []
        source = self.rewrite('getattr(fixture, "absent", default())')
        result = self.run_source(source, lambda: calls.append('called') or 7)
        self.assertEqual((result, calls), (7, ['called']))


if __name__ == '__main__':
    unittest.main()
