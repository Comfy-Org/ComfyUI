"""Run with python -m unittest discover -s deploy/easypanel -p 'test_*.py'."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import start


class StartupTests(unittest.TestCase):
    def launch(self, environment):
        with patch.dict(os.environ, environment, clear=True), \
                patch.object(Path, 'mkdir') as mkdir, \
                patch.object(os, 'execv') as execute:
            start.main()
        return mkdir, execute.call_args.args[1]

    def test_default_persistent_paths_and_port(self):
        mkdir, command = self.launch({})
        self.assertEqual(mkdir.call_count, 6)
        self.assertEqual(command[command.index('--port') + 1], '8188')
        self.assertEqual(command[command.index('--user-directory') + 1], '/data/user')
        self.assertEqual(command[command.index('--database-url') + 1], 'sqlite:////data/user/comfyui.db')
        self.assertEqual(command[command.index('--models-directory') + 1], '/data/models')

    def test_custom_port_and_quoted_arguments(self):
        _, command = self.launch({'COMFYUI_PORT': '9000', 'COMFYUI_ARGS': '--lowvram --extra-model-paths-config "/data/my paths.yaml"'})
        self.assertEqual(command[command.index('--port') + 1], '9000')
        self.assertEqual(command[-3:], ['--lowvram', '--extra-model-paths-config', '/data/my paths.yaml'])

    def test_invalid_port_fails_before_exec(self):
        for port in ('0', '65536', 'abc'):
            with self.subTest(port=port), patch.dict(os.environ, {'COMFYUI_PORT': port}, clear=True), patch.object(os, 'execv') as execute:
                with self.assertRaises(ValueError):
                    start.main()
                execute.assert_not_called()

    def test_managed_options_cannot_break_healthcheck_or_persistence(self):
        for args in ('--port=9000', '--listen 127.0.0.1', '--user-directory /tmp', '--database-url=sqlite://', '--base-directory /tmp'):
            with self.subTest(args=args), patch.dict(os.environ, {'COMFYUI_ARGS': args}, clear=True), patch.object(os, 'execv') as execute:
                with self.assertRaises(ValueError):
                    start.main()
                execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
