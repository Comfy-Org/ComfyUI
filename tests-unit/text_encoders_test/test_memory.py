"""Run the 37 memory regressions in isolated CPU processes.

python -m pytest tests-unit/text_encoders_test/test_memory.py
python tests-unit/text_encoders_test/test_memory.py
"""
import os
from pathlib import Path
import subprocess
import sys
import unittest


_CPU_TEST = """
import sys
import unittest

sys.path[:0] = sys.argv[1:3]
module = sys.argv[3]
sys.argv = [sys.argv[0]]

from comfy.cli_args import args
args.cpu = True
unittest.main(module=module, verbosity=2)
"""


def _run_cpu_tests(module):
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    subprocess.run(
        [sys.executable, "-s", "-c", _CPU_TEST, str(root), str(directory), module],
        cwd=root,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        check=True,
        timeout=120,
    )


class MemoryTests(unittest.TestCase):
    def test_krea2(self):
        _run_cpu_tests("krea2_memory_cases")

    def test_qwen_image(self):
        _run_cpu_tests("qwen_image_memory_cases")


if __name__ == "__main__":
    unittest.main()
