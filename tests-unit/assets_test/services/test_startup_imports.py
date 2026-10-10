import subprocess
import sys
from pathlib import Path

# Runs main.py's top-level imports up to its "Torch already imported" check, in a fresh process.
_REPLAY = """
import ast, sys
tree = ast.parse(open("main.py").read())
check = next(n.lineno for n in tree.body if isinstance(n, ast.If) and "torch" in ast.unparse(n.test))
sys.argv = ["main.py"]
for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)) and node.lineno < check:
        exec(compile(ast.Module([node], []), "main.py", "exec"), {})
assert "torch" not in sys.modules, "torch imported before main.py's check"
"""


def test_nothing_main_imports_before_its_torch_check_loads_torch():
    """cuda_malloc, aimdo and prestartup scripts must all run before torch loads."""
    subprocess.run([sys.executable, "-c", _REPLAY], check=True, cwd=Path(__file__).resolve().parents[3])
