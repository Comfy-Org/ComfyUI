import subprocess
import sys
from pathlib import Path

# Runs main.py's top-level imports that precede `import cuda_malloc`, in a fresh process.
_REPLAY = """
import ast, sys
tree = ast.parse(open("main.py").read())
cut = next(n.lineno for n in ast.walk(tree) if isinstance(n, ast.Import) and any(a.name == "cuda_malloc" for a in n.names))
sys.argv = ["main.py"]
for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)) and node.lineno < cut:
        exec(compile(ast.Module([node], []), "main.py", "exec"), {})
assert "torch" not in sys.modules, "torch imported before cuda_malloc"
"""


def test_nothing_main_imports_before_cuda_malloc_loads_torch():
    """cuda_malloc sets the CUDA allocator, which only takes effect if torch hasn't loaded yet."""
    subprocess.run([sys.executable, "-c", _REPLAY], check=True, cwd=Path(__file__).resolve().parents[3])
