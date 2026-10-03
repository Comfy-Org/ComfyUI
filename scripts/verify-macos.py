import platform
import sys

import torch


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"PyTorch: {torch.__version__}")

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("ERROR: This setup targets Apple Silicon macOS.", file=sys.stderr)
        return 1

    if not torch.backends.mps.is_built():
        print("ERROR: This PyTorch build does not include MPS support.", file=sys.stderr)
        return 1
    if not torch.backends.mps.is_available():
        print("ERROR: MPS is unavailable. Update macOS and install an Apple Silicon PyTorch wheel.", file=sys.stderr)
        return 1

    device = torch.device("mps")
    result = (torch.ones(4, device=device) * 2).cpu().tolist()
    print(f"MPS test: OK ({result})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
