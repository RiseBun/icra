"""Read-only dependency and accelerator environment probe."""

import importlib.util
import os
import platform
import sys


def main() -> None:
    print("python", sys.version.replace("\n", " "))
    print("platform", platform.platform())
    try:
        import torch
        print("torch", torch.__version__)
        print("cuda", torch.cuda.is_available(), "count", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print("gpu", i, torch.cuda.get_device_name(i))
    except Exception as exc:
        print("torch_error", repr(exc))
    for name in ("yaml", "numpy", "open3d", "cv2", "einops"):
        print("module", name, bool(importlib.util.find_spec(name)))
    print("module gymnasium", bool(importlib.util.find_spec("gymnasium")))
    print("module pyrep", bool(importlib.util.find_spec("pyrep")))
    print("module rlbench", bool(importlib.util.find_spec("rlbench")))
    coppelia_root = os.environ.get("COPPELIASIM_ROOT", "")
    print("coppeliasim_root", coppelia_root or "<unset>",
          "exists", bool(coppelia_root and os.path.exists(coppelia_root)))
    print("cwd", os.getcwd())


if __name__ == "__main__":
    main()
