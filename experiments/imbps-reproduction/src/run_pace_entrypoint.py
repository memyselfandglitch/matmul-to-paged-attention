#!/usr/bin/env python3
"""Run an unmodified PACE entrypoint after seeding every relevant RNG."""

from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    entrypoint = args.entrypoint.resolve()
    config = args.config.resolve()
    os.chdir(entrypoint.parent)
    sys.argv = [str(entrypoint), "--config", str(config)]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
