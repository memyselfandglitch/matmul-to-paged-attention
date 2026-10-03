#!/usr/bin/env python3
"""Release a pre-warmed worker and keep uProf alive until its output appears."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-file", type=Path, required=True)
    parser.add_argument("--done-file", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    args = parser.parse_args()

    args.start_file.parent.mkdir(parents=True, exist_ok=True)
    args.start_file.write_text("start\n", encoding="utf-8")
    deadline = time.monotonic() + args.timeout_seconds
    while not args.done_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for worker output {args.done_file}"
            )
        time.sleep(0.01)


if __name__ == "__main__":
    main()
