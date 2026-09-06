"""Summarize paired vLLM latency JSON files without extra dependencies."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def load(path: Path) -> tuple[float, float, int]:
    with path.open(encoding="utf-8") as source:
        result = json.load(source)
    samples = [float(value) for value in result["latencies"]]
    return float(result["avg_latency"]), statistics.median(samples), len(samples)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: summarize_latency.py NHD.json HND.json")

    nhd_avg, nhd_median, nhd_count = load(Path(sys.argv[1]))
    hnd_avg, hnd_median, hnd_count = load(Path(sys.argv[2]))
    best = "NHD" if nhd_median < hnd_median else "HND"

    print("Triton KV-cache layout A/B")
    print(f"NHD samples: {nhd_count}")
    print(f"HND samples: {hnd_count}")
    print(f"NHD average: {nhd_avg:.6f} s")
    print(f"HND average: {hnd_avg:.6f} s")
    print(f"NHD median:  {nhd_median:.6f} s")
    print(f"HND median:  {hnd_median:.6f} s")
    print(f"NHD/HND median ratio: {nhd_median / hnd_median:.4f}x")
    print(f"Faster median: {best}")


if __name__ == "__main__":
    main()
