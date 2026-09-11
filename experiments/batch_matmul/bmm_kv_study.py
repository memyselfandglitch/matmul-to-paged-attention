"""Small study of real PyTorch matrix-multiplication batching.

There is no handwritten matrix-multiplication implementation here.  Both the
explicit-loop baseline and the batched path call torch.matmul; the only
difference is whether Python or PyTorch owns the batch operation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Wait for asynchronous accelerator work before reading the clock."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def paired_median_microseconds(
    loop_work: Callable[[], torch.Tensor],
    batched_work: Callable[[], torch.Tensor],
    device: torch.device,
    repetitions: int,
) -> tuple[float, float]:
    """Time both paths in alternating order to reduce frequency/order bias."""
    for _ in range(2):
        loop_work()
        batched_work()
    synchronize(device)

    loop_samples: list[float] = []
    batched_samples: list[float] = []
    for _ in range(repetitions):
        ordered_work = (
            ((loop_work, loop_samples), (batched_work, batched_samples))
            if len(loop_samples) % 2 == 0
            else ((batched_work, batched_samples), (loop_work, loop_samples))
        )
        for work, samples in ordered_work:
            synchronize(device)
            start = time.perf_counter()
            result = work()
            synchronize(device)
            samples.append((time.perf_counter() - start) * 1.0e6)

    # Make the final result observable after synchronization.
    _ = float(result.reshape(-1)[0].cpu())
    return statistics.median(loop_samples), statistics.median(batched_samples)


def loop_of_torch_matmuls(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Baseline: issue one real torch.matmul call for each batch item."""
    return torch.stack([torch.matmul(a[i], b[i]) for i in range(a.shape[0])])


def batched_torch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """One call: PyTorch owns all leading batch dimensions."""
    return torch.matmul(a, b)


def gflops(batch: int, m: int, n: int, k: int, microseconds: float) -> float:
    return 2.0 * batch * m * n * k / (microseconds * 1.0e3)


def parse_size_bytes(text: str) -> int:
    suffixes = {"K": 1024, "M": 1024**2, "G": 1024**3}
    value = text.strip().upper()
    if value[-1:] in suffixes:
        return int(float(value[:-1]) * suffixes[value[-1]])
    return int(value)


def cpu_cache_sizes() -> dict[int, tuple[int, str]]:
    """Return CPU0's data/unified cache size and sharing domain by level."""
    cache_root = Path("/sys/devices/system/cpu/cpu0/cache")
    result: dict[int, tuple[int, str]] = {}
    if not cache_root.exists():
        return result
    for index in cache_root.glob("index*"):
        try:
            cache_type = (index / "type").read_text().strip()
            if cache_type not in {"Data", "Unified"}:
                continue
            level = int((index / "level").read_text().strip())
            size = parse_size_bytes((index / "size").read_text())
            shared = (index / "shared_cpu_list").read_text().strip()
            result[level] = (size, shared)
        except (OSError, ValueError):
            continue
    return result


def measure_square_case(
    batch: int,
    size: int,
    device: torch.device,
    repetitions: int,
) -> dict[str, float | int | str]:
    a = torch.randn(batch, size, size, device=device)
    b = torch.randn(batch, size, size, device=device)

    loop_result = loop_of_torch_matmuls(a, b)
    batched_result = batched_torch_matmul(a, b)
    torch.testing.assert_close(loop_result, batched_result)
    del loop_result, batched_result

    loop_us, batched_us = paired_median_microseconds(
        lambda: loop_of_torch_matmuls(a, b),
        lambda: batched_torch_matmul(a, b),
        device,
        repetitions,
    )

    element_bytes = a.element_size()
    per_matrix_bytes = 3 * size * size * element_bytes
    batch_bytes = batch * per_matrix_bytes
    ratio = loop_us / batched_us
    if 0.95 <= ratio <= 1.05:
        classification = "near parity"
    elif ratio > 1.05:
        classification = "batched faster"
    else:
        classification = "loop faster"

    result: dict[str, float | int | str] = {
        "batch": batch,
        "size": size,
        "per_matrix_mib": per_matrix_bytes / 1024**2,
        "batch_working_set_mib": batch_bytes / 1024**2,
        "loop_us": loop_us,
        "batched_us": batched_us,
        "loop_over_batched": ratio,
        "classification": classification,
    }
    del a, b
    gc.collect()
    return result


def parse_sweep_sizes(value: str) -> list[int]:
    try:
        sizes = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers") from error
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("every sweep size must be positive")
    return sizes


def cache_region(per_matrix_bytes: float, caches: dict[int, tuple[int, str]]) -> str:
    l2 = caches.get(2, (0, ""))[0]
    l3 = caches.get(3, (0, ""))[0]
    if l2 and per_matrix_bytes <= l2:
        return "fits L2"
    if l3 and per_matrix_bytes <= l3:
        return "L2 < WS <= L3"
    if l3:
        return "exceeds L3"
    return "cache unknown"


def batch_l3_region(batch_bytes: float, caches: dict[int, tuple[int, str]]) -> str:
    l3 = caches.get(3, (0, ""))[0]
    if not l3:
        return "unknown"
    return "> L3" if batch_bytes > l3 else "<= L3"


def write_sweep_csv(
    path: Path,
    rows: list[dict[str, float | int | str]],
    caches: dict[int, tuple[int, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch",
        "size",
        "per_matrix_mib",
        "batch_working_set_mib",
        "cache_region",
        "batch_vs_cpu0_l3",
        "loop_us",
        "batched_us",
        "loop_over_batched",
        "classification",
        "cpu0_l2_mib",
        "cpu0_l3_mib",
    ]
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            per_matrix_bytes = float(row["per_matrix_mib"]) * 1024**2
            batch_bytes = float(row["batch_working_set_mib"]) * 1024**2
            writer.writerow(
                {
                    **row,
                    "cache_region": cache_region(per_matrix_bytes, caches),
                    "batch_vs_cpu0_l3": batch_l3_region(batch_bytes, caches),
                    "cpu0_l2_mib": caches.get(2, (0, ""))[0] / 1024**2,
                    "cpu0_l3_mib": caches.get(3, (0, ""))[0] / 1024**2,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare looped and batched calls to the real torch.matmul"
    )
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda", "mps")
    )
    parser.add_argument(
        "--cache-sweep",
        action="store_true",
        help="sweep square matrix sizes and report nominal working sets",
    )
    parser.add_argument("--sweep-batch", type=int, default=8)
    parser.add_argument("--sweep-repetitions", type=int, default=5)
    parser.add_argument(
        "--sweep-sizes",
        type=parse_sweep_sizes,
        default=parse_sweep_sizes("64,128,256,512,768,1024,1536,2048,3072"),
    )
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if args.sweep_batch <= 0:
        parser.error("--sweep-batch must be positive")
    if args.sweep_repetitions <= 0:
        parser.error("--sweep-repetitions must be positive")
    return args


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    torch.manual_seed(0xB47C4)

    batch, m, n, k_size = 32, 64, 64, 64
    a = torch.randn(batch, m, k_size, device=device)
    b = torch.randn(batch, k_size, n, device=device)

    loop_us, batched_us = paired_median_microseconds(
        lambda: loop_of_torch_matmuls(a, b),
        lambda: batched_torch_matmul(a, b),
        device,
        args.repetitions,
    )

    print("PyTorch matmul batching study")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    if device.type == "cpu":
        print(f"CPU threads: {torch.get_num_threads()}")

    print("\nIndependent matrices")
    print("   A[B,M,K] @ B[B,K,N] -> C[B,M,N]")
    print(f"   shape: batch={batch}, M=N=K={m}")
    print(
        f"   loop of torch.matmul: {loop_us:10.2f} us, "
        f"{gflops(batch, m, n, k_size, loop_us):7.2f} GFLOP/s"
    )
    print(
        f"   one torch.matmul:     {batched_us:10.2f} us, "
        f"{gflops(batch, m, n, k_size, batched_us):7.2f} GFLOP/s"
    )
    print(f"   loop/batched:         {loop_us / batched_us:10.2f}x")

    if args.cache_sweep:
        caches = cpu_cache_sizes() if device.type == "cpu" else {}
        print("\nCache-capacity sweep")
        print(
            "   Nominal working set counts A + B + C once. "
            "Allocation and initialization are not timed."
        )
        for level in (2, 3):
            if level in caches:
                size_bytes, shared_cpus = caches[level]
                print(
                    f"   CPU0 L{level}: {size_bytes / 1024**2:.1f} MiB "
                    f"per reported cache instance; shared CPUs {shared_cpus}"
                )
        if caches:
            print(
                "   Cache thresholds are references: effective capacity depends "
                "on thread placement and cache sharing."
            )

        print(
            f"\n{'n':>6} {'item MiB':>10} {'batch MiB':>11} "
            f"{'item cache':>15} {'batch/L3':>10} {'loop ms':>11} {'batch ms':>11} "
            f"{'loop/batch':>11} {'result':>16}"
        )
        sweep_rows: list[dict[str, float | int | str]] = []
        for size in args.sweep_sizes:
            row = measure_square_case(
                args.sweep_batch, size, device, args.sweep_repetitions
            )
            sweep_rows.append(row)
            per_matrix_bytes = float(row["per_matrix_mib"]) * 1024**2
            batch_bytes = float(row["batch_working_set_mib"]) * 1024**2
            region = cache_region(per_matrix_bytes, caches)
            batch_region = batch_l3_region(batch_bytes, caches)
            print(
                f"{size:6d} {float(row['per_matrix_mib']):10.2f} "
                f"{float(row['batch_working_set_mib']):11.2f} "
                f"{region:>15} {batch_region:>10} "
                f"{float(row['loop_us']) / 1000:11.3f} "
                f"{float(row['batched_us']) / 1000:11.3f} "
                f"{float(row['loop_over_batched']):11.3f}x "
                f"{str(row['classification']):>16}"
            )

        parity_rows = [
            row
            for row in sweep_rows
            if str(row["classification"]) == "near parity"
        ]
        if parity_rows:
            first = parity_rows[0]
            print(
                "\nFirst observed point within ±5%: "
                f"n={first['size']}, per-matrix working set="
                f"{float(first['per_matrix_mib']):.2f} MiB, "
                f"full batch={float(first['batch_working_set_mib']):.2f} MiB."
            )
        else:
            print("\nNo measured sweep point was within ±5%.")

        if args.csv:
            write_sweep_csv(args.csv, sweep_rows, caches)
            print(f"Raw sweep CSV: {args.csv}")

        print(
            "Interpretation caution: approaching 1x can reflect amortized "
            "Python/stacking overhead as O(n^3) GEMM work grows. Crossing a "
            "cache threshold is correlation unless counters establish causation."
        )

    print("\nAll comparisons passed torch.testing.assert_close.")
    print("The loop baseline still uses torch.matmul; no matrix kernel is handwritten.")


if __name__ == "__main__":
    main()
