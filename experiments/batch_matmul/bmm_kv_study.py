"""Small study of real PyTorch matrix-multiplication batching.

There is no handwritten matrix-multiplication implementation here.  Both the
explicit-loop baseline and the batched path call torch.matmul; the only
difference is whether Python or PyTorch owns the batch operation.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

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


def median_microseconds(
    work: Callable[[], torch.Tensor], device: torch.device, repetitions: int
) -> float:
    for _ in range(3):
        result = work()
    synchronize(device)

    samples: list[float] = []
    for _ in range(repetitions):
        synchronize(device)
        start = time.perf_counter()
        result = work()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1.0e6)

    # Make the result observable and keep it alive through synchronization.
    _ = float(result.reshape(-1)[0].cpu())
    return statistics.median(samples)


def loop_of_torch_matmuls(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Baseline: issue one real torch.matmul call for each batch item."""
    return torch.stack([torch.matmul(a[i], b[i]) for i in range(a.shape[0])])


def batched_torch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """One call: PyTorch owns all leading batch dimensions."""
    return torch.matmul(a, b)


def gflops(batch: int, m: int, n: int, k: int, microseconds: float) -> float:
    return 2.0 * batch * m * n * k / (microseconds * 1.0e3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare looped and batched calls to the real torch.matmul"
    )
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda", "mps")
    )
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    return args


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    torch.manual_seed(0xB47C4)

    batch, m, n, k_size = 32, 64, 64, 64
    a = torch.randn(batch, m, k_size, device=device)
    b = torch.randn(batch, k_size, n, device=device)

    loop_result = loop_of_torch_matmuls(a, b)
    batched_result = batched_torch_matmul(a, b)
    torch.testing.assert_close(loop_result, batched_result)

    loop_us = median_microseconds(
        lambda: loop_of_torch_matmuls(a, b), device, args.repetitions
    )
    batched_us = median_microseconds(
        lambda: batched_torch_matmul(a, b), device, args.repetitions
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

    print("\nAll comparisons passed torch.testing.assert_close.")
    print("The loop baseline still uses torch.matmul; no matrix kernel is handwritten.")


if __name__ == "__main__":
    main()
