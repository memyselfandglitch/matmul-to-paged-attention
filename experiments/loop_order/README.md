# Matrix-multiplication loop-order experiments

This directory contains two separate experiments:

- `matmul_optimization_stages.cpp` shows the progression from a baseline
  matrix multiply through transposition, SIMD, cache tiling, register blocking,
  and their combination. This was previously named `ijk.cpp`, even though it
  had grown beyond a single `ijk` kernel.
- `loop_order_study.cpp` is the controlled loop-order experiment. It benchmarks
  all six permutations of the macro-tile loops at two stages: cache tiled, and
  cache tiled plus SIMD register blocked.

## Fair-comparison rules

Within each stage, the six kernels differ only in the order of their `Mc`, `Nc`,
and `Kc` loops:

```text
ijk  ikj  jik  jki  kij  kji
```

All orders share the same row-major matrix layout, `120 x 64 x 240` macro
tiles, edge handling, inputs, output reset, and timing harness. The tiled stage
uses the same compiler-vectorizable `i-k-j` inner kernel for every macro order.
The register-blocked stage uses the same `6 x (2 SIMD vectors)` microkernel for
every macro order. This keeps loop order as the controlled variable.

The benchmark validates every kernel, excludes allocation and output clearing
from timings, shuffles all order/stage combinations on every repetition, and
reports median time and GFLOP/s. A non-tile-aligned `70 x 70` validation case
exercises every edge path before measurements begin.

This is an optimized **direct/unpacked** SGEMM study. Matrix packing and
architecture-specific assembly can improve absolute peak performance, but they
are intentionally outside this experiment because their traversal and packing
costs would introduce additional variables.

## Build and run

```sh
cd loop_order
make
./build/loop_order_study 1920 5 384
```

Arguments are:

```text
loop_order_study [max_n=1920] [repetitions=5] [min_n=384]
```

For a quick correctness and smoke test:

```sh
make check
```

The Makefile uses `-O3` and the native CPU target (`-mcpu=native` on Arm,
`-march=native` elsewhere), so benchmark the binary on the machine whose cache
behavior you want to study.
