# macOS arm64 Phase 1 and Phase 2 baseline

This is a local correctness and smoke-test result, not the target IISc result.

- OS/architecture: macOS arm64
- Compiler: Apple clang 15.0.0
- Fixed physical layout: `BNHD`, matching the installed vLLM NHD default
- Shape: `B=128, H=16, N=16, D=64`
- K + V size: 16 MiB
- Warmups: 1
- Timed repetitions: 5
- Computation: scaled QK, online softmax and weighted V accumulation
- Correctness: all traversal outputs matched within each block-table test

Phase 1 favoured the memory-matched `BNHD` traversal with both block-table
patterns. In the Phase 2 3x3 matrix, the traversal matching physical memory won
for all three layouts. `BNHD` memory with `BNHD` traversal was the global winner
in both patterns. Shuffling particularly penalised the global head-major
`HBND` memory layout.

These are smoke-test findings, not target-machine conclusions. The committed
CSV files and reports make the local result reproducible; performance claims
should use the AMD CPU run with the larger Slurm dimensions.

The local fragmentation smoke test adds contiguous run lengths from 128 down
to 1. With a 2% tie band, HBND is preferred at run lengths 64, 32, and 16;
run length 8 is a tie; and BHND is preferred at 4, 2, and 1. This brackets the
local crossover between run lengths 16 and 4. The kernel times are only about
1.2 ms, so the AMD run is required to locate a defensible target-system
crossover.
