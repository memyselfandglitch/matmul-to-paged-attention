# macOS arm64 Phase 1 baseline

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

With sequential blocks, the memory-matched `BNHD` traversal was fastest. With
shuffled physical blocks, `BHND` was fastest in this small run. That reversal
is exactly why the block table must be included before selecting a Phase 2
memory layout. Performance conclusions should come from repeated runs on the
target AMD system over a parameter sweep.
