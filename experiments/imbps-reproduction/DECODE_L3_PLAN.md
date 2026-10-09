# Decode-shaped L3 sweep

## Question

For OPT-30B decode-shaped MLP calls on one EPYC 9654 socket, does the
384 MiB aggregate package-visible L3 model select a split that improves L3
locality, DRAM traffic, and clean latency across different active-row counts?

This is a standalone MLP experiment. It does not include attention or a KV
cache. In normal decode, active rows equal live batch size. In speculative
target verification, active rows equal live batch size times the number of
verified draft tokens. Context length does not change this standalone MLP
shape and is therefore deferred to an end-to-end experiment.

## Fixed configuration

- Model shape: OPT-30B, H=7168, I=28672, BF16, ReLU.
- Host: mn01, package/NUMA node 0, 96 pinned physical cores, no SMT.
- Allocation: exclusive Slurm measurement jobs.
- Runtime: THP `always` and gperftools tcmalloc required and validated.
- Cache model: 384 MiB aggregate L3 across twelve physical 32 MiB CCD slices.
- Counters: uProf `ipc,l2,l3` in one pass and `memory` in a separate pass.

The 384 MiB capacity is not one physically shared cache. Equation 13 is being
tested as an aggregate placement hypothesis; per-CCD ownership remains a
separate experiment.

## Matrix

- Clean timing active rows: 1, 8, 32, 128, 256, 384, 512, 768, 1024.
- Clean timing variants: TPP K=1 and IMBPS K=2,4,8,16,32.
- Counter active rows: 32, 128, 256, 384, 512, 1024.
- Counter variants: TPP K=1 and IMBPS K=2,4,8,16.
- Clean timing: five randomized paired rounds and five warmups.
- Counters: five randomized rounds per serialized pass and five warmups.

For every tested row count, Equation 13 with 384 MiB gives a strict lower
bound between 1.02 and 1.21, so the author's next-power-of-two policy selects
K=2. K=4/8/16 test increasing split overhead; K=32 is a timing-only
over-splitting control.

## Outcomes

- Primary: clean paired speedup and 95% interval versus TPP.
- Mechanism: L3 accesses, misses, hit rate, and miss latency per invocation.
- Supporting: L2 metrics, DRAM bytes per invocation, DRAM arithmetic intensity,
  and IPC.

Profiler timing is diagnostic only. Performance conclusions use the separate
unprofiled timing run. Report every tested K and do not remove post-hoc
outliers.

## Interpretation boundary

This experiment can establish an MLP-level locality/latency relationship. It
cannot establish end-to-end decode speedup. That requires fixed prompts,
populated KV caches, controlled context lengths, and separately reported MLP,
attention, and total token latency.
