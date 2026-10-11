# Parallel per-CCD IMBPS plan

## Research question

Can the sequential IMBPS parameter-block loop in AMD-PACE v1.0 be accelerated
by running disjoint blocks concurrently on mn01's twelve CCDs, and does a
32-MiB-per-CCD cache model predict the useful split depth?

PACE v1.0 contains an explicit `TODO: Parallelize this loop
(IMBPS_PARALLEL_FLOW)` around the parameter-split loop. Here:

- **global IMBPS** is the unmodified PACE implementation: K blocks execute
  sequentially, while each GEMM uses the full 96-core OpenMP team;
- **parallel per-CCD IMBPS** uses persistent processes, one 8-core OpenMP team
  per CCD. Each process executes disjoint intermediate-dimension blocks, then
  the parent reduces the partial M x H outputs.

This is a process-based prototype, not a claim that PACE itself already
implements the parallel loop.

## Workload and variants

- Standalone OPT-30B MLP: H=7168, I=28672, BF16, ReLU, no bias
- Active rows: 128, 512, 1024 decode/speculative-verification shapes, plus a
  4096-row prefill-shaped/non-resident control
- Controls: all-core PACE TPP; all-core global IMBPS K=12, 24, 48
- CCD scaling with one block per CCD: P/K = 1/1, 2/2, 4/4, 6/6, 12/12
- Split-depth controls at all CCDs: P=12 with K=24 and K=48

K=12 has a fringe block because 28672 is not divisible by 12. That is part of
the real mn01 topology rather than something to hide.

## Per-CCD cache hypothesis

For one active block on a CCD, use the paper-style working set

`W_ccd(K) = alpha*M*H + alpha*M*I_block + alpha*H*I_block`,

where `I_block` is the widest actual block and alpha=2 bytes. Compare this with
32 MiB, not with the socket's aggregate 384 MiB. At M=512, K=12 is predicted
not to fit while K=24 is predicted to fit. At M=1024, K=48 is the fitting
control. At M=4096 the input activation alone is 56 MiB, so no K can satisfy
the 32-MiB equation; it tests parallel execution without a cache-fit claim.

## Timing boundary

Per-CCD wall time includes:

1. release of all persistent CCD workers;
2. each worker's PACE fused MLP work;
3. publication of every partial output;
4. synchronization; and
5. the final cross-CCD FP32 reduction and BF16 conversion.

Process creation, weight preparation, and warmups are excluded. This boundary
prevents an unrealistic claim based on compute-only worker time.

## Outcomes and decision rules

- Latency and speedup versus all-core TPP
- Speedup versus global sequential IMBPS at the same K
- Parallel speedup and efficiency versus one CCD
- Per-CCD modeled fit
- Max, q99, mean absolute difference and output-coordinate argmax agreement
  versus global IMBPS with the same K

Every task is exclusive on mn01 and serialized. Report all seven measured
iterations and independent bootstrap intervals. A useful parallel design must
beat global IMBPS at the same K after reduction cost; cache fit alone is not a
performance conclusion.

## Follow-up

If the prototype wins, move the parallel split loop and reduction into a C++
PACE backend using OpenMP/TBB/pthreads, then repeat clean timing and CCD-scoped
uProf measurements. If it loses, separate compute, publication, barrier, and
reduction costs before rejecting the underlying block placement.
