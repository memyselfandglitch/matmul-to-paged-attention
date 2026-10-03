# Cache-mechanism validation plan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-01
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## Experiment overview

- **Title:** Does IMBPS speedup track reduced cache and DRAM traffic?
- **Objective:** Test whether the split count that minimizes OPT-30B MLP time
  also minimizes cache misses and DRAM traffic, and therefore increases measured
  arithmetic intensity.
- **Hypothesis:** Within each batch size, faster split counts will have fewer L3
  misses and fewer DRAM bytes per MLP invocation. Because algorithmic FLOPs are
  fixed across K, lower DRAM traffic will increase measured arithmetic intensity.
- **Type:** Controlled CPU performance experiment.

This study can provide evidence for the proposed mechanism. It cannot prove
that cache locality is the only cause of speedup. Raw cache-hit count is not the
primary endpoint: a method can reduce total accesses and consequently reduce
both hits and misses.

## Setup

- **Framework:** Python 3.11.4, PyTorch 2.7.0+cpu, PACE 1.0.0.
- **Hardware:** `mn01`, dual-socket EPYC 9654 host; the Slurm allocation is
  restricted to package/socket 0, NUMA node 0, and 96 physical cores.
- **Scheduler:** `jobmn01`, exclusive Slurm allocation; no nested `srun`.
- **Profiler:** AMD uProf PCM using the installed version and metric names
  captured by `slurm/uprof_inventory.sbatch`.
- **Controls:** BF16, OPT-30B MLP shape, ReLU, identical seeds, allocator,
  thread count, affinity, warmups, iterations, and randomized case order.

## Variables

- **Independent variables:** batch in `{16, 32, 64}` and implementation in
  `{TPP K=1, IMBPS K=4, K=8, K=16}`.
- **Primary timing outcome:** paired median speedup from the unprofiled Table II
  confirmatory run.
- **Primary mechanism outcomes:** L3 misses per invocation and DRAM bytes per
  invocation.
- **Secondary outcomes:** L3 accesses, derived L3 miss ratio and hit ratio, L2
  accesses/misses, IPC, achieved bandwidth, GFLOP/s, and arithmetic intensity.

## Cache-resident timing strata

Before counter access is restored, run two timing controls against the 384 MiB
one-socket Equation 12 model:

1. **OPT-30B split-resident stratum:** `(B, SL)` in `{(1, 1920), (2, 1920),
   (16, 128), (16, 256)}` and IMBPS `K` in `{2, 4, 8, 16, 32}`. Every IMBPS
   case fits the modeled cache at K>=2. The TPP K=1 baseline intentionally does
   not fit; its BF16 MLP weight term alone is 392 MiB.
2. **Fully resident control:** OPT-6.7B at `B=16, SL in {128, 256, 384}` and
   OPT-13B at `B=16, SL in {128, 192, 224}`, with TPP K=1 and IMBPS
   `K in {2, 4, 8, 16}`. Equation 12 predicts every baseline and candidate fits.

The first stratum preserves OPT-30B and tests the intended transition from a
non-resident baseline to resident split blocks. The second tests whether IMBPS
still helps once cache-capacity pressure is removed from both implementations.
It is a mechanism control, not a reproduction of Table II.

## Measurement design

1. Keep timing and counter collection in separate exclusive jobs.
2. Warm up and preprocess weights before enabling counters.
3. Profile only the measured loop. Use a barrier/controller when AMD MSR mode
   cannot attach to an existing process.
4. Scope AMD uProf explicitly with `-c package=0`; never use `-a` on this
   dual-socket host. Preserve package-level aggregation in the raw CSV.
5. Use separate counter passes if uProf reports multiplexing or incompatible
   metric groups:
   - cache pass: `ipc,l2,l3`;
   - traffic/compute pass: `fp,memory` or AMD roofline.
6. Run at least five randomized paired rounds with a fresh process per case.
7. Preserve raw profiler output, exact command, tool version, metric scope,
   aggregation level, and measurement duration.
8. Normalize cumulative counters per measured MLP invocation. Never compare raw
   counts from runs with different iteration counts.

## Arithmetic-intensity definitions

- Let `M = batch * sequence`, `H = hidden`, and `I = intermediate`.
- The two dense matrix multiplications perform approximately `4*M*H*I` FLOPs.
  Bias and ReLU operations are recorded separately and are small relative to
  the GEMMs.
- **Algorithmic AI:** algorithmic FLOPs divided by compulsory model/input/output
  bytes. This is K-invariant and is a theoretical reference.
- **Measured DRAM AI:** algorithmic FLOPs divided by measured DRAM read plus
  write bytes. This is the primary intensity metric for testing locality.
- AMD roofline arithmetic intensity, when available, is reported separately as
  a profiler-derived cross-check; it is not silently substituted for the above.
- Do not estimate DRAM bytes as `L3 misses * 64` unless the exact L3 event is
  documented to count cache-line fills at that granularity.

## Analysis plan

For each batch size:

1. Report median and 95% bootstrap CI for speedup, L3 misses/invocation, DRAM
   bytes/invocation, and measured DRAM AI.
2. Report whether the fastest K also has the minimum L3 misses, minimum DRAM
   bytes, and maximum measured AI. Treat differences below measurement noise as
   ties.
3. Compare K=4 versus K=8 with paired differences and confidence intervals.
4. Compute within-batch Spearman association between runtime and each mechanism
   metric. Do not pool batches without batch-stratified reporting; that would
   risk a Simpson's-paradox interpretation.
5. Apply Benjamini-Hochberg correction to the exploratory family of correlation
   tests.

Mechanism support requires all of the following:

- the selected K has a reproducible timing benefit;
- it reproducibly lowers L3 misses or DRAM bytes relative to TPP;
- lower traffic is associated with lower runtime within batch strata;
- the result survives repetition and is not an artifact of multiplexing,
  counter scope, or profiler overhead.

A mismatch is a result, not a failure. It would indicate that overhead,
prefetching, packing, load balance, compute efficiency, or another mechanism is
more important than the selected cache metric.

## Expected outputs

| Output | Format | Success criterion |
|---|---|---|
| uProf inventory | text/CSV | tool/version and supported groups captured |
| Raw cache pass | native uProf output | non-multiplexed L2/L3 metrics with scope |
| Raw traffic pass | native uProf output | memory traffic and FP/roofline metrics |
| Normalized counters | CSV/JSON | per-invocation values for every case/round |
| Mechanism analysis | CSV/JSON | paired CIs, rank agreement, correlations |

## Monitoring configuration

- **Inventory timeout:** 10 minutes.
- **Profiling timeout:** 8 hours per counter pass.
- **Monitor files:** Slurm output/error plus timestamped result directories.
- **Retry policy:** no automatic retry; preserve and diagnose failed cases.
