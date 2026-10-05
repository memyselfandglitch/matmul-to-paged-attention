# L2-aware IMBPS for decode and speculative verification

## Material Passport

- Origin skill: `academic-research-suite` / experiment-agent
- Artifact type: prospective code experiment plan
- Status: preregistered; no server result has been observed
- Target: AMD EPYC 9654 package 0 on `mn01`
- Implementation: AMD-PACE v1.0, OPT-30B MLP, BF16, ReLU
- Last updated: 2026-10-05

## Research question

For small active-row counts, can an L2-derived IMBPS split reduce OPT-30B MLP
latency and memory traffic relative to PACE TPP, and where does the benefit
cross over as decode or speculative-verification parallelism grows?

The standalone MLP observes only `M`, the active row count:

- normal decode: `M = live batch size`;
- speculative verification: `M = live batch size * verified draft tokens`.

Consequently, decompositions with the same `M` are not rerun as if they were
independent workloads. End-to-end speculative decoding is a later phase because
proposal, attention, KV-cache, rejection, and accepted-token effects are absent
from this operator benchmark.

## Prospective hypotheses

1. At least one `K > 1` lowers unprofiled median MLP latency relative to TPP for
   some active-row range.
2. The fastest `K` is near, but is not assumed to equal, the aggregate-L2
   Equation-13 candidate.
3. A beneficial split lowers absolute L2 misses per invocation, L2 miss rate,
   or DRAM bytes per invocation in separately profiled runs.
4. The latency benefit is non-monotonic in `K`; excessive splitting eventually
   loses to loop, kernel, and accumulation overhead.

## Analytical candidates

The paper's Equation 13 is evaluated with OPT-30B `H=7168`, `I=28672`, BF16
`f=2`, active rows `M`, and the affinity-visible aggregate L2 capacity of
96 MiB:

```text
K > f I (M + H) / (L2 - f M H)
```

For `M=1`, the lower bound is 4.084; for `M=1024`, it is 5.463. The author's
next-power-of-two rule therefore selects `K=8` throughout this sweep. `K=7` is
also tested because it is the smallest tested legal split above both bounds and
gives an aligned intermediate width of 4096. At `M=1024`, Equation 12 predicts
78 MiB for K=7 and 70 MiB for K=8, while K=4 is 126 MiB and does not fit.

The 96 MiB value is not one shared cache. It is the sum of 96 private 1 MiB L2
caches. Treating it as capacity is an author-hinted placement hypothesis that
requires pinned workers and stable ownership; the experiment tests rather than
presumes that hypothesis.

## Matrix

- Active rows: `1,2,4,8,16,32,64,128,256,512,1024`.
- Timing K: TPP K=1; IMBPS K=`2,4,7,8,14,16,28,32`.
- uProf rows: `1,8,32,128,512,1024`.
- uProf K: TPP K=1; IMBPS K=`4,7,8,14,16`.
- Timing: five randomized paired rounds, five warmups, at least 20
  measurements and at least one measured second per case.
- Counters: five randomized rounds in separate cache and traffic passes, five
  warmups, at least 10 invocations and at least five counter-active seconds per
  case.

## Fixed controls

- One exclusive Slurm allocation on `mn01`, package/NUMA node 0 only.
- 96 physical cores, no SMT, complete CCDs, fixed CPU list.
- OpenMP close/core binding, dynamic teams disabled, active wait policy.
- THP must be `always`; the job fails before measurement otherwise.
- gperftools tcmalloc 2.18.1 loaded by an exact `LD_PRELOAD` path; the worker
  records mapped allocator libraries and fails if tcmalloc is absent.
- Fresh process per case, deterministic tensors, no automatic retry, no
  post-hoc outlier deletion.

## Outcomes and decision rules

Primary timing outcomes are paired speedup, median invocation latency, active
rows/s, and milliseconds/active-row from unprofiled runs. Counter runs are
mechanism evidence, not primary timing.

The cache pass records retired instructions and converts uProf's L2 access/miss
per-thousand-instruction metrics into absolute accesses and misses per measured
invocation. It also derives L2 miss/hit percentages. The traffic pass records
DRAM bytes per invocation and measured DRAM arithmetic intensity.

Report every K and row count. Call IMBPS beneficial only when its paired timing
interval is above 1 and the direction is not contradicted by the separately
profiled rounds. Do not call the speculative-decode stage improved until an
end-to-end fixed-trace verification experiment shows an accepted-token
throughput gain.
