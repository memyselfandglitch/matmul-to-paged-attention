# IMBPS reproduction experiment plan

## Material Passport

- Origin skill: `academic-research-suite` / experiment-agent
- Artifact type: executable reproduction plan and validation harness
- Status: harness verified locally; hardware-dependent claims unverified
- Primary source: attached IMBPS paper, SHA-256 recorded in `CLAIMS.md`
- Implementation source: AMD-PACE v1.0 at pinned commit
- Last updated: 2026-10-01

## Objective

Determine which IMBPS claims reproduce with the released AMD-PACE operator on
the IISc EPYC 9654 machine, while separating implementation replication from
literal numerical replication of the paper's Turin results.

## Confirmatory hypotheses

1. At the Table II shapes, IMBPS K=4 is faster than the released TPP baseline
   and the paired 95% interval contains the paper's 1.21x, 1.25x, or 1.23x
   target for the corresponding batch.
2. A validated AMD L3 event shows the paper's Table II miss-reduction factor
   under the same measurement scope.
3. Table VIII reproduces the non-monotonic K curve, with timing optima K=4 for
   OPT-13B and K=8 for OPT-30B.
4. BF16 is not bit-identical, but baseline top-1 remains in the candidate top-5
   at the author-described first-token check and MMLU remains within the paper's
   reported rounded scores.

Equation 13 is evaluated as a diagnostic lower bound, not as a hypothesis that
K predicts the empirical optimum.

## Experimental strata

Never pool these strata:

- paper hardware/software, which is incompletely specified and unavailable;
- PACE v1.0 on one full EPYC 9654 socket;
- reduced-core/one-NUMA-node sensitivity runs;
- any changed PACE, PyTorch, Transformers, oneDNN, model, or dataset revision.

The primary EPYC stratum uses every physical core from exactly one socket and
all NUMA memory nodes local to that socket. SMT is disabled or excluded. CPU
affinity and affinity-visible L3 must be captured before every suite.

## Run order

1. Verify the harness and capture preflight metadata.
2. Install the pinned PACE release; rerun preflight inside the environment.
3. Run analytical checks and a small operator/numerics smoke test.
4. Pilot one Table II case to estimate runtime and memory.
5. Run five randomized paired rounds of Table II timing.
6. Run hardware counters as a separate instrumented suite.
7. Run five randomized paired rounds of Table VIII, one model at a time.
8. Run the exploratory decode active-batch/K sweep with a validated L2 event.
9. Run first-token comparisons, then MMLU.
10. Run E2E Table III/VI only after model/dataset revisions and memory capacity
   are frozen in the manifest.

## Controls and measurements

- Same deterministic synthetic tensors for paired standalone variants.
- Same seeded shape-only token stream for separate E2E baseline/candidate cases.
- Fresh process per variant; no automatic retry or post-hoc outlier deletion.
- Fixed physical-core affinity, NUMA policy, frequency governor, THP policy,
  allocator, and thread environment within a stratum.
- Warmups occur before timing and before perf attachment.
- Raw per-iteration times, medians, IQR, coefficient of variation, paired
  speedups, and deterministic bootstrap intervals are retained.
- Generic `cache-misses` is never relabeled as L3. The exact event, profiler,
  kernel, scope, and aggregation method are part of the result.

## Stopping and failure rules

- Stop a case on OOM, timeout, nonzero exit, thermal/frequency instability, or
  profiler multiplexing; preserve the failure and do not retry automatically.
- A pilot may change prospective warmup/iteration counts. Once the confirmatory
  matrix starts, changes create a new result stratum.
- “Exact reproduction” requires the paper target to pass its declared timing,
  counter, and correctness criteria. Directional improvement alone is reported
  as partial replication.

## Known blockers

- The paper's exact PACE commit, dataset, prompt construction, output-token
  count, affinity, allocator, warmup policy, and L3 event/scope are absent.
- Table IV omits K, and Table V requires Intel Sapphire Rapids. These cannot be
  called exact reproductions without additional information or hardware.
- The released PACE v1.0 dependency versions differ from the paper and lacks a
  TPP FP32 registration for the Table III baseline.
- GPU claims require a separate MI210/ROCm 5.7/PyTorch 2.3.1 environment.
