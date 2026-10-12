# Per-CCD concurrent inner-GEMM experiment

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: implementation
- Origin Date: 2026-10-12
- Verification Status: UNVERIFIED
- Version Label: per_ccd_inner_threads_v1

## Question

After row-sharded reduction removed the coordinator bottleneck, can a smaller
OpenMP/oneDNN inner team make twelve concurrent CCD-local PACE operators faster
than the current eight-thread-per-CCD configuration?

## Preregistered matrix

- Active rows: `2048, 4096` (the two shapes where optimized per-CCD IMBPS
  already beat TPP).
- Fixed per-CCD implementation: 12 CCD workers, K=12, zero-copy publication,
  row-sharded reduction.
- Inner threads per CCD operator: `1, 2, 4, 6, 8`.
- Fresh controls: socket-wide TPP and global IMBPS K=6.
- Total: 14 exclusive, shape-isolated Slurm tasks.

Every task uses NUMA node 0, the selected 96 physical cores, THP `always`,
tcmalloc, 20 warmups, at least 30 measured iterations, and at least five
measured seconds. Only the inner-team size changes among per-CCD cases.

## Measurements and decision rule

Primary measurement is unprofiled median end-to-end MLP latency, including
worker synchronization and row-sharded reduction. Worker operator latency and
imbalance are recorded as diagnostics.

Select an inner-team size only if it lowers clean median latency at both tested
row counts without changing the existing numerical agreement checks. If the
winner differs by row count, retain a shape-dependent policy. Run uProf only
after this sweep, and profile the current t8 case, the clean-timing winner, and
global K=6 at M=4096.

## Clean-timing outcome and mechanism follow-up

The clean sweep selected eight inner threads at both row counts. Smaller teams
were monotonically slower, worker imbalance stayed near one, and row-sharded
reduction accounted for only about five percent of latency. Global K=6 was
still faster than per-CCD K=12/t8, especially at `M=4096`.

The follow-up therefore profiles only three `M=4096` cases: socket-wide TPP,
global IMBPS K=6, and per-CCD P=12/K=12/t8. uProf records IPC, utilization,
L2 metrics, and CCD-granular L3 accesses/misses. Profiled timing is mechanism
evidence only; the clean sweep remains authoritative for latency.
