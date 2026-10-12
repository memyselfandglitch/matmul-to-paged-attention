# Per-CCD IMBPS optimization plan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-12
- Verification Status: UNVERIFIED
- Version Label: per_ccd_optimization_v1

## Research question

Can the threaded per-CCD OPT-30B MLP prototype beat socket-wide TPP or global
IMBPS after removing avoidable partial-output copies and the CCD-0-only final
reduction?

## Existing evidence and diagnosis

The current pthread prototype improves L3 locality but concentrates the final
reduction on CCD 0:

- At `M=4096`, `thread-p12-k12` spends about 116 ms of 349 ms under uProf in
  reduction and conversion (33%).
- CCD 0 records 115.3 million L3 misses per invocation, versus about 30-32
  million on each other CCD. Its utilization is 97.7%, versus a 57.3% median
  across the other CCDs.
- The same imbalance is visible at `M=1024`: CCD-0 utilization is 92.0%, versus
  55.3% elsewhere.
- The current thread worker copies every returned partial into a second tensor
  before reduction. At `M=4096`, this republishes twelve 56 MiB outputs before
  the reduction reads them again.
- The earlier clean matrix used only three warmups and seven measured calls for
  four shapes in one process. Shape-isolated uProf runs were 8-43% faster for
  threaded variants, so the optimized timing matrix isolates every shape and
  requires at least 20 warmups, 30 measurements, and five measured seconds.

These observations support two specific implementation changes:

1. **Zero-copy publication:** keep each worker's returned tensor in a shared
   Python reference rather than copying it into a second partial buffer.
2. **Parallel row-sharded reduction:** give every CCD a disjoint output-row
   range and reduce all partials for that range locally, instead of reducing
   the complete output on CCD 0.

## Preregistered matrix

Active rows:

`128, 256, 384, 512, 768, 1024, 2048, 4096`

Controls and variants:

- socket-wide TPP;
- global IMBPS K=6 and K=12;
- one-CCD pthread baseline;
- six-CCD K=6 with the original copy/coordinator reduction;
- twelve-CCD K=12 with the original copy/coordinator reduction;
- twelve-CCD K=12 with zero-copy publication only;
- twelve-CCD K=12 with zero-copy publication and row-sharded reduction;
- twelve-CCD K=24 with zero-copy publication and row-sharded reduction.

Every timing task runs one shape under an exclusive Slurm allocation, THP
`always`, tcmalloc, NUMA node 0, and the same 96-core socket mask.

## Measurements

Primary:

- unprofiled median latency and bootstrap interval versus TPP;
- latency versus global IMBPS at the same K.

Mechanism:

- worker PACE-operator minimum/maximum latency and imbalance ratio;
- publication-copy latency;
- reduction latency and fraction of total latency;
- CCD-level utilization, IPC, L2 metrics, L3 misses, and L3 hit rate;
- CCD-0 versus other-CCD utilization and L3-miss imbalance.

## Decision rules

- The zero-copy change is useful if it lowers latency relative to the original
  K=12 copy variant without changing output agreement beyond the existing BF16
  tolerance.
- The row-sharded reduction fixes the diagnosed hotspot only if it reduces both
  reduction latency and CCD-0/other-CCD counter imbalance.
- A variant is a performance improvement only if its unprofiled confidence
  interval is above 1.0 versus the selected control. Better cache counters alone
  are not sufficient.
- If compute remains dominant after reduction is fixed, select CCD count using
  worker-operator latency and imbalance; do not attribute the remaining gap to
  synchronization without evidence.

## Implementation constraints

The installed PACE v1.0 kernel is not patched. The prototype still launches
concurrent PACE operators from persistent CCD-pinned pthreads. oneDNN documents
OpenMP, TBB, sequential, and explicit threadpool CPU runtimes, and recommends
matching the library runtime to the application runtime. The audit therefore
records the behavior of the installed build rather than assuming that
concurrent inner OpenMP teams are efficient.

Primary references:

- [oneDNN CPU runtime and concurrent-execution build options](https://uxlfoundation.github.io/oneDNN/dev_guide_build_options.html)
- [oneDNN CPU benchmarking and affinity guidance](https://uxlfoundation.github.io/oneDNN/dev_guide_performance_settings.html)
- [OpenMP nested execution model](https://www.openmp.org/spec-html/5.0/openmpse3.html)
- [AMD uProf PCM CCD aggregation](https://docs.amd.com/r/en-US/57368-uProf-user-guide/4.5.-AMDuProfPcm-Command-Line-Options)
