# Paged KV traversal study

This repository follows a three-phase research plan:

1. Keep the installed vLLM default KV-cache memory layout fixed and measure
   different traversal orders for one-token decode attention.
2. Only after analysing Phase 1, choose and test alternative physical memory
   layouts.
3. Sweep physical-block fragmentation from sequential runs to fully shuffled
   blocks and locate where the matched HBND/BHND preference changes.

All three phases are implemented. The lower-level scripts keep later phases
opt-in so each experiment can still be run independently.

## Presentation quick start

On the IISc Slurm server, run the complete Phase 1 + Phase 2 + fragmentation
crossover study with no arguments:

```bash
cd /data/scratch/deveshisingh/matmul-to-paged-attention/experiments/paged-kv-traversal-study
./run.sh
```

`run.sh` waits for that exact job to finish and then prints all three reports. To
redisplay the newest completed result later, use:

```bash
./show_results.sh
```

The launcher resolves paths relative to itself, so it does not depend on the
directory from which Slurm was invoked. The lower-level commands documented
below remain available when custom dimensions or individual phases are needed.

Phase 3 uses contiguous run lengths `512,256,128,64,32,16,8,4,2,1` with five
deterministic trials. Run length 512 is fully sequential; run length 1 shuffles
individual blocks; intermediate values shuffle contiguous chunks. Its report
includes the HBND/BHND ratio, trial agreement, interpretation, and the interval
in which the preference changes.

## What is actually the default?

There are two relevant vLLM versions in this workspace.

### Installed MI210 environment

The installed cluster environment is:

```text
/rhome/deveshisingh/envs/vllm-rocm721
vLLM 0.20.2rc1.dev15+g321fa2d6d
```

Its source only accepts the legacy layouts `NHD` and `HND`. With no explicit
layout and no KV connector requiring another layout, its
`get_kv_connector_cache_layout()` returns `NHD`.

For separate K and V tensors, NHD corresponds to:

```text
[physical_block, token_in_block, kv_head, head_dimension]
 B               N               H        D
```

The fused allocation used by the installed Triton backend is logically:

```text
[physical_block, K_or_V, token_in_block, kv_head, head_dimension]
```

Therefore, **Phase 1 fixes physical memory as `BNHD`**.

### Newer checked-out vLLM source

The newer source checkout at `../vllm` defines a general logical interface
`[L,B,H,N,C]` and multiple physical stride permutations. It chooses a layout
from the active backend's preference list before allocation. In that newer
code there is no universal default independent of backend.

This difference matters: the research baseline must be tied to the installed
version used for the measurement, not an assumed timeless vLLM default.

Primary upstream references:

- [vLLM KVCacheLayout API](https://docs.vllm.ai/en/latest/api/vllm/v1/kv_cache_layout/)
- [VLLM_KV_CACHE_LAYOUT configuration](https://docs.vllm.ai/en/latest/configuration/env_vars/)
- [Layout resolution](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/backends/utils/)

## What Phase 1 computes

There is one decode query per KV head:

```text
Q: [H,D]
K: [B,N,H,D]
V: [B,N,H,D]
```

For every head, the benchmark performs scaled QK dot products, an online
softmax over all cached tokens, and the weighted V accumulation:

```text
scores[h,t] = dot(Q[h,:], K[t,h,:]) / sqrt(D)
output[h,:] = softmax(scores[h,:]) @ V[:,h,:]
```

The online-softmax formulation gives the same result without materialising the
full score array. This is the core computation of one-token decode attention.

The benchmark includes a vLLM-style block table. It runs both:

- sequential physical blocks, to expose best-case spatial locality;
- deterministically shuffled physical blocks, to represent paged allocation.

It does not yet include request scheduling, GPU thread-block scheduling, mixed
sequence lengths, quantised KV cache, or a complete transformer layer. The
reported number is therefore **attention-kernel time**, not total model token
latency.

## Traversals in Phase 1

Memory remains `BNHD` in all three cases:

| Name | Loop order | Purpose |
| --- | --- | --- |
| `BHND` | block, head, token, dimension | requested block-first computation |
| `HBND` | head, block, token, dimension | requested head-first computation |
| `BNHD` | block, token, head, dimension | control matching default memory |

The third case is necessary. Because the installed default has token before
head in physical memory, both requested traversals are otherwise mismatched
with memory. Without the `BNHD` control, we could compare head-first and
block-first but could not tell how either compares with the natural traversal
of the default layout.

All three kernels use identical K, V, Q, block table, arithmetic, precision,
and output validation. Only loop nesting changes.

## Build and run Phase 1

```bash
./scripts/run_study.sh
```

Override dimensions without editing the source:

```bash
BLOCKS=512 HEADS=32 BLOCK_SIZE=16 HEAD_DIM=128 REPEATS=9 \
  ./scripts/run_study.sh
```

The default runner produces:

```text
results/phase1-sequential.csv
results/phase1-shuffled.csv
```

Analyse those two files with:

```bash
python3 python/analyze_phase1.py \
  results/phase1-sequential.csv \
  results/phase1-shuffled.csv
```

On IISc, submit from this repository's root directory:

```bash
sbatch slurm/run_cpu_study.sbatch
```

The Slurm output path is relative to the submission directory.
Each Slurm run writes its CSV files and `analysis.txt` under
`results/job-<job-id>/`, preventing parameter-sweep jobs from overwriting one
another.

## Run Phase 2

Phase 2 crosses every physical layout with every traversal:

| Physical memory | `BNHD` traversal | `BHND` traversal | `HBND` traversal |
| --- | --- | --- | --- |
| `BNHD` | matched | mismatched | mismatched |
| `BHND` | mismatched | matched | mismatched |
| `HBND` | mismatched | mismatched | matched |

The complete matrix is measured with both sequential and shuffled block tables:

```bash
RUN_PHASE2=1 ./scripts/run_study.sh
```

This produces:

```text
results/phase2-sequential.csv
results/phase2-shuffled.csv
results/phase2-analysis.txt
```

The analysis reports the best traversal for each physical layout, the global
layout/traversal winner, the head-first versus block-first ratio, and
sensitivity to block-table shuffling. All nine cases use identical logical K,
V and Q values and are checked against the same output.

The three per-layer layouts correspond to `LBNHC`, `LBHNC`, and `LHBNC` in the
newer vLLM physical-layout interface. Backend support varies, so this standalone
matrix studies the layouts independently of a particular production backend.

## Fragmentation crossover study

Phase 2 showed that matched `HBND` wins with sequential blocks while matched
`BHND` wins with a shuffled block table on the target AMD CPU. The crossover
study replaces those two extremes with a controlled contiguous-run length.
Physical block runs are shuffled, but block numbers inside each run remain
consecutive. A run length equal to the number of blocks is sequential; run
length one is maximally fragmented.

The top-level `./run.sh` includes this as Phase 3 and writes its files beside
the Phase 1 and Phase 2 results under `results/job-<job-id>/`. For the same
integrated run without Slurm, use:

```bash
RUN_PHASE2=1 RUN_CROSSOVER=1 ./scripts/run_study.sh
```

Submit the focused timing study on IISc:

```bash
sbatch slurm/run_crossover.sbatch
```

Its default shape remains `B=512, H=32, N=16, D=128`, and it tests run lengths
`512,256,128,64,32,16,8,4,2,1`. Only the three memory-matched cases are timed.
Each point is repeated with five deterministic block-table seeds. Results are:

```text
results/crossover-<job-id>/crossover-raw.csv
results/crossover-<job-id>/crossover-summary.csv
results/crossover-<job-id>/crossover-analysis.txt
```

The useful arithmetic intensity is `0.5 FLOP/KV byte`: conventional QK and
weighted-V account for four useful floating-point operations per head-dimension
element, while reading K and V consumes eight bytes in float32. This deliberately
excludes online-softmax bookkeeping and assumes Q/output state is cache-resident.

## Hardware-counter study

After locating the crossover, profile the two matched competitors at the
extremes and around the switch:

```bash
RUN_LENGTHS=512,32,16,8,4,1 sbatch --export=ALL slurm/run_perf_counters.sbatch
```

The isolated-case mode ensures each `perf stat` invocation contains only one
layout/traversal pair. It collects cycles, instructions, generic cache events,
L1D and LLC load events, and data-TLB events. A high repetition count makes the
decode kernels dominate process startup; cache scrubbing is disabled inside
this counter run so scrub traffic does not contaminate the counters.

Outputs are:

```text
results/perf-<job-id>/perf-counters.csv
results/perf-<job-id>/perf-analysis.txt
```

Some events may be unavailable under the node's kernel or `perf_event_paranoid`
policy. The raw CSV leaves unsupported counters empty rather than inventing a
value.

### AMD uProf L3 and memory counters

On the IISc EPYC server, Linux perf mode does not expose the `amd_l3` or
`amd_df` PMU devices. The installed uProf binary has capabilities configured
for non-root MSR access, so use the dedicated MSR-mode study instead:

```bash
sbatch slurm/run_uprof_counters.sbatch
```

The default run pins every benchmark process to core 0 with `taskset` and
collects counters for CCX 0, which contains that core. CCX scope is required by
uProf for normalized L3 metrics. The study compares matched `BHND` and `HBND`
at fragmented run lengths `8,4,3,2,1` and repeats each point with block-table
seeds `0,1,2`. It collects IPC, L3 accesses, L3 misses, L3 miss latency, and
Data Fabric memory bandwidth in the same process.

Results are written under:

```text
results/uprof-<job-id>/uprof-counters.csv
results/uprof-<job-id>/uprof-analysis.txt
results/uprof-<job-id>/raw/
```

The `raw/` directory preserves both the benchmark timing CSV and complete
AMDuProfPcm report for every isolated case. The L3 and Data Fabric counters are
hardware-scoped rather than process-attributed, so paired runs should be made
on the same core while the corresponding CCX and system are otherwise quiet.

## Inspect newer vLLM layout definitions

The source probe reuses the newer checkout's actual `KVCacheLayout` enum:

```bash
python python/vllm_layout_probe.py --vllm-source ../vllm
```
