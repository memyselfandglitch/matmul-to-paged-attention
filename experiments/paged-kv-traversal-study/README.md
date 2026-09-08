# Paged KV traversal study

This repository follows a two-phase research plan:

1. Keep the installed vLLM default KV-cache memory layout fixed and measure
   different traversal orders for one-token decode attention.
2. Only after analysing Phase 1, choose and test alternative physical memory
   layouts.

Both phases are implemented. Phase 2 remains opt-in so the fixed-layout
baseline can still be run independently.

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

## Inspect newer vLLM layout definitions

The source probe reuses the newer checkout's actual `KVCacheLayout` enum:

```bash
python python/vllm_layout_probe.py --vllm-source ../vllm
```
