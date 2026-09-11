# Matmul-to-paged-attention study: code and evidence guide

This guide maps every presentation claim to the code and raw result that a
reviewer can inspect. The repository revision used to prepare the deck was:

```text
main repository: https://github.com/memyselfandglitch/matmul-to-paged-attention
revision: b9a68deef18ac3f3aedb485eaa0be8bf7dddb481
vLLM checkout revision: 52358e6e192aeae73dd3764046c98fc4b156ea83
```

The AMD/IISc measurements are primary. The macOS files are retained only as a
correctness and qualitative sanity check.

## Presentation-day quick start

On the IISc server, the complete Phase 1 + Phase 2 study needs no parameters:

```bash
cd /data/scratch/deveshisingh/matmul-to-paged-attention/experiments/paged-kv-traversal-study
./run.sh
```

The command waits for its Slurm job and prints both reports automatically. To
redisplay the newest reports later:

```bash
./show_results.sh
```

## 1. Matrix multiplication optimization and loop order

Code:

- `../loop_order/matmul_optimization_stages.cpp` — baseline, B transpose,
  explicit SIMD, cache tiling, register blocking, and combined stages.
- `../loop_order/loop_order_study.cpp` — all six macro-tile loop orders with
  identical inner work at the tiled and tiled-plus-register-blocked stages.
- `../loop_order/Makefile` — native optimized build and smoke test.

Run:

```bash
cd experiments/loop_order
make check
./build/loop_order_study 1920 5 384
./build/matmul_optimization_stages 1920 5 384
```

Interpretation:

- The named order controls only the `Mc/Nc/Kc` macro-tile loops.
- The `6 x 8` register tile on a 128-bit SIMD machine uses twelve vector
  accumulators (six rows times two four-float vectors).
- In the recorded macOS sanity run used during deck preparation, register
  blocking reduced tiled time by roughly `2.9x–3.1x` across the six macro
  orders. Absolute numbers and small order differences are machine-sensitive.

## 2. Matmul, BMM, and K/V attention structure

Code:

- `../batch_matmul/bmm_kv_study.py` — uses real `torch.matmul` for both the
  looped and batched paths; there is no handwritten matrix kernel.
- `../batch_matmul/README.md` — shape definitions and attention mapping.

Run:

```bash
cd experiments/batch_matmul
python3 bmm_kv_study.py --device cpu --repetitions 21
```

One local CPU sanity run during deck preparation, PyTorch 2.4.0 with four CPU
threads, produced:

| Case | Looped / alternative |
| --- | ---: |
| 32 independent `64x64` products: loop / batched | `1.92x` |
| Shared right operand: broadcast / flattened GEMM | `0.96x` |
| Decode-shaped attention, `B=2,H=8,Q=1,T=512,D=64`: loop / batched | `2.98x` |

These are dispatch/scheduling observations from a macOS sanity run, not the
primary AMD profiling result. The structural conclusion is machine-independent:
BMM performs the same independent products under one batched operation, and
decode attention reduces to batched matrix-vector-like work when `Q=1`.

## 3. vLLM layout facts

Checked-out upstream source:

- `../vllm/vllm/v1/kv_cache_layout.py` — logical `[L,B,H,N,C]` interface and
  physical stride permutations, including `LBHNC`, `LBNHC`, and `LHBNC`.
- `../vllm/vllm/v1/attention/backends/utils.py` — backend layout resolution;
  aliases `NHD -> LBNHC` and `HND -> LBHNC`.
- `../vllm/vllm/v1/attention/backends/rocm_attn.py` — current ROCm backend
  layout preferences in this checkout.

Important distinction:

- `[L,B,H,N,C]` is the stable logical indexing contract.
- Physical layout is represented by strides/permutations and is selected from
  layouts supported by the active backend; there is no backend-independent
  universal physical default in current vLLM.
- The older installed MI210 environment used for the initial baseline accepted
  legacy `NHD/HND`; its default `NHD` maps to per-layer `BNHD`.

## 4. Paged-KV decode microbenchmark

Core code:

- `../paged-kv-traversal-study/src/paged_kv_study.cpp`
  - lines around `offset()` define physical memory layout;
  - `make_block_table()` defines sequential, shuffled, and fragmented access;
  - `visit_token()` performs scaled QK, online softmax, and weighted V;
  - `run_kernel()` defines `BNHD`, `BHND`, and `HBND` traversal orders.
- `../paged-kv-traversal-study/scripts/run_study.sh` — Phase 1/2 runner.
- `../paged-kv-traversal-study/python/analyze_phase1.py` and
  `analyze_phase2.py` — reports.

Default AMD study shape:

```text
one decode query token
B=512 physical blocks
H=32 KV heads
N=16 tokens per block
D=128 head dimension
float32 K and V
8192 cached tokens
256 MiB K+V per layer
useful arithmetic intensity = 0.5 FLOP/KV byte
```

This is a standalone CPU microbenchmark of the core decode-attention data
access and arithmetic. It is not an end-to-end vLLM or FlashInfer latency run.

### Phase 1: fixed BNHD memory, vary traversal

Primary AMD job: `7616`

- `../paged-kv-traversal-study/results/job-7616/phase1-sequential.csv`
- `../paged-kv-traversal-study/results/job-7616/phase1-shuffled.csv`
- `../paged-kv-traversal-study/results/job-7616/analysis.txt`

Run:

```bash
cd experiments/paged-kv-traversal-study
sbatch slurm/run_cpu_study.sbatch
```

Result: the memory-matched `BNHD` traversal wins for both block-table patterns.

### Phase 2: all physical layouts crossed with all traversals

Primary AMD job: `7616`

- `../paged-kv-traversal-study/results/job-7616/phase2-sequential.csv`
- `../paged-kv-traversal-study/results/job-7616/phase2-shuffled.csv`
- `../paged-kv-traversal-study/results/job-7616/phase2-analysis.txt`

Run Phase 1 and Phase 2 together on Slurm:

```bash
cd experiments/paged-kv-traversal-study
sbatch --export=ALL,RUN_PHASE2=1 slurm/run_cpu_study.sbatch
```

Result: matched traversal wins all `3/3` physical layouts. The global winner is
`HBND/HBND` for sequential blocks (`35.461 ms`) but `BHND/BHND` for fully
shuffled blocks (`37.515 ms`).

### Fragmentation crossover

Code:

- `../paged-kv-traversal-study/python/run_crossover.py`
- `../paged-kv-traversal-study/python/analyze_crossover.py`
- `../paged-kv-traversal-study/slurm/run_crossover.sbatch`

Primary AMD job: `7617`

- `../paged-kv-traversal-study/results/crossover-7617/crossover-raw.csv`
- `../paged-kv-traversal-study/results/crossover-7617/crossover-summary.csv`
- `../paged-kv-traversal-study/results/crossover-7617/crossover-analysis.txt`

Result: the timing sweep changes from HBND preference at contiguous run length
`8`, through an effective tie at `4`, to BHND preference at `2` and `1`.

### AMD uProf counter replication

Code:

- `../paged-kv-traversal-study/python/run_uprof_counters.py`
- `../paged-kv-traversal-study/python/analyze_uprof_counters.py`
- `../paged-kv-traversal-study/slurm/run_uprof_counters.sbatch`

Primary repeated AMD job: `7775`

- `../paged-kv-traversal-study/results/uprof-7775/environment.txt`
- `../paged-kv-traversal-study/results/uprof-7775/uprof-counters.csv`
- `../paged-kv-traversal-study/results/uprof-7775/uprof-analysis.txt`
- `../paged-kv-traversal-study/results/uprof-7775/raw/` — complete timing and
  uProf CSV for every layout/run-length/seed case.

Independent earlier AMD run: `7628`

- `../paged-kv-traversal-study/results/uprof-7628/findings.md`
- `../paged-kv-traversal-study/results/uprof-7628/uprof-counters.csv`

Run:

```bash
cd experiments/paged-kv-traversal-study
sbatch slurm/run_uprof_counters.sbatch
```

Defensible counter conclusion:

- Both uProf runs locate the stable refined boundary between run lengths `4`
  (tie) and `3` (BHND wins).
- From run length `8` to `1` in job 7775, HBND IPC falls from about `1.08` to
  `1.01`; BHND stays around `1.09–1.10`.
- HBND has a lower L3 miss percentage but roughly `3–9%` more L3 misses per
  1,000 retired instructions. L3 miss latency stays near `97–99 ns`.
- Therefore the crossover is associated with lower IPC and more normalized L3
  misses, but a simple “higher L3 miss percentage causes the slowdown” claim is
  contradicted by the data. Hardware prefetch effectiveness, memory-level
  parallelism, and backend memory stalls remain hypotheses to test.
- L3 and Data Fabric counters are hardware-scoped, not process-attributed;
  paired runs were pinned to core 0/CCX 0, but a quiet or exclusive CCX would
  strengthen the mechanism claim.

## 5. macOS sanity check

- `../paged-kv-traversal-study/results/macos-arm64-baseline/README.md`
- `../paged-kv-traversal-study/results/macos-arm64-baseline/`

The Mac run also shows an HBND-to-BHND preference reversal as fragmentation
increases, but the boundary differs (`16–4` rather than the AMD refined `4–3`)
and absolute times are much smaller. Use it only as qualitative corroboration.

## 6. Presentation figures

Regenerate all exact charts from the committed CSVs:

```bash
cd experiments/presentation
python3 make_figures.py
```

The PNG files are written to `figures/`. Figure subtitles include the AMD job
number and experimental shape so screenshots remain traceable.
