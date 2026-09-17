# IMBPS CPU Benchmark

This repository independently evaluates Iterative MLP Blocks with Parameter
Splits (IMBPS) for OPT/GPT-style GELU MLPs and Llama-style SwiGLU MLPs. It is a
microbenchmark for the MLP transformation itself; it does not claim end-to-end
Hugging Face or vLLM results.

The harness provides:

- unsplit reference OPT and SwiGLU kernels;
- sequential IMBPS execution with one reusable activation block and output
  accumulator;
- ordinary split views and persistent contiguous prepacked weight blocks;
- correctness checks for every split before it is timed;
- randomized paired baseline/IMBPS observations;
- raw CSV, summary CSV, run manifest, and machine/software metadata;
- support for uneven split counts such as `K=6` or `K=7`.

## Target machine

The initial host, `mn01`, has two AMD EPYC 9654 sockets:

- 192 physical cores total, with SMT disabled;
- 1 MiB private L2 per core;
- 24 independent 32 MiB L3 domains;
- eight cores sharing each L3 domain;
- NUMA node 0 on CPUs 0-95 and node 1 on CPUs 96-191;
- approximately 1.5 TiB RAM;
- `perf_event_paranoid=-1`;
- `schedutil` governor and boost enabled;
- no AMD uProf command currently visible in `PATH`.

The first controlled scope is therefore CPUs `0-7` with memory on NUMA node 0.
Do not use the aggregate 768 MiB L3 value to choose `K`; one participating CCD
has 32 MiB.

## Installation

Python 3.9 or newer and a CPU build of PyTorch 2.3 or newer are required.

```bash
cd imbps-benchmark
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

If the server already has the intended PyTorch build, omit the requirements
installation and use `pip install -e . --no-deps`. Record the exact wheel and
oneDNN configuration in `metadata.json` before comparing runs.

## Validation

Run the unit tests and FP32/BF16 kernel checks:

```bash
./scripts/run_tests.sh
```

The numerical comparison uses both absolute and relative errors. Relative error
can be very large near a zero-valued reference element, so pass/fail is based on
`torch.allclose`; both raw error values are retained for analysis.

## First pinned Zen 4 run

The default script runs an OPT-6.7B-shaped MLP layer (`H=4096`, `I=16384`) with
256 active rows across one eight-core/32-MiB L3 domain:

```bash
./scripts/run_zen4_ccd.sh
```

The command is equivalent to:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
OMP_PROC_BIND=close OMP_PLACES=cores \
numactl --physcpubind=0-7 --membind=0 \
python -m imbps_bench sweep \
  --mlp-kind opt \
  --tokens 256 \
  --hidden-size 4096 \
  --intermediate-size 16384 \
  --dtype bf16 \
  --splits 1,2,4,5,6,7,8,16,32,64 \
  --threads 8 \
  --warmup 3 \
  --repeats 10 \
  --weight-layout prepacked \
  --output-dir results/zen4-opt-first
```

Here `--tokens` means the actual flattened row dimension `M` passed to the MLP
GEMMs. It is deliberately not named batch size: the paper's relationship among
batch size, sequence length, and the analytical `B*C` term is ambiguous. Record
actual runtime tensor shapes before translating end-to-end workloads into `M`.

Override the script through environment variables. For example, a
Llama-3.1-8B-shaped SwiGLU layer is:

```bash
MLP_KIND=swiglu \
HIDDEN_SIZE=4096 \
INTERMEDIATE_SIZE=14336 \
TOKENS=256 \
OUTPUT_DIR=results/zen4-llama8b-m256 \
./scripts/run_zen4_ccd.sh
```

To measure the packing penalty rather than a deployment with persistent blocked
weights, set `WEIGHT_LAYOUT=views`. Prepacking time is always reported separately
from steady-state MLP latency.

## Direct CLI examples

Capture metadata without running a benchmark:

```bash
python -m imbps_bench metadata --output results/mn01-metadata.json
```

Evaluate the paper's Equations 12-13 against one 32 MiB L3 domain:

```bash
python -m imbps_bench predict-k \
  --tokens 256 \
  --hidden-size 4096 \
  --intermediate-size 16384 \
  --dtype bf16 \
  --cache-mib 32 \
  --cache-fraction 1.0
```

The command reports the continuous and ceiling values of `K`, the paper-model
working set at every requested split, and an explicit infeasible result when the
input term alone exceeds usable cache. Treat the answer as a hypothesis: real
usable cache is smaller than nominal capacity and the equation omits backend
packing, scratchpads, output residency, and competing threads.

Test a paper-relevant OPT-13B layer:

```bash
OUTPUT_DIR=results/zen4-opt13b-m256 \
HIDDEN_SIZE=5120 INTERMEDIATE_SIZE=20480 TOKENS=256 \
./scripts/run_zen4_ccd.sh
```

Test a paper-relevant OPT-30B layer:

```bash
OUTPUT_DIR=results/zen4-opt30b-m256 \
HIDDEN_SIZE=7168 INTERMEDIATE_SIZE=28672 TOKENS=256 \
./scripts/run_zen4_ccd.sh
```

The runner estimates its simultaneous allocation before creating tensors and
refuses configurations above 16 GiB by default. Override this deliberately with
`--max-allocation-gib`; the limit is a safety guard, not a model of system RAM.

## Outputs

Every sweep directory contains:

- `raw.csv`: one row per timed call, including the paired execution order;
- `summary.csv`: median, mean, p95, standard deviation, throughput, GFLOP/s, and
  speedup relative to the paired reference for each `K`;
- `metadata.json`: CPU/cache topology, affinity, NUMA output, governors, PyTorch
  configuration, threading environment, tool availability, and CLI arguments;
- `manifest.json`: run state, configuration, paths, and row counts.

The reference appears once for every `K`. This is intentional: each candidate is
paired with a temporally adjacent reference so drift is visible rather than being
hidden behind one baseline measured at the beginning of a long sweep.

## Recommended experiment order

1. Run correctness and the default one-CCD sweep.
2. Repeat the default sweep three times in fresh processes.
3. Compare `prepacked` with `views` to isolate persistent packing benefits.
4. Sweep `M` over `1,16,64,256,1024,4096` while retaining the same CPU domain.
5. Repeat for OPT-13B, OPT-30B, and Llama-3.1-8B shapes.
6. Compare one core, one CCD, one NUMA node, and one socket as distinct studies.
7. Add PMU collection only after confirming the exact events exposed by the
   server kernel with `./scripts/list_perf_events.sh`.

Do not label generic `perf cache-misses` as L3 misses. Use it only as a permission
smoke test. The paper claim requires an AMD L3/LLC event or AMD uProf L3 metric,
plus L3 accesses, miss percentage, miss latency, and DRAM traffic. Counter groups
should be collected in separate repeated runs if multiplexing would be required.

## Methodological boundaries

- Timed regions exclude Python setup, tensor generation, metadata collection,
  correctness comparison, and explicit prepacking.
- Both kernels use persistent workspaces, preventing allocator overhead from
  being misreported as an IMBPS benefit.
- The IMBPS implementation is sequential across expansion splits and relies on
  the GEMM backend for intra-GEMM parallelism. Parallel split-K plus reduction is
  a separate algorithm and should be benchmarked separately.
- This implementation is intentionally at PyTorch operator level. A later native
  oneDNN/C++ kernel is needed to measure post-op fusion, scratchpad control, and
  truly persistent library-level packing without Python dispatch per split.
