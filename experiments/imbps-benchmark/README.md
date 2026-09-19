# IMBPS CPU Benchmark

This repository independently evaluates Iterative MLP Blocks with Parameter
Splits (IMBPS). It contains both an isolated mathematical MLP microbenchmark
and real Hugging Face OPT layer/end-to-end benchmarks.

The harness provides:

- unsplit reference OPT and SwiGLU kernels;
- sequential IMBPS execution with one reusable activation block and output
  accumulator;
- ordinary split views and persistent contiguous prepacked weight blocks;
- correctness checks for every split before it is timed;
- randomized paired baseline/IMBPS observations;
- raw CSV, summary CSV, run manifest, and machine/software metadata;
- support for uneven split counts such as `K=6` or `K=7`;
- Hugging Face OPT `fc1 -> configured activation -> fc2` execution using native
  `nn.Linear` weight layout and both biases;
- complete `OPTForCausalLM` execution with paired reference/IMBPS runs and
  prefill latency, TTFT, decode throughput, and output-token throughput.

The original `sweep` command is deliberately synthetic. The `hf-opt-layer` and
`hf-opt-e2e` commands load actual Transformers classes and, in pretrained mode,
actual model weights. Hugging Face OPT defaults to **ReLU**, not GELU; the real
model commands always use and record `config.activation_function`.

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

On a PEP 668-managed Ubuntu system with no preinstalled PyTorch, create the
environment on the scratch filesystem with:

```bash
cd /data/scratch/deveshisingh/imbps-benchmark
./scripts/bootstrap_venv.sh
source .venv/bin/activate
```

The bootstrap script does not modify system Python. It puts the virtual
environment, pip cache, and temporary downloads below the repository and uses
the official PyTorch CPU wheel index. If the cluster blocks outbound downloads,
load a cluster-provided Python/PyTorch module first or set `TORCH_INDEX_URL` to
the site's wheel mirror.

For the real Hugging Face experiments, install the paper-matched Transformers
4.48 dependency line:

```bash
./scripts/bootstrap_hf_venv.sh
source .venv/bin/activate
```

This remains inside the repository virtual environment and does not modify the
PEP 668-managed system Python. The Zen 4 scripts also place Hugging Face model
downloads under the ignored `.cache/huggingface` directory on the scratch
filesystem instead of consuming the login-home quota. Override `HF_HOME` when a
shared cluster model cache is available.

## Validation

Run the unit tests and FP32/BF16 kernel checks:

```bash
./scripts/run_tests.sh
```

The numerical comparison uses both absolute and relative errors. Relative error
can be very large near a zero-valued reference element, so pass/fail is based on
`torch.allclose`; both raw error values are retained for analysis.

The test suite also checks native Hugging Face Linear layout, uneven splits,
`fc1` and `fc2` bias semantics, model patch enable/restore behavior, and the
expected workspace reduction.

## Real Hugging Face OPT layer benchmark

Run an actual pretrained OPT layer on one Zen 4 CCD:

```bash
MODEL=facebook/opt-125m \
BATCH_SIZE=1 \
SEQUENCE_LENGTH=256 \
OUTPUT_DIR=results/hf-opt125m-layer-m256 \
./scripts/run_hf_opt_layer_zen4.sh
```

For OPT-6.7B:

```bash
MODEL=facebook/opt-6.7b \
BATCH_SIZE=1 \
SEQUENCE_LENGTH=1024 \
OUTPUT_DIR=results/hf-opt6.7b-layer-m1024 \
./scripts/run_hf_opt_layer_zen4.sh
```

The default `--input-source captured` registers a pre-hook on the selected
layer's real `fc1`, begins a normal model forward, clones the tensor delivered
to that projection, and stops the model immediately. Thus the timed MLP sees an
activation produced by embeddings, attention, residuals, and layer norm rather
than a normal-distribution placeholder. `--input-source random` is available
for shape-controlled ablations.

The reference is exactly:

```python
layer.fc2(layer.activation_fn(layer.fc1(hidden_states)))
```

Hugging Face stores `fc1.weight` as `[intermediate, hidden]` and `fc2.weight` as
`[hidden, intermediate]`. IMBPS slices rows of `fc1` and columns of `fc2`; it
adds each `fc1` bias slice before activation and adds the `fc2` bias exactly once
after accumulating all partial down projections. `prepacked` makes persistent
contiguous block copies and excludes the recorded one-time packing cost from
steady-state latency.

Use `--model-mode random-config --input-source random` for a download-light
framework smoke test. That path still uses the real Transformers
`OPTDecoderLayer` class, activation, biases, and Linear layouts, but not
pretrained values.

## Complete OPT benchmark

Run the full model, with every decoder MLP toggled between the Hugging Face
reference and IMBPS replacement:

```bash
MODEL=facebook/opt-125m \
BATCH_SIZE=1 \
INPUT_TOKENS=256 \
OUTPUT_TOKENS=16 \
OUTPUT_DIR=results/hf-opt125m-e2e \
./scripts/run_hf_opt_e2e_zen4.sh
```

The replacement does not duplicate or reimplement the OPT decoder layer. It
temporarily makes the existing `fc1` and activation slots pass-throughs and
lets the replacement `fc2` slot execute the complete split MLP. Hugging Face's
attention, SDPA selection, layer norms, residuals, dropout-in-eval behavior,
position handling, LM head, and KV-cache path remain unchanged. The patch is
restored after every K experiment.

Generation is fixed-length greedy decoding so every variant performs identical
work and EOS cannot shorten a sample. Model loading, weight packing, input
creation, correctness checks, and warmups are outside timed samples. The
reported metrics are:

- `prefill_ms`: prompt forward through `OPTForCausalLM`, including KV-cache and
  prompt logits creation;
- `ttft_ms`: prefill plus first-token argmax;
- `decode_ms`: KV-cached forwards for the remaining `output_tokens - 1` tokens;
- `decode_tokens_per_second`: batch-scaled decode tokens divided by decode time;
- `output_tokens_per_second`: all requested generated tokens divided by total
  generation time.

Before timing each K, the harness compares first-token logits with
`torch.allclose` and records whether all generated token IDs match exactly.
Raw observations go to `raw.csv`; medians, p95 values, and paired speedups go to
`summary.csv`.

Memory columns distinguish logical activation size from implementation-owned
storage. `logical_activation_bytes` (layer CSV) and
`prefill_activation_bytes_per_layer` (end-to-end CSV) compare the full
intermediate width with the largest split width. `workspace_bytes` reports only
the persistent reusable buffers owned by the IMBPS adapter; native Hugging Face
allocations are transient and therefore show zero in that column rather than
implying that the reference uses no activation memory.

Omitting `--prompt` uses deterministic random vocabulary IDs, matching AMD
PACE's controlled synthetic-input approach. Passing `--prompt 'text...'`
loads the model tokenizer and pads/truncates to `--input-tokens`.

## Relationship to the paper and AMD PACE

The paper reports Transformers 4.48, oneDNN 3.7, SDPA for Hugging Face, BF16
and FP32, sequence lengths 256 and 1920, and TTFT as its principal end-to-end
metric. Its batch sizes make the effective MLP row count `M = batch * sequence`
much larger than the initial synthetic `M=256/1024/4096` experiments. Start
with OPT-125M to validate the workflow, then use OPT-6.7B with the paper's
batch/sequence grid as memory and run time permit.

AMD PACE's offline benchmark informed four measurement choices here: fixed
input/output token counts, deterministic reusable inputs, explicit warmup/run
counts, and separate TTFT/token-throughput reporting. Like PACE, this harness
does not enable a background CPU/RAM monitor during absolute timing because
monitoring can perturb the measurement. PACE itself is not a runtime dependency.

Transformers versions before 4.48 may reject OPT `--attn-implementation sdpa`;
use the pinned HF bootstrap above. `--attn-implementation eager` remains
available as an explicit ablation, but results from different attention
implementations must not be combined.

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
GEMMs. It is deliberately not named batch size: for the paper's three-dimensional
input `[B, C, H]`, the Linear modules flatten the active row count to `M = B*C`.
Record the actual runtime tensor shape when comparing end-to-end workloads.

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
