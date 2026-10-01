# IMBPS reproduction harness

This project reproduces the measurable claims in **IMBPS - Iterative MLP
Blocks with Parameter Splits for Improving LLM Inference** while keeping the
paper, the authors' follow-up email, and the released AMD-PACE code as separate
sources of evidence.

The target CPU is an AMD EPYC 9654 (Genoa, 96 cores). That is not the paper's
primary Turin system (128 cores, stated as 512 MiB L3), so the harness reports
both:

1. **implementation replication** - the same PACE IMBPS operator and workloads;
2. **result replication** - whether the paper's numerical targets are met.

Do not call a directional improvement an exact reproduction. Every result is
compared against the paper target and retains the machine, software, affinity,
counter, and run metadata needed to explain a mismatch.

## What is implemented

- Equation 12/13 working-set calculations, including strict-integer,
  paper-nearest, and author-reported power-of-two policies.
- Version-pinned AMD-PACE v1.0 setup at commit
  `cfbe8b551cca18c686b771144c18129242796ea1`.
- Standalone OPT-style MLP timing with the released
  `torch.ops.pace.mlp_mlp_fusion` operator.
- A PACE TPP baseline, which is the baseline named in the authors' email.
- Table II and Table VIII timing/split matrices.
- A warmup-before-attach `perf stat` path for process-scoped counters.
- BF16/FP32 numerical comparison across split counts.
- PACE v1.0 E2E and MMLU configuration generation.
- Machine, NUMA, cache, kernel, compiler, Python, PyTorch, PACE, affinity,
  transparent-huge-page, and profiler provenance capture.
- A preregistered execution and decision protocol in `EXPERIMENT_PLAN.md`.

The numeric target registry is in `configs/claims.json`; the evidence and open
reproducibility gaps are in `CLAIMS.md`.

## Important blockers to literal reproduction

The released artifacts do not completely specify the paper experiment:

- PACE v1.0 uses PyTorch 2.7.0, Transformers 4.51.3, and oneDNN 3.8. The paper
  states Transformers 4.48 and oneDNN 3.7.
- The paper does not state its PACE commit, output-token count, exact
  summarization dataset, thread/NUMA placement, allocator, warmup/repetition
  policy, or L3 counter command/scope.
- The v1.0 performance script calls its workload "real data" but hard-codes
  GSM8K questions, not a summarization dataset.
- Equation 13 is a lower bound derived from a strict inequality, yet the paper
  says it is rounded to the nearest integer. The authors' email instead says
  they used the next power of two and then overrode it empirically.
- On one full EPYC 9654 socket, aggregate L3 is 384 MiB. For Table II's
  `B=16, C=1920, H=7168, BF16` shape, the unsplit input term alone is 420 MiB,
  so Equation 13 has no finite solution at 384 MiB.

These are experiment inputs, not nuisances to smooth over. The harness records
them and avoids silently choosing favorable values.

## Copy to the IISc machine

From the local workspace:

```bash
rsync -a imbps-reproduction/ USER@HOST:/data/scratch/USER/imbps-reproduction/
ssh USER@HOST
cd /data/scratch/USER/imbps-reproduction
```

Use the actual SSH alias and scratch path configured for your account.

## Stage 0 - preflight

First verify the hardware-independent harness locally or on the remote host:

```bash
./scripts/verify_harness.sh
```

Then run preflight with the same one-socket physical-core allocation that will
be used for measurement. On Slurm, review the site-specific resource flags and
submit:

```bash
sbatch slurm/preflight.sbatch
```

On a dedicated host, `./scripts/preflight.sh` is sufficient only if the shell's
CPU affinity already covers exactly the intended socket. Inspect the generated
`results/preflight-*/environment.json` before installing or benchmarking.
Confirm at least:

- CPU model is EPYC 9654;
- the selected CPU set includes every physical core from exactly one socket and
  no SMT siblings;
- all NUMA nodes local to that socket are understood and used consistently;
- AVX-512 and BF16 support are present;
- frequency governor, SMT, THP, `perf_event_paranoid`, machine-wide L3, and
  affinity-visible L3 capacity are recorded;
- no other job is sharing the node.

## Stage 1 - install the released implementation

PACE v1.0 builds oneDNN and libXSMM from source and requires network access,
GCC 12+, substantial RAM, and time:

```bash
./scripts/bootstrap_pace_v1.sh
source .venv/bin/activate
```

The script refuses a PACE checkout whose resolved commit does not match the
pinned v1.0 commit. It does not install or change system packages.

PACE's performance guide recommends `tcmalloc`, but the paper does not report
its allocator. Check whether `libtcmalloc.so` is already installed with
`ldconfig -p | grep tcmalloc`. If you use it, set `LD_PRELOAD` to the exact
library path before preflight and keep all paired cases under that setting. Do
not pool allocator-on and allocator-off results.

## Stage 2 - check the analytical model

```bash
source .venv/bin/activate
python src/cache_model.py \
  --batch 16 --sequence 1920 --hidden 7168 --intermediate 28672 \
  --bytes-per-element 2 --cache-mib 512 --splits 4

python src/cache_model.py \
  --batch 16 --sequence 1920 --hidden 7168 --intermediate 28672 \
  --bytes-per-element 2 --cache-mib 750 --splits 4
```

Expected diagnostics:

- 512 MiB: Equation 13 lower bound about 23; `K=4` working set about 938 MiB.
- 750 MiB: lower bound about 6.3; next power of two is 8.
- 384 MiB: no finite K because the input term is already about 420 MiB.

## Stage 3 - smoke test

For this small smoke test, using a subset of cores is fine. Replace node/core
choices after reading preflight output:

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=48
export OMP_PROC_BIND=close
export OMP_PLACES=cores
numactl --cpunodebind=0 --membind=0 \
  python src/pace_mlp_bench.py \
    --backend imbps --batch 1 --sequence 64 \
    --hidden 768 --intermediate 3072 --splits 4 \
    --warmups 2 --iterations 5 --output results/smoke.json
```

Then run the numerical check:

```bash
numactl --cpunodebind=0 --membind=0 \
  python src/check_numerics.py \
    --hidden 768 --intermediate 3072 --rows 128 \
    --splits 1,2,4,8,16 --dtype bf16 --activation relu \
    --output results/numerics-opt125m.json
```

The BF16 split result is not expected to be bit-identical. Report maximum and
quantile errors, exact-element fraction, and argmax/top-5 containment instead
of reducing correctness to a single permissive `allclose`.

For the author-described selected-logit check, dump identical first-token
inputs from separate clean model loads and compare them:

```bash
python src/dump_first_token.py \
  --model facebook/opt-125m --backend tpp --splits 1 \
  --samples 100 --sequence 32 --batch-size 4 \
  --output results/opt125m-tpp-k1.npz

python src/dump_first_token.py \
  --model facebook/opt-125m --backend imbps --splits 4 \
  --samples 100 --sequence 32 --batch-size 4 \
  --output results/opt125m-imbps-k4.npz

python src/compare_first_token.py \
  --baseline results/opt125m-tpp-k1.npz \
  --candidate results/opt125m-imbps-k4.npz \
  --output results/opt125m-k4-first-token-comparison.json
```

This records full-vocabulary last-position log probabilities, greedy-token
agreement, and whether the baseline top-1 remains in the candidate top-5.

## Stage 4 - Table II standalone timing

The paper target is OPT-30B, BF16, sequence length 1920, batch sizes 16/32/64,
and empirical `K=4`. The preferred cluster launch uses one full EPYC 9654
socket (96 physical cores). It evaluates K=4, 8, and 16 so the author's stated
empirical selection of K=4 is tested, not assumed:

```bash
sbatch slurm/table_ii.sbatch
```

For a direct launch, substitute the exact CPU list and every NUMA node belonging
to that socket, as reported by `lscpu` and `numactl`:

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=96
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export LIBXSMM_BLOCK_SIZE=32
numactl --physcpubind="CPU_LIST" --membind="NUMA_NODE_LIST" \
  python src/run_standalone_matrix.py \
    --claim table_ii --rounds 5 --warmups 3 --iterations 7
```

The runner alternates a TPP baseline and IMBPS cases in deterministic shuffled
order, creates a fresh process per case, never retries a failed case, and writes
raw JSON plus `summary.csv` under a timestamped result directory.

OPT's released model configuration uses ReLU. The primary standalone and decode
matrices therefore use ReLU; GELU results are a separate sensitivity stratum.

Use three rounds for a pilot and five or more for the confirmatory run. Increase
iterations only after checking job time and thermal stability. A fresh process
per case prevents packed weights and oneDNN primitive state from leaking
between variants.

## Stage 5 - Table VIII split sensitivity

Start with one model, because the full matrix is expensive:

```bash
MODELS=opt13b sbatch slurm/table_viii.sbatch
```

Then submit with `MODELS=opt30b`. Table VIII tests K=1/2/4/8/16/32/64 at batch 512
and sequence length 256. In this harness, K=1 is the TPP no-split baseline; K>1
uses the IMBPS fused operator.

## Stage 6 - hardware counters

First inventory events:

```bash
perf list | grep -Ei 'amd_l3|l3|llc|cache-miss'
cat /proc/sys/kernel/perf_event_paranoid
```

Then profile the same cases after warmup:

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=96
export OMP_PROC_BIND=close
export OMP_PLACES=cores
numactl --physcpubind="CPU_LIST" --membind="NUMA_NODE_LIST" \
  python src/run_perf_matrix.py \
    --claim table_ii --iterations 20 \
    --events cycles,instructions,cache-references,cache-misses
```

`run_perf_matrix.py` warms up first, discovers all worker TIDs, attaches
`perf stat` to those threads, and only then releases the measured loop. Generic
`cache-misses` is **not automatically labeled L3**. Add an AMD event only after
confirming its definition and scope on this kernel, for example with
`--events ... --l3-event EVENT`. Keep AMD uProf and `perf` results in separate
columns; do not compare raw counts collected with different scope.

For the exploratory decode claim, identify a validated AMD L2 event and run the
same profiler with `--claim decode_exploratory --models opt125m --l2-event
EVENT`. This is a threshold study, not a paper-number replication.

## Stage 7 - E2E and MMLU

Generate explicit configs rather than editing PACE files in place:

```bash
source .venv/bin/activate
python src/generate_pace_configs.py --suite table_iii --output-dir generated/table_iii
python src/generate_pace_configs.py --suite table_vi --output-dir generated/table_vi
python src/generate_pace_configs.py --suite mmlu --output-dir generated/mmlu
```

Run a generated performance config from the PACE checkout:

```bash
export PACE_ROOT="$PWD/vendor/AMD-PACE"
export IMBPS_BLOCK_SIZE=4   # omit for a TPP baseline config
python src/run_pace_entrypoint.py \
  --entrypoint "$PACE_ROOT/benchmarks/llm/performance/benchmark_llm_offline.py" \
  --config "$PWD/generated/table_iii/FILE.json" --seed 0
```

Run an accuracy config:

```bash
python src/run_pace_entrypoint.py \
  --entrypoint "$PACE_ROOT/benchmarks/llm/accuracy/evaluation.py" \
  --config "$PWD/generated/mmlu/FILE.json" --seed 0
```

Or execute every generated case with commit verification and no automatic
retry:

```bash
python src/run_generated_suite.py \
  --manifest generated/table_iii/run-manifest.json \
  --pace-root vendor/AMD-PACE
```

Summarize completed suites against the paper tables:

```bash
python src/summarize_performance.py \
  --suite table_iii --suite-dir generated/table_iii \
  --output generated/table_iii/comparison.csv

python src/summarize_performance.py \
  --suite table_vi --suite-dir generated/table_vi \
  --output generated/table_vi/comparison.csv

python src/summarize_mmlu.py \
  --suite-dir generated/mmlu \
  --output generated/mmlu/comparison.csv
```

The paper omits output-token count and dataset details for E2E. Generated
configs therefore use one output token and deterministic shape-only input by
default. The wrapper seeds Python, NumPy, and PyTorch before importing the
unmodified PACE entrypoint; this matters because PACE v1.0's input generator
otherwise leaves Python's RNG unseeded. Each manifest records the seed and the
other assumptions beside the JSON files. Change those inputs only as a declared
sensitivity analysis or after author clarification.

## Stage 8 - decode threshold study

The paper gives no decode table, batch sizes, or active-row threshold. Start
with OPT-125M, then repeat on progressively larger models only if the small
pilot is stable:

```bash
source .venv/bin/activate
python src/run_standalone_matrix.py \
  --claim decode_exploratory --models opt125m \
  --rounds 5 --warmups 5 --iterations 20
```

This sweeps active batches 1/8/32/128/512 and K=2/4/8 against TPP at sequence
length 1. Report crossover points and L2 counters; do not describe a slowdown
at batch 1 as contradicting a claim for which the paper supplied no shape.

## Decision rules

- Timing: report every round, median, IQR, coefficient of variation, and the
  baseline/IMBPS ratio. Do not discard outliers after seeing the result.
- Hardware-sensitive timing is not compared byte-for-byte across machines.
- Table II is an exact numerical match only if the 95% interval covers the
  paper speedup and the named L3 counter/scope also reproduces its miss ratio.
- Table VIII requires the shape of the K curve and the reported optimum, not
  merely one K beating baseline.
- Accuracy: report the full MMLU configuration, task version, few-shot count,
  batch size, and raw `lm_eval` output. A rounded score match alone is weak
  evidence.
- Any changed library, model revision, dataset revision, topology, or profiler
  scope creates a new result stratum; never merge it into the original one.

## Directory layout

```text
configs/       paper targets and experiment matrix
generated/     generated PACE configs (ignored by git if desired)
results/       immutable per-run outputs
scripts/       setup and preflight entry points
slurm/         cluster launchers
src/           benchmark, analytical model, and analysis code
tests/         hardware-independent unit tests
vendor/        pinned AMD-PACE checkout created by bootstrap
```
