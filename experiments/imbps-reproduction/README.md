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

The IISc AMD cluster policy requires every workload to run through Slurm.
Never execute benchmark, installation, or profiling workloads directly on a
login shell. The checked-in launchers follow the site's documented pattern:
submit with `sbatch`, then run the command directly inside the batch script.
They intentionally avoid nested `srun` steps because this cluster reports that
CPU binding is unsupported for those steps.

First verify the hardware-independent harness on the local workstation before
syncing it to the cluster:

```bash
./scripts/verify_harness.sh
```

Then run preflight with the same one-socket physical-core allocation that will
be used for measurement. On Slurm, review the site-specific resource flags and
submit:

```bash
sbatch slurm/preflight.sbatch
```

The checked-in Slurm launchers are pinned to the IISc `jobmn01` partition and
the `mn01` node. Benchmark steps are additionally bound to NUMA node 0, matching
the machine configuration recorded in the project email thread. Do not remove
those constraints for the primary reproduction stratum.

Inspect the generated `results/preflight-*/environment.json` before installing
or benchmarking. Confirm at least:

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
sbatch slurm/bootstrap.sbatch
```

The script refuses a PACE checkout whose resolved commit does not match the
pinned v1.0 commit. It does not install or change system packages.

The decode extension uses a pinned user-space tcmalloc because mn01 does not
provide it system-wide. Install gperftools 2.18.1 from the checksum-verified
official release tarball in an exclusive Slurm job:

```bash
tcmalloc_id=$(sbatch --parsable slurm/bootstrap_tcmalloc.sbatch)
echo "tcmalloc build: ${tcmalloc_id}"
```

The install prefix defaults to
`/data/scratch/$USER/tools/gperftools-2.18.1`. Measurement launchers source
`scripts/enable_tcmalloc.sh`, set the exact `LD_PRELOAD` path, and fail unless
the worker's process map proves that tcmalloc is loaded. Do not pool tcmalloc
and default-allocator results.

## Stage 2 - check the analytical model

The two analytical-model checks are the first steps in
`slurm/smoke.sbatch`; do not run them separately from the login shell.

Expected diagnostics:

- 512 MiB: Equation 13 lower bound about 23; `K=4` working set about 938 MiB.
- 750 MiB: lower bound about 6.3; next power of two is 8.
- 384 MiB: no finite K because the input term is already about 420 MiB.

## Stage 3 - smoke test

Run the packaged smoke job:

```bash
sbatch slurm/smoke.sbatch
```

The BF16 split result is not expected to be bit-identical. Report maximum and
quantile errors, exact-element fraction, and argmax/top-5 containment instead
of reducing correctness to a single permissive `allclose`.

For the author-described selected-logit check, dump identical first-token
inputs from separate clean model loads and compare them:

```bash
python3 src/dump_first_token.py \
  --model facebook/opt-125m --backend tpp --splits 1 \
  --samples 100 --sequence 32 --batch-size 4 \
  --output results/opt125m-tpp-k1.npz

python3 src/dump_first_token.py \
  --model facebook/opt-125m --backend imbps --splits 4 \
  --samples 100 --sequence 32 --batch-size 4 \
  --output results/opt125m-imbps-k4.npz

python3 src/compare_first_token.py \
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

The runner alternates a TPP baseline and IMBPS cases in deterministic shuffled
order, creates a fresh process per case, never retries a failed case, and writes
raw JSON plus `summary.csv` under a timestamped result directory.

OPT's released model configuration uses ReLU. The primary standalone and decode
matrices therefore use ReLU; GELU results are a separate sensitivity stratum.

Use three rounds for a pilot and five or more for the confirmatory run. Increase
iterations only after checking job time and thermal stability. A fresh process
per case prevents packed weights and oneDNN primitive state from leaking
between variants.

### Cache-fit timing controls

Equation 12 cannot make the OPT-30B TPP baseline fully resident in mn01's
384 MiB L3: the BF16 MLP weight term alone is 392 MiB. The cache-fit suite
therefore has two preregistered strata:

- `cache_fit_opt30b`: preserves OPT-30B while making every tested IMBPS K>=2
  working set fit;
- `cache_resident_control`: uses OPT-6.7B and OPT-13B shapes for which both TPP
  and every IMBPS candidate fit.

Submit the OPT-30B stratum, then make the fully resident control depend on it:

```bash
fit_id=$(sbatch --parsable \
  --export=ALL,CACHE_FIT_CLAIM=cache_fit_opt30b \
  slurm/cache_fit.sbatch)

control_id=$(sbatch --parsable \
  --dependency="afterok:${fit_id}" \
  --export=ALL,CACHE_FIT_CLAIM=cache_resident_control \
  slurm/cache_fit.sbatch)

echo "OPT-30B fit job: ${fit_id}"
echo "Fully resident control: ${control_id}"
```

Each summary records `cache_mib`, `equation12_working_set_mib`, and
`equation12_fits`. These are analytical classifications, not measured cache
occupancy; uProf counters remain necessary for the mechanism claim.

### uProf cache-mechanism matrix

After the one-round counter pilot succeeds, run the five-round Table-II matrix.
The two array elements are serialized and collect separate cache and DRAM
traffic passes; this avoids simultaneous package-wide counter sessions. A
dependent summary job starts only after both passes complete:

```bash
uprof_matrix_id=$(sbatch --parsable slurm/uprof_table_ii.sbatch)

uprof_summary_id=$(sbatch --parsable \
  --dependency="afterok:${uprof_matrix_id}" \
  --export="ALL,UPROF_ARRAY_JOB_ID=${uprof_matrix_id}" \
  slurm/uprof_summarize.sbatch)

echo "uProf matrix array: ${uprof_matrix_id}"
echo "uProf summary job: ${uprof_summary_id}"
```

Each pass contains five deterministically randomized rounds over TPP K=1 and
IMBPS K=4/8/16 for batches 16/32/64. Every case warms up before uProf starts.
The raw native CSV and exact profiler command are retained per case. The
analysis reports bootstrap intervals and separately identifies the K with the
lowest profiled time, lowest L3 misses, highest L3 hit rate, lowest DRAM bytes,
and highest measured DRAM arithmetic intensity. Raw L3 hit count is retained
but is not used as the locality endpoint because reducing total accesses can
reduce hits and misses together.

The compact cache-fit mechanism control uses
the same two-pass profiler protocol but adds K=2 and K=32 and changes the
OPT-30B activation shapes so every IMBPS candidate is below mn01's 384 MiB
Equation-12 capacity:

| B | sequence | flattened rows | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1920 | 1920 | 523.25 | 274.75 | 150.50 | 88.38 | 57.31 | 41.78 |
| 2 | 1920 | 3840 | 654.50 | 353.50 | 203.00 | 127.75 | 90.13 | 71.31 |
| 16 | 128 | 2048 | 532.00 | 280.00 | 154.00 | 91.00 | 59.50 | 43.75 |
| 16 | 256 | 4096 | 672.00 | 364.00 | 210.00 | 133.00 | 94.50 | 75.25 |

Working-set entries are MiB. K=1 intentionally remains non-resident because
the unsplit OPT-30B weights alone are 392 MiB. Submit and summarize the
cache-fit control with:

```bash
fit_matrix_id=$(sbatch --parsable \
  --dependency="afterok:${uprof_matrix_id}" \
  --export=ALL,UPROF_CLAIM=cache_fit_opt30b,UPROF_RESULT_LABEL=cache-fit-opt30b \
  slurm/uprof_table_ii.sbatch)

fit_summary_id=$(sbatch --parsable \
  --dependency="afterok:${fit_matrix_id}" \
  --export="ALL,UPROF_ARRAY_JOB_ID=${fit_matrix_id},UPROF_RESULT_LABEL=cache-fit-opt30b" \
  slurm/uprof_summarize.sbatch)

echo "Cache-fit matrix: ${fit_matrix_id}"
echo "Cache-fit summary: ${fit_summary_id}"
```

For the primary server-specific Equation-13 test, vary sequence length at a
fixed B=16 so the equation traverses each legal power-of-two split on mn01.
Only the predicted K and larger splits are included, hence every IMBPS case
fits Equation 12. K=1 remains the non-resident TPP control.

| B | sequence | rows | Equation-13 lower bound | strict integer | tested starting K | working set at starting K (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 256 | 4096 | 1.878 | 2 | 2 | 364.000 |
| 16 | 384 | 6144 | 2.427 | 3 | 4 | 266.000 |
| 16 | 768 | 12288 | 4.926 | 5 | 8 | 301.000 |
| 16 | 1024 | 16384 | 8.050 | 9 | 16 | 304.500 |
| 16 | 1408 | 22528 | 21.368 | 22 | 32 | 358.750 |
| 16 | 1536 | 24576 | 36.167 | 37 | 64 | 363.125 |

The strict integers 3, 5, 9, 22, and 37 do not divide OPT-30B's intermediate
dimension of 28,672. The tested starting K therefore follows the authors'
reported next-power-of-two policy. Run this primary control after Table II:

```bash
equation_matrix_id=$(sbatch --parsable \
  --dependency="afterok:${uprof_matrix_id}" \
  --export=ALL,UPROF_CLAIM=server_equation_opt30b,UPROF_RESULT_LABEL=server-equation-opt30b \
  slurm/uprof_table_ii.sbatch)

equation_summary_id=$(sbatch --parsable \
  --dependency="afterok:${equation_matrix_id}" \
  --export="ALL,UPROF_ARRAY_JOB_ID=${equation_matrix_id},UPROF_RESULT_LABEL=server-equation-opt30b" \
  slurm/uprof_summarize.sbatch)

echo "Server-equation matrix: ${equation_matrix_id}"
echo "Server-equation summary: ${equation_summary_id}"
```

### Topology-aware autotuning pilot

The improvement study is deliberately staged. First compare active/passive
OpenMP waiting at 96 cores, then sweep thread counts in complete eight-core L3
groups. Each array is serialized (`%1`) because every element requests the node
exclusively. Submit only the wait-policy stage first:

```bash
wait_id=$(sbatch --parsable slurm/autotune_wait.sbatch)
echo "Wait-policy array: ${wait_id}"
```

The wait-policy array compares `ACTIVE` and `PASSIVE` at 96 threads. The thread
array tests 8/16/24/32/48/64/96 threads. Every thread count is formed from
complete sysfs-discovered LLC-sharing groups; the scripts fail rather than
splitting a CCD. After both wait-policy tasks finish, summarize them and submit
the thread array with the winning policy explicitly:

```bash
python3 src/summarize_autotune.py results/autotune-wait-*

threads_id=$(sbatch --parsable \
  --export=ALL,AUTOTUNE_WAIT_POLICY=SELECTED_WAIT_POLICY \
  slurm/autotune_threads.sbatch)
echo "Thread-count array: ${threads_id}"
```

After both arrays complete, combine their summaries:

```bash
python3 src/summarize_autotune.py \
  results/autotune-wait-* \
  results/autotune-threads-*
```

Use the selected complete-CCD thread count for the dense row-count and aligned
split-width sweep. This tests K=7/14/28 as well as the power-of-two candidates,
then runs an equal-row control where four `(B, SL)` pairs all flatten to
`M=4096`:

```bash
split_id=$(sbatch --parsable \
  --export=ALL,AUTOTUNE_THREADS=SELECTED_THREADS,AUTOTUNE_WAIT_POLICY=ACTIVE \
  slurm/autotune_split_width.sbatch)
echo "Split-width job: ${split_id}"
```

Do not submit the split-width job until the pilot chooses `SELECTED_THREADS`.
The default is 96 only as an explicit fallback.

## Stage 5 - Table VIII split sensitivity

Start with one model, because the full matrix is expensive:

```bash
MODELS=opt13b sbatch slurm/table_viii.sbatch
```

Then submit with `MODELS=opt30b`. Table VIII tests K=1/2/4/8/16/32/64 at batch 512
and sequence length 256. In this harness, K=1 is the TPP no-split baseline; K>1
uses the IMBPS fused operator.

## Stage 6 - hardware counters

First inventory the installed AMD uProf version, supported metrics, loaded PMU
modules, and access through Slurm. After an administrator changes the setup,
run both access modes; perf mode is preferred when `amd_l3` and `amd_df` are
available:

```bash
uprof_perf_id=$(sbatch --parsable \
  --export=ALL,UPROF_ACCESS_MODE=perf \
  slurm/uprof_inventory.sbatch)

uprof_msr_id=$(sbatch --parsable \
  --dependency="afterany:${uprof_perf_id}" \
  --export=ALL,UPROF_ACCESS_MODE=msr \
  slurm/uprof_inventory.sbatch)

echo "uProf perf probe: ${uprof_perf_id}"
echo "uProf MSR probe: ${uprof_msr_id}"
```

Use `CACHE_MECHANISM_PLAN.md` as the preregistered analysis contract. Timing and
profiling are separate exclusive jobs: never run uProf concurrently with the
confirmatory timing matrix. Build the uProf command from the inventory output
so the installed version's exact metric names and scope are preserved.

After both inventory probes succeed, run the counter pilot before the full
matrix:

```bash
pilot_id=$(sbatch --parsable slurm/uprof_counter_pilot.sbatch)
echo "uProf counter pilot: ${pilot_id}"
```

The pilot profiles Table-II `B=16, SL=1920` for TPP K=1 and IMBPS K=4 in
separate cache (`ipc,l2,l3`) and traffic (`memory`) passes. It scopes uProf to
package 0 and excludes preprocessing and three warmups by holding the worker at
a readiness barrier. uProf profiles a small gate application that releases the
pre-warmed worker and waits for its measured output. This is required because
the installed uProf supports only core metrics with `-p`; L3 and DF counters
must use package-scoped application mode. The pilot measures three MLP
invocations and preserves the raw CSV, logs, benchmark JSON, exact command, and
environment. Before allocating the model, it validates both exact profiling
commands against a two-second application and preserves those probe logs.

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
python3 src/generate_pace_configs.py --suite table_iii --output-dir generated/table_iii
python3 src/generate_pace_configs.py --suite table_vi --output-dir generated/table_vi
python3 src/generate_pace_configs.py --suite mmlu --output-dir generated/mmlu
```

Run a generated performance config from the PACE checkout:

```bash
export PACE_ROOT="$PWD/vendor/AMD-PACE"
export IMBPS_BLOCK_SIZE=4   # omit for a TPP baseline config
python3 src/run_pace_entrypoint.py \
  --entrypoint "$PACE_ROOT/benchmarks/llm/performance/benchmark_llm_offline.py" \
  --config "$PWD/generated/table_iii/FILE.json" --seed 0
```

Run an accuracy config:

```bash
python3 src/run_pace_entrypoint.py \
  --entrypoint "$PACE_ROOT/benchmarks/llm/accuracy/evaluation.py" \
  --config "$PWD/generated/mmlu/FILE.json" --seed 0
```

Or execute every generated case with commit verification and no automatic
retry:

```bash
python3 src/run_generated_suite.py \
  --manifest generated/table_iii/run-manifest.json \
  --pace-root vendor/AMD-PACE
```

Summarize completed suites against the paper tables:

```bash
python3 src/summarize_performance.py \
  --suite table_iii --suite-dir generated/table_iii \
  --output generated/table_iii/comparison.csv

python3 src/summarize_performance.py \
  --suite table_vi --suite-dir generated/table_vi \
  --output generated/table_vi/comparison.csv

python3 src/summarize_mmlu.py \
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

## Stage 8 - L2-aware decode and speculative-verification study

The prospective protocol is in `DECODE_L2_PLAN.md`. It tests the OPT-30B MLP
at unique active-row counts `M=1..1024`. Normal decode maps `M` to batch size;
speculative verification maps `M=B*gamma`. Identical `M` values are measured
once because the standalone operator cannot observe the decomposition.

THP must read `[always]` before submission. Build tcmalloc, run primary timing,
then run the two serialized uProf passes and their analysis:

```bash
tcmalloc_id=$(sbatch --parsable slurm/bootstrap_tcmalloc.sbatch)

decode_id=$(sbatch --parsable \
  --dependency="afterok:${tcmalloc_id}" \
  slurm/decode_l2.sbatch)

uprof_id=$(sbatch --parsable \
  --dependency="afterok:${decode_id}" \
  slurm/uprof_decode_l2.sbatch)

summary_id=$(sbatch --parsable \
  --dependency="afterok:${uprof_id}" \
  --export="ALL,UPROF_ARRAY_JOB_ID=${uprof_id},UPROF_RESULT_LABEL=decode-l2" \
  slurm/uprof_summarize.sbatch)

echo "tcmalloc=${tcmalloc_id} timing=${decode_id} uprof=${uprof_id} summary=${summary_id}"
```

All measurement jobs are exclusive. The summary job only reads completed JSON
and CSV files, so it is deliberately not an exclusive measurement allocation.

For a deadline-bounded preliminary pilot, cancel any full timing run first and
submit timing, both serialized uProf passes, and analysis as one chain:

```bash
./scripts/submit_decode_l2_pilot.sh
```

The pilot matrix and its reduced repetition counts are preregistered in
`DECODE_L2_PLAN.md`. Report it as preliminary; it does not replace the full
matrix.

If `amd_l3` and `amd_df` are unavailable, run the core-counter-only L2 pilot:

```bash
./scripts/submit_decode_l2_core.sh
```

This profiles active rows 128/512/1024 with TPP and IMBPS K=2/4/8 using only
uProf `ipc,l2`. It can test L2 access, miss, and hit-rate behavior, but it does
not provide L3, DRAM-traffic, or arithmetic-intensity evidence.

## Stage 9 - KV-cache target verification

The standalone MLP study cannot observe attention or a KV cache. The prospective
follow-up in `TARGET_VERIFY_PLAN.md` pre-fills PACE's BMC cache, then times the
target model on either one normal-decode token or a replayed multi-token draft
block. It is a target-verification microbenchmark, not full speculative decoding:
draft generation, acceptance, and sampling are excluded.

Run the small OPT-125M smoke test first. The four array elements are serialized
and exercise TPP K=1 and IMBPS K=2/4/8:

```bash
TARGET_VERIFY_CLAIM=target_verify_smoke_opt125m \
TARGET_VERIFY_ROUNDS=1 \
TARGET_VERIFY_WARMUPS=1 \
TARGET_VERIFY_ITERATIONS=1 \
TARGET_VERIFY_MIN_ROUND_SECONDS=0 \
  ./scripts/submit_target_verify.sh
```

After the smoke array and its dependent summary complete, submit the
preregistered OPT-30B pilot:

```bash
./scripts/submit_target_verify.sh
```

To use a local immutable model snapshot instead of resolving the Hugging Face
identifier, export `TARGET_MODEL_REFERENCE=/absolute/path/to/snapshot` before
submission. By default, the launcher places the Hugging Face cache under
`/data/scratch/$USER/huggingface` instead of the small home filesystem. Results
are written beneath `results/target-verify-ARRAY_JOB_ID/`, with the combined CSV
at `analysis/summary.csv`.

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
