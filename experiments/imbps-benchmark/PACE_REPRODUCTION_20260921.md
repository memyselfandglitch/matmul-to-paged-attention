# AMD PACE reproduction: 21 September 2026

## Bottom line

The optimized PACE implementation reproduces the IMBPS mechanism, but not all
reported magnitudes or any universal choice of `K`.

- At the exact OPT-30B/Table-II shape, PACE v1.0 makes `K=4` the fastest tested
  choice: **1.108x on one socket and 1.122x on two sockets** versus the same
  fused PACE operator at `K=1`.
- This remains below the paper's approximately 1.21x result.
- The capacity-equation candidate `K=23` is not optimal: it reaches 1.052x on
  one socket and falls to **0.931x** on two sockets.
- For PACE end-to-end OPT-125M at batch 1, 256 input tokens and 16 output
  tokens, every IMBPS setting loses to the PACE TPP MLP. The least costly is
  `K=1`, which is 6.34% slower in generation and 7.31% slower in TTFT. `K=4`
  is 41.62% slower in generation.
- BF16 split results are close, not bit-identical: the standalone maximum
  absolute difference is 0.0625 for every tested `K>1`. No element has absolute
  error at least 0.1, and the full output passes `rtol=atol=0.01`.
- The current PACE-main correctness suite passes TPP and all tested IMBPS
  settings under its symmetric top-5 rule, while still reporting one or two
  token divergences from Hugging Face in 256 generated-token decisions.

These findings support an empirical, workload- and placement-aware `K` policy.
They do not support choosing `K` from aggregate nominal cache capacity alone.

## Machine and software

| Item | Value |
|---|---|
| Host | `mn01` |
| CPU | 2 x AMD EPYC 9654, 96 cores/socket, SMT disabled |
| NUMA | node 0 CPUs 0-95; node 1 CPUs 96-191 |
| L3 | 24 independent 32 MiB CCD domains; 384 MiB/socket; 768 MiB machine total |
| L2 | 1 MiB private per core, 192 MiB machine total |
| Governor | `schedutil`, boost enabled |
| Standalone/E2E code | AMD PACE v1.0, commit `cfbe8b551cca18c686b771144c18129242796ea1` |
| Standalone/E2E PyTorch | 2.7.0+cpu |
| Correctness harness | PACE main commit `32bdcf0846273e9a7add697d09f32dcb38b5cb4b` |
| Correctness PyTorch | 2.12.0+cpu |

The correctness result is deliberately separated because the author-linked
top-5 harness is newer than PACE v1.0.

## 1. Exact OPT-30B standalone PACE operator

The test calls `torch.ops.pace.mlp_mlp_fusion` directly. It uses `M=30,720`
active token rows, `H=7,168`, `I=28,672`, BF16, GELU, both biases, and PACE-test
style random tensors scaled by 0.01. Up weights are split on dimension 0 and
down weights on dimension 1 exactly as PACE's IMBPS backend does. Prepacking,
allocation and correctness are outside the timed region. Five measurement
blocks contain every `K` exactly once in randomized order.

| Placement | K | Median (ms) | Speedup vs PACE K=1 |
|---|---:|---:|---:|
| One socket, 96 threads, local memory | 1 | 2128.7 | 1.000x |
| | 4 | 1920.7 | **1.108x** |
| | 8 | 1924.4 | 1.106x |
| | 16 | 1976.1 | 1.077x |
| | 23 | 2022.5 | 1.052x |
| Two sockets, 192 threads, interleaved memory | 1 | 1837.7 | 1.000x |
| | 4 | 1638.5 | **1.122x** |
| | 8 | 1688.4 | 1.088x |
| | 16 | 1858.0 | 0.989x |
| | 23 | 1974.0 | **0.931x** |

Full precision and dispersion fields are in
`results/pace-v1-standalone-opt30-20260921.csv`.

### Interpretation

The authors' explanation that the result machine had about 750 MiB of cache is
consistent with the aggregate 768 MiB on this two-socket host. It is not a
single shared cache, however: it is 24 separate 32 MiB L3 domains. The
two-socket result shows why aggregate capacity cannot by itself predict `K`.
`K=23` satisfies the paper equation under one interpretation yet becomes 6.9%
slower than unsplit PACE across two sockets. Over-splitting adds dispatch,
accumulation, and locality costs that a capacity inequality does not model.

The PACE result also changes the earlier prototype conclusion: with the native
fused implementation, `K=4`, not `K=8`, is best on this exact workload. This
agrees qualitatively with the author's empirical selection, while the 1.21x
magnitude still does not reproduce.

## 2. PACE v1.0 end-to-end OPT-125M

This uses the official `benchmark_llm_offline.py` entry point. All operators
except the MLP remain fixed: native norm, TPP QKV/out projection/LM head, JIT
attention, and BMC KV cache. Conditions are one CCD (CPUs 0-7), node-0 memory,
8 threads, BF16, batch 1, 256 input tokens, 16 generated tokens, two warmups and
five timed generations. Python's RNG is explicitly seeded before the unmodified
PACE entry point because its data generator otherwise ignores
`generation_args.manual_seed` when constructing synthetic token IDs.

| MLP backend | K | Generation (s) | TTFT (ms) | Output tok/s | Generation vs TPP |
|---|---:|---:|---:|---:|---:|
| TPP | - | 0.2046 | 52.95 | 78.19 | baseline |
| IMBPS | 1 | 0.2176 | 56.82 | 73.53 | 6.34% slower |
| IMBPS | 2 | 0.2237 | 61.32 | 71.53 | 9.31% slower |
| IMBPS | 4 | 0.2898 | 63.95 | 55.21 | 41.62% slower |
| IMBPS | 8 | 0.3749 | 69.90 | 42.67 | 83.23% slower |

This is a concrete counterexample to a universal inference-speedup claim, not a
claim that every model or larger active-row workload will regress. OPT-125M's
small MLP and batch-1 decode give too little reusable work to amortize the fused
operator and split overheads. The PACE harness warns that its TTFT streamer adds
some overhead; it is enabled identically for every row because TTFT is an
explicit study metric.

## 3. Correctness

### Standalone full-tensor comparison

For all `K>1`, maximum absolute error is 0.0625, mean absolute error is
`0.93e-6` to `1.59e-6`, a deterministic sample of about one million elements
has p99 error zero, and no output element differs by at least 0.1. Thus
"lossless" is defensible as an algorithmic or task-accuracy description, not as
bitwise BF16 equality.

### Author-linked top-5 suite

The current PACE correctness suite compares eight fixed prompts, 32 generated
tokens each, against Hugging Face. A mismatch passes only if each side's chosen
token occurs in the other's top five.

| Backend | Top-5 result | Tolerated token divergences vs HF |
|---|---|---:|
| TPP | 8/8 prompts pass | 2 |
| IMBPS K=1 | 8/8 prompts pass | 1 |
| IMBPS K=2 | 8/8 prompts pass | 1 |
| IMBPS K=4 | 8/8 prompts pass | 2 |
| IMBPS K=8 | 8/8 prompts pass | 2 |

Passing this criterion establishes plausible greedy-output stability, not exact
token equality and not MMLU accuracy. MMLU remains a separate required run.

## 4. L3 counter status

PACE L3 collection is currently blocked by host permissions:

- AMD uProf 5.3.521 is present and retains `cap_sys_rawio,cap_perfmon`;
- `/dev/cpu/0/msr` is now `crw------- root root`, so uProf MSR mode fails;
- perf mode reports that this kernel does not expose `amd_l3` for this CPU and
  requests `amd_uncore` or MSR mode;
- `sudo -n` is unavailable.

Therefore no new PACE-v1 L3-reduction number is claimed here. The older
prototype L3 measurements in `CLAIM_VERIFICATION_20260919.md` are not silently
substituted for native PACE measurements. An administrator must restore safe
read access to the MSR devices or load/expose `amd_uncore`, after which
`scripts/run_pace_uprof_l3_zen4.sh` can run unchanged.

## Reproduction

```bash
# Exact standalone shape; aggregate-two-socket and one-socket studies
PLACEMENT=two-socket ./scripts/run_pace_table2_zen4.sh
PLACEMENT=one-socket ./scripts/run_pace_table2_zen4.sh

# Official v1.0 E2E harness, matched deterministic synthetic tokens
PLACEMENT=ccd SPLITS=1,2,4,8 ./scripts/run_pace_e2e_opt125m_zen4.sh

# Author-linked newer correctness suite
SPLITS=1,2,4,8 ./scripts/run_pace_correctness_opt125m_zen4.sh

# Once MSR/amd_uncore access is restored
UPROF_MODE=msr ./scripts/run_pace_uprof_l3_zen4.sh
```

## Remaining work before a publication-grade conclusion

1. Restore L3 PMU access and collect at least three counter runs for PACE K=1,
   4, 8 and 23 under both socket placements.
2. Repeat timing in independent processes and report bootstrap confidence
   intervals; five in-process pairs are preliminary.
3. Run PACE E2E on a large model/high-active-row workload. The OPT-125M result
   establishes a boundary case, not the paper's main large-model regime.
4. Run the authors' MMLU configuration and report aggregate accuracy, exact item
   agreement, and confidence intervals separately.
5. Record the authors' exact PACE commit, model files, affinity, memory policy,
   governor and configuration JSON for the published table.
