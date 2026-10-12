# Batched normal-decode L2 study

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-12
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## Experiment overview

- **Objective:** Determine how normal end-to-end decode changes with live batch
  size, populated KV-cache length, and IMBPS split size.
- **Hypothesis:** Equation 13's aggregate-L2 candidate K=2 is a useful starting
  point, but the runtime-optimal backend/K depends on batch and KV-cache traffic.
- **Type:** CPU inference benchmark.

This is normal autoregressive generation only. Speculative decoding is excluded.

## Setup

- Target: `Qwen/Qwen2.5-7B-Instruct`
- PACE: v1.0 / commit `cfbe8b551cca18c686b771144c18129242796ea1`
- Dtype: BF16
- KV cache: BMC
- Sampling: greedy
- Host: mn01, socket/NUMA node 0, 96 physical cores
- Runtime controls: exclusive Slurm allocation, THP `always`, tcmalloc, fixed
  OpenMP affinity
- Entry command: `./scripts/submit_batched_decode_l2.sh`

## Independent variables

| Variable | Values |
|---|---|
| Batch | 1, 2, 4, 8, 16 |
| Prompt/KV context per sequence | 128, 512, 2048 tokens |
| MLP implementation | TPP K=1; IMBPS K=2, 4, 8 |
| Generated tokens | 32 per sequence |

All sequences in a batch have the same prompt length. This intentionally avoids
padding so batch size and KV-cache size are not confounded with padding ratio.

## L2 hypothesis

The paper-style placement proxy is

`W(K) = alpha * [M*H + (M*I + H*I)/K]`, with `M=batch` during normal decode.

The proxy uses mn01's aggregate 96 MiB across 96 private L2 caches. For the
registered Qwen2.5-7B dimensions it selects K=2 throughout the tested batch
range. K=4 and K=8 are nearby controls.

This is not an exact cache model: aggregate private L2 is not one shared cache,
and Qwen2.5 uses a gated MLP while Equation 13 is used in its paper-style
two-GEMM form.

## Outcomes and decision rule

- **Primary metric:** post-first-emission decode-tail speedup versus TPP.
- **Secondary metrics:** decode-step latency, aggregate token latency,
  post-first token throughput, total generation latency, TTFT, and nominal KV
  payload.
- Three independently launched rounds are summarized with bootstrap 95%
  intervals.
- Claim an end-to-end decode-tail speedup only when the interval's lower bound
  exceeds 1.
- Generated sequences must agree with the corresponding TPP control.

## Expected outputs

| Output | Path | Success criterion |
|---|---|---|
| Raw results | `results/batched-decode-l2-<job>/b*-*/result.json` | 20 complete variants |
| Summary | `results/batched-decode-l2-<job>/analysis/summary.csv` | 60 rows |
| Manifest | `results/batched-decode-l2-<job>/analysis/manifest.json` | Limitations retained |

## Monitoring

- Smoke test: batch 2, context 16, four output tokens, TPP and IMBPS K=2.
- The full matrix starts only if both smoke tasks succeed.
- The summary starts only if all 20 full tasks succeed.
- Per-task hard timeout: 12 hours.

## Not tested here

Mixed-length padded batches, continuous batching, service-level concurrency,
speculative decoding, and hardware counters are outside this clean timing pass.
