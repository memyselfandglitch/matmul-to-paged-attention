# Full normal and speculative decode plan

## Research question

Does IMBPS improve complete CPU generation once attention, KV-cache traffic,
sampling, and repeated decode iterations are included, and how does it interact
with PACE v1.0 PARD speculative decoding?

This is distinct from the existing target-forward microbenchmark. The timed
boundary is the public `LLMModel.generate` call and includes prompt prefill,
normal iterative decode, and every PARD stage when PARD is enabled.

## Supported model pair

- Target: `Qwen/Qwen2.5-7B-Instruct`
- Parallel draft: `amd/PARD-Qwen2.5-0.5B`
- PACE: v1.0 / commit `cfbe8b551cca18c686b771144c18129242796ea1`
- Dtype: BF16
- KV cache: BMC
- Sampling: greedy
- Batch: 1, because PACE v1.0 rejects PARD batches larger than one

These are the model pair and restrictions documented by AMD-PACE v1.0. This
study does not claim to be an OPT-30B result.

## Independent variables

- Context tokens: 128, 1024, 2048
- Generation mode: normal autoregressive decode or PARD
- PARD speculative block: 4, 8, 12
- MLP backend: TPP K=1 or IMBPS K=2, 4, 8
- Requested generated tokens: 64

There are 16 backend/mode variants. Each loads the model independently and
measures every context length in randomized order for three rounds after one
warmup per context.

## Outcomes

- Total generate latency and output tokens/s
- Time to first generated emission, which includes prefill
- Post-first-emission throughput
- PARD mean kept tokens per iteration and accepted-draft fraction
- Exact generated-token sequence agreement with the same-mode TPP run
- Three speedup views:
  - versus normal TPP;
  - versus TPP in the same normal/PARD mode;
  - PARD versus normal generation under the same backend and K.

## Hypotheses and interpretation

The standalone OPT-30B crossover occurred at hundreds of active MLP rows.
Normal batch-one decode presents one active row, and PARD presents at most its
small speculative block. Therefore IMBPS is not assumed to win; a slowdown is
a valid result. PARD may still improve end-to-end generation by reducing target
forward passes when acceptance is high.

PACE v1.0 applies one `OperatorConfig` to both the target and draft model.
Consequently a PARD IMBPS result measures IMBPS on both models, not an isolated
target-only intervention. This limitation must remain visible in reporting.

## Controls and decision rule

Every Slurm task is exclusive on mn01, pinned to the same 96 physical cores,
NUMA node 0, THP `always`, tcmalloc 2.18.1, and identical PACE/PyTorch versions.
Tasks are serialized to prevent co-tenancy. Report all rounds and independent
bootstrap 95% intervals. Claim a speedup only when the lower interval bound is
above 1. Do not discard runs after observing their values.

## Not included

Random sampling, batches above one, a separate target/draft backend setting,
service-level concurrency, and uProf counter collection are outside this first
full-decode timing matrix.
