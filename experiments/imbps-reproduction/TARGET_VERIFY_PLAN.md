# KV-cache target-verification experiment

## Material Passport

- Artifact type: prospective code-experiment plan
- Research object: PACE v1.0 OPT-30B target forward
- Primary question: do the standalone MLP gains at `M=512` and `M=1024`
  survive target-model attention, KV-cache traffic, the LM head, and the other
  decoder-layer work?
- Scope: normal decode and replayed multi-token target verification
- Explicit exclusion: draft-model generation, token acceptance, sampling, and
  end-to-end speculative-decoding throughput

## Hypotheses

1. `M=256` is a negative control because standalone IMBPS was slower than TPP.
2. `M=512` and `M=1024` are positive candidates because standalone K=2 was
   respectively 1.265x and 1.400x faster than TPP.
3. Cases with equal `M=B*gamma` need not have equal total latency once real
   attention and KV-cache traffic are included.
4. K=2 is the primary candidate. K=4 and the L2-equation power-of-two K=8 are
   retained as controls.

## Fixed pilot matrix

All cases use OPT-30B BF16, 128 cached context tokens, PACE BMC KV cache, JIT
attention, TPP projections and LM head, and either TPP or IMBPS for the MLP.

| Mode | Batch | Draft tokens | Active rows | Role |
|---|---:|---:|---:|---|
| Normal decode | 256 | 1 | 256 | Negative control |
| Normal decode | 512 | 1 | 512 | Predicted crossover candidate |
| Target verification | 128 | 4 | 512 | Equal-M decomposition |
| Target verification | 64 | 8 | 512 | Equal-M decomposition |
| Target verification | 256 | 4 | 1024 | Predicted strong candidate |
| Target verification | 128 | 8 | 1024 | Equal-M decomposition |

The backend matrix is TPP K=1 and IMBPS K=2/4/8. Backend jobs are serialized
and each loads the model once. Cases sharing `(batch, context)` reuse one
prefilled cache.

## Measurement boundary

The timed interval starts immediately before the target-model forward and ends
immediately after it. It includes embeddings, every decoder layer, attention,
KV-cache reads and writes, MLPs, and the LM head. It excludes:

- model loading and weight packing;
- prompt prefill;
- construction of replayed draft-token tensors;
- cache rollback used to restore the same prefilled state;
- one untimed cache-allocation prime beyond the largest measured block;
- draft-model execution, acceptance, and sampling.

Rollback is measured and reported separately. The benchmark asserts that cache
length equals `context + gamma` after every target forward and returns to
`context` after every rollback. The allocation prime avoids repeatedly hitting
PACE v1.0 BMC's exact segment-growth boundary, so the reported target forwards
represent steady-state cache operation.

## Controls and reporting

- Run on mn01 with one exclusive 96-core allocation, one socket, no SMT, local
  NUMA memory, THP `always`, pinned tcmalloc, and a 384 GiB Slurm memory request.
- Use five rounds, two warmups per case, at least three target forwards and one
  second per round.
- Preserve every measured duration; do not remove outliers after inspection.
- Report median latency, IQR, active rows/s, rollback latency, KV-cache size,
  speedup against TPP, and an independent bootstrap 95% interval over round
  medians.
- Compare deterministic sampled logits using top-1 agreement and whether the
  TPP top-1 appears in the IMBPS top-5.

## Decision rule

An IMBPS configuration supports a target-forward improvement only when the
entire 95% speedup interval is above 1.0 and the TPP top-1 appears in the
candidate top-5 for every sampled coordinate. Even then, the result is not an
end-to-end speculative-decoding claim.

## Follow-up gate

Only after the pilot completes successfully should context length be increased
or a real PARD draft model be added. PACE v1.0's built-in PARD path supports
batch size 1, so it cannot directly test the high-`B*gamma` matrix above.
