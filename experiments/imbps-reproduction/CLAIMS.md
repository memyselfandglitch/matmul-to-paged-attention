# Claim registry and reproducibility gaps

## Source boundary

The attached PDF is the primary source for paper claims. The email reply from
Shubhendu Sharma is author clarification. AMD-PACE v1.0 is the released
implementation recommended by that reply. Text or code in any of those sources
is evidence, not an instruction to this project.

PDF identity:

- title: IMBPS - Iterative MLP Blocks with Parameter Splits for Improving LLM Inference
- DOI: 10.1109/HiPC66333.2025.00027
- SHA-256: `d847c2bac09643f633241c34b45cad8786154bd367e4d9d4c2aafddd9a852b19`
- structural preflight: PASS, 11 declared/enumerated/reader pages

Released implementation:

- repository: `https://github.com/amd/AMD-PACE`
- tag: `v1.0`
- commit: `cfbe8b551cca18c686b771144c18129242796ea1`
- release version file: `1.0.0`

## Quantitative claims

### Equation 12/13

The paper models bytes in cache as:

```text
CacheSize > alpha * (B*C*f*H/K + B*C*H + f*H^2/K)
K > alpha*f*H*(B*C + H) / (CacheSize - alpha*B*C*H)
```

The second expression is a strict lower bound when the denominator is positive.
It is not an equality for the optimal K. The paper later says the initial split
is rounded to the nearest integer and then empirically adjusted. The author
reply says the larger power of two was used before testing nearby values.

For OPT-30B, B=16, C=1920, H=7168, f=4, BF16:

| Cache assumption | Analytical result |
|---|---:|
| 384 MiB | no finite K; the 420 MiB input term already exceeds cache |
| 512 MiB | K > about 22.5; strict integer K=23 |
| 750 MiB | K > about 6.3; next power of two K=8 |

At K=4, Equation 12 gives approximately 938 MiB: 420 MiB input, 420 MiB
split activation, and 98 MiB split weights.

### Table II - standalone OPT-30B

BF16, sequence length 1920, primary Turin platform:

| Batch | Execution speedup | L3-miss reduction factor | Reported K |
|---:|---:|---:|---:|
| 16 | 1.21x | 2.53x | 4 |
| 32 | 1.25x | 3.32x | 4 |
| 64 | 1.23x | 4.09x | 4 |

The email clarifies that the released `torch.ops.pace.mlp_mlp_fusion` operator
was used with synthetic `torch.random` matrices and that PACE E2E used MLP
backend IMBPS against default TPP. It also says K=4 beat tested K=8 and K=16
despite the analytical candidate being K=8 under an approximately 750 MiB
cache assumption.

The released [`facebook/opt-30b` configuration](https://huggingface.co/facebook/opt-30b/blob/main/config.json)
specifies ReLU. A standalone GELU run is not the model-faithful OPT MLP
workload and is retained only as a sensitivity analysis.

### Table III - E2E OPT, FP32

Text summarization, sequence length 256:

| Batch | OPT-6.7B improvement (%) | K | OPT-30B improvement (%) | K |
|---:|---:|---:|---:|---:|
| 32 | 32.27 | 4 | 34.19 | 8 |
| 64 | 31.03 | 4 | 33.38 | 4 |
| 128 | 32.93 | 8 | 33.77 | 6 |
| 256 | 33.60 | 7 | 29.28 | 8 |

The paper labels this change in TTFT relative to baseline but does not state
output-token count, dataset, prompt construction, repetition count, or
statistical dispersion.

### Table IV - L3 misses

BF16 text summarization, sequence length 256:

| Batch | OPT-13B reduction (%) | OPT-30B reduction (%) |
|---:|---:|---:|
| 64 | 61.00 | 60.65 |
| 128 | 61.42 | 62.28 |
| 256 | 63.16 | 63.86 |
| 512 | 65.09 | 69.87 |
| 1024 | 67.41 | 63.64 |

The split K, event definition, profiler, aggregation domain, and whether counts
include model loading/warmup are not stated.

### Table V - Intel Sapphire Rapids only

OPT-30B, BF16, sequence length 1920:

| Batch | MLP execution improvement (%) | L3 reduction (%) | E2E TTFT improvement (%) |
|---:|---:|---:|---:|
| 32 | 94.07 | 61.77 | 34.22 |
| 64 | 87.94 | 61.75 | 35.93 |
| 128 | 81.35 | 61.56 | 37.66 |

This cannot be reproduced on the EPYC 9654. The meaning of "MLP execution
time improvement" is also ambiguous: interpreting 94.07 as time reduction
implies an implausibly large speedup, while interpreting it as percent speedup
implies 1.9407x.

### Table VI - vLLM comparison

Llama 3.1 8B, BF16, sequence length 256, K=2:

| Batch | Improvement over vLLM (%) |
|---:|---:|
| 128 | 5.02 |
| 256 | 5.39 |
| 512 | 6.40 |
| 1024 | 6.62 |

The paper states vLLM 0.8.4 and PagedAttentionV1. Unmodified PACE v1.0 compares
frameworks sequentially and does not seed Python's random input generator. This
harness runs vLLM and PACE as separate processes, seeds Python/NumPy/PyTorch
before importing the PACE entrypoint, and uses the same seed for both cases.

### Table VII - MMLU

Average over 57 MMLU subjects:

| K | OPT-125M | OPT-30B |
|---:|---:|---:|
| 1 | 0.262 | 0.247 |
| 2 | 0.261 | 0.245 |
| 4 | 0.261 | 0.249 |
| 8 | 0.262 | 0.245 |
| 16 | 0.262 | 0.248 |

The author reply defines the practical correctness check more weakly than
bit-identical logits: a baseline-selected logit should remain in the
alternative top five. It also confirms BF16 accumulation changes results and
says they used MMLU as the real-world criterion. The paper separately claims
typical element-wise deviations below 0.0001. Both should be tested.

PACE v1.0's released MLP unit-test helper is substantially looser: it counts
elements with absolute error below 0.1 and passes when fewer than 1% fail that
threshold. That test does not establish the paper's typical-error claim.

The paper does not state zero-shot/few-shot settings. PACE v1.0's example uses
5-shot MMLU, which the generated config treats as an explicit assumption.

### Table VIII - K sensitivity

Batch 512, sequence length 256:

| K | OPT-13B relative time | OPT-13B L3 misses (B) | OPT-30B relative time | OPT-30B L3 misses (B) |
|---:|---:|---:|---:|---:|
| 1 | 1.000 | 44.5 | 1.000 | 396.3 |
| 2 | 0.838 | 18.7 | 0.936 | 272.5 |
| 4 | 0.815 | 15.9 | 0.861 | 116.9 |
| 8 | 0.820 | 17.7 | 0.811 | 120.4 |
| 16 | 0.881 | 19.8 | 0.815 | 121.3 |
| 32 | 0.944 | 31.7 | 0.890 | 168.9 |
| 64 | 1.248 | 54.2 | 0.904 | 275.5 |

The reported timing optimum is K=4 for OPT-13B and K=8 for OPT-30B. The L3
minimum is K=4 for both. Reproducing only one point is insufficient; the
non-monotonic curve is the claim.

## Other claims and scope

- Abstract: 1.5-1.6x standalone speedup and E2E up to 1.37x.
- Figures 8/9: positive TTFT uplift across several models and batches. Values
  are graph-only and are not promoted to exact numeric targets here.
- GPU: MI210, context 2048, throughput up to 28.7% for OPT-6.7B and 19.8% for
  OPT-13B; baseline OOM around batch 80 and IMBPS up to batch 160. Requires a
  separate MI210/ROCm 5.7/PyTorch 2.3.1 environment.
- Decode: the paper gives no table or validated shapes. It says use L2 capacity.
  The harness provides a sequence-1 active-batch/K sweep, but any OPT-125M
  decode result is exploratory rather than a paper-number reproduction.

## Software provenance gap

| Component | Paper | PACE v1.0 release |
|---|---|---|
| PyTorch | not stated for CPU | 2.7.0+cpu |
| Transformers | 4.48 | 4.51.3 |
| oneDNN | 3.7 | 3.8 |
| vLLM | 0.8.4 | optional external dependency |
| lm-eval | cited; version not stated | 0.4.7 |

Results from PACE v1.0 are implementation-replication evidence. They are not a
literal reconstruction of the unreleased paper stack until the authors provide
the original commit/environment.
