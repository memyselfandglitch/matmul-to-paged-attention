# Consolidated IMBPS claim-verification report — 2026-09-19

## Executive conclusion

IMBPS is a real, workload-dependent CPU optimization, not a universal one. On
the IISc dual-socket AMD EPYC 9654 (Zen 4/Genoa) host, splitting a large MLP can
reduce LLC traffic and improve standalone latency. The benefit appears only
after the active-row matrix is large enough, and its optimum depends strongly
on model dimensions, thread count, memory placement, weight layout, and K.

The strongest positive result is a synthetic OPT-30B-shaped BF16 MLP at the
paper's batch-16/sequence-1,920 row count: K=8 reaches **1.115x** standalone
speedup and 56.8% fewer L3 misses. A Llama-3.1-8B-shaped SwiGLU MLP reaches
**1.250x** at K=4. The strongest negative results are real OPT-125M end-to-end
inference, where every K loses, and BF16 numerical tests, where 7 of 400 split
runs change the first greedy token.

The study therefore reaches five main conclusions:

1. **The mechanism is valid at large M.** Exact OPT-30B and Llama-shaped
   standalone kernels speed up on Genoa.
2. **The paper's quantitative K=4 result does not reproduce.** At the exact
   OPT-30B dimensions and M, K=4 is 1.072x here, not 1.21x, and reduces L3
   misses 28.2%, not 60.5%. K=8/16 is required to approach that miss reduction.
3. **Small and decode workloads are counterexamples to universal benefit.** On
   pretrained OPT-125M, splitting loses until roughly 2,048 active rows, and
   all end-to-end B1/S256 variants are slower.
4. **The published cache equation is not an optimal-K formula and is internally
   inconsistent with several paper tables.** It omits the live output
   accumulator; even its own formula predicts K=23, not K=4, for Table II's
   OPT-30B/B16 case.
5. **“Lossless” must mean algebraically equivalent, not numerically identical.**
   BF16 primitive/layout changes cause logit drift and occasional token changes.

The exact Turin, Intel, GPU, vLLM, and MMLU point estimates remain unverified.

## System, implementation, and measurement rules

| Item | Verification setup |
|---|---|
| Host | 2x AMD EPYC 9654, 96 cores/socket, SMT off |
| Cache | 32 KiB L1D/core, 1 MiB L2/core, 32 MiB L3/8-core CCD, 12 CCDs/socket |
| Memory | Two NUMA nodes, approximately 756 GiB/node; local distance 10, remote 32 |
| Primary CPU placement | Socket 0, CPUs 0–95, memory node 0; 8-core tests use CCD 0 |
| Precision | BF16 primary; FP32 numerical control |
| Real model | `facebook/opt-125m`, Transformers 4.48.3 |
| Synthetic shapes | OPT-6.7B, OPT-30B, and Llama-3.1-8B MLP dimensions |
| Counters | AMD uProf 5.3.521, non-multiplexed Zen 4 L3/L2 event subsets |
| Timing | Warmed, paired and randomized native/IMBPS measurements; medians reported |
| Statistics | Paired bootstrap 95% intervals and exact two-sided sign tests |

K=1 through the adapter is kept separate from the native reference. It changes
the primitive and weight layout but does not split the intermediate dimension;
it is a layout control, not an IMBPS result.

## Consolidated claim verdicts

| Claim family | Verdict | Main evidence |
|---|---|---|
| Exact-arithmetic equivalence | **Verified** | OPT/GELU and SwiGLU block decompositions pass correctness tests, including uneven K. |
| Floating-point identity / lossless output | **Counterexample** | BF16 max final-logit error reaches 0.3125; K>1 changes 7/400 first-token decisions across 100 seeds. |
| Standalone speedup at large M | **Verified qualitatively** | OPT-30B shape peaks at 1.115x; Llama SwiGLU shape peaks at 1.250x. |
| Paper Table II K=4 magnitude | **Not reproduced** | 1.072x and 28.2% fewer L3 misses here versus paper's 1.21x and about 60.5%. |
| Small/prefill/decode benefit | **Counterexample outside target regime** | Real OPT-125M splitting loses below about 2,048 rows and slows B1/S256 E2E. |
| Reduced L3 misses imply speedup | **False as a general rule** | OPT-6.7B K=8 has 31.0% fewer misses but is 10.2% slower. |
| L2-derived decode split | **Counterexample** | OPT-125M M=1 L2 misses rise 51.5–110.7% and decode slows. |
| Equation 13 predicts optimal K | **Refuted as stated** | It is only a capacity lower bound, omits live output and overhead, and disagrees with table K values. |
| 1/K intermediate-memory saving | **Verified logically; bounded in process RSS** | Logical activation shrinks with K, but measured peak RSS falls only 19.2–25.9%. |
| Packing is eliminated | **Not generally true** | Persistent packing improves large-M speed by 2–3% but copies 784 MiB and costs about 15 ms. |
| No MMLU loss | **Unverified** | Paper lacks item-level evidence; local `lm-eval`/MMLU dataset and exact revision were unavailable. |
| Turin, Sapphire Rapids, MI210, vLLM gains | **Unverified** | Those hardware/runtime configurations are not present on this host. |

The exhaustive paper-by-paper inventory is in `PAPER_CLAIM_MATRIX.md`.

## 1. The analytical K formula needs correction

The paper's Equation 12 models the resident bytes as

`s * [M*H + (M*I + H*I)/K]`,

where `s` is element size, `M` active rows, `H` hidden width, and `I` MLP
intermediate width. Algorithm 1, however, keeps both input `X` and output
accumulator `Y` live while iterating over blocks. A still-conservative lower
bound is therefore

`s * [2*M*H + (M*I + H*I)/K]`.

This correction still omits kernel scratchpads, cache conflicts, runtime
metadata, and other simultaneously live model tensors. It is a lower bound,
not a full cache simulator.

Equation 12 is strict (`working_set < cache`). Therefore the safe integer is
`floor(K_continuous)+1`; ordinary rounding, and `ceil` at an exact integer,
can leave the two sides equal. If the persistent input/output term already
exceeds cache, no finite K can satisfy the inequality.

The audit exposes a direct paper inconsistency:

| Published case | Reported K | K from paper equation | Resident-corrected result | Working set at reported K |
|---|---:|---:|---:|---:|
| Table II OPT-30B, B16, SL1920, BF16 | 4 | 23 | infeasible | 1,358 MiB / 512 MiB |
| Table II OPT-30B, B32, SL1920, BF16 | 4 | infeasible | infeasible | 2,618 MiB / 512 MiB |
| Table II OPT-30B, B64, SL1920, BF16 | 4 | infeasible | infeasible | 5,138 MiB / 512 MiB |
| Table III OPT-30B, B32, SL256, FP32 | 8 | 6 | 27 | 658 MiB / 512 MiB |
| Table VIII OPT-30B, B512, SL256, BF16 | 8 | infeasible | infeasible | 4,529 MiB / 512 MiB |

This does not show that the measured K values are bad empirical choices. It
shows that they cannot have been derived as stated from Equation 13 using the
paper's listed dimensions and 512 MiB cache. The exact derivation, cache domain,
and assumed tensor residency need clarification.

Artifact: `results/paper-capacity-audit-20260919.csv`.

## 2. A measured active-row phase boundary

The real pretrained OPT-125M layer sweep used one eight-core CCD and compared
native Hugging Face with K=2/4/8/16. The best true split changes sign around
M=2,048:

| Active rows M | Best split K | Best split speedup |
|---:|---:|---:|
| 1 | 2 | 0.747x |
| 8 | 2 | 0.785x |
| 32 | 2 | 0.859x |
| 128 | 2 | 0.895x |
| 256 | 2 | 0.946x |
| 512 | 2 | 0.957x |
| 1,024 | 4 | 0.981x |
| 2,048 | 2 | 1.014x |
| 4,096 | 4 | 1.057x |
| 8,192 | 2 | 1.243x |

This is the clearest deployment rule from the study: choose K from active rows
and layer shape, and bypass splitting below a calibrated threshold. K must be
selected separately for prefill and decode.

Artifact: `results/hf-opt125m-row-sweep-20260919/phase_summary.csv`.

## 3. Exact OPT-30B-shaped Table II experiment

The closest direct reproduction uses the exact Table II tensor dimensions:
`M=30,720`, `H=7,168`, `I=28,672`, BF16, with one 96-core Genoa socket. It uses
synthetic weights and inputs rather than downloading the full pretrained model,
which is sufficient for the dense-kernel/cache claim but not model quality.

| K | Median native ms | Median IMBPS ms | Speedup | Faster paired samples |
|---:|---:|---:|---:|---:|
| 1 adapter | 1,797.6 | 1,803.7 | 0.997x | 1/5 |
| 2 | 1,800.2 | 1,793.0 | 1.004x | 5/5 |
| 4 | 1,805.2 | 1,684.4 | **1.072x** | 5/5 |
| 8 | 1,807.0 | 1,621.0 | **1.115x** | 5/5 |
| 16 | 1,803.2 | 1,618.1 | **1.114x** | 4/5 |
| 23, paper-equation prediction | 1,801.3 | 1,668.4 | 1.080x | 5/5 |
| 32 | 1,801.4 | 1,624.0 | 1.109x | 5/5 |

Five pairs are too few for a two-sided exact sign test to cross 0.05 even at
5/5 wins (`p=0.0625`). The effect at K=8 is nevertheless consistent and has a
paired bootstrap interval of 1.109–1.117x in this run. A publication-quality
reproduction should increase independent process-level repetitions.

Exact L3 counters, three runs of three calls each across all 12 CCDs, show:

| Variant | L3 misses/call | Reduction vs native | Counter CV |
|---|---:|---:|---:|
| Native | 7.715 B | baseline | 3.42% |
| K=4 | 5.541 B | 28.2% | 0.43% |
| K=8 | 3.335 B | 56.8% | 1.43% |
| K=16 | 3.077 B | 60.1% | 1.35% |

Thus the cache mechanism clearly exists, but the reported K is not portable:
K=4 delivers about half the paper's stated miss reduction and a much smaller
speedup. K=16 reaches approximately 60% fewer misses, yet K=8 and K=16 tie in
latency. Minimizing LLC misses is not equivalent to minimizing time.

Artifacts: `results/synthetic-opt30-table2-m30720-t96-20260919/` and
`results/synthetic-opt30-table2-m30720-t96-uprof-l3-20260919/`.

## 4. Thread count and NUMA placement can dominate the result

At the same OPT-30B shape:

| Placement | Best tested K | Best speedup |
|---|---:|---:|
| 32 local cores | 16 | 1.097x |
| 64 local cores | 16 | 1.153x |
| 96 local cores | 8 | 1.115x |
| 64 cores, memory forced remote | 16 | **2.064x** |

The 2.064x remote-NUMA result is not a desirable deployment configuration.
Native latency more than doubles to about 4.55 s, while the split path is less
damaged. It demonstrates that an omitted NUMA/first-touch policy can create a
large apparent speedup. Paper results without thread affinity and memory
placement are not fully reproducible.

Artifacts: `results/synthetic-opt30-table2-m30720-t{32,64,96}-20260919/` and
`results/synthetic-opt30-table2-m30720-t64-remote-numa-20260919/`.

## 5. A real end-to-end counterexample at small scale

For pretrained OPT-125M, batch 1, 256 input tokens, and 16 greedy output
tokens, every K loses. Each row below uses nine paired randomized measurements:

| K | TTFT speedup | Total speedup | Decode speedup |
|---:|---:|---:|---:|
| 1 adapter | 0.971x | 0.922x | 0.908x |
| 2 | 0.957x | 0.898x | 0.882x |
| 4 | 0.929x | 0.841x | 0.818x |
| 8 | 0.875x | 0.745x | 0.713x |
| 16 | 0.787x | 0.607x | 0.569x |

For every metric and K, IMBPS loses all 9/9 pairs; the exact two-sided sign
test is `p=0.00390625`. K=2 TTFT paired speedup is 0.9561x with bootstrap 95%
CI 0.9543–0.9577x. This is strong evidence that Python/operator and small-GEMM
overheads dominate in this regime.

At decode shape M=1, exact L2 misses also move in the wrong direction: +51.5%
at K=2, +51.7% at K=4, +108.7% at K=8, and +110.7% at K=16. Native execution
should be the default decode path for this layer size.

Artifacts: `results/hf-opt125m-e2e-audit-20260919/` and
`results/hf-opt125m-uprof-l2-decode-20260919/`.

## 6. A second positive shape: Llama/SwiGLU

A standalone BF16 SwiGLU kernel with Llama-3.1-8B dimensions (`H=4,096`,
`I=14,336`, `M=32,768`) shows:

| K | Speedup |
|---:|---:|
| 2 | 1.212x |
| 4 | **1.250x** |
| 8 | 1.139x |
| 16 | 1.002x |

This verifies that the blocking mechanism extends to gated MLPs and that too
large a K eventually loses the advantage. It does not verify the paper's vLLM
or full-model claim; this is a synthetic standalone kernel with three samples.

Artifact: `results/synthetic-llama31-8b-table6-m32768-t96-20260919/`.

## 7. BF16 is algebraically equivalent but not numerically identical

The broadest audit uses 100 deterministic random prompts/seeds on pretrained
OPT-125M and compares final first-token logits and greedy token decisions:

| K | Median max-abs logit error | Worst max-abs | Model allclose | Token matches |
|---:|---:|---:|---:|---:|
| 1 adapter | 0.1099 | 0.3125 | 57/100 | 100/100 |
| 2 | 0.1250 | 0.2500 | 50/100 | **99/100** |
| 4 | 0.1250 | 0.2656 | 51/100 | **99/100** |
| 8 | 0.1250 | 0.2500 | 44/100 | **97/100** |
| 16 | 0.1250 | 0.3125 | 47/100 | **98/100** |

The four disagreement seeds include low-margin decisions, as expected, but not
only exact ties. For seed 98, K=2/4/8/16 all change token 1438 to 1942 with a
native top-two margin of 0.03125. K=8 also changes seeds 50 and 95; K=16 changes
seed 35. These are direct counterexamples to an unconditional “no output
change” interpretation.

Controls narrow the likely cause:

- Five FP32 seeds pass all model/layer comparisons for every K; worst final
  max-abs error is `1.016e-5`, and errors are identical across K.
- All-FP32 down GEMMs and accumulation inside an otherwise BF16 model do not
  restore native logits.
- BF16 partial GEMMs with FP32-only summation also do not restore native logits
  and can worsen the maximum error.
- K=1 already differs, so primitive/layout/epilogue semantics dominate; split
  reduction order is not the sole cause.

The correct wording is “exactly equivalent in real arithmetic, within a tested
numerical tolerance in floating point.” MMLU remains unverified: the server
does not contain `lm_eval`, the MMLU dataset, or the paper's exact evaluator
revision/settings. A rounded aggregate score would still not prove item-level
identity.

Artifacts: `results/hf-opt125m-numerics-100seed-bf16-20260919/`,
`results/hf-opt125m-numerical-disagreements-20260919/`, and
`results/hf-opt125m-numerics-multiseed-{fp32,fp32accum,fp32sum}-20260919/`.

## 8. Memory and packing tradeoffs

For the exact OPT-30B-shaped process, peak RSS is:

| Variant | Peak RSS | Reduction vs native |
|---|---:|---:|
| Native | 4.20 GiB | baseline |
| Adapter K=1 | 4.23 GiB | -0.8% |
| K=4 | 3.39 GiB | 19.2% |
| K=8 | 3.15 GiB | 25.0% |
| K=16 | 3.11 GiB | 25.9% |

The logical split activation falls approximately as 1/K, but input, output,
weights, runtime scratchpads, and packed storage do not. Capacity claims must
report process/device peak, not only one logical intermediate tensor.

At this shape, zero-copy views give 1.053x/1.082x/1.087x for K=4/8/16.
Persistent contiguous prepacking gives 1.072x/1.115x/1.114x, but duplicates
822,083,584 bytes (784 MiB) and costs about 15 ms to create. Packing is a
startup/memory tradeoff that can be amortized; it is not eliminated.

Artifacts: `results/synthetic-opt30-table2-rss-20260919/`,
`results/synthetic-opt30-table2-m30720-t96-views-20260919/`, and
`results/synthetic-opt30-table2-m30720-t96-20260919/`.

## 9. Matrix-multiply-inspired 2D row tiling did not help this prototype

Because K-only blocking cannot shrink the live input/output, a final experiment
combined K=8 intermediate blocking with M-row tiling on the exact OPT-30B
shape. It reduces the temporary up-projection tile but creates more GEMM calls:

| Row tile | Temporary up workspace | Speedup vs native |
|---:|---:|---:|
| 1,024 | 7 MiB | 0.899x |
| 2,048 | 14 MiB | 0.973x |
| 4,096 | 28 MiB | 0.941x |
| 8,192 | 56 MiB | 1.047x |
| 15,360 | 105 MiB | 1.067x |
| 30,720, no row split | 210 MiB | **1.083x** |

The untiled path wins. This is a counterexample to blindly applying smaller
matrix tiles at Python/operator granularity: extra GEMM dispatches and reduced
matrix efficiency outweigh the locality gain. Row tiling remains worth testing
only inside one fused C++/oneDNN kernel where it can preserve packing and avoid
operator boundaries.

Artifact: `results/synthetic-opt30-2d-tiling-k8-20260919/summary.csv`.

## 10. What should be improved next

Prioritized implementation and research work:

1. **Use a runtime dispatcher, not a fixed K.** Key it by layer shape, active
   rows, dtype, thread count, cache/NUMA domain, and phase. Keep native decode.
2. **Fuse the block pipeline below Python.** FC1+bias+activation and partial FC2
   accumulation should live in a C++/oneDNN or AOCL kernel to reduce dispatch,
   maintain packed operands, and enable useful inner-kernel row tiling.
3. **Match native BF16 semantics.** Reuse the same oneDNN primitive descriptors,
   accumulator rules, bias placement, and epilogue before making lossless claims.
4. **Make cache ownership explicit.** Derive K per CCD/NUMA domain, not from
   aggregate socket L3, and first-touch weights/workspaces on the executing node.
5. **Autotune K after a safe analytical filter.** The resident bound can reject
   impossible fits; timing must select among feasible candidates. L3 misses are
   a diagnostic, not the objective.
6. **Amortize and account for prepacking.** Cache packed weights per model/layer,
   expose its 784 MiB cost for OPT-30B-shaped weights, and include startup time.
7. **Test huge pages and TLB counters separately.** Large matrices may be
   translation-limited even after L3 traffic improves; do not conflate this with
   LLC capacity.
8. **Integrate at full-model prefill.** Measure TTFT, tokens/s, energy, memory,
   and output disagreements on OPT/Llama models large enough to cross the phase
   boundary; then compare against a tuned oneDNN and vLLM baseline.
9. **Run a specified quality suite.** Pin model revision, tokenizer, lm-eval
   revision, prompts, seeds, and dataset; publish per-item disagreement counts
   alongside MMLU averages.

## 11. Reproduction status and limitations

Completed:

- Reference OPT/GELU and SwiGLU kernels, K sweeps, layouts, metadata, raw CSV.
- Real pretrained Hugging Face OPT layer and end-to-end replacement.
- Active-row, thread-count, NUMA, layout, RSS, L2, L3, and 2D-tiling studies.
- Paired randomized measurements, bootstrap intervals, and exact sign tests.
- Multi-layer, multi-seed BF16/FP32 numerical audits and token disagreements.
- 19 passing correctness/analytical tests.

Still not directly verified:

- The authors' 128-core Turin host and aggregate 512 MiB L3 configuration.
- Full pretrained OPT-6.7B/13B/30B TTFT and MMLU.
- Sapphire Rapids, MI210, FlexGen, and vLLM/PagedAttention results.
- Power/energy and frequency-normalized measurements.

These gaps limit direct acceptance or rejection of those exact point estimates.
They do not weaken the measured counterexamples to universal performance,
cache, numerical-identity, or Equation-13 claims.

## Authoritative artifacts

- Claim matrix: `PAPER_CLAIM_MATRIX.md`
- Paper-capacity audit: `results/paper-capacity-audit-20260919.csv`
- Real-model phase boundary: `results/hf-opt125m-row-sweep-20260919/`
- End-to-end statistics: `results/hf-opt125m-e2e-audit-20260919/`
- OPT-30B timing/cache: `results/synthetic-opt30-table2-m30720-t96-20260919/`
- Thread/NUMA studies: `results/synthetic-opt30-table2-m30720-t{32,64,96}-20260919/`
- SwiGLU study: `results/synthetic-llama31-8b-table6-m32768-t96-20260919/`
- Numerical studies: `results/hf-opt125m-numerics-*-20260919/`
- RSS/layout studies: `results/synthetic-opt30-table2-rss-20260919/`
- 2D tiling: `results/synthetic-opt30-2d-tiling-k8-20260919/`
- Analysis scripts: `scripts/analyze_paper_capacity_claims.py`,
  `scripts/paired_statistics.py`, `scripts/audit_hf_opt_numerics.py`,
  `scripts/summarize_hf_row_sweep.py`, `scripts/summarize_uprof.py`, and
  `scripts/benchmark_2d_tiling.py`
