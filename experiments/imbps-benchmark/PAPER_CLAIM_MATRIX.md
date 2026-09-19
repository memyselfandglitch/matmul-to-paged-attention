# IMBPS paper claim matrix

This matrix separates mathematical facts, claims supported by the paper's own
tables, claims tested on the IISc Genoa host, and claims that still require the
authors' target hardware or raw data. “Counterexample” means that the claim is
false in at least the measured regime; it does not imply that the paper's exact
Turin/OPT-30B result is false.

## Method, memory, and correctness

| ID | Paper claim | Assessment | Evidence or limitation |
|---|---|---|---|
| M1 | Column-partitioned up projection and row-partitioned down projection are algebraically equivalent to an unsplit OPT/GPT MLP. | **Proved in exact arithmetic.** | Block matrix multiplication gives `Gψ = Σ Giψi`. Biases are omitted from the paper's proof; equivalence requires partitioning the FC1 bias and adding the FC2 bias once after the reduction. |
| M2 | The same blocking applies to Llama/SwiGLU MLPs. | **Algebraically valid and unit-tested.** | Partitioning both up and gate outputs along the same intermediate dimension preserves element-wise SiLU/multiplication and the down-projection sum. Synthetic uneven-K tests pass for views and prepacked layouts. |
| M3 | IMBPS is “lossless” and results are element-wise identical in principle. | **True only over exact real arithmetic; false as a bitwise floating-point claim.** | Reordered primitives and reductions change floating-point behavior. Across 100 real OPT-125M seeds, K>1 changes 7/400 first-token decisions; even adapter K=1 differs numerically because it uses a different primitive/layout path. |
| M4 | Practical deviations are typically `<0.0001`. | **Counterexample on real BF16 OPT-125M.** | Across 100 seeds, final-logit maximum absolute error has median 0.110–0.125 and reaches 0.3125. FP32 stays <=1.016e-5, showing strong precision dependence; both all-FP32 partials and FP32-only summation fail to restore native BF16 logits. |
| M5 | Splitting causes no accuracy loss and MMLU is unchanged. | **Token-level counterexamples; MMLU unverified.** | K=2/4/8/16 match 99/99/97/98 of 100 first-token decisions. Paper Table VII itself varies: OPT-125M 0.261–0.262 and OPT-30B 0.245–0.249. No pinned evaluator or item-level paper data are available, and the local server lacks `lm_eval`/MMLU. |
| M6 | There is no systematic bias toward any MMLU task. | **Unsupported by the presented evidence.** | The paper provides only the 57-task average, not per-task scores, item predictions, uncertainty, or disagreement counts. |
| M7 | Peak intermediate activation storage falls with K. | **Proved, but process-level saving is bounded.** | The block activation width is `ceil(I/K)`. At the exact OPT-30B shape, K=4/8/16 reduce measured process peak RSS only 19.2%/25.0%/25.9% because input, output, weights, packed copies and scratch remain live. |
| M8 | IMBPS eliminates packing overhead. | **Not a general consequence of the algorithm.** | At the exact OPT-30B shape, views score 1.053×/1.082×/1.087× for K=4/8/16, while persistent prepacking scores 1.072×/1.115×/1.114× but duplicates 784 MiB and costs about 15 ms. Packing is an amortizable tradeoff, not eliminated work. |
| M9 | Equation 13 determines the optimal K. | **Refuted as stated.** | For Table II OPT-30B/B16, the paper equation itself predicts minimum K=23, not reported K=4. It is only a cache-capacity lower bound and omits output residency, overhead, conflicts, topology, packing, and GEMM efficiency. |
| M10 | The Equation 13 result guarantees Equation 12's cache inequality. | **Incorrect for multiple published rows.** | Algorithm 1 keeps input and output live, so a lower bound contains `2·M·H`. At Table II B16/K4 this is 1,358 MiB against 512 MiB; B32/B64 are infeasible even under the published formula. The strict inequality also requires `floor(K_continuous)+1`. |
| M11 | K-only blocking handles large working sets. | **Counterexample to sufficiency.** | For synthetic OPT-6.7B dimensions at M=4096 on one Genoa CCD, the BF16 input alone is 32 MiB while a conservative usable-cache budget is 24 MiB. No K can satisfy the model; M-dimension tiling is required. |
| M12 | Mixed-precision blocks with higher-precision accumulation are naturally supported. | **Plausible extension, but two straightforward implementations fail.** | All-FP32 down GEMMs/accumulation and BF16 partial GEMMs with FP32 summation both leave roughly 0.125 median final-logit error and do not restore native behavior. The former also doubles packed storage and slows TTFT. Native BF16 primitive/epilogue matching is required. |
| M13 | IMBPS integrates with pruning and can guide pruning thresholds/batch sizes. | **Speculative/unverified.** | No pruned-model experiment, pruning threshold study, or accuracy/performance table is presented. |

## CPU performance and cache claims

| ID | Paper claim | Assessment | Evidence or limitation |
|---|---|---|---|
| P1 | Standalone MLP obtains 1.5–1.6× speedup (abstract). | **Not reproduced and not supported by Table II.** | Table II's maximum is 1.25×. Exact OPT-30B dimensions/M on Genoa peak at 1.115×; a Llama-3.1-8B-shaped SwiGLU kernel reaches 1.250×. The abstract's exact workload/raw data are absent. |
| P2 | OPT-30B, SL=1920, BF16 obtains 1.21–1.25× speedup and 2.53–4.09× fewer L3 misses at BS=16–64, K=4. | **Mechanism reproduced; K=4 magnitude not reproduced.** | At exact OPT-30B dimensions and B16/SL1920 rows on 96-core Genoa, K=4 gives 1.072× and 28.2% fewer L3 misses. K=8 gives 1.115×/56.8%; K=16 gives 1.114×/60.1%. This is synthetic and Genoa, not pretrained OPT-30B on Turin. |
| P3 | TTFT improves consistently across all tested long-context models (Figure 8). | **Counterexample outside the tested model regime.** | Real OPT-125M/B1/SL256 TTFT is slower for every K: 0.957× at K=2, 0.894× at K=5, 0.787× at K=16. Figure 8's exact raw values are not supplied. |
| P4 | Table III reports 29.28–34.19% TTFT improvements for OPT-6.7B/30B, SL=256, FP32. | **Untested.** | Requires the large pretrained models and high batches 32–256. The table's “improvement” should be reported together with absolute times and a precise formula. |
| P5 | Table IV reports 60.65–69.87% fewer L3 misses for OPT-13B/30B, SL=256, BF16, BS=64–1024. | **Not reproduced at small scale; target regime untested.** | Non-multiplexed AMD L3 counters on OPT-125M/B1 show 0.42% fewer misses at K=2 (effectively zero), then 15.4% more at K=4, 33.7% more at K=5, 101% more at K=8 and 177% more at K=16. |
| P6 | Reduced L3 misses directly correlate with latency improvements. | **Sometimes correlated, not sufficient.** | At exact OPT-30B dimensions, larger reductions accompany speedups, but K=8 and K=16 have different miss counts and tied latency. At OPT-6.7B dimensions/M=30,720, K=8 reduces L3 misses 31.0% yet is 10.2% slower. |
| P7 | Table V shows large standalone, L3 and E2E gains on Intel Sapphire Rapids. | **Untested.** | No Sapphire Rapids host is in scope. The “MLP Execution Time Improvement” entries 81.35–94.07% are ambiguous without absolute times or a stated speedup/reduction formula and do not align transparently with the abstract's 1.5–1.6× wording. |
| P8 | IMBPS beats vLLM by 5.02–6.62% for Llama-3.1-8B. | **Standalone mechanism supported; vLLM claim untested.** | A synthetic Llama-3.1-8B-shaped SwiGLU MLP reaches 1.250× at K=4, but the paper's vLLM 0.8.4/PagedAttention integration and end-to-end workload were not reproduced. |
| P9 | Excessive K eventually degrades performance and there is a workload-dependent optimum. | **Qualitatively reproduced.** | Layer, E2E and exact L3 results worsen as K becomes large. This also disproves any universal fixed K. |
| P10 | Decode MLPs retain substantial L2 misses and should use an L2-derived split. | **Counterexample for OPT-125M.** | At M=1, exact raw L2 totals rise 51.5% at K=2, 51.7% at K=4, 108.7% at K=8 and 110.7% at K=16; E2E decode also slows. Native decode should be the default in this regime. |
| P11 | Benefits generalize across BF16 and FP32. | **Numerical behavior generalizes differently; performance unverified.** | FP32 is much more accurate than BF16 in this adapter. Target-scale FP32 performance has not been reproduced. |
| P12 | Benefits generalize across Genoa, Turin and Sapphire Rapids. | **Workload- and topology-dependent on Genoa; other CPUs unverified.** | Genoa ranges from 0.747× at M=1 to 1.243× at M=8192 for one real layer. Exact OPT-30B shape changes optimum with 32/64/96 threads; remote NUMA inflates apparent speedup to 2.064×. Intel/Turin reproduction is absent. |

## GPU and system-integration claims

| ID | Paper claim | Assessment | Evidence or limitation |
|---|---|---|---|
| G1 | MI210 throughput improves up to 28.7% for OPT-6.7B and 19.8% for OPT-13B. | **Untested.** | No MI210 is available in this study. Raw batch/latency data are not tabulated in the paper. |
| G2 | Baseline fails around BS>80 while IMBPS reaches BS=160 by reducing activation storage. | **Mechanism is credible; quantitative result unverified.** | The 1/K activation bound supports higher capacity, but actual OOM depends on complete model/runtime allocations. |
| G3 | IMBPS is device-agnostic and consistently beneficial. | **Overgeneralized.** | Algebra is device-agnostic; performance is not. OPT-125M/B1 on Genoa is slower, and only one GPU generation is reported. |
| G4 | IMBPS improves FlexGen throughput and supports longer contexts/larger batches. | **Unsupported in the paper's presented results.** | No FlexGen performance table, configuration or raw result is provided. |
| G5 | IMBPS integrates seamlessly with Hugging Face and vLLM with modest changes. | **Partially demonstrated, not independently reproduced for vLLM.** | Real Hugging Face layer and model replacement works in this repository. vLLM integration code/artifacts from the paper are unavailable. |

## Reproducibility and reporting observations

| ID | Observation | Consequence |
|---|---|---|
| R1 | Figures 8 and 9 have bars but no numeric labels or raw-data link. | Exact reproduction/error analysis requires digitizing plots and still lacks variance. |
| R2 | Tables II–VIII report point estimates without repetitions, dispersion, confidence intervals, absolute latency, power, or raw CSV. | Statistical significance and run-to-run stability cannot be assessed. |
| R3 | “Improvement,” “speedup,” and normalized timing are used with different units across tables. | A reproduction must define `baseline/optimized`, percent time reduction, and percent throughput increase separately. |
| R4 | K=1 in Table VIII is native vanilla execution. | Adapter K=1 must be kept as a separate layout/primitive control; treating it as native would attribute non-splitting effects to IMBPS. |
| R5 | The paper's software versions are mostly specified, but model revisions, prompts/token IDs, seeds, thread affinity, NUMA policy, warmups and repetitions are not. | Bitwise correctness and performance results are not fully reproducible. On this host, forcing 64-core execution to remote memory changes the best measured speedup from 1.153× local to 2.064× remote by damaging the native baseline. |
| R6 | The IISc setup now has working non-root AMD uProf MSR access. | `AMDuProfPcm` 5.3.521 has `cap_sys_rawio,cap_perfmon=eip`; a minimal two-event L3 configuration avoids the built-in latency-event multiplexing. |

## Current authoritative artifacts

- Full narrative: `CLAIM_VERIFICATION_20260919.md`
- Paper-equation audit: `results/paper-capacity-audit-20260919.csv`
- Real-model active-row phase boundary: `results/hf-opt125m-row-sweep-20260919/`
- Non-multiplexed L3 reports: `results/hf-opt125m-uprof-l3-minimal-20260919/`
- L3 parser: `scripts/summarize_uprof.py`
- Zen 4 L3 event config: `config/uprof_l3_access_miss_zen4.xml`
- Exact OPT-30B timing/L3: `results/synthetic-opt30-table2-m30720-t96-20260919/` and `results/synthetic-opt30-table2-m30720-t96-uprof-l3-20260919/`
- Thread/NUMA studies: `results/synthetic-opt30-table2-m30720-t{32,64,96}-20260919/`
- SwiGLU study: `results/synthetic-llama31-8b-table6-m32768-t96-20260919/`
- Decode L2 reports and config: `results/hf-opt125m-uprof-l2-decode-20260919/` and `config/uprof_l2_access_miss_zen4.xml`
- Packing/RSS ablations: `results/synthetic-opt30-table2-m30720-t96-views-20260919/` and `results/synthetic-opt30-table2-rss-20260919/`
- Numerical audits: `results/hf-opt125m-numerics-100seed-bf16-20260919/` and `results/hf-opt125m-numerics-multiseed-{fp32,fp32accum,fp32sum}-20260919/`
- 2D tiling: `results/synthetic-opt30-2d-tiling-k8-20260919/`
- End-to-end audit: `results/hf-opt125m-e2e-audit-20260919/summary.csv`
