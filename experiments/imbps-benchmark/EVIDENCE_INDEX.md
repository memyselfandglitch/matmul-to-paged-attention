# IMBPS code-to-evidence index

This index maps every headline conclusion in
[`CLAIM_VERIFICATION_20260919.md`](CLAIM_VERIFICATION_20260919.md) to the code
that produced it and the committed raw/summary evidence.

Native PACE findings are reported separately in
[`PACE_REPRODUCTION_20260921.md`](PACE_REPRODUCTION_20260921.md).

## Shared benchmark path

- Synthetic weights, split ranges, persistent packing, reference OPT/SwiGLU,
  and IMBPS accumulation: [`imbps_bench/kernels.py`](imbps_bench/kernels.py)
- Randomized paired K sweeps, correctness checks, metadata, and CSV emission:
  [`imbps_bench/runner.py`](imbps_bench/runner.py)
- Real Hugging Face OPT MLP adapter:
  [`imbps_bench/hf_opt.py`](imbps_bench/hf_opt.py)
- Real layer and complete-model timing, TTFT, decode, and token comparison:
  [`imbps_bench/hf_runner.py`](imbps_bench/hf_runner.py)
- External counter/RSS workload loop:
  [`scripts/profile_mlp_kernel.py`](scripts/profile_mlp_kernel.py)
- Paired bootstrap intervals and exact sign tests:
  [`scripts/paired_statistics.py`](scripts/paired_statistics.py)

## Claim-specific provenance

| Conclusion | Producing code | Evidence |
|---|---|---|
| Exact OPT-30B-shaped BF16 K=8: 1.115x | `kernels.py`, `runner.py`; dimensions and K are captured by the manifest | [`results/synthetic-opt30-table2-m30720-t96-20260919/`](results/synthetic-opt30-table2-m30720-t96-20260919/) (`raw.csv`, `summary.csv`, `paired_statistics.csv`, `manifest.json`, `metadata.json`) |
| K=8 has 56.8% fewer L3 misses; K=4 has 28.2% fewer | [`scripts/profile_mlp_kernel.py`](scripts/profile_mlp_kernel.py), [`scripts/summarize_uprof.py`](scripts/summarize_uprof.py), [`config/uprof_l3_access_miss_zen4.xml`](config/uprof_l3_access_miss_zen4.xml) | [`results/synthetic-opt30-table2-m30720-t96-uprof-l3-20260919/`](results/synthetic-opt30-table2-m30720-t96-uprof-l3-20260919/) (three raw reports per variant plus `raw_metrics.csv` and `summary.csv`) |
| Llama-3.1-8B-shaped SwiGLU reaches 1.250x at K=4 | SwiGLU branch in `kernels.py`; `runner.py` | [`results/synthetic-llama31-8b-table6-m32768-t96-20260919/`](results/synthetic-llama31-8b-table6-m32768-t96-20260919/) |
| Real OPT-125M changes from loss to gain around M=2,048 | [`scripts/run_hf_opt_layer_zen4.sh`](scripts/run_hf_opt_layer_zen4.sh), `hf_opt.py`, `hf_runner.py`, [`scripts/summarize_hf_row_sweep.py`](scripts/summarize_hf_row_sweep.py) | [`results/hf-opt125m-row-sweep-20260919/`](results/hf-opt125m-row-sweep-20260919/) (`m1` through `m8192` raw/summary/statistics plus `phase_summary.csv`) |
| OPT-125M E2E loses every 9/9 pair; K=2 TTFT is 4.6% slower | [`scripts/run_hf_opt_e2e_zen4.sh`](scripts/run_hf_opt_e2e_zen4.sh), `hf_runner.py`, `paired_statistics.py` | [`results/hf-opt125m-e2e-audit-20260919/`](results/hf-opt125m-e2e-audit-20260919/) |
| Decode M=1 L2 misses rise 51.5-110.7% | `profile_mlp_kernel.py`, [`scripts/summarize_uprof_l2.py`](scripts/summarize_uprof_l2.py), [`config/uprof_l2_access_miss_zen4.xml`](config/uprof_l2_access_miss_zen4.xml) | [`results/hf-opt125m-uprof-l2-decode-20260919/`](results/hf-opt125m-uprof-l2-decode-20260919/) |
| Paper equation predicts K=23 and resident-corrected model is infeasible | [`imbps_bench/analytical.py`](imbps_bench/analytical.py), [`scripts/analyze_paper_capacity_claims.py`](scripts/analyze_paper_capacity_claims.py) | [`results/paper-capacity-audit-20260919.csv`](results/paper-capacity-audit-20260919.csv) |
| BF16 K>1 changes 7/400 first-token decisions | `hf_opt.py`, `hf_runner.py`, [`scripts/audit_hf_opt_numerics.py`](scripts/audit_hf_opt_numerics.py) | [`results/hf-opt125m-numerics-100seed-bf16-20260919/`](results/hf-opt125m-numerics-100seed-bf16-20260919/) and [`results/hf-opt125m-numerical-disagreements-20260919/`](results/hf-opt125m-numerical-disagreements-20260919/) |
| FP32 control stays accurate | `audit_hf_opt_numerics.py`; FP32/BF16 accumulation modes in `hf_opt.py` | [`results/hf-opt125m-numerics-multiseed-fp32-20260919/`](results/hf-opt125m-numerics-multiseed-fp32-20260919/), plus the `fp32accum` and `fp32sum` sibling directories |
| Peak RSS falls only 19-26% | `profile_mlp_kernel.py`, invoked under `/usr/bin/time -v`; each artifact retains the exact command | [`results/synthetic-opt30-table2-rss-20260919/`](results/synthetic-opt30-table2-rss-20260919/) |
| Remote NUMA inflates apparent speedup to 2.064x | `runner.py`, launched with CPUs 0-63 and memory forced to node 1 | [`results/synthetic-opt30-table2-m30720-t64-remote-numa-20260919/`](results/synthetic-opt30-table2-m30720-t64-remote-numa-20260919/); compare with the local `t64` sibling directory |
| Python-level 2D row tiling does not beat the full-row path | [`scripts/benchmark_2d_tiling.py`](scripts/benchmark_2d_tiling.py) | [`results/synthetic-opt30-2d-tiling-k8-20260919/summary.csv`](results/synthetic-opt30-2d-tiling-k8-20260919/summary.csv) |
| PACE v1 exact OPT-30B: K=4 reaches 1.108x one-socket and 1.122x two-socket | [`scripts/benchmark_pace_mlp.py`](scripts/benchmark_pace_mlp.py), [`scripts/run_pace_table2_zen4.sh`](scripts/run_pace_table2_zen4.sh) | [`results/pace-v1-standalone-opt30-20260921.csv`](results/pace-v1-standalone-opt30-20260921.csv) |
| PACE v1 OPT-125M E2E: all IMBPS K values lose to TPP | [`scripts/run_pace_e2e_opt125m_zen4.sh`](scripts/run_pace_e2e_opt125m_zen4.sh), [`config/pace_opt125m_tpp.json`](config/pace_opt125m_tpp.json), [`config/pace_opt125m_imbps.json`](config/pace_opt125m_imbps.json) | [`results/pace-v1-e2e-opt125m-20260921.csv`](results/pace-v1-e2e-opt125m-20260921.csv) |
| PACE-main top-5 correctness passes despite 1-2 token divergences from HF | [`scripts/run_pace_correctness_opt125m_zen4.sh`](scripts/run_pace_correctness_opt125m_zen4.sh), `config/pace_correctness_opt125m_*.json` | [`results/pace-main-correctness-opt125m-20260921.csv`](results/pace-main-correctness-opt125m-20260921.csv) |

## Deliberately unverified claims

There is no MMLU, Turin, Sapphire Rapids, MI210, or vLLM result directory. Those
experiments were not run, so the report marks them unverified rather than
inferring results from unrelated workloads.
