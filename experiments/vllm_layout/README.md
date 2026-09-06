# vLLM Triton KV-cache layout study

This experiment compares the two FlashInfer-style physical KV-cache layouts
while using vLLM's `TRITON_ATTN` backend on AMD ROCm:

- `NHD`, alias `LBNHC`: layer, block, token, head, content
- `HND`, alias `LBHNC`: layer, block, head, token, content

Both cases run sequentially in one Slurm allocation, on the same GPU, with the
same model and benchmark parameters. This isolates layout while keeping the
attention backend fixed. It does not measure FlashInfer.

## Setup

From the repository on `mn01`:

```bash
sbatch experiments/vllm_layout/setup_rocm_vllm.sbatch
```

After setup completes successfully:

```bash
sbatch experiments/vllm_layout/run_triton_layout_ab.sbatch
```

To reverse the order as a simple bias check:

```bash
LAYOUT_ORDER=HND,NHD sbatch --export=ALL \
  experiments/vllm_layout/run_triton_layout_ab.sbatch
```

Results are written to:

```text
/rhome/deveshisingh/matmul-to-paged-attention/experiments/vllm_layout/results/job-<job-id>/
```

The directory contains raw logs, per-iteration JSON data, and `summary.txt`.

## Validated MI210 result

These runs used `facebook/opt-125m`, dummy model weights, FP16 KV cache,
batch size 8, 1024 input tokens, 32 output tokens, 3 warmups, 10 measured
iterations, eager execution, and `TRITON_ATTN`. Both completed on the same
AMD Instinct MI210 (`gn03`) with ROCm 7.2.1.

| Job | Order | NHD median | HND median | Faster |
| --- | --- | ---: | ---: | ---: |
| 7504 | NHD, HND | 301.095 ms | 294.631 ms | HND by 2.1% |
| 7506 | HND, NHD | 301.144 ms | 297.315 ms | HND by 1.3% |

The mean of the two run medians is 301.119 ms for NHD and 295.973 ms for
HND, so HND was about 1.7% lower-latency in this small MHA case. Treat this as
a first signal, not a general conclusion about all models, sequence lengths,
batch sizes, or KV-cache dtypes.

Raw validated results:

```text
/rhome/deveshisingh/matmul-to-paged-attention/experiments/vllm_layout/results/job-7504/
/rhome/deveshisingh/matmul-to-paged-attention/experiments/vllm_layout/results/job-7506/
```
