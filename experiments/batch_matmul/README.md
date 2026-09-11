# Matrix multiplication and BMM

## Structural difference

Ordinary matrix multiplication computes one product:

```text
A[M,K] * B[K,N] -> C[M,N]
```

Batched matrix multiplication adds a leading batch dimension:

```text
A[batch,M,K] * B[batch,K,N] -> C[batch,M,N]
```

Mathematically, BMM is just:

```cpp
for (int b = 0; b < batch; ++b)
    C[b] = A[b] * B[b];
```

There is no interaction between batch items and no reduction in arithmetic.
With strided contiguous storage, an implementation finds each matrix using:

```text
A address = base_A + b * M*K
B address = base_B + b * K*N
C address = base_C + b * M*N
```

`torch.matmul` performs this batching when at least one input has more than two
dimensions. It treats the final two dimensions as matrix dimensions and
broadcasts the preceding dimensions. The study now calls the real PyTorch
operation; it contains no handwritten matrix-multiplication kernel.

## Why compare the two calls?

BMM can help when matrices are individually too small to use the machine well:

- one API/kernel dispatch replaces many dispatches;
- independent products provide additional parallel work;
- uniform shapes and strides simplify scheduling;

BMM does not inherently perform fewer FLOPs. For large matrices, looping over
well-optimized GEMMs can be just as fast.

## Cache-capacity hypothesis

The presentation run also sweeps square matrices from `64x64` through
`3072x3072` at batch size 8. The largest nominal `A+B+C` batch footprint is
864 MiB, which exceeds the 384 MiB aggregate L3 reported by the target EPYC
9654. It reports:

- the nominal `A + B + C` working set for one matrix product;
- the nominal working set for the complete batch;
- CPU0's reported L2 and L3 size per cache instance;
- whether one product and the complete batch exceed that L3-instance size;
- looped and batched median time and their ratio;
- effective useful GFLOP/s for both paths;
- nominal square-GEMM arithmetic intensity in FLOP per `A+B+C` byte;
- the first observed point within 5% of parity.

This separates measurement from interpretation. A ratio approaching one can
happen because Python dispatch and output-stacking overhead becomes negligible
beside cubic GEMM work. If it happens after a cache threshold, that is a useful
correlation, but cache misses require hardware counters before being claimed as
the cause. The raw sweep is saved as `cache-sweep.csv`.
The nominal GEMM arithmetic intensity is identical for both paths because they
perform the same useful arithmetic on the same A, B and final C tensors. The
loop's additional stack copy is discussed separately rather than represented
as a different matrix-multiplication algorithm.

## Presentation quick start on IISc

Run the complete PyTorch matmul-versus-BMM comparison with one command:

```bash
cd /data/scratch/deveshisingh/matmul-to-paged-attention/experiments/batch_matmul
./run.sh
```

The launcher finds an existing cluster Python environment that can import
PyTorch, submits a CPU Slurm job, waits, and prints the report. Raw output and
machine information are saved under `results/job-<job-id>/`.

## Run the study

```sh
cd batch_matmul
make check
python3 bmm_kv_study.py --repetitions 9
```

The explicit-loop baseline also calls `torch.matmul` for every individual
matrix. This isolates the difference between issuing many PyTorch calls and
giving all leading batch dimensions to one `torch.matmul` call. The script
automatically uses CUDA, MPS, or CPU in that order; pass `--device cpu` to
select CPU explicitly.

References:

- [`torch.matmul`](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html)
- [`torch.bmm`](https://docs.pytorch.org/docs/stable/generated/torch.bmm.html),
  the stricter three-dimensional operation that does not broadcast
