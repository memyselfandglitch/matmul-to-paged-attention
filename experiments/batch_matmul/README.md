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
