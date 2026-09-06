# Matrix multiplication, BMM, and K/V cache

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

## When BMM helps

BMM can help when matrices are individually too small to use the machine well:

- one API/kernel dispatch replaces many dispatches;
- independent products provide additional parallel work;
- uniform shapes and strides simplify scheduling;
- a shared operand can sometimes be reused from cache.

BMM does not inherently perform fewer FLOPs. For large matrices, looping over
well-optimized GEMMs can be just as fast. If every batch item shares the same
right-hand matrix, contiguous `A[batch,M,K]` can instead be viewed as
`A[batch*M,K]` and processed as one larger ordinary GEMM.

## Mapping attention to BMM

For attention, batch and head are commonly flattened into one group dimension:

```text
groups = batch * heads

Q[groups,Q,D] * K^T[groups,D,T] -> scores[groups,Q,T]
softmax(scores)
probabilities[groups,Q,T] * V[groups,T,D] -> output[groups,Q,D]
```

Thus K/V-cache attention contains two batched matrix products with a softmax
between them. During prompt processing, `Q` can contain many query tokens and
these are genuine matrix products. During token-by-token decoding, `Q=1`, so
each product degenerates into a batched matrix-vector operation. Reading the
growing K/V cache often becomes more important than arithmetic at that point.

The included benchmark uses PyTorch tensors with the cache layout:

```text
[batch, heads, context_tokens, head_dim]
```

Production systems may use blocked or paged cache layouts, but the underlying
`QK^T -> softmax -> probabilities*V` structure remains the same.

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
