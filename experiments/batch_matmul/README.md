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

On a CPU, a BMM library may loop over GEMM kernels or parallelize matrices
across cores. On a GPU, the batch index normally becomes another grid/scheduling
dimension, allowing one launch to schedule many small products. Libraries may
also use strided-batched or grouped-GEMM kernels.

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

The included benchmark uses a simple contiguous cache layout:

```text
[batch, heads, context_tokens, head_dim]
```

Production systems may use blocked or paged cache layouts, but the underlying
`QK^T -> softmax -> probabilities*V` structure remains the same.

## Run the study

```sh
cd batch_matmul
make check
./build/bmm_kv_study 9
```

The optional argument is the number of timing repetitions. This is portable
C++ intended to expose structure and dispatch overhead; it is not a benchmark
of Accelerate, oneDNN, cuBLAS, or another vendor library.
