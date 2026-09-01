# vLLM KV-cache physical-layout study

This study separates two independent layout questions in current vLLM:

1. Within one layer and cache page, is `head` or `token slot` physically first?
2. Across layers and pages, is `layer` or `block` physically first?

It follows vLLM `main` as inspected on 2026-09-01. Layout APIs are actively
evolving, so older releases may expose only the legacy `HND` and `NHD` names.

## vLLM does not have one universal fixed physical layout

The current logical shape is:

```text
[L, B, H, N, C]

L = layer slot
B = physical cache block/page
H = KV head
N = token/state slot inside the block
C = contiguous backend-specific content, commonly packed K and V data
```

The allocator can permute the first four axes. Current layout names include
`LBHNC`, `LBNHC`, `LHBNC`, `BLHNC`, `BLNHC`, and `BHLNC`. vLLM resolves one
layout for a run from the attention backends' supported layouts, an optional
`VLLM_KV_CACHE_LAYOUT` request, and a KV connector preference.

Legacy aliases map as follows:

```text
HND -> LBHNC -> [layer, block, head, token, content]
NHD -> LBNHC -> [layer, block, token, head, content]
```

Notice that `block` is before both `head` and `token` in each legacy layout.
Therefore “head-major” versus “block-major” is not a valid either/or choice.
The usual comparison is **HND versus NHD inside a block**. A genuinely
block-major-across-layers layout is `BLHNC`, where all layers for a block are
adjacent.

## Does the kernel remain the same?

The attention equations remain identical. The physical address changes:

```text
address = base
        + block * block_stride
        + head  * head_stride
        + token * token_stride
        + dim
```

For `[B,H,N,C]` (HND):

```text
head_stride  = N*C
token_stride = C
```

For `[B,N,H,C]` (NHD):

```text
head_stride  = C
token_stride = H*C
```

A stride-aware kernel can use exactly the same source and receive different
strides. A kernel that assumes a particular contiguous order cannot: vLLM must
select a compatible specialization/backend or convert the layout. Physical
conversion is a real permutation/copy; a tensor `transpose` alone normally
creates only a different strided view.

Current FlashAttention and Triton backend wrappers take the logical cache,
transpose it into the view expected by the attention call, and pass the
resulting strides into the underlying path. Backend support still matters:
vLLM explicitly resolves the intersection of layouts supported by all active
backends. Some platform-specific write kernels retain stricter assumptions and
fall back to stride-aware paths for non-native layouts.

## Expected trade-offs

- HND makes the token sequence for one head contiguous, matching a common
  decode attention traversal: head, then token, then head dimension.
- NHD makes all heads for one token adjacent, which can favor inserting a newly
  produced token's K/V values.
- LBHNC keeps a layer's cache pages together, which is natural for per-layer
  attention compute.
- BLHNC keeps the same block across layers together, allowing a connector to
  transfer a whole cross-layer block using larger contiguous operations.

Which is faster depends on the backend, GPU, block size, number of KV heads,
head dimension, quantization, attention phase, and whether compute or KV
transfer is being measured.

## Run the educational CPU model

```sh
cd vllm_layout
make check
./build/kv_layout_study 9
```

The executable uses one stride-aware attention-like kernel for both HND and
NHD, then separately compares layer-major and block-major block gathering. It
demonstrates the structural effects; it does not predict vLLM GPU performance.

## Primary references

- [vLLM physical layout enum](https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_cache_layout.py)
- [Layout resolution and legacy aliases](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/utils.py)
- [FlashAttention KV-cache views and attention call](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py)
- [Triton attention KV-cache views](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py)
- [KV-transfer layout conversion](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/utils.py)
