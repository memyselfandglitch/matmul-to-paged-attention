# From Matrix Multiplication to Paged KV-Cache Layouts

Audience: systems/architecture faculty and research collaborators.

Style: technical research presentation; white background, deep navy text,
electric blue/teal data accents, amber for contrasts, restrained diagrams,
large readable labels, no decorative stock photography. Preserve exact chart
images rather than redrawing measured values.

## Slide 1 — From Matrix Multiplication to Paged KV-Cache Layouts

Goal: introduce the progression from loop locality to a fragmentation-dependent
KV-cache layout result.

- Matrix multiplication → BMM → decode attention → paged KV cache.
- Deveshi Singh.
- Target evidence: AMD EPYC CPU on IISc `mn01`.

Visual: clean four-stage pipeline.

## Slide 2 — The study asks one systems question at increasing depth

Goal: show the research ladder.

- How does loop order affect row-major matrix multiplication?
- What remains after cache tiling and SIMD register blocking?
- What is structurally different about BMM and decode attention?
- How do logical indexing, physical layout, and traversal differ in vLLM?
- When does physical-block fragmentation reverse the best layout/traversal?

Visual: staircase or vertical research ladder.

## Slide 3 — Loop order is a memory-access decision

Goal: connect `i,j,k` loops to row-major locality.

- `C[i,j] += A[i,k] * B[k,j]` performs the same arithmetic for all six orders.
- Row-major address: `base + (row * n + column)`; `static_cast<size_t>` makes
  the index an unsigned size type suitable for array addressing.
- Inner `j` gives contiguous B/C access; inner `k` gives a strided B access in
  the naive layout.
- Performance changes because cache lines and reuse differ, not because FLOPs
  differ.

Visual: three small matrix grids with highlighted row/column access.

## Slide 4 — Tiling and register blocking attack different levels

Goal: separate cache reuse from register reuse.

- Macro tiles: `Mc×Nc×Kc = 120×64×240` keep working sets closer to cache.
- SIMD width on the Mac study: 128 bits = four float32 lanes.
- Register microkernel: `6×8` output tile = six rows × two SIMD vectors.
- Twelve vector accumulators hold 48 partial C values across the K range.
- A scalar is broadcast and reused across two B vectors; each B vector is
  reused across six rows.

Visual: 6×8 C tile divided into twelve 1×4 register strips.

## Slide 5 — After identical tiling, macro loop order is a small effect

Goal: report the controlled loop-order result without overclaiming.

- All six `Mc/Nc/Kc` orders share the same inner kernel and tile sizes.
- The tiled-plus-register-blocked stage is about `2.9×–3.1×` faster than the
  tiled stage in the recorded macOS sanity run.
- No single macro order dominates across all sizes.
- Register reuse is the large signal; residual order differences are smaller
  and machine/noise sensitive.

Visual: compact bar showing ≈3.0× geometric speedup, plus six nearly level dots.

## Slide 6 — BMM is many independent matrix products under one operation

Goal: clarify structure and practical advantage.

- GEMM: `A[M,K] @ B[K,N] → C[M,N]`.
- BMM: `A[B,M,K] @ B[B,K,N] → C[B,M,N]`.
- There is no cross-batch reduction and no inherent FLOP reduction.
- Advantage comes from lower dispatch overhead and more parallel work for
  small matrices.
- Local PyTorch CPU sanity: 32 independent `64×64` products were `1.92×`
  faster as one batched `torch.matmul` call than a Python loop of calls.

Visual: stack of matrix pairs entering one batched operator.

## Slide 7 — K/V attention contains two batched products

Goal: map BMM to attention.

- Flatten batch and head into groups.
- `Q @ Kᵀ → scores`, softmax, then `probabilities @ V → output`.
- Prefill has many query tokens and true matrix products.
- One-token decode has `Q=1`; products become batched matrix-vector-like work.
- Cache reads grow with context, so decode is deliberately low intensity.

Visual: QKᵀ → softmax → PV pipeline with shapes.

## Slide 8 — Hugging Face and vLLM solve different cache-management problems

Goal: explain why blocks appear in vLLM.

- A conventional transformer cache is commonly viewed as
  `[batch, head, sequence, head_dimension]`.
- Blocks are not required by attention mathematics.
- vLLM adds a block table and physical pages for allocation, sharing, and
  avoiding large contiguous per-request reservations.
- Logical token order remains the attention order; physical block numbers may
  be non-contiguous.

Visual: contiguous per-request tensor versus logical blocks mapped to physical
pages.

## Slide 9 — vLLM keeps a logical contract while backends choose strides

Goal: state the layout facts precisely.

- Current checked-out source exposes logical `[L,B,H,N,C]` indexing.
- Physical permutations include `LBHNC`, `LBNHC`, and `LHBNC`.
- Legacy aliases: `NHD → LBNHC`; `HND → LBHNC`.
- The active attention backend advertises supported/preferred layouts; there is
  no timeless backend-independent physical default.
- The installed older MI210 environment used for the initial baseline defaulted
  to `NHD`, corresponding to per-layer `BNHD`.

Visual: one logical tensor feeding three physical stride permutations.

## Slide 10 — The controlled decode microbenchmark isolates layout and traversal

Goal: establish what was measured and what was not.

- One decode query token; scaled QK, online softmax, weighted V.
- Shape: `B=512, H=32, N=16, D=128`, float32.
- 8192 cached tokens; K+V footprint = 256 MiB per layer.
- Useful arithmetic intensity = 0.5 FLOP per KV byte.
- Standalone C++ CPU microbenchmark—not end-to-end vLLM/FlashInfer latency.
- Identical logical values and correctness checks across all cases.

Visual: test-harness diagram with controlled variables.

## Slide 11 — Phase 1: fixed BNHD memory favors BNHD traversal

Goal: answer the first controlled question.

- Physical memory held fixed as BNHD.
- Traversals tested: BNHD, BHND (block-first), HBND (head-first).
- BNHD wins with sequential and fully shuffled block tables.
- Sequential medians: 38.8 ms BNHD, 44.7 ms BHND, 74.0 ms HBND.
- Result: traversal should first match the contiguous physical dimension.

Visual: use uploaded figure “Phase 1 fixed BNHD traversal results” unchanged.

## Slide 12 — Phase 2: matched traversal wins every physical layout

Goal: show the full 3×3 experiment.

- Cross BNHD/BHND/HBND memory with all three traversals.
- Memory-matched traversal wins `3/3` layouts for both block-table patterns.
- Sequential global winner: HBND/HBND at 35.461 ms.
- Fully shuffled global winner: BHND/BHND at 37.515 ms.
- HBND/HBND slows 17.4% under full shuffling; BHND/BHND slows 1.2%.

Visual: use uploaded figure “Phase 2 layout traversal heatmaps” unchanged.

## Slide 13 — Fragmentation is modeled by contiguous physical-block runs

Goal: make the independent variable concrete.

- Run length 512: one sequential scan of all blocks.
- Run length 16: 32 shuffled runs, each internally contiguous.
- Run length 4: 128 shuffled runs.
- Run length 1: every physical block independently shuffled.
- Logical token order is preserved by the block table; only physical address
  locality changes.

Visual: three block tables mapping logical blocks to colored physical runs.

## Slide 14 — Timing sweep: fragmentation flips HBND to BHND

Goal: report the broad AMD crossover.

- Job 7617; five deterministic block-table seeds.
- HBND is faster from run length 512 through 8.
- Run length 4 is an effective tie.
- BHND is 2.7% faster at run 2 and 5.9% faster at run 1.
- BNHD remains stable but is not the pairwise focus.

Visual: use uploaded figure “AMD fragmentation crossover job 7617” unchanged.

## Slide 15 — uProf repeat refines the stable boundary to 4 → 3

Goal: link time to hardware counters cautiously.

- `mn01`, AMD uProf 5.3.521 MSR mode, core 0/CCX 0.
- Job 7775 repeats runs 8,4,3,2,1 across three seeds.
- Run lengths 8 and 4 are ties in this repeat; BHND wins at 3, 2, and 1.
- HBND/BHND time reaches 1.088× at run 1.
- HBND IPC falls from ≈1.08 to 1.01; BHND stays ≈1.09–1.10.

Visual: use uploaded figure “AMD uProf counters job 7775” unchanged.

## Slide 16 — The counters rule out the simplest cache-miss story

Goal: distinguish observation from mechanism.

- HBND has lower L3 miss percentage than BHND, even when slower.
- HBND has about 3–9% more L3 misses per 1,000 retired instructions.
- L3 miss latency stays roughly 97–99 ns as HBND slows.
- DRAM bandwidth remains around 7.2–7.7 GB/s and is hardware scoped.
- Defensible claim: fragmentation causes a repeatable crossover associated
  with lower HBND IPC; L3 miss percentage alone does not explain it.

Visual: evidence matrix with “supports / does not support / still unknown”.

## Slide 17 — What the Mac sanity check contributes

Goal: use local results only at the correct evidential level.

- macOS arm64 used smaller `B=128,H=16,N=16,D=64` and 16 MiB K+V.
- Matched traversal again wins each physical layout.
- Increasing fragmentation again reverses HBND vs BHND preference.
- Crossover location differs (between runs 16 and 4), as expected across
  architecture and footprint.
- Use it as qualitative corroboration, not target-machine evidence.

Visual: AMD and Mac arrows showing same direction, different boundary.

## Slide 18 — Research implication: no universal best physical layout

Goal: state the actionable hypothesis.

- HBND benefits when each head can scan long contiguous block runs.
- BHND is more robust when physical block references are fragmented.
- A production decision must include allocator/scheduler block-table locality,
  backend kernel mapping, and layout-conversion cost.
- Candidate direction: fragmentation-aware layout/traversal choice or an
  allocator that preserves longer physical runs.
- This standalone CPU result motivates—but does not yet prove—a vLLM GPU win.

Visual: decision boundary with locality on the x-axis.

## Slide 19 — Next experiments turn the observation into a mechanism

Goal: present a focused research plan.

- Profile an exclusive/quiet CCX to strengthen hardware-scoped counters.
- Add prefetch, memory-level parallelism, and backend-stall metrics.
- Replay block tables captured from real continuous-batching schedules.
- Port matched layouts/traversals to an actual vLLM ROCm attention backend.
- Then sweep context length, block size, MHA/GQA/MQA, dtype, and request mix.

Visual: five-step validation pipeline.

## Slide 20 — Reproducibility: code, raw data, and exact revision

Goal: make every result inspectable.

- Repository: `github.com/memyselfandglitch/matmul-to-paged-attention`.
- Main revision: `b9a68deef18ac3f3aedb485eaa0be8bf7dddb481`.
- vLLM checkout: `52358e6e192aeae73dd3764046c98fc4b156ea83`.
- Code: `experiments/loop_order`, `experiments/batch_matmul`, and
  `experiments/paged-kv-traversal-study`.
- Primary results: jobs 7616, 7617, and uProf 7775; raw per-case counter CSVs
  are committed.
- Detailed map: `experiments/presentation/STUDY_EVIDENCE_GUIDE.md`.

Visual: compact repository tree and QR-code placeholder for the GitHub URL.

## Slide 21 — Discussion

Goal: close on the defensible result.

- Matching traversal to physical layout is necessary.
- Physical block locality determines whether HBND or BHND wins.
- On the AMD CPU, the refined crossover occurs between contiguous run lengths
  4 and 3 for this 256 MiB, float32, one-token decode workload.
- The mechanism likely involves lost memory-system efficiency, but requires
  prefetch/MLP/stall evidence and production-backend validation.

Visual: one-sentence takeaway and questions.
