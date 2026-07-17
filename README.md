# gemm

![CI](https://github.com/mohabdr0805/gemm/actions/workflows/ci.yml/badge.svg)

> **TL;DR**: hand-written CUDA SGEMM reaching 74–83% of cuBLAS at n ≥ 2048, and
> ~105% at n=1024, through shared-memory tiling, register tiling, and vectorized
> double-buffered loads (v3), plus a FlashAttention-style attention kernel that
> gains up to ~9.4× from query tiling. Every kernel is validated against a CPU
> oracle. Device figures are measured on an RTX 3080 over ~200 ms of work per
> kernel, with cuBLAS timed in the same run.

Optimized GEMM (`C = α·A·B + β·C`) in C++/OpenMP and CUDA, single precision,
row-major. The repo goes from a naive reference to tuned CPU and GPU kernels and
measures each step. The CUDA side also has a fused bias+activation epilogue, the
basic building block of an inference layer.

## Contents

- CPU: naive reference + cache-tiled, OpenMP-parallel, SIMD-vectorized version
  (~90× naive; 209 GFLOP/s at n=4096 on a 12700F).
- GPU v1: shared-memory tiled kernel (coalesced loads, bank-conflict padding,
  border handling).
- GPU v2: register tiling, each thread computing an 8×8 micro-block of C.
  6.6–8.7× over v1 (grows with size), ~61–66% of cuBLAS SGEMM on the same card.
- GPU v3: v2 + vectorized `float4` loads and double buffering, each step chosen
  from a Nsight Compute profile. 74–83% of cuBLAS at n ≥ 2048, ~105% at n=1024.
- Inference: fused GEMM + bias + activation (ReLU/GELU) in a single kernel.
- Softmax: numerically stable row-wise softmax (CPU oracle + CUDA kernel with
  shared-memory tree reductions); the attention kernels build on it.
- Attention: a FlashAttention-style kernel (online softmax, no N×N matrix) with
  an optional causal mask, a query-tiled v2 that reuses K/V across a block (up to
  ~9.4× over v1), and an FA-2-style warp-partitioned kernel that splits the head
  dimension across a warp (kills v2's d=128 register spill; 1.6–5.7× over v2 there).

## Results

Measured on an RTX 3080 10 GB (Ampere, `sm_86`) + i7-12700F (12C/20T), CUDA 13.0,
Windows/MSVC. Methodology: each GEMM kernel is timed over ~200 ms of work after a
warm-up, with the iteration count sized per kernel from a probe run and printed
alongside the result; the kernels and cuBLAS are timed back-to-back in the same
run, so the ratios (% of cuBLAS, v1→v2) hold even though absolute GFLOP/s move
with the card's boost clocks (not locked here).

Sizing by time replaced a flat 50 iterations, which was not good enough at either
end. At n=1024 v3 runs in 0.15 ms, so 50 iterations measured 7 ms and launch
overhead left the v3/cuBLAS ratio swinging 87–113% between runs; it now sits at
105% ± 1. At n=4096 v1 runs in 100 ms, so the same 50 iterations soaked the card
for 5 s and moved the clocks under everything timed after it. Equal time per
kernel fixes both.

*Scope: FP32 on CUDA cores throughout (no tensor cores). The cuBLAS baseline
runs in its default math mode (plain `cublasSgemm`, no `cublasSetMathMode`), so
TF32 is disabled and the comparison is like-for-like. Cross-check: cuBLAS
matches the FP32 CPU oracle to ~7e-6, where TF32's 10-bit mantissa would give
errors around 1e-3.*

End-to-end GFLOP/s (whole wrapper: cudaMalloc + H2D/D2H + kernel):

| n    | naive | CPU tiled (OpenMP+SIMD) | GPU v1 wrapper |
|------|-------|-------------------------|----------------|
| 1024 | 0.90  | 83                      | 580            |
| 2048 | —     | 161                     | 911            |
| 4096 | —     | 209                     | 1 292          |

*Footnote: the end-to-end row is one timed call including `cudaMalloc` and both
transfers, so it measures the wrapper, not the kernel. It climbs with n while the
bare v1 kernel below is flat (~1 370 GFLOP/s), because the copies scale as n² and
the kernel as n³: at n=1024 the transfers cost more than the kernel itself (580
vs 1 331), by n=4096 they are nearly amortised away (1 292 vs 1 383). Absolute
GFLOP/s depend on clock state; the ratios within each table hold.*

CPU tiled is ~90× the naive version at n=1024 and reaches 209 GFLOP/s at
n=4096, about 16% of the 12700F's AVX2 peak, with no cache cliff. The
three-level blocking explains it: at any instant a thread touches only three
64×64 sub-blocks (3 × 16 KB, L1-resident at any n), so growing n only adds
bandwidth pressure from panel re-streaming: 172 GFLOP/s at n=6144, 154 at
n=8192 (about −10% once twenty threads' panel re-reads hit the shared 25 MB
L3). We initially predicted a cliff at n=8192 ("the 64·n·4 = 2 MB panel exceeds
the 1.25 MB per-core L2") and the measurement refuted it: the panel governs
inter-tile reuse, not the instantaneous working set. Single-threaded,
vectorization alone is worth 7.8× (3.8 → 29.4 GFLOP/s); see the `gemm_tiled`
section for the MSVC details.

GPU kernels vs cuBLAS SGEMM (device timing, no transfers, GFLOP/s):

| n    | v1 shared-tiled | v2 register | v3 float4+2×buf | cuBLAS SGEMM | v1 vs cuBLAS | v2 vs cuBLAS | v3 vs cuBLAS |
|------|-----------------|-------------|-----------------|--------------|--------------|--------------|--------------|
| 1024 | 1 331           | 8 799       | 14 443          | 13 680       | ~10%         | ~64%         | **~105%**    |
| 2048 | 1 378           | 11 865      | 14 657          | 19 158       | ~7%          | ~62%         | ~77%         |
| 3072 | 1 370           | 11 799      | 14 827          | 17 958       | ~8%          | ~66%         | ~83%         |
| 4096 | 1 383           | 12 033      | 14 678          | 19 693       | ~7%          | ~61%         | ~75%         |

Register tiling is 6.6–8.7× the v1 kernel, and the gap grows with n. Grid
underfill at n=1024 is visible in v2's absolute throughput, not in its ratio:
128×128 tiles produce 64 blocks, and v2 fits 2 per SM, so it has 136 slots for
the 3080's 68 SMs and fills fewer than half of them (wave quantization): 8 799
GFLOP/s against ~12 000 at larger n. The ratio hides it because cuBLAS underfills
at n=1024 too (13 680, its own worst size).

v3 is the interesting row. Its throughput is flat, ~14 400–14 800 GFLOP/s at every
size, so the "% of cuBLAS" column is really a picture of cuBLAS's variation, and
v3 *beats* cuBLAS at n=1024 for the same reason it costs at n=4096: it needs 130
registers, so only one block fits per SM (see below), which means 68 slots, and
n=1024 supplies 64 blocks. The property that halves its occupancy is what leaves
it nearly full when the grid is small. `ctest` checks every kernel, cuBLAS
included, against the CPU oracle (max error ~1e-5, tolerance 1e-3). The benchmark
does not check correctness.

## CPU — `gemm_tiled`

C is split into 64×64 tiles to fit in cache. `#pragma omp parallel for collapse(2)`
hands one whole tile to each thread, so there is no data race and no reduction. The
inner i-k-j order keeps B and C accessed row by row.

Row access is only half the point of that loop order; the other half is SIMD.
The inner `Crow[j] += a * Brow[j]` over contiguous floats maps onto 8-wide AVX2
FMAs, worth 7.8× single-threaded (3.8 → 29.4 GFLOP/s). MSVC needed three
explicit steps to emit them: `/arch:AVX2` (x64 defaults to 4-wide SSE2; the
flag is guarded by `GEMM_NATIVE` since AVX2 binaries crash on pre-2013 CPUs),
`#pragma omp simd` + `__restrict` (MSVC refuses to auto-vectorize
`C[i*N+j] += a*B[k*N+j]` because it assumes C may alias B, `/Qvec-report`
reason 1200; GCC emits a runtime overlap check instead), and hoisted row
pointers (the vectorizer also rejects the mixed `(size_t)i*N + j` addressing).
`/Qvec-report:2` confirms: `C5001: omp simd loop vectorized`.

## GPU v1 — `gemm_kernel`

16×16 tile per block, with A and B staged in shared memory so global traffic drops
by roughly a factor of TILE. `threadIdx.x` indexes the column, so a warp reads
contiguous addresses (coalesced). Sizes that aren't multiples of 16 are guarded,
and the shared tiles are padded to `[TILE][TILE+1]` to avoid bank conflicts.

## GPU v2 — `gemm_reg_kernel`

128×128 block tile, K stepped in chunks of 8, 256 threads. Each thread keeps an
8×8 micro-block of C in registers, so every shared-memory load feeds 8 FMAs (an
8×8 outer product is 64 FMAs for 16 loads). That raises arithmetic intensity
(see below).

Why these numbers: a 128×128 block tile with a K-step of 8 needs 2 × (128×8)
floats of shared memory = 8 KB per block. 256 threads each owning an 8×8
micro-block cover the tile exactly (256 × 64 = 128×128). At 128 registers per
thread, one block uses 32 K of the SM's 64 K register file, so exactly two
blocks fit per SM. The 33% occupancy below is deliberate: the extra registers
raise arithmetic intensity, which is worth more than occupancy at this point.

## Occupancy vs arithmetic intensity

`ptxas` on `sm_86` (256-thread blocks; an `sm_86` SM runs up to 1536 threads):

| kernel | registers/thread | shared/block | occupancy          | GFLOP/s (n=4096) |
|--------|------------------|--------------|--------------------|------------------|
| v1     | 37 (0 spill)     | 2176 B       | 100% (6 blocks/SM) | 1 383            |
| v2     | 128 (0 spill)    | 8192 B       | 33% (2 blocks/SM)  | 12 033           |
| v3     | 130 (0 spill)    | 16384 B      | 17% (1 block/SM)   | 14 678           |

Occupancy falls 100% → 33% → 17% down the table while throughput rises 1 383 →
12 033 → 14 678. v1 sits at 100% theoretical occupancy yet runs 8.7× slower than
v2 at a third of the occupancy: one output per thread gives too little reuse per
load, so v1 is limited by arithmetic intensity, not occupancy. Past full
occupancy, arithmetic intensity is the only lever left; past that, latency
hiding, which is what v3 buys with the rest of its occupancy.

The v3 row also shows how sharp the cliff is. Two blocks per SM need
`regs × 256 × 2 ≤ 65536`, so the limit is 128 registers/thread: v2 sits exactly
on it, and v3's prefetch pushes it to 130. Two registers over, and the second
block is gone. The 16 KB of shared is not what binds (it would still allow six).

## GPU v3 — `gemm_reg_v3db_kernel` (float4 + double buffering)

v2 sits at ~61% of cuBLAS. Nsight Compute on the v2 kernel at n=4096 (full grid)
showed where the rest goes: DRAM at only ~11% (so not bandwidth-bound), and the
top warp stalls were global-load latency (`long_scoreboard`) and shared-load
bank conflicts, with ~268M shared-load bank conflicts reported. Two changes
followed, each checked by re-profiling.

First, vectorized `float4` global loads plus a transposed As tile (so the
compute loop reads contiguous shared rows). This was worth several points of
cuBLAS, but the re-profile was a useful correction: the bank-conflict count did
not move (the dominant conflict is on the Bs read, which this did not touch), so
the gain came from issuing a quarter as many global-load instructions, not from
the conflicts. It also left global-load latency as the new top stall
(`long_scoreboard` rose from ~12% to ~16%).

Second, double buffering. Two shared buffers: each step issues the next tile's
global loads into registers up front (in flight while the current tile is
computed), then lands them in the other buffer and swaps, so the latency
overlaps the FMAs instead of stalling in front of them. The prediction was that
`long_scoreboard` would collapse and large-n throughput would recover, and both
held: the stall fell from ~16% to ~2%, and v3 reaches 74–83% of cuBLAS at
n ≥ 2048 (75% at n=4096, 77% at n=2048, 83% at n=3072). At n=1024 it passes
cuBLAS outright, ~105%, for the occupancy reason given in Results: one block per
SM needs only 64 of the card's 68 slots to fill it.

| stall (n=4096, ncu)        | v2    | v3 float4 | v3 + double-buf |
|----------------------------|-------|-----------|-----------------|
| global latency (long_sb)   | 11.8% | 15.5%     | 2.3%            |
| shared latency (short_sb)  | 2.9%  | 18.0%     | 24.3%           |
| barrier                    | 9.8%  | 9.3%      | 4.9%            |
| mio throttle (LSU/bank)    | 7.4%  | 7.1%      | 5.5%            |

Double buffering also trades occupancy for latency hiding, and `ptxas` prices the
trade exactly: 130 registers against v2's 128. The cliff for two blocks per SM is
128, so v3 drops to one block and occupancy halves (32% → 17%). The doubled
shared tile is not what binds; 16 KB/block would still allow six. It wins anyway,
because hiding the global latency is worth more than the lost occupancy (the
v1→v2 lesson again). And it moves the bottleneck: with `long_scoreboard` gone the
top stall is `short_scoreboard`, the latency of the shared→register reads
themselves, which the halved occupancy no longer covers. Winning those two
registers back is the cheapest v4 lever, and it feeds the other: the occupancy it
returns is exactly what would hide the shared latency.

The fast path assumes aligned sizes (M,N % 128 == 0, K % 8 == 0, needed for the
unguarded float4 loads); any other shape falls back to v2. `benchmark_gemm_versions`
prints v1, v2, v3 (float4), v3 (double-buffered) and cuBLAS.

## Fused inference epilogue

A linear layer computes `Y = act(X·W + bias)`. Naively that is two kernels: the
GEMM, then an element-wise pass for bias and activation, which writes the output
and reads it all back. The fused kernel does the bias and activation in the GEMM
epilogue, before the first write: one launch, one global-memory pass. cuBLASLt,
CUTLASS and TensorRT do the same.

The activation is a template parameter (resolved at compile time, no branch), and
its math (ReLU, tanh-GELU) lives in `activation.hpp`, shared between the CPU oracle
and the GPU kernel so the two cannot disagree.

Fusion saves one M×N pass plus a launch, so the relative gain is about `TILE/K`
plus launch overhead: useful for shallow layers and small problems, negligible on a
large square GEMM where the GEMM dominates. On the RTX 3080 (GELU): within noise
(0.97–1.16×) on big square GEMMs; mixed at thin K, from a small net loss
(0.91× at 1024²·64) to 1.35× at 2048²·64 and 1.17× at 4096²·64; and ~1.5–2× on
tiny matrices where the saved launch matters most.

One caveat: the epilogue is currently fused into the v1 tiling, while the
register-tiled v2 is ~10× faster as a plain GEMM. Templating the epilogue into
`gemm_reg_kernel` is the next step (see Roadmap).

## Softmax — `softmax_rows_kernel`

Row-wise safe softmax (`out[i,:] = softmax(in[i,:])`, subtracting the row max
before `exp` so nothing overflows): one block per row, two cooperative tree
reductions in shared memory (row max, then sum of exp), threads striding over
rows longer than the block. Memory-bound, so the benchmark reports effective
bandwidth: ~500 GB/s at 1024² (the 4 MB matrix mostly lives in the 3080's 5 MB
L2), dropping to ~240 GB/s at 4096² once the matrix is DRAM-resident. The card's
DRAM peak is 760 GB/s, and the kernel's real traffic is ~2.5× the algorithmic
figure. The kernel exists mostly as a stepping stone to the attention kernels
below, which reuse its reduction idiom and replace its global softmax with an
online one.

## FlashAttention v1 — `flash_attention_kernel`

Attention is `O = softmax(scale · Q·Kᵀ) · V`. Done literally, that builds the
`N×N` score matrix `S`, softmaxes it, then multiplies by `V`: three passes and
`O(N²)` memory, which blows up on long sequences. FlashAttention never writes
`S`: it streams the keys and keeps a running softmax, so scores, softmax and
the `P·V` product all happen in one fused pass with `O(N·d)` extra memory.

The trick is the online softmax. Walking the keys in tiles, each query row
carries three running quantities: the max score `m`, the normalizer
`l = Σ exp(sⱼ − m)`, and the unnormalized output `acc = Σ exp(sⱼ − m)·vⱼ`. When
a new tile raises the max from `m` to `m'`, the old state is rescaled by
`exp(m − m')` before the tile is folded in, and `O = acc / l` once at the end.
The rescale is what keeps the result identical to a global safe softmax while
only ever seeing one tile at a time.

The layout mirrors the softmax kernel: one block per query row. The block loads
its query once, then for each key tile stages `K` and `V` in shared memory,
computes the tile's scores (one thread per key), reduces the tile max and sum
with the same tree reductions as `softmax_rows_kernel`, updates `(m, l, acc)`,
and moves on. The shared `Ks`/`Vs` tiles are padded to `[BC][D+1]` against bank
conflicts, like the GEMM tiles. The causal mask costs nothing extra: row `i`
stops its key loop at `i+1`, so tiles past the diagonal are never loaded.

Because each block reads all of `K` and `V` once, v1's global traffic is
`O(M·N·d)` with no `N×N` allocation. That is the central FlashAttention
property, and also v1's ceiling: at long sequence length it re-reads `K`/`V`
`M` times.

## Attention v2 — query tiling (`flash_attention_v2_kernel`)

*Naming note: "v2" is this repo's own v1→v2 progression (like the GEMM
kernels), not the FlashAttention-2 algorithm. FA-2's warp-partitioned layout,
where lanes cooperate on the head dimension, is a further step (see Roadmap).*

Same online softmax, but a block now owns a tile of `BR` query rows instead of
one, so each streamed `K`/`V` tile is read once and reused by all `BR` rows:
`K`/`V` traffic drops from `M` reads to `M/BR`. It is the same v1→v2 move as
the GEMM kernels (shared tile → register tile): each thread owns one query row
and keeps its running `(m, l)` and accumulator `acc[d]` in registers, so rows
are independent and there are no cross-thread reductions. The head dimension is
a template parameter (dispatched at runtime for `d ∈ {32, 64, 128}`, like the
fused GEMM's activation) so the inner loops fully unroll; any other `d` falls
back to v1.

`q[d]`/`acc[d]` stay register-resident up to d = 64: `ptxas -v` reports 128
registers at d=32 and 218 at d=64, both with 0 spill. At d = 128 the arrays
exceed the 255-register/thread limit and spill (~1 KB/thread). v2 still wins
there, because the `K`/`V` reuse more than pays for the spill traffic, but by a
smaller margin.

The reproducible number here is the v2/v1 speedup: the two kernels are timed
back-to-back over equal ~200 ms windows, whereas absolute GFLOP/s moves with the
card's clock state. Sizing by time matters for the ratio, not just the absolute:
under the flat 50 iterations this table used to run, v1 (20 ms/iter at n=4096)
soaked the card for a second before v2 was timed, and the speedup came out ~11×
against the 9.4× measured here. "Timed in the same power state" was the claim;
the fixed count was quietly breaking it. On an RTX 3080 (`sm_86`), `n×n`, full
attention, device timing:

| n    | speedup d=64 | speedup d=128 |
|------|--------------|---------------|
| 1024 | ~2.5×        | ~1.1×         |
| 2048 | ~4.8×        | ~2.0×         |
| 4096 | ~9.4×        | ~4.0×         |

The gain grows with n: the longer the sequence, the more each cached `K`/`V`
tile is reused. d=128 gains less than d=64 because of the register spill. Since
this is a single head, the benchmark only exposes `M/BR` blocks, so at small n
(especially causal, which halves the work) v2 can be block-starved and v1 wins;
the crossover is around n = 1024. A real workload adds batch·heads more blocks
and v2 wins across the board.

`./build/bench n` prints v1 vs v2 for both table dims (d = 64 and d = 128),
full and causal.

## Attention FA-2 — warp-partitioned head dim (`flash_attention_fa2_kernel`)

v2 gives a query row to one thread, which holds `q[d]` and `acc[d]` in registers.
At d=128 that is 256+ floats per thread, over the 255-register limit, so it
spills (ptxas: 255 registers, ~4 KB of spill traffic per thread). FA-2 gives a
row to a whole warp instead and splits d across its 32 lanes: lane `l` owns dims
l, l+32, ..., so at d=128 each lane holds only d/32 = 4 elements of q and acc.
ptxas then reports 64 registers and 0 spill.

Splitting d breaks one thing: the score `s = <q, k>` is a sum over d, now spread
across the lanes. Each lane computes its partial dot, and a warp butterfly
(`__shfl_xor`, registers only) sums the 32 partials into the full score, present
in every lane. That is the only cross-lane communication. The running softmax
scalars (m, l) are per-row, so every lane recomputes them identically; the P·V
product is lane-independent, each lane accumulating its own output dims. The one
reduction stays in registers, so it never touches shared memory or a block sync.

Measured on an RTX 3080, `n×n`, full attention, GFLOP/s:

| n    | v2 d=128 | FA-2 d=128 | FA-2/v2 | v2 d=64 | FA-2 d=64 | FA-2/v2 |
|------|----------|------------|---------|---------|-----------|---------|
| 1024 | 275      | 1573       | 5.71×   | 505     | 1069      | 2.12×   |
| 2048 | 488      | 1425       | 2.92×   | 1047    | 1275      | 1.22×   |
| 4096 | 919      | 1505       | 1.64×   | 1886    | 1328      | 0.70×   |

At d=128 FA-2 wins at every size: no spill, and a flat ~1400–1600 GFLOP/s
regardless of n. At d=64, where v2 does not spill, it is a trade-off. FA-2 gives
a row to a warp, so a block serves only 8 rows against v2's 64, and each shared
K/V tile is reused 8× per block instead of 64×. At large n the K/V traffic
dominates, so v2's higher reuse wins (0.70× at n=4096); at small n v2 is
block-starved (16 blocks for 68 SMs, wave quantization) while FA-2's smaller row
tile fills the card, so FA-2 wins. The right kernel depends on the regime: FA-2
for d=128, v2 for d≤64 at long sequence length.

## Build & run

CPU (Linux/GCC, or Windows/MSVC. On MSVC the build forces `/openmp:experimental`:
the LLVM OpenMP runtime, so the tiled GEMM's `collapse(2)` is actually
parallelized (classic `/openmp` is stuck at OpenMP 2.0 and silently ignores it),
plus the `omp simd` directive the inner loop needs to vectorize):
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench 1024
```
on Windows, run from a "x64 Native Tools Command Prompt", or install CMake standalone.

With CUDA (NVIDIA GPU + nvcc):
```bash
cmake -S . -B build -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench 2048
```
Default target architectures are `75;86` (Turing + Ampere); pass
`-DCMAKE_CUDA_ARCHITECTURES=89` (Ada, etc.) to target your own card.

## Docker

```bash
# CPU image: builds, runs the test, then the benchmark
docker build -t gemm-cpu .
docker run --rm gemm-cpu 2048

# CUDA image: compiles the GPU kernels (running them needs an NVIDIA GPU)
docker build -f Dockerfile.cuda -t gemm-cuda .
docker run --rm --gpus all gemm-cuda 2048
```
Images build with `-DGEMM_NATIVE=OFF` (no `-march=native`) so they run on any
x86-64 host. The CPU image is published on each tagged release to
`ghcr.io/mohabdr0805/gemm`.

## CI / CD

- CI (`.github/workflows/ci.yml`): on every push/PR, builds and tests the CPU code,
  compile-checks the CUDA build, and builds both Docker images (CPU and CUDA). The
  runners have no GPU, so the CUDA kernels are compiled but not run.
- CD (`.github/workflows/release.yml`): on a `v*` tag, builds the CPU image and
  pushes it to GHCR. There is no service to deploy here; the artifact is the image.

## Layout

```
include/gemm/   headers (gemm_cpu.hpp, gemm_cuda.cuh, activation.hpp,
                softmax*.hpp/cuh, attention*.hpp/cuh)
src/            gemm_cpu.cpp (naive + tiled), gemm_cuda.cu (kernels),
                softmax_{cpu,cuda}, attention_{cpu,cuda}
benchmarks/     GFLOP/s, v1/v2/v3 vs cuBLAS, fusion, softmax, attention benchmarks
tests/          correctness vs the CPU oracle (gemm, softmax, attention)
```

## Roadmap

- [x] CPU: naive + OpenMP tiled
- [x] GPU v1: shared-memory tiled
- [x] GPU v2: register tiling (8×8 per thread)
- [x] Fused inference epilogue (bias + activation)
- [x] Docker + CI/CD (GitHub Actions, image on GHCR)
- [x] Row-wise softmax kernel (safe softmax: CPU oracle + CUDA)
- [x] Fused attention kernel (FlashAttention-style, online softmax + causal mask)
- [x] Attention v2: query tiling (K/V reused across the block, up to ~9.4× over v1)
- [x] Baseline: cuBLAS SGEMM, same card, same run (v2 ≈ 61–66%)
- [ ] Baseline: PyTorch SDPA for the attention kernels
- [x] GEMM v3: vectorized `float4` loads + double buffering (~61% → 74–83% of cuBLAS,
      ~105% at n=1024)
- [ ] GEMM v4: win back 2 registers with `__launch_bounds__(256,2)` (ptxas says 130;
      the 2-blocks/SM cliff is 128, so occupancy 17% → 33%), then v3's new top stall,
      shared latency (`short_scoreboard` 24%)
- [ ] Fused epilogue on the register-tiled v2 GEMM
- [x] Attention FA-2: warp-partitioned head dim (kills the d=128 spill; 1.6–5.7× over v2 at d=128)
- [ ] Multi-device StarPU variant
