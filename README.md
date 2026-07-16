# gemm

![CI](https://github.com/mohabdr0805/gemm/actions/workflows/ci.yml/badge.svg)

> **TL;DR**: hand-written CUDA SGEMM reaching 54–61% of cuBLAS at n ≥ 2048
> through shared-memory and register tiling, plus a FlashAttention-style
> attention kernel that gains up to ~11× from query tiling. Every kernel is
> validated against a CPU oracle. Device figures are 50-iteration averages on
> an RTX 3080, with cuBLAS timed in the same run.

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
  7–11× over v1 (grows with size), ~44–61% of cuBLAS SGEMM on the same card
  (54–61% at n ≥ 2048; the n=1024 grid underfills the card, see Results).
- Inference: fused GEMM + bias + activation (ReLU/GELU) in a single kernel.
- Softmax: numerically stable row-wise softmax (CPU oracle + CUDA kernel with
  shared-memory tree reductions); the attention kernels build on it.
- Attention: a FlashAttention-style kernel (online softmax, no N×N matrix) with
  an optional causal mask, a query-tiled v2 that reuses K/V across a block (up to
  ~11× over v1), and an FA-2-style warp-partitioned kernel that splits the head
  dimension across a warp (kills v2's d=128 register spill; 1.6–6× over v2 there).

## Results

Measured on an RTX 3080 10 GB (Ampere, `sm_86`) + i7-12700F (12C/20T), CUDA 13.0,
Windows/MSVC. Methodology: every device figure is a 50-iteration average after a
warm-up; the kernels and cuBLAS are timed back-to-back in the same run, so the
ratios (% of cuBLAS, v1→v2) are reproducible even though absolute GFLOP/s move
with the card's boost clocks (not locked here).

*Scope: FP32 on CUDA cores throughout (no tensor cores). The cuBLAS baseline
runs in its default math mode (plain `cublasSgemm`, no `cublasSetMathMode`), so
TF32 is disabled and the comparison is like-for-like. Cross-check: cuBLAS
matches the FP32 CPU oracle to ~7e-6, where TF32's 10-bit mantissa would give
errors around 1e-3.*

End-to-end GFLOP/s (whole wrapper: cudaMalloc + H2D/D2H + kernel):

| n    | naive | CPU tiled (OpenMP+SIMD) | GPU v1 wrapper |
|------|-------|-------------------------|----------------|
| 1024 | 0.90  | 83                      | 593            |
| 2048 | —     | 161                     | 414            |
| 4096 | —     | 209                     | 354            |

*Footnote: the end-to-end row is a single timed call, which catches the card at
full boost. The device-only table below averages 50 back-to-back iterations at
sustained clocks. This is why the v1 wrapper can nominally exceed the bare v1
kernel at n = 1024 (593 vs 537). Absolute GFLOP/s depend on clock state; the
ratios within each table are reproducible.*

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

| n    | v1 shared-tiled | v2 register | cuBLAS SGEMM | v1 vs cuBLAS | v2 vs cuBLAS |
|------|-----------------|-------------|--------------|--------------|--------------|
| 1024 | 537             | 3 798       | 8 719        | ~6%          | ~44%         |
| 2048 | 457             | 4 809       | 7 858        | ~6%          | ~61%         |
| 4096 | 460             | 5 210       | 9 602        | ~5%          | ~54%         |

Register tiling is 7–11× the v1 kernel, and the gap grows with n. The lower
~44% at n=1024 is grid underfill, not the kernel: 128×128 tiles produce only 64
blocks for the 3080's 68 SMs × 2 resident blocks, so the card runs half-empty
(wave quantization). At the larger sizes v2 sits at ~54–61% of cuBLAS; the rest
of the gap is what the roadmap items target (vectorized `float4` loads, double
buffering). `ctest` checks every kernel, cuBLAS included, against the CPU
oracle (max error ~1e-5, tolerance 1e-3). The benchmark does not check
correctness.

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
| v1     | 37 (0 spill)     | 2176 B       | 100% (6 blocks/SM) | 460              |
| v2     | 128 (0 spill)    | 8192 B       | 33% (2 blocks/SM)  | 5 210            |

v1 sits at 100% theoretical occupancy yet runs 11× slower than v2 at a third of
the occupancy. One output per thread gives too little reuse per load: v1 is
limited by arithmetic intensity, not occupancy. Past full occupancy, arithmetic
intensity is the only lever left.

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
bandwidth: ~530 GB/s at 1024² (the 4 MB matrix mostly lives in the 3080's 5 MB
L2), dropping to ~120–220 GB/s once the matrix is DRAM-resident. The card's
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
back-to-back in the same power state, whereas absolute GFLOP/s swings ~2–3×
with boost clocks. On an RTX 3080 (`sm_86`), `n×n`, full attention, device
timing:

| n    | speedup d=64 | speedup d=128 |
|------|--------------|---------------|
| 1024 | ~3×          | ~1.3×         |
| 2048 | ~5.5–6.3×    | ~2.7–2.8×     |
| 4096 | ~11×         | ~5.4×         |

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
| 1024 | 283      | 1689       | 5.97×   | 504     | 1076      | 2.13×   |
| 2048 | 526      | 1613       | 3.07×   | 1007    | 1305      | 1.30×   |
| 4096 | 1038     | 1705       | 1.64×   | 1996    | 1442      | 0.72×   |

At d=128 FA-2 wins at every size: no spill, and a flat ~1600–1700 GFLOP/s
regardless of n. At d=64, where v2 does not spill, it is a trade-off. FA-2 gives
a row to a warp, so a block serves only 8 rows against v2's 64, and each shared
K/V tile is reused 8× per block instead of 64×. At large n the K/V traffic
dominates, so v2's higher reuse wins (0.72× at n=4096); at small n v2 is
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
benchmarks/     GFLOP/s, v1 vs v2, fusion, softmax, attention benchmarks
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
- [x] Attention v2: query tiling (K/V reused across the block, up to ~11× over v1)
- [x] Baseline: cuBLAS SGEMM, same card, same run (v2 ≈ 44–61%; 54–61% at n ≥ 2048)
- [ ] Baseline: PyTorch SDPA for the attention kernels
- [ ] GEMM v3: vectorized `float4` loads + double buffering (close the cuBLAS gap)
- [ ] Fused epilogue on the register-tiled v2 GEMM
- [x] Attention FA-2: warp-partitioned head dim (kills the d=128 spill; 1.6–6× over v2 at d=128)
- [ ] Multi-device StarPU variant
