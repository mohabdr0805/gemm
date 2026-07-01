# gemm-optim

![CI](https://github.com/MohaBdr0805/gemm/actions/workflows/ci.yml/badge.svg)

Optimized GEMM (`C = α·A·B + β·C`) in C++/OpenMP and CUDA, single precision,
row-major. The repo goes from a naive reference to tuned CPU and GPU kernels and
measures each step. The CUDA side also has a fused bias+activation epilogue, the
basic building block of an inference layer.

## Contents

- CPU: naive reference + cache-tiled OpenMP version.
- GPU v1: shared-memory tiled kernel (coalesced loads, bank-conflict padding,
  border handling).
- GPU v2: register tiling, each thread computing an 8×8 micro-block of C. About
  6.6× over v1, ~85% of the card's FP32 peak.
- Inference: fused GEMM + bias + activation (ReLU/GELU) in a single kernel.
- Attention: a FlashAttention-style kernel (online softmax, no N×N matrix) with
  an optional causal mask, plus a query-tiled v2 that reuses K/V across a block
  (up to ~10× over v1 at long sequence length).
- A correctness test against the naive oracle and a GFLOP/s benchmark for each.

## Results

Measured on a GTX 1650 (Turing, `sm_75`, ~2.5 TFLOP/s FP32) with a 12-thread CPU,
CUDA 13.3.

End-to-end GFLOP/s (the CUDA figures include H2D/D2H transfers):

| n    | naive | CPU tiled (OpenMP) | GPU v1 |
|------|-------|--------------------|--------|
| 1024 | 0.80  | 46.5               | 104.6  |
| 2048 | —     | 21.0               | 159.4  |

CPU tiled is about 58× the naive version at n=1024.

GPU kernel only (device timing, no transfers, n=2048):

| kernel                 | GFLOP/s | FP32 peak |
|------------------------|---------|-----------|
| v1 shared-memory tiled | 333     | ~13%      |
| v2 register tiled      | 2213    | ~85%      |

Register tiling is about 6.6× the v1 kernel. Every run also checks each kernel
against the naive oracle (max error ~5e-6, tolerance 1e-3).

## CPU — `gemm_tiled`

C is split into 64×64 tiles to fit in cache. `#pragma omp parallel for collapse(2)`
hands one whole tile to each thread, so there is no data race and no reduction. The
inner i-k-j order keeps B and C accessed row by row.

## GPU v1 — `gemm_kernel`

16×16 tile per block, with A and B staged in shared memory so global traffic drops
by roughly a factor of TILE. `threadIdx.x` indexes the column, so a warp reads
contiguous addresses (coalesced). Sizes that aren't multiples of 16 are guarded,
and the shared tiles are padded to `[TILE][TILE+1]` to avoid bank conflicts.

## GPU v2 — `gemm_reg_kernel`

128×128 block tile, K stepped in chunks of 8, 256 threads. Each thread keeps an
8×8 micro-block of C in registers, so every shared-memory load feeds 8 FMAs (an
8×8 outer product is 64 FMAs for 16 loads). That raises arithmetic intensity, which
is what actually moves the needle here (see below).

## Occupancy vs arithmetic intensity

`ptxas` on `sm_75`:

| kernel | registers/thread | shared/block | occupancy | throughput |
|--------|------------------|--------------|-----------|------------|
| v1     | 36 (0 spill)     | 2176 B       | 100%      | ~13% peak  |
| v2     | 128 (0 spill)    | 8192 B       | 50%       | ~85% peak  |

v1 is already at 100% theoretical occupancy but only ~13% of peak: it is limited by
arithmetic intensity, not occupancy (one output per thread is too little reuse per
load). v2 spends more registers, which drops occupancy to 50% (2 blocks/SM), but
each thread now does 64 outputs from registers. The result is 6.6× and ~85% of
peak. Past full occupancy, the lever is arithmetic intensity, not more occupancy.

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
large square GEMM where the GEMM dominates. On the GTX 1650 (GELU): 1.01× at 2048³,
1.20× at 2048²·64, up to 1.7× on tiny matrices.

## FlashAttention v1 — `flash_attention_kernel`

Attention is `O = softmax(scale · Q·Kᵀ) · V`. Done literally, that builds the
`N×N` score matrix `S`, softmaxes it, then multiplies by `V` — three passes and
`O(N²)` memory, which is what blows up on long sequences. FlashAttention never
writes `S`: it streams the keys and keeps a *running* softmax, so scores, softmax
and the `P·V` product all happen in one fused pass with `O(N·d)` extra memory.

The trick is the **online softmax**. Walking the keys in tiles, each query row
carries three running quantities — the max score `m`, the normalizer
`l = Σ exp(sⱼ − m)`, and the (unnormalized) output `acc = Σ exp(sⱼ − m)·vⱼ`. When
a new tile raises the max from `m` to `m'`, the old state is simply rescaled by
`exp(m − m')` before the tile is folded in; `O = acc / l` at the very end. That
rescale is exactly what keeps the result identical to a global safe softmax while
only ever seeing one tile at a time.

The layout mirrors the softmax kernel: **one block per query row**. The block
loads its query once, then for each key tile stages `K` and `V` in shared memory,
computes the tile's scores (one thread per key), reduces the tile max and sum
with the same tree reductions as `softmax_cuda`, updates `(m, l, acc)`, and moves
on. The shared `Ks`/`Vs` tiles are padded to `[BC][D+1]` to dodge bank conflicts,
just like the GEMM tiles. The **causal** mask costs nothing extra: row `i` simply
stops its key loop at `i+1`, so tiles past the diagonal are never loaded.

Because each block reads all of `K` and `V` once, v1's global traffic is
`O(M·N·d)` with no `N×N` allocation at all — the headline FlashAttention property,
but also its ceiling: at long sequence length it re-reads `K`/`V` `M` times.

## FlashAttention v2 — `flash_attention_v2_kernel`

Same online softmax, but a block now owns a **tile of `BR` query rows** instead of
one, so each streamed `K`/`V` tile is read once and reused by all `BR` rows —
`K`/`V` traffic drops from `M` reads to `M/BR`. It is the exact v1→v2 move the GEMM
kernels make (shared tile → register tile): here **each thread owns one query row**
and keeps its running `(m, l)` and accumulator `acc[d]` in **registers**, so the
rows are independent and there are no cross-thread reductions. The head dimension is
a template parameter (dispatched at runtime for `d ∈ {32, 64, 128}`, like the fused
GEMM's activation) so `q`/`acc` stay register-resident and the inner loops unroll;
any other `d` transparently falls back to v1.

Measured on an RTX 3080 (`sm_86`), head dim 64, `n×n`, device timing (GFLOP/s):

| n    | v1 full | v2 full | speedup | v2 causal | speedup |
|------|---------|---------|---------|-----------|---------|
| 1024 | 61      | 195     | 3.2×    | 107       | 1.2×    |
| 2048 | 61      | 354     | 5.8×    | 199       | 2.3×    |
| 4096 | 67      | 688     | 10.3×   | 468       | 5.7×    |

The gain grows with `n`: the longer the sequence, the more each cached `K`/`V` tile
is reused. Because this is a single head, the benchmark only exposes `M/BR` blocks,
so at tiny `n` — especially causal, which halves the work — v2 can be block-starved
and v1's simplicity wins; the crossover is around `n = 1024`. A real workload adds
`batch·heads` more blocks and stays firmly in v2's favour.

`./build/bench n` prints v1 vs v2 (full and causal) on your card.

## Build & run

CPU (Linux/GCC or Windows/MinGW):
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench 1024
```

With CUDA (NVIDIA GPU + nvcc):
```bash
cmake -S . -B build -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench 2048
```
Target architectures are `75;86` (Turing + Ampere) in `CMakeLists.txt`.

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
  compile-checks the CUDA build, and builds the CPU image. The runners have no GPU,
  so the CUDA kernels are compiled but not run.
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
- [x] Fused attention kernel (FlashAttention-style, online softmax + causal mask)
- [x] Attention v2: query tiling (K/V reused across the block, up to ~10× over v1)
- [ ] Multi-device StarPU variant
