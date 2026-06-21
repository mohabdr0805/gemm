# gemm-optim

Optimized **GEMM** (`C = α·A·B + β·C`) in C++/OpenMP and CUDA, with performance
analysis. Row-major matrices, single precision.

GEMM is the workhorse of dense linear algebra (BLAS level 3): the classic ground
for demonstrating mastery of memory optimization and parallelism, on both CPU and
GPU. This repo starts from a naive reference and applies, step by step, the
optimizations that matter — measuring each gain.

## What this project demonstrates

- **CPU**: cache blocking (tiling) + **OpenMP** parallelization, race-free.
- **GPU**: **shared-memory tiled CUDA kernel**, **coalesced** accesses, border
  handling (sizes not multiples of the tile), **bank-conflict padding**.
- **Method**: systematic correctness test against the naive oracle, GFLOP/s
  benchmark, and **occupancy analysis** to pinpoint the real bottleneck.

## Results

Measured on a **GTX 1650** (Turing, `sm_75`) + a 12-logical-core CPU, CUDA 13.3.
GFLOP/s (higher is better); the GPU time deliberately includes H2D/D2H transfers
(end-to-end performance).

| n     | naive | CPU tiled (OpenMP) | GPU CUDA |
|-------|-------|--------------------|----------|
| 1024  | 0.80  | **46.5**           | **104.6** |
| 2048  | —     | 21.0               | **159.4** |

- CPU tiled vs naive: **~58×** at n=1024 (cache blocking + 12 threads).
- GPU vs CPU tiled: ×2.2 at n=1024, **×7.6** at n=2048 — the GPU advantage grows
  with size (more parallelism, transfers amortized).
- Correctness validated on every run: max GPU vs naive error `4.8e-6` (tolerance `1e-3`).

## The optimizations

### CPU — `gemm_tiled`
- Split `C` into `64×64` tiles to fit in cache.
- `#pragma omp parallel for collapse(2)` over the tiles: each thread owns a full
  tile → **no data race**, no reduction.
- Inner `i-k-j` loop order: `B` and `C` traversed row-wise (contiguous accesses).

### GPU — `gemm_kernel` (CUDA)
- `16×16` tile: each block computes one tile of `C`, streaming `A` and `B` through
  **shared memory** → global-memory traffic reduced by ~TILE.
- **Coalescing**: `threadIdx.x` indexes the column, so consecutive threads in a
  warp read contiguous global-memory addresses.
- **Borders**: sizes not multiples of 16 handled with guards (out-of-bounds loads
  set to 0, conditional final write).
- **Bank conflicts**: shared tiles padded to `[TILE][TILE+1]` to map columns onto
  distinct banks.

## Performance analysis (occupancy)

`ptxas` stats for the kernel on `sm_75`: **36 registers/thread, 0 spills, 2176 B
shared memory/block**. With 256-thread blocks, the binding limit is warps per SM
(32) → **100% theoretical occupancy** (4 blocks × 256 = 1024 threads/SM); neither
registers nor shared memory limit it.

Takeaway: since occupancy is already maxed, the kernel is **not occupancy-bound**
but **arithmetic-intensity-bound** — each thread computes a single element of `C`,
i.e. too little reuse per memory access (~5% of FP32 peak). The next lever is
therefore not occupancy but **register tiling** (each thread computes a micro-block
of `C`, e.g. `4×4`), which raises reuse and arithmetic intensity. *(Next step of
the project.)*

## Build & run

**CPU** (Linux/GCC, or Windows/MinGW):
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure   # correctness test
./build/bench 1024                            # benchmark for n=1024
```

**With CUDA** (NVIDIA GPU + nvcc):
```bash
cmake -S . -B build -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure   # adds the GPU vs naive check
./build/bench 1024                            # adds the CUDA line
```
Target GPU architectures: `75;86` (Turing + Ampere), tunable in `CMakeLists.txt`.

## Layout

```
include/gemm/   public headers (gemm_cpu.hpp, gemm_cuda.cuh)
src/            implementations: gemm_cpu.cpp (naive + tiled), gemm_cuda.cu (kernel)
benchmarks/     GFLOP/s measurements
tests/          correctness test (optimized vs naive, borders included)
```

## Roadmap

- [x] CPU: naive + OpenMP tiled
- [x] GPU: shared-memory tiled CUDA kernel
- [ ] CUDA kernel v2: register tiling (micro-block per thread)
- [ ] Dockerfile (reproducible CPU + CUDA build)
- [ ] GitHub Actions CI (build + CPU tests)
- [ ] Multi-device StarPU variant (CPU+GPU tasks)
