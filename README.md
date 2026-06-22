# gemm-optim

Optimized **GEMM** (`C = α·A·B + β·C`) in C++/OpenMP and CUDA, with performance
analysis — from optimized CPU compute to **GPU kernels for neural-network
inference**. Row-major matrices, single precision.

GEMM is the workhorse of dense linear algebra (BLAS level 3): the classic ground
for demonstrating mastery of memory optimization and parallelism, on both CPU and
GPU. This repo starts from a naive reference and applies, step by step, the
optimizations that matter — measuring each gain.

## What this project demonstrates

- **CPU**: cache blocking (tiling) + **OpenMP** parallelization, race-free.
- **GPU v1**: **shared-memory tiled CUDA kernel**, **coalesced** accesses, border
  handling, **bank-conflict padding**.
- **GPU v2**: **register tiling** (each thread computes an 8×8 micro-block of `C`)
  → **6.6× over v1, ~85% of the GPU's FP32 peak**.
- **Inference**: **fused GEMM + bias + activation (ReLU/GELU)** epilogue in a
  single launch — the core pattern of an inference engine.
- **Method**: correctness test against the naive oracle, GFLOP/s benchmark, and
  **occupancy analysis** that drives the v1 → v2 optimization.

## Results

Measured on a **GTX 1650** (Turing, `sm_75`, ~2.5 TFLOP/s FP32) + a 12-logical-core
CPU, CUDA 13.3.

**CPU** — GFLOP/s, GPU end-to-end (incl. H2D/D2H transfers):

| n     | naive | CPU tiled (OpenMP) | GPU v1 (end-to-end) |
|-------|-------|--------------------|---------------------|
| 1024  | 0.80  | 46.5               | 104.6               |
| 2048  | —     | 21.0               | 159.4               |

CPU tiled vs naive: **~58×** at n=1024 (cache blocking + 12 threads).

**GPU GEMM kernel** — device timing (no transfers), n=2048:

| kernel                       | GFLOP/s | % of FP32 peak |
|------------------------------|---------|----------------|
| v1 — shared-memory tiled     | 333     | ~13%           |
| **v2 — register tiled**      | **2213**| **~85%**       |

→ register tiling buys **6.6×**. Correctness validated on every run (all kernels
vs the naive oracle, max error `~5e-6`, tolerance `1e-3`).

## The optimizations

### CPU — `gemm_tiled`
- Split `C` into `64×64` tiles to fit in cache.
- `#pragma omp parallel for collapse(2)` over the tiles: each thread owns a full
  tile → **no data race**, no reduction.
- Inner `i-k-j` loop order: `B` and `C` traversed row-wise (contiguous accesses).

### GPU v1 — `gemm_kernel` (shared-memory tiling)
- `16×16` tile: each block computes one tile of `C`, streaming `A` and `B` through
  **shared memory** → global-memory traffic reduced by ~TILE.
- **Coalescing**: `threadIdx.x` indexes the column, so consecutive threads in a
  warp read contiguous global-memory addresses.
- **Borders**: sizes not multiples of 16 handled with guards.
- **Bank conflicts**: shared tiles padded to `[TILE][TILE+1]`.

### GPU v2 — `gemm_reg_kernel` (register tiling)
- `128×128` block tile, `K` stepped in chunks of 8; **256 threads**, each computing
  an **8×8 micro-block of `C` held in registers**.
- Each value loaded from shared memory is reused 8× (an 8×8 outer product = 64 FMAs
  per 16 shared loads) → **arithmetic intensity way up**, which is the real lever.

## Performance analysis (occupancy → arithmetic intensity)

`ptxas` stats on `sm_75`:

| kernel | registers/thread | shared mem/block | occupancy | throughput |
|--------|------------------|------------------|-----------|------------|
| v1     | 36 (0 spill)     | 2176 B           | **100%**  | ~13% of peak |
| v2     | 128 (0 spill)    | 8192 B           | **50%**   | **~85% of peak** |

v1 already hits **100% theoretical occupancy**, yet only ~13% of FP32 peak — it is
**not occupancy-bound but arithmetic-intensity-bound** (each thread computes a
single element of `C`, too little reuse per memory access).

v2 deliberately **drops occupancy to 50%** (128 registers/thread caps it at 2
blocks/SM) but each thread now computes 64 outputs from registers — far more
compute per memory access. Net result: **6.6× faster**, ~85% of peak. The lesson:
once occupancy is saturated, the lever is **arithmetic intensity / ILP**, not more
occupancy — even at the cost of occupancy.

## Fused inference epilogue (GEMM + bias + activation)

A linear neural-network layer computes `Y = act(X·W + bias)`. Done naively that is
**two kernels**: the GEMM, then an element-wise pass for the bias and activation —
i.e. writing *then* re-reading the whole output through global memory. The fused
version computes the bias and activation **in the GEMM kernel's epilogue**, before
the first write: **one launch, one global-memory pass**. This is the optimization
cuBLASLt, CUTLASS and TensorRT perform.

- The activation is a **template parameter** of the kernel → selected at compile
  time, no branch in the inner loop.
- The activation math (ReLU, tanh-GELU) is **shared** between the CPU reference
  (correctness oracle) and the GPU kernel — one source of truth (`activation.hpp`).

Fusion saves one `M×N` global-memory pass plus a kernel launch, so its relative
gain scales as ~`TILE/K` plus launch overhead — it matters for **shallow layers
and small problems**, not big square GEMMs (where the GEMM dominates). Measured on
a GTX 1650 (GELU): 1.01× at 2048³, 1.20× at 2048²·64, up to 1.7× on tiny matrices.

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
ctest --test-dir build --output-on-failure   # adds GPU v1/v2 + fused-epilogue checks
./build/bench 2048                            # adds v1-vs-v2 and the fusion benchmark
```
Target GPU architectures: `75;86` (Turing + Ampere), tunable in `CMakeLists.txt`.

## Layout

```
include/gemm/   public headers (gemm_cpu.hpp, gemm_cuda.cuh, activation.hpp)
src/            implementations: gemm_cpu.cpp (naive + tiled), gemm_cuda.cu (kernels)
benchmarks/     GFLOP/s measurements, v1-vs-v2, fusion benchmark
tests/          correctness: tiled, CUDA v1/v2, and fused epilogue vs CPU oracle
```

## Roadmap

- [x] CPU: naive + OpenMP tiled
- [x] GPU v1: shared-memory tiled CUDA kernel
- [x] GPU v2: register tiling (8×8 micro-block per thread) — 6.6× over v1
- [x] Fused inference epilogue (GEMM + bias + activation)
- [ ] Dockerfile (reproducible CPU + CUDA build)
- [ ] GitHub Actions CI (build + CPU tests)
- [ ] Multi-device StarPU variant (CPU+GPU tasks)
