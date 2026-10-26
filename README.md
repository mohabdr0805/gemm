# gemm

![CI](https://github.com/mohabdr0805/gemm/actions/workflows/ci.yml/badge.svg)

> **TL;DR**: hand-written CUDA SGEMM at **106.7% of cuBLAS SGEMM** on the
> geometric mean of 25 square aligned sizes from n=1024 to 4096, ahead of it on
> 21, with a Stream-K kept in [`experiments/stream-k/`](experiments/stream-k/).
> Without it the library kernel is at 102.4%, ahead on 15, and its low end is
> wave quantization, modelled below. Plus a FlashAttention-style attention kernel, up to ~8.4× from query
> tiling. Every step was chosen from a Nsight Compute profile, every kernel is
> checked against a CPU oracle, and device figures come from an RTX 3080, ~200 ms
> of work per kernel, cuBLAS timed in the same run.

Optimized GEMM (`C = α·A·B + β·C`) in C++/OpenMP and CUDA, single precision,
row-major. The repo goes from a naive reference to tuned CPU and GPU kernels and
measures each step; the full write-up, with the profiles, is in [`docs/`](docs/).

## Results

Measured on an RTX 3080 10 GB (Ampere, `sm_86`) with CUDA 13.0, inputs rotated
over twice the L2 so no iteration reads what the last one left in cache
(protocol: [`docs/methodology.md`](docs/methodology.md)). GPU kernels as a share
of cuBLAS SGEMM:

| n    | v1  | v2  | v3   | v4   | v5   | v6       |
|------|-----|-----|------|------|------|----------|
| 1024 | 11% | 64% | 104% | 103% | 116% | **122%** |
| 2048 | 7%  | 60% | 77%  | 79%  | 91%  | **93%**  |
| 3072 | 8%  | 64% | 83%  | 86%  | 98%  | **101%** |
| 4096 | 7%  | 60% | 77%  | 79%  | 90%  | **93%**  |

Four sizes are the round numbers everyone quotes, and they hide how much the
ratio moves with shape. Over all 25 sizes, in steps of 128:

|                                     | geometric mean | median | range       | above cuBLAS |
|-------------------------------------|----------------|--------|-------------|--------------|
| v6, clocks locked at 1710 MHz       | 99.8%          | 100.6% | 78.1–122.0% | 13 of 25     |
| **v7**, default clocks, two runs    | **102.4%**     | 101.8% | 85.5–123.8% | **15 of 25** |

At default clocks the card runs these kernels at its 370 W limit, and v6 timed in
the same runs gives 100.4 and 99.9%. The v7 figures are the mean of the two runs,
size by size.

With 68 SMs, a grid takes `ceil(blocks/68)` rounds for `blocks/68` of work, and
[a model](docs/wave-quantization.md) whose only free parameter is the plateau
predicts 24 of the 25 sizes within 3%. The 78% is n=1152, 1.19 blocks per SM
paid at the price of 2.

## From v1 to v7

Each step in detail is in [`docs/gemm.md`](docs/gemm.md).

| version | the change | the result |
|---------|------------|------------|
| v1 | shared-memory tiling | 7–11% of cuBLAS |
| v2 | an 8×8 register tile per thread | 6–8.5× over v1, 60–64% of cuBLAS |
| v3 | `float4` loads and double buffering | 77–83% at n ≥ 2048, global-load stalls from 16% to 2% |
| v4 | `__launch_bounds__(256, 2)` | back to 128 registers and two blocks per SM, +2–3 points at n ≥ 2048 |
| v5 | each thread's columns split into two groups, half a tile apart | bank conflicts from 268M to 0 as predicted, +11 to +13 points |
| v6 | a warp tier between the block tile and the thread tile | a third fewer shared-memory wavefronts at the same register count, +4% on average |
| + | a second tile, 256×128, picked per size by the quantization model | the faster of the two on 13 of 13 sizes, without timing them |
| v7 | both tiles staged with `cp.async`, A transposed by 4-byte copies, BK=16 on 3 stages, the next fragments read a step ahead, cuBLAS's FFMA order | from 93.7% to 98.8% at n=4096, +2 points on the 25-size mean, in the same runs |

A Stream-K on v7's last wave reaches 106.7% on the mean, ahead on 21 of 25, and
103.0% at n=4096, but it needs a workspace and a second kernel on a second
stream, so it stays out of the library, in
[`experiments/stream-k/`](experiments/stream-k/). Split-K does not ship either. `cp.async` on B alone costs 6.5%, which is why v7 copies
both operands.

## Attention

Each kernel in detail is in [`docs/attention.md`](docs/attention.md).

- **v1**: FlashAttention-style, with an online softmax, no N×N matrix and an
  optional causal mask.
- **v2**: query tiling, each K/V tile reused by 64 query rows, up to ~8.4× over
  v1.
- **FA-2-style**: the head dimension split across a warp. It removes v2's
  register spill at d=128 and runs 1.6–6.2× over v2 there, while at d=64 and
  long sequences v2 stays ahead.
- **v3**: a register micro-tile per thread, so the exponential stays at one per
  (row, key). 1.25× to 2.99× over the best of the three above.

Against PyTorch on the same buffers, v3 beats its FP32 path, which materializes
the N×N matrix, on 5 of 8 shapes, and runs 2.2× to 5.5× behind cuDNN's FP16
kernel on tensor cores.

## Also in the repo

- A CPU version, cache-tiled with OpenMP and AVX2, ~214× the naive reference at
  n=1024 and ~200 GFLOP/s from n=2048 (i7-12700F).
- A fused GEMM + bias + activation epilogue, ~1.1× where the saved pass is a
  real share of the work.
- A row-wise softmax, and an online variant 1.69× faster once the working set
  leaves the L2.

## Scope

FP32 on CUDA cores throughout (no tensor cores). The cuBLAS baseline runs in its
default math mode (plain `cublasSgemm`, no `cublasSetMathMode`), so TF32 is
disabled and the comparison is like-for-like. Cross-check: cuBLAS matches the
FP32 CPU oracle to ~1e-5, where TF32's 10-bit mantissa would give errors around
1e-3. The baseline is cuBLAS's default heuristic kernel choice via
`cublasSgemm`, not an exhaustive `cublasLt` algorithm search; that is the field
standard for this comparison, and it is what sets the ceiling. One card
(RTX 3080, `sm_86`), one OS; the ratios can move on Ada/Hopper.

The figures are for square aligned shapes. Any shape with `M % 128 == 0`,
`N % 128 == 0` and `K % 8 == 0` takes the fast path; anything else falls back to
the register-tiled v2 at around 61% of cuBLAS.

## Next

- A v7 for the remaining `barrier` and shared-latency stalls.
- The fused epilogue on the register-tiled kernel, which currently sits on v1's
  tiling.

## Build & run

CPU (Linux/GCC, or Windows/MSVC. On MSVC the build forces `/openmp:experimental`
for one reason: it is the only mode that accepts the `omp simd` directive the
tiled GEMM's inner loop needs to vectorize. It does not support `collapse`, so the
tile loops are fused by hand instead; see [`docs/gemm.md`](docs/gemm.md)):
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

The attention baseline also needs PyTorch with CUDA, and loads the `gemm_capi`
library the CUDA build produces:
```bash
python benchmarks/attention_baseline.py
```

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
benchmarks/     GFLOP/s, v1/v2/v3 vs cuBLAS, fusion, softmax, attention benchmarks;
                attention_baseline.py, the attention kernels against PyTorch
tests/          correctness vs the CPU oracle (gemm, softmax, attention)
docs/           the detailed write-up: methodology, GEMM steps, wave
                quantization, attention
experiments/    stream-k, kept out of the library
```
