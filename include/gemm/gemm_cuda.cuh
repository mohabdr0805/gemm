#pragma once
#include "gemm/activation.hpp"

// GEMM on the GPU (CUDA). Same conventions as gemm_cpu.hpp:
// row-major, A (MxK), B (KxN), C (MxN), C = alpha*A*B + beta*C.
// No CUDA dependency here -> this header is also included from host code
// (tests, bench) compiled by g++/cl, without nvcc.
namespace gemm {

// Base GEMM, v1 (shared-memory tiled kernel).
void gemm_cuda(int M, int N, int K, float alpha,
               const float* A, const float* B, float beta, float* C);

// GEMM v2 (register tiled): each thread computes an 8x8 micro-block of C in
// registers -> higher arithmetic intensity. Same convention, same result.
void gemm_cuda_reg(int M, int N, int K, float alpha,
                   const float* A, const float* B, float beta, float* C);

// GEMM v3+ (currently ships v5): v2 + float4 vectorized global loads, a
// transposed As tile, double buffering (the next K-tile's loads are in flight
// while the current tile computes), and conflict-free Bs reads (each thread's
// 8 output columns split into two groups of 4, half a tile apart, so a warp's
// shared reads cover all 32 banks). Picks its __launch_bounds__ build from the
// grid size. Fast path for M,N % 128 == 0 and K % 8 == 0; other shapes fall
// back to v2. Same result.
void gemm_cuda_v3(int M, int N, int K, float alpha,
                  const float* A, const float* B, float beta, float* C);

// cuBLAS SGEMM baseline, same row-major convention (internally computes
// C^T = B^T * A^T in cuBLAS's column-major world -- the standard swap). Same
// beta==0 write-only semantics. Validated against the naive oracle like every
// other kernel; used by the benchmark as the vendor reference.
void gemm_cublas(int M, int N, int K, float alpha,
                 const float* A, const float* B, float beta, float* C);

// Device-timed comparison of the whole ladder -- v1 (shared-memory tiled), v2
// (register tiled), v3 (float4, then float4 + double buffering), v4 (the
// launch_bounds build of v3), v5 (conflict-free Bs reads, both builds) -- and
// cuBLAS SGEMM, all back-to-back in the same power state. v3..v5 are timed only
// on the aligned fast path.
void benchmark_gemm_versions(int M, int N, int K);

// Fused inference epilogue: C = act( alpha*A*B + beta*C + bias[col] ).
// The bias add and the activation are computed in the GEMM kernel's epilogue ->
// a single global-memory pass, a single launch (vs GEMM then a separate
// element-wise kernel). bias: length-N vector, or nullptr.
void gemm_bias_act_cuda(int M, int N, int K, float alpha,
                        const float* A, const float* B, float beta, float* C,
                        const float* bias, Activation act);

// Measures (device timing, no transfers) the gain from fusion: compares the
// fused kernel to the two-pass path (GEMM + separate bias/activation kernel).
// Prints both times and the speedup.
void benchmark_fusion(int M, int N, int K, Activation act);

} // namespace gemm
