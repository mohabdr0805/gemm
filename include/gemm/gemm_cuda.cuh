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

// Device-timed comparison of v1 (shared-memory tiled) vs v2 (register tiled).
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
