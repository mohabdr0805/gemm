#pragma once
#include "gemm/activation.hpp"

// GEMM on the GPU, C = alpha*A*B + beta*C, row-major like gemm_cpu.hpp. The
// header has no CUDA types, so host code compiles it without nvcc.
namespace gemm {

// v1 tiles A and B in shared memory.
void gemm_cuda(int M, int N, int K, float alpha,
               const float* A, const float* B, float beta, float* C);

// v2 keeps an 8x8 block of C in registers per thread.
void gemm_cuda_reg(int M, int N, int K, float alpha,
                   const float* A, const float* B, float beta, float* C);

// The fast kernel, v6 today. M and N must be multiples of 128 and K of 8,
// otherwise it falls back to v2.
void gemm_cuda_v3(int M, int N, int K, float alpha,
                  const float* A, const float* B, float beta, float* C);

// cuBLAS SGEMM with the same row-major convention, the benchmark's reference.
void gemm_cublas(int M, int N, int K, float alpha,
                 const float* A, const float* B, float beta, float* C);

// Times v1 to v6 and cuBLAS back to back on the device.
void benchmark_gemm_versions(int M, int N, int K);

// C = act(alpha*A*B + beta*C + bias[col]) in one launch. bias has length N, or
// is nullptr.
void gemm_bias_act_cuda(int M, int N, int K, float alpha,
                        const float* A, const float* B, float beta, float* C,
                        const float* bias, Activation act);

// Times the fused kernel against GEMM followed by a separate bias/activation pass.
void benchmark_fusion(int M, int N, int K, Activation act);

} // namespace gemm
