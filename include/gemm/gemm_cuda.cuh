#pragma once

// GEMM on the GPU (CUDA). Same conventions as gemm_cpu.hpp:
// row-major, A (MxK), B (KxN), C (MxN), C = alpha*A*B + beta*C.
// No CUDA dependency here -> this header is also included from host code
// (tests, bench) compiled by g++/cl, without nvcc.
namespace gemm {

void gemm_cuda(int M, int N, int K, float alpha,
               const float* A, const float* B, float beta, float* C);

} // namespace gemm
