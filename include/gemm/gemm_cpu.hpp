#pragma once
#include <cstddef>

// All matrices are stored row-major.
//   A : M x K     B : K x N     C : M x N
// Computes: C = alpha * A * B + beta * C
namespace gemm {

// Naive reference (triple loop). Used as the correctness oracle.
void gemm_naive(int M, int N, int K, float alpha,
                const float* A, const float* B, float beta, float* C);

// Optimized version: cache tiling + OpenMP parallelization.
void gemm_tiled(int M, int N, int K, float alpha,
                const float* A, const float* B, float beta, float* C);

} // namespace gemm
