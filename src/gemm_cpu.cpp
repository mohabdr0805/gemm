#include "gemm/gemm_cpu.hpp"
#include <algorithm>

namespace gemm {

// Naive reference: simple, slow, but obviously correct.
void gemm_naive(int M, int N, int K, float alpha,
                const float* A, const float* B, float beta, float* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * acc + beta * C[i * N + j];
        }
    }
}

// Optimized: cache blocking + OpenMP.
// Idea: split C into BS x BS tiles. Each thread owns a full (ii,jj) tile (all its
// kk contributions), so no thread writes into another's tile -> no data race, no
// reduction needed. The inner i-k-j order keeps B and C traversed row-wise
// (contiguous, cache-friendly).
void gemm_tiled(int M, int N, int K, float alpha,
                const float* A, const float* B, float beta, float* C) {
    constexpr int BS = 64; // tile size (tune for L1/L2)

    // 1) Pre-scale C by beta (kept separate to simplify the rest).
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[i * N + j] *= beta;

    // 2) Blocked product. collapse(2) spreads the (ii,jj) tiles over the threads.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < M; ii += BS) {
        for (int jj = 0; jj < N; jj += BS) {
            const int i_max = std::min(ii + BS, M);
            const int j_max = std::min(jj + BS, N);
            for (int kk = 0; kk < K; kk += BS) {
                const int k_max = std::min(kk + BS, K);
                for (int i = ii; i < i_max; ++i) {
                    for (int k = kk; k < k_max; ++k) {
                        const float a = alpha * A[i * K + k];
                        for (int j = jj; j < j_max; ++j) {
                            C[i * N + j] += a * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

} // namespace gemm
