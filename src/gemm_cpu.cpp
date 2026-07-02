#include "gemm/gemm_cpu.hpp"
#include <algorithm>

namespace gemm {

// Naive reference: simple, slow, but obviously correct.
// BLAS semantics for beta == 0: C is write-only (never read), so garbage or NaN
// in C cannot leak into the result -- 0*NaN would be NaN, hence the branch.
void gemm_naive(int M, int N, int K, float alpha,
                const float* A, const float* B, float beta, float* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[(size_t)i * K + k] * B[(size_t)k * N + j];
            }
            const size_t idx = (size_t)i * N + j;
            float out = alpha * acc;
            if (beta != 0.0f) out += beta * C[idx];
            C[idx] = out;
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

    // 1) Pre-scale C by beta (kept separate to simplify the rest). beta == 0
    //    overwrites instead of scaling: BLAS treats C as write-only then, and
    //    0 * NaN would leak a NaN out of uninitialized C.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            const size_t idx = (size_t)i * N + j;
            C[idx] = (beta == 0.0f) ? 0.0f : beta * C[idx];
        }

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
                        const float a = alpha * A[(size_t)i * K + k];
                        for (int j = jj; j < j_max; ++j) {
                            C[(size_t)i * N + j] += a * B[(size_t)k * N + j];
                        }
                    }
                }
            }
        }
    }
}

// CPU reference for the fused epilogue (oracle for the GPU kernel).
// Naively computes C = act( alpha*A*B + beta*C + bias[col] ). Slow but obviously
// correct: used to validate the fused GPU kernel.
void gemm_bias_act_ref(int M, int N, int K, float alpha,
                       const float* A, const float* B, float beta, float* C,
                       const float* bias, Activation act) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[(size_t)i * K + k] * B[(size_t)k * N + j];
            }
            const size_t idx = (size_t)i * N + j;
            float v = alpha * acc;
            if (beta != 0.0f) v += beta * C[idx]; // beta==0: C write-only (BLAS)
            if (bias) v += bias[j];               // per-output-column bias
            C[idx] = apply_act(act, v);           // activation last
        }
    }
}

} // namespace gemm
