#include "gemm/gemm_cpu.hpp"
#ifdef USE_CUDA
#include "gemm/gemm_cuda.cuh"
#endif
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>

// Checks tiled (and CUDA if enabled) against the naive oracle.
int main() {
    const int M = 128, N = 96, K = 112; // not multiples of 16, to exercise the borders
    const float alpha = 1.3f, beta = 0.7f;

    std::vector<float> A(M * K), B(K * N), C0(M * N), C1(M * N);
#ifdef USE_CUDA
    std::vector<float> C2(M * N);
#endif

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);
    for (int i = 0; i < M * N; ++i) {
        float v = dist(rng);
        C0[i] = v; C1[i] = v;
#ifdef USE_CUDA
        C2[i] = v;
#endif
    }

    gemm::gemm_naive(M, N, K, alpha, A.data(), B.data(), beta, C0.data()); // oracle

    const double tol = 1e-3; // float: margin for rounding
    int rc = 0;

    // tiled vs naive
    gemm::gemm_tiled(M, N, K, alpha, A.data(), B.data(), beta, C1.data());
    double err_tiled = 0.0;
    for (int i = 0; i < M * N; ++i)
        err_tiled = std::max(err_tiled, (double)std::fabs(C0[i] - C1[i]));
    std::printf("Max abs error (tiled vs naive) : %.3e (tol %.1e)\n", err_tiled, tol);
    if (err_tiled > tol) { std::printf("FAIL (tiled)\n"); rc = 1; }

    // cuda vs naive
#ifdef USE_CUDA
    gemm::gemm_cuda(M, N, K, alpha, A.data(), B.data(), beta, C2.data());
    double err_cuda = 0.0;
    for (int i = 0; i < M * N; ++i)
        err_cuda = std::max(err_cuda, (double)std::fabs(C0[i] - C2[i]));
    std::printf("Max abs error (cuda  vs naive) : %.3e (tol %.1e)\n", err_cuda, tol);
    if (err_cuda > tol) { std::printf("FAIL (cuda)\n"); rc = 1; }
#endif

    std::printf(rc == 0 ? "OK\n" : "FAIL\n");
    return rc;
}
