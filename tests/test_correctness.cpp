#include "gemm/gemm_cpu.hpp"
#ifdef USE_CUDA
#include "gemm/gemm_cuda.cuh"
#endif
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>

// Checks tiled (and CUDA + the fused epilogue if enabled) against the naive oracle.
int main() {
    const int M = 128, N = 96, K = 112; // not multiples of 16, to exercise the borders
    const float alpha = 1.3f, beta = 0.7f;

    std::vector<float> A(M * K), B(K * N), C0(M * N), C1(M * N);
#ifdef USE_CUDA
    std::vector<float> C2(M * N), C3(M * N);
#endif

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);
    for (int i = 0; i < M * N; ++i) {
        float v = dist(rng);
        C0[i] = v; C1[i] = v;
#ifdef USE_CUDA
        C2[i] = v; C3[i] = v;
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

#ifdef USE_CUDA
    // cuda vs naive
    gemm::gemm_cuda(M, N, K, alpha, A.data(), B.data(), beta, C2.data());
    double err_cuda = 0.0;
    for (int i = 0; i < M * N; ++i)
        err_cuda = std::max(err_cuda, (double)std::fabs(C0[i] - C2[i]));
    std::printf("Max abs error (cuda  vs naive) : %.3e (tol %.1e)\n", err_cuda, tol);
    if (err_cuda > tol) { std::printf("FAIL (cuda)\n"); rc = 1; }

    // register-tiled v2 vs naive
    gemm::gemm_cuda_reg(M, N, K, alpha, A.data(), B.data(), beta, C3.data());
    double err_reg = 0.0;
    for (int i = 0; i < M * N; ++i)
        err_reg = std::max(err_reg, (double)std::fabs(C0[i] - C3[i]));
    std::printf("Max abs error (reg   vs naive) : %.3e (tol %.1e)\n", err_reg, tol);
    if (err_reg > tol) { std::printf("FAIL (reg)\n"); rc = 1; }

    // fused epilogue (GEMM + bias + activation) vs CPU oracle, for ReLU and GELU
    std::vector<float> bias(N);
    for (auto& x : bias) x = dist(rng);

    auto check_fused = [&](gemm::Activation act, const char* nm) {
        std::vector<float> Cinit(M * N);
        for (int i = 0; i < M * N; ++i) Cinit[i] = dist(rng);
        std::vector<float> Cref = Cinit, Cgpu = Cinit;
        gemm::gemm_bias_act_ref(M, N, K, alpha, A.data(), B.data(), beta, Cref.data(), bias.data(), act);
        gemm::gemm_bias_act_cuda(M, N, K, alpha, A.data(), B.data(), beta, Cgpu.data(), bias.data(), act);
        double err = 0.0;
        for (int i = 0; i < M * N; ++i)
            err = std::max(err, (double)std::fabs(Cref[i] - Cgpu[i]));
        std::printf("Max abs error (fused %-4s gpu vs ref) : %.3e (tol %.1e)\n", nm, err, tol);
        if (err > tol) { std::printf("FAIL (fused %s)\n", nm); rc = 1; }
    };
    check_fused(gemm::Activation::ReLU, "ReLU");
    check_fused(gemm::Activation::GELU, "GELU");
#endif

    std::printf(rc == 0 ? "OK\n" : "FAIL\n");
    return rc;
}
