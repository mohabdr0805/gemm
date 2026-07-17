#include "gemm/gemm_cpu.hpp"
#ifdef USE_CUDA
#include "gemm/gemm_cuda.cuh"
#endif
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <limits>

// Checks tiled (and CUDA + the fused epilogue if enabled) against the naive oracle.
int main() {
    // Deliberately NOT multiples of 16 (v1 tile), 128 (v2 block tile) or 8 (v2 BK
    // step), so every border guard -- row < M, col < N, and the K tail -- actually
    // fires. (The old 128/96/112 were all multiples of 16: the v1 guards never
    // took the out-of-bounds branch.)
    const int M = 125, N = 93, K = 115;
    const float alpha = 1.3f, beta = 0.7f;

    std::vector<float> A(M * K), B(K * N), C0(M * N), C1(M * N);
#ifdef USE_CUDA
    std::vector<float> C2(M * N), C3(M * N), C4(M * N), Cseed(M * N);
#endif

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);
    for (int i = 0; i < M * N; ++i) {
        float v = dist(rng);
        C0[i] = v; C1[i] = v;
#ifdef USE_CUDA
        C2[i] = v; C3[i] = v; C4[i] = v; Cseed[i] = v;
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

    // BLAS beta==0 semantics (CPU): C is write-only, so NaN-filled C must not
    // leak into the result -- 0*NaN would be NaN, the implementations must skip
    // the read instead of multiplying by zero.
    {
        const float qnan = std::numeric_limits<float>::quiet_NaN();
        std::vector<float> Cn(M * N, qnan), Ct(M * N, qnan);
        gemm::gemm_naive(M, N, K, alpha, A.data(), B.data(), 0.0f, Cn.data());
        gemm::gemm_tiled(M, N, K, alpha, A.data(), B.data(), 0.0f, Ct.data());
        double e = 0.0; bool finite = true;
        for (int i = 0; i < M * N; ++i) {
            if (!std::isfinite(Cn[i]) || !std::isfinite(Ct[i])) finite = false;
            e = std::max(e, (double)std::fabs(Cn[i] - Ct[i]));
        }
        std::printf("beta=0, NaN-filled C (tiled vs naive)    : finite=%s, err %.3e\n",
                    finite ? "yes" : "NO", e);
        if (!finite || e > tol) { std::printf("FAIL (beta0 cpu)\n"); rc = 1; }
    }

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

    // v3 (float4 + double buffering). Its fast path needs aligned sizes; the main
    // M/N/K here is unaligned, so this call exercises v3's fall-back-to-v2 path.
    // Fresh buffer seeded from the pristine init (C3 was overwritten above), so the
    // beta term acts on the original C, not on the reg result.
    std::vector<float> Cfb = Cseed;
    gemm::gemm_cuda_v3(M, N, K, alpha, A.data(), B.data(), beta, Cfb.data());
    double err_v3fb = 0.0;
    for (int i = 0; i < M * N; ++i)
        err_v3fb = std::max(err_v3fb, (double)std::fabs(C0[i] - Cfb[i]));
    std::printf("Max abs error (v3 fb vs naive) : %.3e (tol %.1e)\n", err_v3fb, tol);
    if (err_v3fb > tol) { std::printf("FAIL (v3 fallback)\n"); rc = 1; }

    // v3 fast path (float4 + double buffering) on aligned shapes, with the
    // file-level alpha/beta and the NaN beta=0 write-only contract. Three K depths
    // exercise the double-buffer kernel's distinct paths: 256 (32 K-tiles, even),
    // 24 (3 tiles, odd -> the buffer index ends flipped), and 8 (a single tile ->
    // prologue only, has_next never fires).
    auto check_v3_aligned = [&](int m, int n, int k) {
        std::vector<float> AA(m * k), BB(k * n), Cref(m * n), Cgpu(m * n);
        for (auto& x : AA) x = dist(rng);
        for (auto& x : BB) x = dist(rng);
        for (int i = 0; i < m * n; ++i) { float v = dist(rng); Cref[i] = v; Cgpu[i] = v; }
        gemm::gemm_naive  (m, n, k, alpha, AA.data(), BB.data(), beta, Cref.data());
        gemm::gemm_cuda_v3(m, n, k, alpha, AA.data(), BB.data(), beta, Cgpu.data());
        double e = 0.0;
        for (int i = 0; i < m * n; ++i) e = std::max(e, (double)std::fabs(Cref[i] - Cgpu[i]));
        std::printf("Max abs error (v3 aligned %dx%dx%d vs naive) : %.3e (tol %.1e)\n", m, n, k, e, tol);
        if (e > tol) { std::printf("FAIL (v3 aligned %dx%dx%d)\n", m, n, k); rc = 1; }

        const float qnan = std::numeric_limits<float>::quiet_NaN();
        std::vector<float> Cn(m * n, qnan), Cv(m * n, qnan);
        gemm::gemm_naive  (m, n, k, 1.0f, AA.data(), BB.data(), 0.0f, Cn.data());
        gemm::gemm_cuda_v3(m, n, k, 1.0f, AA.data(), BB.data(), 0.0f, Cv.data());
        double e2 = 0.0; bool fin = true;
        for (int i = 0; i < m * n; ++i) {
            if (!std::isfinite(Cv[i])) fin = false;
            e2 = std::max(e2, (double)std::fabs(Cn[i] - Cv[i]));
        }
        std::printf("beta=0, NaN-filled C (v3 aligned %dx%dx%d)   : finite=%s, err %.3e\n", m, n, k, fin ? "yes" : "NO", e2);
        if (!fin || e2 > tol) { std::printf("FAIL (v3 beta0 %dx%dx%d)\n", m, n, k); rc = 1; }
    };
    check_v3_aligned(256, 256, 256); // even K-tile count
    check_v3_aligned(128, 128,  24); // odd K-tile count -> buffer parity
    check_v3_aligned(128, 128,   8); // single K-tile -> prologue only, no swap
    // 144 blocks >= 2 blocks/SM x 68 SMs: crosses the wrapper's grid-dispatch
    // threshold, so this is the only shape here that exercises the
    // __launch_bounds__(256,2) build. K=24 keeps the naive oracle cheap.
    check_v3_aligned(1536, 1536, 24);

    // cuBLAS baseline vs naive: validates the row-major <-> column-major swap
    // (C^T = B^T A^T) before the benchmark quotes any "% of cuBLAS" number.
    gemm::gemm_cublas(M, N, K, alpha, A.data(), B.data(), beta, C4.data());
    double err_cb = 0.0;
    for (int i = 0; i < M * N; ++i)
        err_cb = std::max(err_cb, (double)std::fabs(C0[i] - C4[i]));
    std::printf("Max abs error (cublas vs naive): %.3e (tol %.1e)\n", err_cb, tol);
    if (err_cb > tol) { std::printf("FAIL (cublas)\n"); rc = 1; }

    // fused epilogue (GEMM + bias + activation) vs CPU oracle. ReLU/GELU with a
    // bias, plus the documented-but-otherwise-untested bias=nullptr + None combo.
    std::vector<float> bias(N);
    for (auto& x : bias) x = dist(rng);

    auto check_fused = [&](gemm::Activation act, const char* nm, const float* b) {
        std::vector<float> Cinit(M * N);
        for (int i = 0; i < M * N; ++i) Cinit[i] = dist(rng);
        std::vector<float> Cref = Cinit, Cgpu = Cinit;
        gemm::gemm_bias_act_ref(M, N, K, alpha, A.data(), B.data(), beta, Cref.data(), b, act);
        gemm::gemm_bias_act_cuda(M, N, K, alpha, A.data(), B.data(), beta, Cgpu.data(), b, act);
        double err = 0.0;
        for (int i = 0; i < M * N; ++i)
            err = std::max(err, (double)std::fabs(Cref[i] - Cgpu[i]));
        std::printf("Max abs error (fused %-4s gpu vs ref) : %.3e (tol %.1e)\n", nm, err, tol);
        if (err > tol) { std::printf("FAIL (fused %s)\n", nm); rc = 1; }
    };
    check_fused(gemm::Activation::ReLU, "ReLU", bias.data());
    check_fused(gemm::Activation::GELU, "GELU", bias.data());
    check_fused(gemm::Activation::None, "None", nullptr); // pure GEMM through the fused path

    // BLAS beta==0 semantics on the GPU: C is write-only, so NaN-filled C must
    // not leak into the result (0*NaN would). Also proves the wrappers skip the
    // C upload without breaking anything.
    {
        const float qnan = std::numeric_limits<float>::quiet_NaN();
        std::vector<float> Cn(M * N, qnan), Cg(M * N, qnan), Cr(M * N, qnan);
        gemm::gemm_naive   (M, N, K, alpha, A.data(), B.data(), 0.0f, Cn.data());
        gemm::gemm_cuda    (M, N, K, alpha, A.data(), B.data(), 0.0f, Cg.data());
        gemm::gemm_cuda_reg(M, N, K, alpha, A.data(), B.data(), 0.0f, Cr.data());
        double e1 = 0.0, e2 = 0.0; bool finite = true;
        for (int i = 0; i < M * N; ++i) {
            if (!std::isfinite(Cn[i]) || !std::isfinite(Cg[i]) || !std::isfinite(Cr[i])) finite = false;
            e1 = std::max(e1, (double)std::fabs(Cn[i] - Cg[i]));
            e2 = std::max(e2, (double)std::fabs(Cn[i] - Cr[i]));
        }
        std::printf("beta=0, NaN-filled C (cuda/reg vs naive) : finite=%s, err %.3e / %.3e\n",
                    finite ? "yes" : "NO", e1, e2);
        if (!finite || e1 > tol || e2 > tol) { std::printf("FAIL (beta0 gpu)\n"); rc = 1; }
    }
#endif

    std::printf(rc == 0 ? "OK\n" : "FAIL\n");
    return rc;
}
