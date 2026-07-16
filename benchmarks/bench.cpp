#include "gemm/gemm_cpu.hpp"
#ifdef USE_CUDA
#include "gemm/gemm_cuda.cuh"
#include "gemm/softmax_cuda.cuh"
#include "gemm/attention_cuda.cuh"
#endif
#include <vector>
#include <functional>
#include <cstdlib>
#include <random>
#include <chrono>
#include <cstdio>

using clk = std::chrono::high_resolution_clock;

static double seconds(std::function<void()> f) {
    auto t0 = clk::now();
    f();
    auto t1 = clk::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 1024; // n x n matrices
    const float alpha = 1.0f, beta = 0.0f;

    std::vector<float> A(n * n), B(n * n), C(n * n, 0.0f);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    const double flops = 2.0 * n * n * n; // 1 mul + 1 add per product element

    // (naive gets very slow past ~1024, skip it when n is large)
    if (n <= 1024) {
        double t = seconds([&]{ gemm::gemm_naive(n, n, n, alpha, A.data(), B.data(), beta, C.data()); });
        std::printf("naive : %8.3f s   %7.2f GFLOP/s\n", t, flops / t / 1e9);
    } else {
        std::printf("naive : (skipped, too slow for n > 1024)\n");
    }

    // Warm-up: the first OpenMP region pays the thread-pool spin-up, which would
    // otherwise be billed to the timed run (same discipline as the device benches).
    gemm::gemm_tiled(n, n, n, alpha, A.data(), B.data(), beta, C.data());
    double t = seconds([&]{ gemm::gemm_tiled(n, n, n, alpha, A.data(), B.data(), beta, C.data()); });
    std::printf("tiled : %8.3f s   %7.2f GFLOP/s\n", t, flops / t / 1e9);

#ifdef USE_CUDA
    gemm::gemm_cuda(n, n, n, alpha, A.data(), B.data(), beta, C.data()); // warm-up (context init)
    double tc = seconds([&]{ gemm::gemm_cuda(n, n, n, alpha, A.data(), B.data(), beta, C.data()); });
    // whole-wrapper time: includes cudaMalloc/free and H2D/D2H, not just the kernel
    std::printf("cuda  : %8.3f s   %7.2f GFLOP/s\n", tc, flops / tc / 1e9);

    // GEMM kernels vs the vendor baseline, device timing. The "% of cuBLAS"
    // ratios come from the same run, so they hold even when clocks swing.
    std::printf("\n[GEMM v1 vs v2 vs cuBLAS]  device timing (no transfers)\n");
    gemm::benchmark_gemm_versions(n, n, n);

    // Fused inference epilogue: fusion vs two-pass (pure device timing, no transfers).
    // The gain grows as K shrinks (the saved epilogue pass is a bigger share) and
    // for smaller problems (launch overhead matters) -> a square GEMM shows ~1x.
    std::printf("\n[Fused GEMM+bias+GELU]  device timing, fusion vs two-pass\n");
    const int shapes[][3] = { {n, n, n}, {n, n, 64}, {256, 256, 256} };
    for (const auto& s : shapes) {
        std::printf("M=%d N=%d K=%d:\n", s[0], s[1], s[2]);
        gemm::benchmark_fusion(s[0], s[1], s[2], gemm::Activation::GELU);
    }

    // Softmax (row-wise): memory-bound, so the figure of merit is bandwidth.
    std::printf("\n[Softmax row-wise]  device timing (no transfers)\n");
    gemm::benchmark_softmax(n, n);
    gemm::benchmark_softmax(n, 1024);

    // Attention (FlashAttention-style): fused scores + softmax + P*V in one pass,
    // never materializing the n x n score matrix. v1 (one query row per block) vs
    // v2 (query tiling: K/V reused across the block's rows). Both head dims from
    // the README table: 64 (register-resident) and 128 (spills, smaller gain).
    std::printf("\n[Attention FlashAttention-style]  v1 vs v2, device timing (no transfers)\n");
    for (int d : { 64, 128 }) {
        std::printf("d=%d:\n", d);
        gemm::benchmark_attention_versions(n, n, d, /*causal=*/false);
        gemm::benchmark_attention_versions(n, n, d, /*causal=*/true);
    }

    // v2 (one thread per row) vs FA-2 (one warp per row, head dim split across
    // lanes). FA-2 removes v2's d=128 register spill; the win is largest at d=128.
    std::printf("\n[Attention v2 vs FA-2 warp-partitioned]  device timing (no transfers)\n");
    for (int d : { 64, 128 }) {
        std::printf("d=%d:\n", d);
        gemm::benchmark_attention_fa2(n, n, d, /*causal=*/false);
    }
#endif

    return 0;
}
